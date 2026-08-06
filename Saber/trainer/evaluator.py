import torch
import numpy as np
import logging
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader
from Saber.trainer.metrics import compute_retrieval_metrics

logger = logging.getLogger("saber")

class Evaluator:
    """
    Evaluates the retrieval performance of the trained model.
    Supports same-modal (S1->S1, S2->S2) and cross-modal (S1->S2) retrieval directions.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        config: Dict[str, Any]
    ) -> None:
        """
        Args:
            model: The trained REJEPA model.
            dataloader: DataLoader with evaluation images (returns unaugmented 'image' inputs).
            device: Computing device (CPU or CUDA).
            config: Configurations dictionary.
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.config = config

    def extract_all_embeddings(self) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Runs inference over the entire dataloader to extract embeddings, labels, and filenames.
        (Same-modal extraction)
        """
        self.model.eval()
        embeddings_list = []
        labels_list = []
        filenames_list = []

        logger.info("Extracting embeddings for evaluation...")
        num_batches = len(self.dataloader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                images = batch.get("image", batch.get("image1")).to(self.device)
                
                # Auto-resize on GPU to prevent CPU resize bottleneck
                if images.shape[-1] != 224 or images.shape[-2] != 224:
                    import torch.nn.functional as F
                    images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
                    
                labels = batch["label"]
                names = batch["name"]

                # Extract L2-normalized embeddings
                embeds = self.model.get_retrieval_embedding(images)
                
                embeddings_list.append(embeds.cpu().numpy())
                labels_list.append(labels.numpy())
                filenames_list.extend(names)
                
                if batch_idx % 100 == 0:
                    logger.info(f"Extraction Batch [{batch_idx}/{num_batches}] completed.")

        all_embeddings = np.concatenate(embeddings_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)

        return all_embeddings, all_labels, filenames_list

    def evaluate(self, top_k: int = 5) -> Dict[str, Any]:
        """
        Evaluates retrieval metrics on the dataloader.
        Supports same-modal and cross-modal retrieval divisions.
        """
        num_samples = len(self.dataloader.dataset)
        if num_samples < 5:
            raise ValueError(f"Dataset has only {num_samples} samples, which is too small for retrieval split.")

        # Use randomized splitting with fixed seed to select held-out query items
        rng = np.random.RandomState(42)
        shuffled_indices = rng.permutation(num_samples)
        query_size = max(1, num_samples // 5)
        query_indices = np.sort(shuffled_indices[:query_size])
        
        eval_split = str(self.config.dataset.get("split", "all")).lower()
        if eval_split == "test":
            gallery_indices = np.sort(shuffled_indices[query_size:])
        else:
            gallery_indices = np.arange(num_samples)


        dataset_cfg = self.config.get("dataset", {}) if isinstance(self.config, dict) else getattr(self.config, "dataset", {})
        modality = str(dataset_cfg.get("modality", "both") if isinstance(dataset_cfg, dict) else getattr(dataset_cfg, "modality", "both")).lower()
        is_cross_modal = (modality == "both" or getattr(self.dataloader.dataset, "modality", "s2") == "both")

        if not is_cross_modal:
            # Same-modal path
            embeddings, labels, names = self.extract_all_embeddings()
            query_embeds = embeddings[query_indices]
            query_labels = labels[query_indices]
            query_names = np.array([names[i] for i in query_indices])
            
            gallery_embeds = embeddings[gallery_indices]
            gallery_labels = labels[gallery_indices]
            gallery_names = np.array([names[i] for i in gallery_indices])
        else:
            logger.info("Extracting bimodal embeddings for cross-modal evaluation (S1 query, S2 gallery)...")
            self.model.eval()
            
            s1_embeds_list = []
            s2_embeds_list = []
            labels_list = []
            filenames_list = []
            
            s1_channels = getattr(self.model, "s1_channels", 2)
            num_batches = len(self.dataloader)
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.dataloader):
                    images = batch.get("image", batch.get("image1")).to(self.device)

                    # Auto-resize on GPU to prevent CPU resize bottleneck
                    if images.shape[-1] != 224 or images.shape[-2] != 224:
                        import torch.nn.functional as F
                        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
                        
                    labels = batch["label"]
                    names = batch["name"]
                    
                    x_s1 = images[:, :s1_channels, :, :]
                    x_s2 = images[:, s1_channels:, :, :]
                    
                    embed_s1 = self.model.get_retrieval_embedding(x_s1)
                    embed_s2 = self.model.get_retrieval_embedding(x_s2)
                    
                    s1_embeds_list.append(embed_s1.cpu().numpy())
                    s2_embeds_list.append(embed_s2.cpu().numpy())
                    labels_list.append(labels.numpy())
                    filenames_list.extend(names)
                    
                    if batch_idx % 100 == 0:
                        logger.info(f"Bimodal Extraction Batch [{batch_idx}/{num_batches}] completed.")
                    
            all_s1_embeds = np.concatenate(s1_embeds_list, axis=0)
            all_s2_embeds = np.concatenate(s2_embeds_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            names = filenames_list
            
            direction = self.config.get("retrieval", {}).get("direction", "s1_to_s2").lower() if isinstance(self.config, dict) else getattr(self.config.retrieval, "direction", "s1_to_s2").lower()
            if direction == "s2_to_s1":
                logger.info("Setting up retrieval direction: S2 (query) -> S1 (gallery)")
                query_embeds = all_s2_embeds[query_indices]
                gallery_embeds = all_s1_embeds[gallery_indices]
            else:
                logger.info("Setting up retrieval direction: S1 (query) -> S2 (gallery)")
                query_embeds = all_s1_embeds[query_indices]
                gallery_embeds = all_s2_embeds[gallery_indices]
            
            query_labels = labels[query_indices]
            gallery_labels = labels[gallery_indices]
            query_names = np.array([names[i] for i in query_indices])
            gallery_names = np.array([names[i] for i in gallery_indices])
            
            # Synthesize final embeddings array for FAISS building compatibility
            embeddings = np.zeros((num_samples, query_embeds.shape[1]), dtype=np.float32)
            embeddings[query_indices] = query_embeds
            embeddings[gallery_indices] = gallery_embeds

        # Apply Mean-Centering Vector Calibration (z - mu) / ||z - mu||
        use_pca = self.config.get("retrieval", {}).get("use_pca", False) if isinstance(self.config, dict) else getattr(self.config.retrieval, "use_pca", False)
        use_dba = self.config.get("retrieval", {}).get("use_dba", False) if isinstance(self.config, dict) else getattr(self.config.retrieval, "use_dba", False)
        use_qe = self.config.get("retrieval", {}).get("use_qe", False) if isinstance(self.config, dict) else getattr(self.config.retrieval, "use_qe", False)

        mu = np.mean(gallery_embeds, axis=0, keepdims=True)
        gallery_embeds = gallery_embeds - mu
        query_embeds = query_embeds - mu

        if use_pca:
            whiten_dim = min(512, query_embeds.shape[1])
            logger.info(f"Stage 1: PCA Whitening (768-D -> {whiten_dim}-D)...")
            cov = np.cov(gallery_embeds, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[idx][:whiten_dim], 1e-7)
            eigenvectors_d = eigenvectors[:, idx[:whiten_dim]]
            W = np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors_d.T
            query_embeds = query_embeds @ W.T
            gallery_embeds = gallery_embeds @ W.T

        # L2 normalize
        query_embeds = query_embeds / (np.linalg.norm(query_embeds, axis=1, keepdims=True) + 1e-8)
        gallery_embeds = gallery_embeds / (np.linalg.norm(gallery_embeds, axis=1, keepdims=True) + 1e-8)

        if use_dba:
            dba_k = 5
            dba_alpha = 3.0
            logger.info(f"Stage 2: DBA Gallery Smoothing (k={dba_k})...")
            g_sims = gallery_embeds @ gallery_embeds.T
            np.fill_diagonal(g_sims, -1.0)
            gallery_augmented = np.zeros_like(gallery_embeds)
            for i in range(len(gallery_embeds)):
                top_k_idx = np.argpartition(g_sims[i], -dba_k)[-dba_k:]
                weights = np.maximum(g_sims[i, top_k_idx], 0.0) ** dba_alpha
                gallery_augmented[i] = gallery_embeds[i] + (weights[:, None] * gallery_embeds[top_k_idx]).sum(axis=0)
            gallery_embeds = gallery_augmented / (np.linalg.norm(gallery_augmented, axis=1, keepdims=True) + 1e-8)

        if use_qe:
            qe_k = 3
            qe_alpha = 3.0
            logger.info(f"Stage 3: Alpha Query Expansion (k={qe_k})...")
            qg_sims = query_embeds @ gallery_embeds.T
            query_expanded = np.zeros_like(query_embeds)
            for i in range(len(query_embeds)):
                top_k_idx = np.argpartition(qg_sims[i], -qe_k)[-qe_k:]
                weights = np.maximum(qg_sims[i, top_k_idx], 0.0) ** qe_alpha
                query_expanded[i] = query_embeds[i] + (weights[:, None] * gallery_embeds[top_k_idx]).sum(axis=0)
            query_embeds = query_expanded / (np.linalg.norm(query_expanded, axis=1, keepdims=True) + 1e-8)

        logger.info(f"Retrieval Split: {len(query_indices)} queries, {len(gallery_indices)} gallery items (Mean-Calibrated).")

        # Calculate metrics by computing chunked similarities to avoid OOM
        is_multilabel = True
        exclude_self = not is_cross_modal
        metrics5 = compute_retrieval_metrics(
            query_embeds=query_embeds,
            gallery_embeds=gallery_embeds,
            query_labels=query_labels,
            gallery_labels=gallery_labels,
            top_k=5,
            is_multilabel=is_multilabel,
            rerank_config=self.config.get("retrieval", None),
            query_names=query_names,
            gallery_names=gallery_names,
            exclude_self_matches=exclude_self
        )
        metrics10 = compute_retrieval_metrics(
            query_embeds=query_embeds,
            gallery_embeds=gallery_embeds,
            query_labels=query_labels,
            gallery_labels=gallery_labels,
            top_k=10,
            is_multilabel=is_multilabel,
            rerank_config=self.config.get("retrieval", None),
            query_names=query_names,
            gallery_names=gallery_names,
            exclude_self_matches=exclude_self
        )
        metrics = {}
        metrics.update(metrics5)
        metrics.update(metrics10)

        return {
            "metrics": metrics,
            "embeddings": embeddings,
            "labels": labels,
            "names": names,
            "query_indices": query_indices,
            "gallery_indices": gallery_indices,
            "similarity_matrix": None  # Removed to prevent OOM
        }
