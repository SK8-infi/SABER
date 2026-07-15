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
                images = batch["image"].to(self.device)
                
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

        # Use randomized splitting with fixed seed to prevent geographic data leakage
        rng = np.random.RandomState(42)
        shuffled_indices = rng.permutation(num_samples)
        query_size = max(1, num_samples // 5)
        query_indices = np.sort(shuffled_indices[:query_size])
        gallery_indices = np.sort(shuffled_indices[query_size:])

        is_cross_modal = (self.config.dataset.get("modality", "s2").lower() == "both")

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
                    images = batch["image"].to(self.device)
                    
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
            
            direction = self.config.get("retrieval", {}).get("direction", "s1_to_s2").lower()
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

        logger.info(f"Retrieval Split: {len(query_indices)} queries, {len(gallery_indices)} gallery items.")

        # Calculate metrics by computing chunked similarities to avoid OOM
        is_multilabel = (self.config.dataset.name.lower() == "ben14k")
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
