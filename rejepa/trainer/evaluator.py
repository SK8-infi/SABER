import torch
import numpy as np
import logging
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader
from rejepa.trainer.metrics import compute_retrieval_metrics

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
        with torch.no_grad():
            for batch in self.dataloader:
                images = batch["image"].to(self.device)
                labels = batch["label"]
                names = batch["name"]

                # Extract L2-normalized embeddings
                embeds = self.model.get_retrieval_embedding(images)
                
                embeddings_list.append(embeds.cpu().numpy())
                labels_list.append(labels.numpy())
                filenames_list.extend(names)

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

        is_cross_modal = (self.config.dataset.name.lower() == "ben14k" and self.config.dataset.get("modality", "s2").lower() == "both")

        if not is_cross_modal:
            # Same-modal path
            embeddings, labels, names = self.extract_all_embeddings()
            query_embeds = embeddings[query_indices]
            query_labels = labels[query_indices]
            
            gallery_embeds = embeddings[gallery_indices]
            gallery_labels = labels[gallery_indices]
        else:
            # Cross-modal path (SAR S1 -> Optical S2)
            logger.info("Extracting bimodal embeddings for cross-modal evaluation (S1 query, S2 gallery)...")
            self.model.eval()
            
            query_embeds_list = []
            gallery_embeds_list = []
            labels_list = []
            filenames_list = []
            
            with torch.no_grad():
                idx_counter = 0
                for batch in self.dataloader:
                    images = batch["image"].to(self.device)  # shape: (B, 14, 224, 224)
                    labels = batch["label"]
                    names = batch["name"]
                    
                    batch_size = images.shape[0]
                    for b in range(batch_size):
                        global_idx = idx_counter + b
                        img_slice = images[b:b+1]  # shape: (1, 14, 224, 224)
                        
                        if global_idx in query_indices:
                            # S1 Query embedding: pass first 2 channels (SAR)
                            x_q = img_slice[:, :2, :, :]
                            embed = self.model.get_retrieval_embedding(x_q)
                            query_embeds_list.append(embed.cpu().numpy())
                        else:
                            # S2 Gallery embedding: pass remaining 12 channels (MS)
                            x_g = img_slice[:, 2:, :, :]
                            embed = self.model.get_retrieval_embedding(x_g)
                            gallery_embeds_list.append(embed.cpu().numpy())
                            
                    labels_list.append(labels.numpy())
                    filenames_list.extend(names)
                    idx_counter += batch_size
                    
            query_embeds = np.concatenate(query_embeds_list, axis=0)
            gallery_embeds = np.concatenate(gallery_embeds_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            names = filenames_list
            
            query_labels = labels[query_indices]
            gallery_labels = labels[gallery_indices]
            
            # Synthesize final embeddings array for FAISS building compatibility
            embeddings = np.zeros((num_samples, query_embeds.shape[1]), dtype=np.float32)
            embeddings[query_indices] = query_embeds
            embeddings[gallery_indices] = gallery_embeds

        logger.info(f"Retrieval Split: {len(query_indices)} queries, {len(gallery_indices)} gallery items.")

        # Calculate metrics by computing chunked similarities to avoid OOM
        is_multilabel = (self.config.dataset.name.lower() == "ben14k")
        metrics = compute_retrieval_metrics(
            query_embeds=query_embeds,
            gallery_embeds=gallery_embeds,
            query_labels=query_labels,
            gallery_labels=gallery_labels,
            top_k=top_k,
            is_multilabel=is_multilabel
        )

        return {
            "metrics": metrics,
            "embeddings": embeddings,
            "labels": labels,
            "names": names,
            "query_indices": query_indices,
            "gallery_indices": gallery_indices,
            "similarity_matrix": None  # Removed to prevent OOM
        }
