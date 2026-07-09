import torch
import numpy as np
import logging
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader
from project.trainer.metrics import compute_retrieval_metrics

logger = logging.getLogger("saber")

class Evaluator:
    """
    Evaluates the retrieval performance of the trained model.
    Extracts L2-normalized embeddings for the entire evaluation dataset,
    partitions them into query/gallery splits, and computes standard retrieval metrics.
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
        
        Returns:
            A tuple of (embeddings, labels, filenames) as numpy arrays/lists.
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

        # Concatenate all batch outputs
        all_embeddings = np.concatenate(embeddings_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)

        return all_embeddings, all_labels, filenames_list

    def evaluate(self, top_k: int = 5) -> Dict[str, Any]:
        """
        Evaluates retrieval metrics on the dataloader.
        Partitions data into query (20%) and gallery (80%) subsets deterministically.
        
        Args:
            top_k: Retrieve top K matches.
            
        Returns:
            Dictionary containing metrics and extracted tensors/metadata.
        """
        embeddings, labels, names = self.extract_all_embeddings()
        num_samples = len(embeddings)
        
        if num_samples < 5:
            raise ValueError(f"Dataset has only {num_samples} samples, which is too small for retrieval split.")

        # Determine indices (Query: every 5th image, Gallery: remainder)
        query_indices = np.arange(0, num_samples, 5)
        gallery_indices = np.array([i for i in range(num_samples) if i not in query_indices])

        query_embeds = embeddings[query_indices]
        query_labels = labels[query_indices]
        
        gallery_embeds = embeddings[gallery_indices]
        gallery_labels = labels[gallery_indices]

        logger.info(f"Retrieval Split: {len(query_indices)} queries, {len(gallery_indices)} gallery items.")

        # Compute cosine similarity matrix
        # Since embeddings are L2 normalized, similarity is the dot product
        similarity_matrix = query_embeds @ gallery_embeds.T

        # Detect multilabel dataset configuration
        is_multilabel = (self.config.dataset.name.lower() == "ben14k")

        # Calculate metrics
        metrics = compute_retrieval_metrics(
            similarity_matrix=similarity_matrix,
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
            "similarity_matrix": similarity_matrix
        }
