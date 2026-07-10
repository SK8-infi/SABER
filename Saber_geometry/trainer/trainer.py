import os
import torch
import logging
from typing import Any, Dict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

logger = logging.getLogger("saber")

class Trainer:
    """
    Manages the training loop for the REJEPA system.
    Supports Automated Mixed Precision (AMP), gradient clipping,
    AdamW optimization, cosine learning rate decay, and TensorBoard logging.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        criterion: torch.nn.Module,
        config: Dict[str, Any],
        device: torch.device
    ) -> None:
        """
        Args:
            model: The REJEPA model instance.
            train_loader: DataLoader containing the training data.
            optimizer: Optimizes model parameters.
            scheduler: Adjusts learning rate during training.
            criterion: Evaluates composite training loss.
            config: Full configurations dictionary.
            device: Training device (CPU or CUDA).
        """
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.device = device
        
        self.epochs = config.train.epochs
        self.checkpoint_dir = config.checkpoint_dir
        self.grad_clip = config.train.grad_clip
        self.amp_enabled = config.train.amp and ("cuda" in str(device))

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=os.path.join(config.log_dir, "tensorboard"))
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Runs a single epoch of training."""
        self.model.train()
        
        # We only train Projection Head, Predictor, and Input Adapter
        # Freezing of Backbone is guaranteed in backbone setup.
        epoch_losses = {
            "loss": 0.0,
            "jaccard_loss": 0.0,
            "ranking_loss": 0.0,
            "invariance_loss": 0.0,
            "variance_loss": 0.0,
            "covariance_loss": 0.0
        }
        
        num_batches = len(self.train_loader)
        if num_batches == 0:
            return {k: 0.0 for k in epoch_losses}

        for batch_idx, batch in enumerate(self.train_loader):
            # Move images and labels to target device
            x1 = batch["image1"].to(self.device)
            x2 = batch["image2"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            # Execute forward pass under autocast for mixed precision
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                z1, z2, z1_pred = self.model(x1, x2)
                loss_dict = self.criterion(z1, z2, z1_pred, labels)
                loss = loss_dict["loss"]

            # Backward pass using gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient Clipping
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip
                )

            # Step optimizer & learning rate scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate loss metrics
            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k].item()

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        # Step Cosine Scheduler (if step-level is preferred, run after each batch; here epoch-level)
        if self.scheduler is not None:
            self.scheduler.step()

        return epoch_losses

    def fit(self) -> None:
        """Main loop that executes the full training timeline."""
        logger.info(f"Starting training for {self.epochs} epochs on device: {self.device}")
        
        for epoch in range(1, self.epochs + 1):
            losses = self.train_epoch(epoch)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log metrics to stdout and TensorBoard
            logger.info(
                f"Epoch [{epoch}/{self.epochs}] "
                f"Loss: {losses['loss']:.4f} | "
                f"Jac: {losses['jaccard_loss']:.4f} | "
                f"Rank: {losses['ranking_loss']:.4f} | "
                f"Var: {losses['variance_loss']:.4f} | "
                f"Cov: {losses['covariance_loss']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # TensorBoard logging
            self.tb_writer.add_scalar("Train/Total_Loss", losses["loss"], epoch)
            self.tb_writer.add_scalar("Train/Jaccard_Loss", losses["jaccard_loss"], epoch)
            self.tb_writer.add_scalar("Train/Ranking_Loss", losses["ranking_loss"], epoch)
            self.tb_writer.add_scalar("Train/Invariance_Loss", losses["invariance_loss"], epoch)
            self.tb_writer.add_scalar("Train/Variance_Loss", losses["variance_loss"], epoch)
            self.tb_writer.add_scalar("Train/Covariance_Loss", losses["covariance_loss"], epoch)
            self.tb_writer.add_scalar("Train/Learning_Rate", current_lr, epoch)

            # Save epoch checkpoint
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "loss": losses["loss"]
            }
            
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(checkpoint_state, checkpoint_path)
            
            # Save latest checkpoint
            latest_path = os.path.join(self.checkpoint_dir, "latest.pth")
            torch.save(checkpoint_state, latest_path)

        self.tb_writer.close()
        logger.info("Training complete. Models checkpoints saved successfully.")
