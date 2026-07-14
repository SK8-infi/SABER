import os
import torch
import logging
import copy
from typing import Any, Dict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

logger = logging.getLogger("saber")

class Trainer:
    """
    Manages the training loop for the REJEPA/SABER system.
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
            model: The REJEPA/SABER model instance.
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
        self.accum_steps = config.train.get("grad_accumulation_steps", 1)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=os.path.join(config.log_dir, "tensorboard"))
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        # Configurable EMA target encoder path for cross-modal prediction stability
        self.use_ema = config.train.get("use_ema", False)
        self.ema_decay = config.train.get("ema_decay", 0.99)
        if self.use_ema:
            logger.info("Initializing EMA target model copy for training.")
            self.target_model = copy.deepcopy(model).to(device)
            for p in self.target_model.parameters():
                p.requires_grad = False
        else:
            self.target_model = None

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Runs a single epoch of training."""
        self.model.train()
        if self.target_model is not None:
            self.target_model.eval()
        
        self.optimizer.zero_grad()
        epoch_losses = {}
        num_batches = len(self.train_loader)
        if num_batches == 0:
            return {}

        for batch_idx, batch in enumerate(self.train_loader):
            # Move images and labels to target device
            x1 = batch["image1"].to(self.device)
            x2 = batch["image2"].to(self.device)
            
            # Auto-resize on GPU to prevent CPU resize bottleneck
            if x1.shape[-1] != 224 or x1.shape[-2] != 224:
                import torch.nn.functional as F
                x1 = F.interpolate(x1, size=(224, 224), mode="bilinear", align_corners=False)
                x2 = F.interpolate(x2, size=(224, 224), mode="bilinear", align_corners=False)
                
            labels = batch.get("label", None)
            if labels is not None:
                labels = labels.to(self.device)

            # Execute forward pass under autocast for mixed precision
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                if self.use_ema and self.target_model is not None:
                    # 1. Online forward pass (computes z1 and z1_pred)
                    z1, _, z1_pred = self.model(x1, x2)
                    
                    # 2. Target forward pass (using EMA target model with stop-gradient)
                    with torch.no_grad():
                        _, z2, _ = self.target_model(x1, x2)
                        z2 = z2.detach()
                else:
                    # Standard online dual projection path
                    z1, z2, z1_pred = self.model(x1, x2)
                
                # Forward to loss criterion (with labels if supported)
                if labels is not None:
                    # Check for cached soft codes in case of hashing head
                    soft1 = getattr(self.model, "soft_codes1", None)
                    soft2 = getattr(self.model, "soft_codes2", None)
                    
                    # If using EMA, compute target soft codes from target projection
                    if self.use_ema and self.target_model is not None:
                        if getattr(self.target_model, "hashing_head", None) is not None:
                            soft2 = self.target_model.hashing_head(z2)
                    
                    try:
                        if soft1 is not None:
                            loss_dict = self.criterion(z1, z2, z1_pred, labels, soft1, soft2)
                        else:
                            loss_dict = self.criterion(z1, z2, z1_pred, labels)
                    except TypeError:
                        # Fallback if loss function doesn't accept labels
                        loss_dict = self.criterion(z1, z2, z1_pred)
                else:
                    loss_dict = self.criterion(z1, z2, z1_pred)
                
                loss = loss_dict["loss"] / self.accum_steps

            # Backward pass using gradient scaling
            self.scaler.scale(loss).backward()

            # Step optimizer every accum_steps batches (or at the last batch)
            if (batch_idx + 1) % self.accum_steps == 0 or (batch_idx + 1) == num_batches:
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
                self.optimizer.zero_grad()

                # Update EMA target model parameters
                if self.use_ema and self.target_model is not None:
                    with torch.no_grad():
                        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                            target_param.data.mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)

            # Accumulate loss metrics dynamically (using unscaled values)
            for k in loss_dict:
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += loss_dict[k].item()
                
            if batch_idx % 50 == 0:
                logger.info(f"Epoch [{epoch}] - Batch [{batch_idx}/{num_batches}] - Loss: {loss.item() * self.accum_steps:.4f}")

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        # Step Cosine Scheduler
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
            loss_str = " | ".join(f"{k.capitalize()[:4]}: {v:.4f}" for k, v in losses.items())
            logger.info(
                f"Epoch [{epoch}/{self.epochs}] "
                f"{loss_str} | "
                f"LR: {current_lr:.6f}"
            )

            # TensorBoard logging
            for k, v in losses.items():
                self.tb_writer.add_scalar(f"Train/{k.capitalize()}_Loss", v, epoch)
            self.tb_writer.add_scalar("Train/Learning_Rate", current_lr, epoch)

            # Save epoch checkpoint
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "loss": losses.get("loss", 0.0)
            }
            
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(checkpoint_state, checkpoint_path)
            
            # Save latest checkpoint
            latest_path = os.path.join(self.checkpoint_dir, "latest.pth")
            torch.save(checkpoint_state, latest_path)

        self.tb_writer.close()
        logger.info("Training complete. Models checkpoints saved successfully.")
