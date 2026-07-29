import os
import time
import torch
import logging
import copy
from typing import Any, Dict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        amp_dtype_str = config.train.get("amp_dtype", "bfloat16").lower()
        if self.amp_enabled and amp_dtype_str == "bfloat16" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            self.use_scaler = False
            logger.info("Using Native CUDA bfloat16 Mixed Precision (GradScaler disabled for maximum speed).")
        elif self.amp_enabled:
            self.amp_dtype = torch.float16
            self.use_scaler = True
            if hasattr(torch.amp, "GradScaler"):
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            else:
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            logger.info("Using Standard CUDA float16 Mixed Precision with GradScaler.")
        else:
            self.amp_dtype = torch.float32
            self.use_scaler = False



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

        pbar = tqdm(
            enumerate(self.train_loader),
            total=num_batches,
            desc=f"Epoch [{epoch:02d}/{self.epochs:02d}]",
            dynamic_ncols=True,
            unit="batch",
            leave=True
        )

        for batch_idx, batch in pbar:
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

            # Execute forward pass under autocast for mixed precision (bfloat16 or float16)
            autocast_cm = torch.amp.autocast("cuda", enabled=self.amp_enabled, dtype=self.amp_dtype) if hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype)
            with autocast_cm:
                if self.use_ema and self.target_model is not None:
                    outputs = self.model(x1, x2)
                    if len(outputs) == 5:
                        z1, z2_online, z1_pred, logits_s1, logits_s2 = outputs
                        with torch.no_grad():
                            target_outputs = self.target_model(x1, x2)
                            _, target_z2, _, _, _ = target_outputs
                            z2 = target_z2.detach()
                    else:
                        z1, _, z1_pred = outputs
                        with torch.no_grad():
                            _, z2, _ = self.target_model(x1, x2)
                            z2 = z2.detach()
                else:
                    outputs = self.model(x1, x2)
                    if len(outputs) == 5:
                        z1, z2, z1_pred, logits_s1, logits_s2 = outputs
                    else:
                        z1, z2, z1_pred = outputs
                
                # Forward to loss criterion (with labels if supported)
                if labels is not None:
                    # Check for cached soft codes in case of hashing head
                    soft1 = getattr(self.model, "soft_codes1", None)
                    soft2 = getattr(self.model, "soft_codes2", None)
                    
                    if self.use_ema and self.target_model is not None:
                        if getattr(self.target_model, "hashing_head", None) is not None:
                            soft2 = self.target_model.hashing_head(z2)
                    
                    if len(outputs) == 5:
                        loss_dict = self.criterion(
                            z1=z1,
                            z2=z2 if self.use_ema else z2_online,
                            z1_pred=z1_pred,
                            targets=labels,
                            soft_codes1=soft1,
                            soft_codes2=soft2,
                            logits_s1=logits_s1,
                            logits_s2=logits_s2
                        )
                    else:
                        try:
                            if soft1 is not None:
                                loss_dict = self.criterion(z1, z2, z1_pred, labels, soft1, soft2)
                            else:
                                loss_dict = self.criterion(z1, z2, z1_pred, labels)
                        except TypeError:
                            loss_dict = self.criterion(z1, z2, z1_pred)
                else:
                    if len(outputs) == 5:
                        loss_dict = self.criterion(
                            z1=z1,
                            z2=z2 if self.use_ema else z2_online,
                            z1_pred=z1_pred,
                            targets=None,
                            logits_s1=logits_s1,
                            logits_s2=logits_s2
                        )
                    else:
                        loss_dict = self.criterion(z1, z2, z1_pred)
                
                loss = loss_dict["loss"] / self.accum_steps

            # Backward pass (bfloat16 skips GradScaler overhead, float16 uses GradScaler)
            if self.use_scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step optimizer every accum_steps batches (or at the last batch)
            if (batch_idx + 1) % self.accum_steps == 0 or (batch_idx + 1) == num_batches:
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                # Gradient Clipping
                if self.grad_clip > 0:
                    if self.use_scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.grad_clip)

                # Step optimizer & learning rate scaler
                if self.use_scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
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

            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{loss.item() * self.accum_steps:.4f}",
                "lr": f"{current_lr:.2e}"
            })

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
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            losses = self.train_epoch(epoch)
            epoch_duration = time.time() - epoch_start
            
            elapsed_total = time.time() - start_time
            avg_epoch_time = elapsed_total / epoch
            remaining_epochs = self.epochs - epoch
            eta_seconds = avg_epoch_time * remaining_epochs
            
            eta_hours = int(eta_seconds // 3600)
            eta_mins = int((eta_seconds % 3600) // 60)
            eta_secs = int(eta_seconds % 60)
            if eta_hours > 0:
                eta_str = f"{eta_hours:02d}h {eta_mins:02d}m {eta_secs:02d}s"
            else:
                eta_str = f"{eta_mins:02d}m {eta_secs:02d}s"

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log metrics to stdout and TensorBoard
            loss_str = " | ".join(f"{k.capitalize()[:4]}: {v:.4f}" for k, v in losses.items())
            logger.info(
                f"Epoch [{epoch}/{self.epochs}] completed in {epoch_duration:.1f}s | "
                f"Est. Time Remaining: {eta_str} | "
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
