# 💡 Idea: Google Drive Automated Checkpoint Resuming (`resume.md`)

## 📌 Problem Context
Google Colab T4 runtime sessions have a strict duration limit (5–12 hours). Running long 50+ epoch training runs on large satellite datasets (BEN-14K + DSRSID) in a single session risks data loss if Colab disconnects or times out midway.

---

## 🛠️ Proposed Solution Architecture

### 1. Per-Epoch Google Drive Checkpoint Syncing
At the end of each training epoch in `train_unified.py`, the training state is serialized into Google Drive:
* **Target Path**: `/content/drive/MyDrive/SABER_Data/checkpoints/saber_unified.pth`
* **Payload**:
  ```python
  checkpoint_payload = {
      "epoch": epoch,
      "model_state_dict": model.state_dict(),
      "ema_state_dict": ema_model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "loss": avg_loss,
      "config": config
  }
  ```

### 2. Automated Smart Resume Workflow
When starting a new Colab session and invoking:
```bash
python Saber/train_unified.py --epochs 50 --resume true
```

The script will:
1. Check for existing checkpoint files on Google Drive (`saber_unified.pth` or `latest.pth`).
2. If detected, load model weights, target EMA model weights, and AdamW optimizer states.
3. Extract `start_epoch = saved_epoch + 1`.
4. Seamlessly resume Phase 1 joint training from `start_epoch` $\rightarrow$ `50`.
5. Skip Phase 1 automatically if Phase 1 epochs are already completed, proceeding straight to Phase 2 (Master Patch CFM Bridge).

---

## 📋 Status
* **Status**: IDEA / PROPOSED
* **Action**: Saved for future evaluation when scaling training beyond 10+ epochs.
