# 🚀 SABER 40-Epoch Master Unified Training Logs

This document tracks the 40-epoch Master Unified Training execution of SABER across BigEarthNet-14K and DSRSID.

---

## 📊 Training Progress Summary (Epochs 1 – 8 Completed)

| Epoch | Phase 1 Loss | Soft Jaccard | Ranking Loss | Invariance | Variance | Covariance | Learning Rate | Drive Sync Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Epoch 1** | `20.5883` | `0.3323` | `0.2999` | `0.0816` | `0.6589` | `0.3334` | `3.00e-04` | 💾 Synced (`checkpoints_40epochs`) |
| **Epoch 2** | `19.7751` | `0.1717` | `0.3233` | `0.0694` | `0.6447` | `0.3404` | `3.00e-04` | 💾 Synced (`checkpoints_40epochs`) |
| **Epoch 3** | `19.5474` | `0.1442` | `0.3143` | `0.0646` | `0.6378` | `0.3575` | `2.98e-04` | 💾 Synced (`checkpoints_40epochs`) |
| **Epoch 4** | `19.4143` | `0.1307` | `0.3061` | `0.0619` | `0.6338` | `0.3668` | `2.96e-04` | 💾 Synced (`checkpoints_40epochs`) |
| **Epoch 5** | `19.3116` | `0.1222` | `0.3016` | `0.0594` | `0.6304` | `0.3756` | `2.93e-04` | 💾 Synced (`checkpoints_40epochs`) |
| **Epoch 6** | `19.2338` | `0.1149` | `0.2954` | `0.0579` | `0.6281` | `0.3809` | `2.89e-04` | 💾 Synced (`checkpoints_40epochs`) |
| **Epoch 7** | `19.1619` | `0.1067` | `0.2901` | `0.0563` | `0.6260` | `0.3866` | `2.84e-04` | 💾 Synced (`checkpoints_40epochs`) |
| **Epoch 8** | `19.0932` | `0.1011` | `0.2857` | `0.0546` | `0.6241` | `0.3908` | `2.78e-04` | 💾 Synced (`checkpoints_40epochs`) |

---

## 📜 Raw Console Log Stream

```text
================================================================================
 🚀 UNIFIED SENSOR-AGNOSTIC SABER MASTER TRAINING ENGINE (SPEED OPTIMIZED)
================================================================================
[2026-08-01 21:42:47] [INFO] [train_unified.py:62]: Computation Device: cuda | Execution Mode: 'ALL' | CuDNN Benchmark: ACTIVE
[2026-08-01 21:42:47] [INFO] [train_unified.py:96]: Initializing BEN-14K Sentinel-1/2 dataset from 'Datasets/benv1_14k'...
[2026-08-01 21:42:47] [INFO] [train_unified.py:109]: Initializing DSRSID Gaofen PAN/MS dataset from 'Datasets/DSRSID'...
[2026-08-01 21:42:47] [INFO] [train_unified.py:137]: BEN-14K Batches: 216 | DSRSID Batches: 204 (Batch Size: 48)
[2026-08-01 21:42:51] [INFO] [train_unified.py:235]: ============================================================
[2026-08-01 21:42:51] [INFO] [train_unified.py:236]:  PHASE 1: MASTER ENCODER JOINT TRAINING (40 Epochs | SPEED OPTIMIZED)
[2026-08-01 21:42:51] [INFO] [train_unified.py:237]: ============================================================
Phase 1 Epoch 1/40: 100% 420/420 [28:38<00:00,  4.09s/it, loss=20.0167, jacc=0.227, invar=0.066, var=0.646, cov=0.354, lr=3.00e-04]
[2026-08-01 22:11:29] [INFO] [train_unified.py:350]: Epoch [1/40] completed in 1718.2s | Loss: 20.5883 | Jacc: 0.3323 | Rank: 0.2999 | Invar: 0.0816 | Var: 0.6589 | Cov: 0.3334
[2026-08-01 22:12:20] [INFO] [train_unified.py:378]: 💾 Synced Epoch [1/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 2/40: 100% 420/420 [28:48<00:00,  4.11s/it, loss=19.2379, jacc=0.145, invar=0.047, var=0.634, cov=0.375, lr=3.00e-04]
[2026-08-01 22:41:08] [INFO] [train_unified.py:350]: Epoch [2/40] completed in 1728.2s | Loss: 19.7751 | Jacc: 0.1717 | Rank: 0.3233 | Invar: 0.0694 | Var: 0.6447 | Cov: 0.3404
[2026-08-01 22:42:43] [INFO] [train_unified.py:378]: 💾 Synced Epoch [2/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 3/40: 100% 420/420 [28:50<00:00,  4.12s/it, loss=19.5111, jacc=0.128, invar=0.053, var=0.632, cov=0.391, lr=2.98e-04]
[2026-08-01 23:11:33] [INFO] [train_unified.py:350]: Epoch [3/40] completed in 1730.1s | Loss: 19.5474 | Jacc: 0.1442 | Rank: 0.3143 | Invar: 0.0646 | Var: 0.6378 | Cov: 0.3575
[2026-08-01 23:13:29] [INFO] [train_unified.py:378]: 💾 Synced Epoch [3/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 4/40: 100% 420/420 [28:48<00:00,  4.12s/it, loss=18.7713, jacc=0.128, invar=0.032, var=0.623, cov=0.393, lr=2.96e-04]
[2026-08-01 23:42:17] [INFO] [train_unified.py:350]: Epoch [4/40] completed in 1728.8s | Loss: 19.4143 | Jacc: 0.1307 | Rank: 0.3061 | Invar: 0.0619 | Var: 0.6338 | Cov: 0.3668
[2026-08-01 23:43:47] [INFO] [train_unified.py:378]: 💾 Synced Epoch [4/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 5/40: 100% 420/420 [28:50<00:00,  4.12s/it, loss=19.2156, jacc=0.150, invar=0.060, var=0.624, cov=0.404, lr=2.93e-04]
[2026-08-02 00:12:38] [INFO] [train_unified.py:350]: Epoch [5/40] completed in 1730.7s | Loss: 19.3116 | Jacc: 0.1222 | Rank: 0.3016 | Invar: 0.0594 | Var: 0.6304 | Cov: 0.3756
[2026-08-02 00:14:09] [INFO] [train_unified.py:378]: 💾 Synced Epoch [5/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 6/40: 100% 420/420 [28:51<00:00,  4.12s/it, loss=19.0941, jacc=0.117, invar=0.045, var=0.625, cov=0.397, lr=2.89e-04]
[2026-08-02 00:43:00] [INFO] [train_unified.py:350]: Epoch [6/40] completed in 1731.5s | Loss: 19.2338 | Jacc: 0.1149 | Rank: 0.2954 | Invar: 0.0579 | Var: 0.6281 | Cov: 0.3809
[2026-08-02 00:44:19] [INFO] [train_unified.py:378]: 💾 Synced Epoch [6/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 7/40: 100% 420/420 [28:51<00:00,  4.12s/it, loss=18.9028, jacc=0.111, invar=0.042, var=0.623, cov=0.398, lr=2.84e-04]
[2026-08-02 01:13:10] [INFO] [train_unified.py:350]: Epoch [7/40] completed in 1731.4s | Loss: 19.1619 | Jacc: 0.1067 | Rank: 0.2901 | Invar: 0.0563 | Var: 0.6260 | Cov: 0.3866
[2026-08-02 01:14:49] [INFO] [train_unified.py:378]: 💾 Synced Epoch [7/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 8/40: 100% 420/420 [28:52<00:00,  4.12s/it, loss=19.0534, jacc=0.112, invar=0.047, var=0.620, cov=0.422, lr=2.78e-04]
[2026-08-02 01:43:41] [INFO] [train_unified.py:350]: Epoch [8/40] completed in 1732.2s | Loss: 19.0932 | Jacc: 0.1011 | Rank: 0.2857 | Invar: 0.0546 | Var: 0.6241 | Cov: 0.3908
[2026-08-02 01:45:46] [INFO] [train_unified.py:378]: 💾 Synced Epoch [8/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 9/40:  20% 82/420 [05:41<23:08,  4.11s/it, loss=18.9845, jacc=0.102, invar=0.035, var=0.620, cov=0.407, lr=2.71e-04]
```
