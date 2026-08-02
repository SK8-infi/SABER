# 🚀 SABER 40-Epoch Master Unified Training Logs

This document tracks the 40-epoch Master Unified Training execution of SABER across BigEarthNet-14K and DSRSID.

---

## 📊 Training Progress Summary (Epochs 1 – 19 Completed)

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
| **Epoch 9** | `19.0338` | `0.0933` | `0.2813` | `0.0536` | `0.6225` | `0.3942` | `2.46e-04` | 💾 Synced (`checkpoints_40epochs`) |
| **Epoch 10** | `18.9670` | `0.0881` | `0.2746` | `0.0522` | `0.6207` | `0.3982` | `2.39e-04` | 💾 Synced (`checkpoints_40epochs`) |
| **Epoch 11** | `18.9364` | `0.0831` | `0.2705` | `0.0517` | `0.6200` | `0.4003` | `2.32e-04` | 💾 Synced (`checkpoints_40epochs`) |
| **Epoch 12** | `18.9048` | `0.0817` | `0.2696` | `0.0510` | `0.6190` | `0.4021` | `2.24e-04` | 💾 Synced (`checkpoints_40epochs`) |
| **Epoch 13** | `18.8663` | `0.0776` | `0.2630` | `0.0505` | `0.6178` | `0.4056` | `2.16e-04` | 💾 Synced (`checkpoints_40epochs`) |
| **Epoch 14** | `18.8391` | `0.0745` | `0.2629` | `0.0496` | `0.6171` | `0.4072` | `2.07e-04` | 💾 Synced (`checkpoints_40epochs`) |
| **Epoch 15** | `18.8073` | `0.0728` | `0.2596` | `0.0488` | `0.6162` | `0.4097` | `1.98e-04` | 💾 Synced (`checkpoints_40epochs`) |
| **Epoch 16** | `18.7921` | `0.0718` | `0.2613` | `0.0483` | `0.6155` | `0.4114` | `1.88e-04` | 💾 Synced (`checkpoints_40epochs`) |
| **Epoch 17** | `18.7685` | `0.0704` | `0.2574` | `0.0478` | `0.6149` | `0.4130` | `1.78e-04` | 💾 Synced (`checkpoints_40epochs`) |
| **Epoch 18** | `18.7429` | `0.0679` | `0.2537` | `0.0471` | `0.6145` | `0.4140` | `1.68e-04` | 💾 Synced (`checkpoints_40epochs`) |
| **Epoch 19** | `18.7222` | `0.0644` | `0.2508` | `0.0470` | `0.6141` | `0.4150` | `1.57e-04` | 💾 Synced (`checkpoints_40epochs`) |

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
Phase 1 Epoch 9/40: 100% 420/420 [26:56<00:00,  3.85s/it, loss=19.2842, jacc=0.099, invar=0.054, var=0.617, cov=0.445, lr=2.46e-04]
[2026-08-02 07:18:53] [INFO] [train_unified.py:353]: Epoch [9/40] completed in 1616.4s | Loss: 19.0338 | Jacc: 0.0933 | Rank: 0.2813 | Invar: 0.0536 | Var: 0.6225 | Cov: 0.3942
[2026-08-02 07:20:11] [INFO] [train_unified.py:381]: 💾 Synced Epoch [9/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 10/40: 100% 420/420 [26:52<00:00,  3.84s/it, loss=18.9222, jacc=0.094, invar=0.036, var=0.618, cov=0.420, lr=2.39e-04]
[2026-08-02 07:47:03] [INFO] [train_unified.py:353]: Epoch [10/40] completed in 1612.8s | Loss: 18.9670 | Jacc: 0.0881 | Rank: 0.2746 | Invar: 0.0522 | Var: 0.6207 | Cov: 0.3982
[2026-08-02 07:48:04] [INFO] [train_unified.py:381]: 💾 Synced Epoch [10/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 11/40: 100% 420/420 [26:51<00:00,  3.84s/it, loss=19.0891, jacc=0.083, invar=0.048, var=0.621, cov=0.425, lr=2.32e-04]
[2026-08-02 08:14:56] [INFO] [train_unified.py:353]: Epoch [11/40] completed in 1611.7s | Loss: 18.9364 | Jacc: 0.0831 | Rank: 0.2705 | Invar: 0.0517 | Var: 0.6200 | Cov: 0.4003
[2026-08-02 08:15:43] [INFO] [train_unified.py:381]: 💾 Synced Epoch [11/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 12/40: 100% 420/420 [26:51<00:00,  3.84s/it, loss=18.5020, jacc=0.092, invar=0.035, var=0.614, cov=0.413, lr=2.24e-04]
[2026-08-02 08:42:35] [INFO] [train_unified.py:353]: Epoch [12/40] completed in 1611.7s | Loss: 18.9048 | Jacc: 0.0817 | Rank: 0.2696 | Invar: 0.0510 | Var: 0.6190 | Cov: 0.4021
[2026-08-02 08:43:46] [INFO] [train_unified.py:381]: 💾 Synced Epoch [12/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 13/40: 100% 420/420 [26:51<00:00,  3.84s/it, loss=19.0638, jacc=0.077, invar=0.050, var=0.614, cov=0.453, lr=2.16e-04]
[2026-08-02 09:10:37] [INFO] [train_unified.py:353]: Epoch [13/40] completed in 1611.2s | Loss: 18.8663 | Jacc: 0.0776 | Rank: 0.2630 | Invar: 0.0505 | Var: 0.6178 | Cov: 0.4056
[2026-08-02 09:13:12] [INFO] [train_unified.py:381]: 💾 Synced Epoch [13/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 14/40: 100% 420/420 [26:51<00:00,  3.84s/it, loss=18.8521, jacc=0.081, invar=0.049, var=0.613, cov=0.440, lr=2.07e-04]
[2026-08-02 09:40:03] [INFO] [train_unified.py:353]: Epoch [14/40] completed in 1611.4s | Loss: 18.8391 | Jacc: 0.0745 | Rank: 0.2629 | Invar: 0.0496 | Var: 0.6171 | Cov: 0.4072
[2026-08-02 09:41:38] [INFO] [train_unified.py:381]: 💾 Synced Epoch [14/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 15/40: 100% 420/420 [26:49<00:00,  3.83s/it, loss=18.6678, jacc=0.070, invar=0.036, var=0.614, cov=0.417, lr=1.98e-04]
[2026-08-02 10:08:27] [INFO] [train_unified.py:353]: Epoch [15/40] completed in 1609.6s | Loss: 18.8073 | Jacc: 0.0728 | Rank: 0.2596 | Invar: 0.0488 | Var: 0.6162 | Cov: 0.4097
[2026-08-02 10:09:33] [INFO] [train_unified.py:381]: 💾 Synced Epoch [15/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 16/40: 100% 420/420 [26:51<00:00,  3.84s/it, loss=18.8946, jacc=0.090, invar=0.046, var=0.614, cov=0.440, lr=1.88e-04]
[2026-08-02 10:36:24] [INFO] [train_unified.py:353]: Epoch [16/40] completed in 1611.0s | Loss: 18.7921 | Jacc: 0.0718 | Rank: 0.2613 | Invar: 0.0483 | Var: 0.6155 | Cov: 0.4114
[2026-08-02 10:37:36] [INFO] [train_unified.py:381]: 💾 Synced Epoch [16/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 17/40: 100% 420/420 [26:51<00:00,  3.84s/it, loss=18.8100, jacc=0.078, invar=0.037, var=0.612, cov=0.443, lr=1.78e-04]
[2026-08-02 11:04:28] [INFO] [train_unified.py:353]: Epoch [17/40] completed in 1611.7s | Loss: 18.7685 | Jacc: 0.0704 | Rank: 0.2574 | Invar: 0.0478 | Var: 0.6149 | Cov: 0.4130
[2026-08-02 11:05:27] [INFO] [train_unified.py:381]: 💾 Synced Epoch [17/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 18/40: 100% 420/420 [26:51<00:00,  3.84s/it, loss=18.4752, jacc=0.075, invar=0.035, var=0.612, cov=0.421, lr=1.68e-04]
[2026-08-02 11:32:19] [INFO] [train_unified.py:353]: Epoch [18/40] completed in 1612.0s | Loss: 18.7429 | Jacc: 0.0679 | Rank: 0.2537 | Invar: 0.0471 | Var: 0.6145 | Cov: 0.4140
[2026-08-02 11:33:21] [INFO] [train_unified.py:381]: 💾 Synced Epoch [18/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 19/40: 100% 420/420 [26:52<00:00,  3.84s/it, loss=18.7567, jacc=0.069, invar=0.040, var=0.612, cov=0.441, lr=1.57e-04]
[2026-08-02 12:00:13] [INFO] [train_unified.py:353]: Epoch [19/40] completed in 1612.2s | Loss: 18.7222 | Jacc: 0.0644 | Rank: 0.2508 | Invar: 0.0470 | Var: 0.6141 | Cov: 0.4150
[2026-08-02 12:01:41] [INFO] [train_unified.py:381]: 💾 Synced Epoch [19/40] checkpoint to Google Drive: '/content/drive/MyDrive/SABER_Data/checkpoints_40epochs'
Phase 1 Epoch 20/40:  92% 387/420 [24:44<02:06,  3.84s/it, loss=18.6745, jacc=0.055, invar=0.060, var=0.615, cov=0.392, lr=1.47e-04]
```
