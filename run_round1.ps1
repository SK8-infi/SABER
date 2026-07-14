# run_round1.ps1
# Sequential execution of Round 1 improvements for SABER on real data.

Write-Output "=========================================================="
Write-Output "STARTING SABER ROUND 1 PIPELINE (BEN-14K & DSRSID)"
Write-Output "=========================================================="

# Create checkpoints directory if it doesn't exist
New-Item -ItemType Directory -Force -Path checkpoints

# --------------------------------------------------------
# PART 1: BEN-14K Pipeline
# --------------------------------------------------------
Write-Output "`n[1/10] Training BEN-14K Encoder (20 epochs)..."
.venv\Scripts\python.exe Saber/train.py --dataset_name ben14k --modality both --data_dir Datasets/benv1_14k --epochs 20 --synthetic false
if ($LASTEXITCODE -ne 0) { Write-Error "BEN-14K training failed"; exit $LASTEXITCODE }

Write-Output "`n[2/10] Copying BEN-14K encoder checkpoint..."
Copy-Item checkpoints/latest.pth checkpoints/latest_ben14k.pth -Force

Write-Output "`n[3/10] Extracting features for BEN-14K bridge..."
.venv\Scripts\python.exe Saber/extract_features.py --dataset_name ben14k --data_dir Datasets/benv1_14k --synthetic false --checkpoint checkpoints/latest_ben14k.pth --output_dir checkpoints/extracted_ben14k
if ($LASTEXITCODE -ne 0) { Write-Error "BEN-14K feature extraction failed"; exit $LASTEXITCODE }

Write-Output "`n[4/10] Training BEN-14K CFM Bridge (20 epochs)..."
.venv\Scripts\python.exe Saber/train_bridge.py --features_dir checkpoints/extracted_ben14k --epochs 20 --ode_steps 5
if ($LASTEXITCODE -ne 0) { Write-Error "BEN-14K bridge training failed"; exit $LASTEXITCODE }

Write-Output "`n[5/10] Copying BEN-14K bridge checkpoint..."
Copy-Item checkpoints/bridge_best.pth checkpoints/bridge_best_ben14k.pth -Force

# --------------------------------------------------------
# PART 2: DSRSID Pipeline
# --------------------------------------------------------
Write-Output "`n[6/10] Training DSRSID Encoder (20 epochs)..."
.venv\Scripts\python.exe Saber/train.py --dataset_name dsrsid --data_dir Datasets/DSRSID/DSRSID-001.mat --epochs 20 --synthetic false --modality both
if ($LASTEXITCODE -ne 0) { Write-Error "DSRSID training failed"; exit $LASTEXITCODE }

Write-Output "`n[7/10] Copying DSRSID encoder checkpoint..."
Copy-Item checkpoints/latest.pth checkpoints/latest_dsrsid.pth -Force

Write-Output "`n[8/10] Extracting features for DSRSID bridge..."
.venv\Scripts\python.exe Saber/extract_features.py --dataset_name dsrsid --data_dir Datasets/DSRSID/DSRSID-001.mat --synthetic false --checkpoint checkpoints/latest_dsrsid.pth --output_dir checkpoints/extracted_dsrsid
if ($LASTEXITCODE -ne 0) { Write-Error "DSRSID feature extraction failed"; exit $LASTEXITCODE }

Write-Output "`n[9/10] Training DSRSID CFM Bridge (20 epochs)..."
.venv\Scripts\python.exe Saber/train_bridge.py --features_dir checkpoints/extracted_dsrsid --epochs 20 --ode_steps 5
if ($LASTEXITCODE -ne 0) { Write-Error "DSRSID bridge training failed"; exit $LASTEXITCODE }

Write-Output "`n[10/10] Copying DSRSID bridge checkpoint..."
Copy-Item checkpoints/bridge_best.pth checkpoints/bridge_best_dsrsid.pth -Force

# --------------------------------------------------------
# PART 3: Evaluation
# --------------------------------------------------------
Write-Output "`n=========================================================="
Write-Output "RUNNING EVALUATIONS"
Write-Output "=========================================================="

# 1. BEN-14K Same-Modal
Write-Output "`nEvaluating BEN-14K Same-Modal (S2 -> S2)..."
.venv\Scripts\python.exe Saber/evaluate.py --architecture saber --dataset_name ben14k --modality s2 --synthetic false --data_dir Datasets/benv1_14k --checkpoint checkpoints/latest_ben14k.pth

# 2. BEN-14K Cross-Modal (using correct bridge checkpoint)
Write-Output "`nEvaluating BEN-14K Cross-Modal (S1 -> S2)..."
Copy-Item checkpoints/bridge_best_ben14k.pth checkpoints/bridge_best.pth -Force
.venv\Scripts\python.exe Saber/evaluate.py --architecture saber --dataset_name ben14k --modality both --synthetic false --data_dir Datasets/benv1_14k --checkpoint checkpoints/latest_ben14k.pth

# 3. DSRSID Same-Modal
Write-Output "`nEvaluating DSRSID Same-Modal (MS -> MS)..."
.venv\Scripts\python.exe Saber/evaluate.py --architecture saber --checkpoint checkpoints/latest_dsrsid.pth --dataset_name dsrsid --modality ms --synthetic false --data_dir Datasets/DSRSID/DSRSID-001.mat

# 4. DSRSID Cross-Modal (using correct bridge checkpoint)
Write-Output "`nEvaluating DSRSID Cross-Modal (PAN -> MS)..."
Copy-Item checkpoints/bridge_best_dsrsid.pth checkpoints/bridge_best.pth -Force
.venv\Scripts\python.exe Saber/evaluate.py --architecture saber --checkpoint checkpoints/latest_dsrsid.pth --dataset_name dsrsid --modality both --synthetic false --data_dir Datasets/DSRSID/DSRSID-001.mat

Write-Output "`n=========================================================="
Write-Output "ROUND 1 PIPELINE COMPLETED SUCCESSFULLY"
Write-Output "=========================================================="
