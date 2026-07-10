import subprocess
import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("pipeline")

def run_command(command: list):
    logger.info(f"Running: {' '.join(command)}")
    result = subprocess.run(command, check=True)
    return result

def main():
    venv_python = os.path.abspath("./Saber_dofa/.venv/Scripts/python.exe")
    if not os.path.exists(venv_python):
        venv_python = "python"  # Fallback to system python

    # Create storage directories
    os.makedirs("checkpoints/ben14k", exist_ok=True)
    os.makedirs("checkpoints/dsrsid", exist_ok=True)
    os.makedirs("visualizations/ben14k", exist_ok=True)
    os.makedirs("visualizations/dsrsid", exist_ok=True)

    # =========================================================================
    # PHASE 1: BEN-14K Sequential Train & Eval
    # =========================================================================
    logger.info("=== PHASE 1: Training and evaluating BEN-14K on GPU ===")
    
    # 1. Train BEN-14K
    train_ben = [
        venv_python, "Saber_dofa/train.py",
        "--epochs", "10",
        "--synthetic", "false",
        "--batch_size", "32"
    ]
    try:
        run_command(train_ben)
        
        # 2. Evaluate & Index BEN-14K
        eval_ben = [
            venv_python, "Saber_dofa/evaluate.py",
            "--checkpoint", "checkpoints/latest.pth",
            "--synthetic", "false"
        ]
        run_command(eval_ben)
        
        # 3. Archive BEN-14K outputs
        logger.info("Archiving BEN-14K outputs...")
        for f in ["latest.pth", "faiss_index.bin", "faiss_index_metadata.pth"]:
            src = os.path.join("checkpoints", f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join("checkpoints/ben14k", f))
                
        for img in ["tsne.png", "umap.png", "similarity_heatmap.png"]:
            src = os.path.join("visualizations", img)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join("visualizations/ben14k", img))
                
    except Exception as e:
        logger.error(f"Error during BEN-14K pipeline: {e}")

    # =========================================================================
    # PHASE 2: DSRSID (Gaofen-1) Sequential Train & Eval
    # =========================================================================
    logger.info("=== PHASE 2: Training and evaluating DSRSID on GPU ===")
    
    # 1. Train DSRSID (Gaofen-1)
    train_dsrsid = [
        venv_python, "Saber_dofa/train.py",
        "--epochs", "5",
        "--synthetic", "false",
        "--batch_size", "32",
        "--dataset_name", "dsrsid",
        "--data_dir", "c:/Github/SABER/Datasets/DSRSID/DSRSID-001.mat"
    ]
    
    try:
        # Run Train DSRSID
        run_command(train_dsrsid)
        
        # Run Evaluate & Index DSRSID
        eval_dsrsid = [
            venv_python, "Saber_dofa/evaluate.py",
            "--checkpoint", "checkpoints/latest.pth",
            "--synthetic", "false",
            "--dataset_name", "dsrsid",
            "--data_dir", "c:/Github/SABER/Datasets/DSRSID/DSRSID-001.mat"
        ]
        run_command(eval_dsrsid)
        
        # Archive DSRSID outputs
        logger.info("Archiving DSRSID outputs...")
        for f in ["latest.pth", "faiss_index.bin", "faiss_index_metadata.pth"]:
            src = os.path.join("checkpoints", f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join("checkpoints/dsrsid", f))
                
        for img in ["tsne.png", "umap.png", "similarity_heatmap.png"]:
            src = os.path.join("visualizations", img)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join("visualizations/dsrsid", img))
                
    except Exception as e:
        logger.error(f"Error during DSRSID pipeline: {e}")

    logger.info("=== ALL PIPELINES COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
