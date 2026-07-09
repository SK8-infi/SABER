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
    venv_python = os.path.abspath("./project/.venv/Scripts/python.exe")
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
        venv_python, "project/train.py",
        "--epochs", "10",
        "--synthetic", "false",
        "--batch_size", "32"
    ]
    try:
        run_command(train_ben)
        
        # 2. Evaluate & Index BEN-14K
        eval_ben = [
            venv_python, "project/evaluate.py",
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
        venv_python, "project/train.py",
        "--epochs", "5",
        "--synthetic", "false",
        "--batch_size", "32"
    ]
    
    config_path = "project/configs/config.yaml"
    try:
        # Temporarily rewrite the dataset name in config.yaml to run DSRSID.
        logger.info("Temporarily switching config.yaml to DSRSID...")
        with open(config_path, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            if "name: \"ben14k\"" in line:
                new_lines.append(line.replace("name: \"ben14k\"", "name: \"dsrsid\""))
            elif "data_dir: \"C:/Users/praba/Downloads/benv1_14k\"" in line:
                new_lines.append(line.replace("data_dir: \"C:/Users/praba/Downloads/benv1_14k\"", "data_dir: \"C:/Users/praba/Downloads/DSRSID.mat\""))
            else:
                new_lines.append(line)
                
        with open(config_path, "w") as f:
            f.writelines(new_lines)

        # Run Train DSRSID
        run_command(train_dsrsid)
        
        # Run Evaluate & Index DSRSID
        eval_dsrsid = [
            venv_python, "project/evaluate.py",
            "--checkpoint", "checkpoints/latest.pth",
            "--synthetic", "false"
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
        
    finally:
        # Restore config.yaml to BEN-14K defaults
        logger.info("Restoring config.yaml to BEN-14K defaults...")
        with open(config_path, "r") as f:
            lines = f.readlines()
            
        restored_lines = []
        for line in lines:
            if "name: \"dsrsid\"" in line:
                restored_lines.append(line.replace("name: \"dsrsid\"", "name: \"ben14k\""))
            elif "data_dir: \"C:/Users/praba/Downloads/DSRSID.mat\"" in line:
                restored_lines.append(line.replace("data_dir: \"C:/Users/praba/Downloads/DSRSID.mat\"", "data_dir: \"C:/Users/praba/Downloads/benv1_14k\""))
            else:
                restored_lines.append(line)
                
        with open(config_path, "w") as f:
            f.writelines(restored_lines)

    logger.info("=== ALL PIPELINES COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
