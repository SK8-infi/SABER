import subprocess
import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("modalities_runner")

def run_command(command: list):
    logger.info(f"Running: {' '.join(command)}")
    result = subprocess.run(command, check=True)
    return result

def main():
    venv_python = os.path.abspath("./Saber/.venv/Scripts/python.exe")
    if not os.path.exists(venv_python):
        venv_python = "python"  # Fallback to system python

    # Create storage directories
    os.makedirs("checkpoints/sar", exist_ok=True)
    os.makedirs("checkpoints/crossmodal", exist_ok=True)
    os.makedirs("visualizations/sar", exist_ok=True)
    os.makedirs("visualizations/crossmodal", exist_ok=True)

    config_path = "Saber/configs/config.yaml"

    # =========================================================================
    # PHASE 1: Same-Modal SAR (Sentinel-1, 2 channels)
    # =========================================================================
    logger.info("=== PHASE 1: Training and evaluating Same-Modal SAR on GPU ===")
    
    try:
        # 1. Train SAR
        run_command([
            venv_python, "Saber/train.py",
            "--epochs", "5",
            "--synthetic", "false",
            "--batch_size", "32",
            "--modality", "s1"
        ])
        
        # 2. Evaluate & Index SAR
        run_command([
            venv_python, "Saber/evaluate.py",
            "--checkpoint", "checkpoints/latest.pth",
            "--synthetic", "false",
            "--modality", "s1"
        ])
        
        # 3. Archive outputs
        logger.info("Archiving SAR outputs...")
        for f in ["latest.pth", "faiss_index.bin", "faiss_index_metadata.pth"]:
            src = os.path.join("checkpoints", f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join("checkpoints/sar", f))
                
        for img in ["tsne.png", "umap.png", "similarity_heatmap.png"]:
            src = os.path.join("visualizations", img)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join("visualizations/sar", img))
                
    except Exception as e:
        logger.error(f"Error during SAR pipeline: {e}")

    # =========================================================================
    # PHASE 2: Cross-Modal Retrieval (Sentinel-1 SAR query -> Sentinel-2 MS gallery)
    # =========================================================================
    logger.info("=== PHASE 2: Training and evaluating Cross-Modal (S1 -> S2) on GPU ===")
    
    try:
        # 1. Train Cross-Modal
        run_command([
            venv_python, "Saber/train.py",
            "--epochs", "5",
            "--synthetic", "false",
            "--batch_size", "32",
            "--modality", "both"
        ])
        
        # 2. Evaluate & Index Cross-Modal
        run_command([
            venv_python, "Saber/evaluate.py",
            "--checkpoint", "checkpoints/latest.pth",
            "--synthetic", "false",
            "--modality", "both"
        ])
        
        # 3. Archive outputs
        logger.info("Archiving Cross-Modal outputs...")
        for f in ["latest.pth", "faiss_index.bin", "faiss_index_metadata.pth"]:
            src = os.path.join("checkpoints", f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join("checkpoints/crossmodal", f))
                
        for img in ["tsne.png", "umap.png", "similarity_heatmap.png"]:
            src = os.path.join("visualizations", img)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join("visualizations/crossmodal", img))
                
    except Exception as e:
        logger.error(f"Error during Cross-Modal pipeline: {e}")

    logger.info("=== RUN COMPLETE ===")

if __name__ == "__main__":
    main()
