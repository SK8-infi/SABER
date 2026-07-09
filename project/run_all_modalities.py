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
    venv_python = os.path.abspath("./project/.venv/Scripts/python.exe")
    if not os.path.exists(venv_python):
        venv_python = "python"  # Fallback to system python

    # Create storage directories
    os.makedirs("checkpoints/sar", exist_ok=True)
    os.makedirs("checkpoints/crossmodal", exist_ok=True)
    os.makedirs("visualizations/sar", exist_ok=True)
    os.makedirs("visualizations/crossmodal", exist_ok=True)

    config_path = "project/configs/config.yaml"

    # =========================================================================
    # PHASE 1: Same-Modal SAR (Sentinel-1, 2 channels)
    # =========================================================================
    logger.info("=== PHASE 1: Training and evaluating Same-Modal SAR on GPU ===")
    
    try:
        # Set modality to "s1" in config
        logger.info("Switching config.yaml to modality: s1...")
        with open(config_path, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            if "modality: " in line:
                new_lines.append("  modality: \"s1\"                  # \"s2\" (optical), \"s1\" (SAR), or \"both\" (cross-modal)\n")
            else:
                new_lines.append(line)
                
        with open(config_path, "w") as f:
            f.writelines(new_lines)

        # 1. Train SAR
        run_command([venv_python, "project/train.py", "--epochs", "5", "--synthetic", "false", "--batch_size", "32"])
        
        # 2. Evaluate & Index SAR
        run_command([venv_python, "project/evaluate.py", "--checkpoint", "checkpoints/latest.pth", "--synthetic", "false"])
        
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
        # Set modality to "both" in config
        logger.info("Switching config.yaml to modality: both...")
        with open(config_path, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            if "modality: " in line:
                new_lines.append("  modality: \"both\"                # \"s2\" (optical), \"s1\" (SAR), or \"both\" (cross-modal)\n")
            else:
                new_lines.append(line)
                
        with open(config_path, "w") as f:
            f.writelines(new_lines)

        # 1. Train Cross-Modal
        run_command([venv_python, "project/train.py", "--epochs", "5", "--synthetic", "false", "--batch_size", "32"])
        
        # 2. Evaluate & Index Cross-Modal
        run_command([venv_python, "project/evaluate.py", "--checkpoint", "checkpoints/latest.pth", "--synthetic", "false"])
        
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
        
    finally:
        # Restore default modality to "s2"
        logger.info("Restoring config.yaml to default modality: s2...")
        with open(config_path, "r") as f:
            lines = f.readlines()
            
        restored_lines = []
        for line in lines:
            if "modality: " in line:
                restored_lines.append("  modality: \"s2\"                  # \"s2\" (optical), \"s1\" (SAR), or \"both\" (cross-modal)\n")
            else:
                restored_lines.append(line)
                
        with open(config_path, "w") as f:
            f.writelines(restored_lines)

    logger.info("=== RUN COMPLETE ===")

if __name__ == "__main__":
    main()
