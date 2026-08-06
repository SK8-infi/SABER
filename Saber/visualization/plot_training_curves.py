import os
import matplotlib.pyplot as plt
import numpy as np

def plot_all_curves(output_dir="docs/assets"):
    os.makedirs(output_dir, exist_ok=True)
    
    # ── 1. Main Encoder 20-Epoch Training Data ───────────────────────
    epochs_enc = np.arange(1, 21)
    loss_enc = [27.3668, 25.9101, 25.5505, 25.4731, 25.2843, 25.0882, 24.8329, 24.6082, 24.4160, 24.2498, 
                24.1150, 24.0276, 23.9065, 23.8307, 23.7607, 23.7202, 23.6551, 23.6309, 23.6242, 23.5892]
    jaccard = [0.5047, 0.4166, 0.3995, 0.4026, 0.3986, 0.3934, 0.3826, 0.3746, 0.3684, 0.3588,
               0.3536, 0.3489, 0.3410, 0.3358, 0.3322, 0.3279, 0.3246, 0.3213, 0.3212, 0.3200]
    rank = [2.4504, 2.3337, 2.2988, 2.2896, 2.2456, 2.2249, 2.1828, 2.1489, 2.1134, 2.0897,
            2.0631, 2.0457, 2.0082, 1.9899, 1.9633, 1.9631, 1.9390, 1.9343, 1.9287, 1.9087]
    clas = [0.8103, 0.3727, 0.2270, 0.2027, 0.1921, 0.1855, 0.1807, 0.1772, 0.1736, 0.1704,
            0.1675, 0.1658, 0.1642, 0.1622, 0.1605, 0.1582, 0.1570, 0.1563, 0.1560, 0.1552]
    inva = [0.1894, 0.1363, 0.1280, 0.1239, 0.1212, 0.1196, 0.1201, 0.1190, 0.1181, 0.1174,
            0.1152, 0.1152, 0.1136, 0.1126, 0.1113, 0.1106, 0.1100, 0.1096, 0.1090, 0.1095]
    vari = [0.7332, 0.7438, 0.7422, 0.7419, 0.7381, 0.7319, 0.7222, 0.7140, 0.7068, 0.6992,
            0.6940, 0.6882, 0.6848, 0.6821, 0.6806, 0.6787, 0.6775, 0.6765, 0.6760, 0.6754]
    cova = [0.2527, 0.1852, 0.2037, 0.2156, 0.2304, 0.2463, 0.2815, 0.3151, 0.3509, 0.3975,
            0.4386, 0.4865, 0.5173, 0.5407, 0.5590, 0.5739, 0.5821, 0.5931, 0.6043, 0.6073]
    lr = [0.000340, 0.000670, 0.001000, 0.000991, 0.000966, 0.000925, 0.000870, 0.000802, 0.000723, 0.000637,
          0.000547, 0.000454, 0.000364, 0.000278, 0.000199, 0.000131, 0.000076, 0.000035, 0.000010, 0.000001]

    # Style configuration
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # Subplot 1: Total Loss & Learning Rate
    color_loss = '#e67e22'
    color_lr = '#2980b9'
    
    ax1.set_title("SABER Main Encoder Total Loss & Cosine Learning Rate", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Total Loss", color=color_loss, fontsize=11, fontweight='bold')
    line1 = ax1.plot(epochs_enc, loss_enc, color=color_loss, linewidth=2.5, marker='o', label='Total Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.set_xticks(range(1, 21))
    ax1.grid(True, alpha=0.3)
    
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel("Learning Rate", color=color_lr, fontsize=11, fontweight='bold')
    line2 = ax1_twin.plot(epochs_enc, lr, color=color_lr, linewidth=2, linestyle='--', label='Cosine LR')
    ax1_twin.tick_params(axis='y', labelcolor=color_lr)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True)
    
    # Subplot 2: Loss Components Breakdown
    ax2.set_title("Multi-Task Supervised Loss Component Breakdown", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss Component Value", fontsize=11)
    ax2.plot(epochs_enc, rank, color='#e74c3c', linewidth=2, marker='s', label='Neighborhood Rank Loss')
    ax2.plot(epochs_enc, clas, color='#8e44ad', linewidth=2, marker='^', label='Classification Loss')
    ax2.plot(epochs_enc, jaccard, color='#f39c12', linewidth=2, marker='d', label='Jaccard Target Loss')
    ax2.plot(epochs_enc, cova, color='#27ae60', linewidth=1.5, linestyle=':', label='VICReg Covariance')
    ax2.plot(epochs_enc, vari, color='#16a085', linewidth=1.5, linestyle='-.', label='VICReg Variance')
    ax2.plot(epochs_enc, inva, color='#d35400', linewidth=1.5, linestyle='--', label='VICReg Invariance')
    ax2.set_xticks(range(1, 21))
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', frameon=True, fontsize=9)
    
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "encoder_training_curves.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig1_path}")

    # ── 2. CFM Latent Bridge 80-Epoch Data ───────────────────────────
    bridge_sub_epochs = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    bridge_loss = [1.2497, -0.0965, -0.4514, -0.5865, -0.6811, -0.9547, -1.0926, -1.1856, -1.2550, -1.3109,
                   -1.3603, -1.4030, -1.4349, -1.4640, -1.4897, -1.5104, -1.5287, -1.5404, -1.5486, -1.5536]
    bridge_f1_step1 = [73.19, 74.77, 75.20, 75.28, 75.27, 75.06, 74.89, 75.05, 74.92, 75.11,
                       74.95, 75.40, 75.08, 75.08, 75.12, 75.42, 75.27, 75.43, 75.26, 75.19]
    bridge_f1_step10 = [73.33, 74.82, 75.25, 75.32, 75.32, 74.79, 74.71, 74.69, 74.39, 74.58,
                        74.54, 75.17, 74.69, 74.48, 74.73, 75.24, 75.02, 75.26, 75.16, 75.04]

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    
    color_bloss = '#8e44ad'
    color_f1 = '#27ae60'
    
    ax1.set_title("CFM Latent Bridge 80-Epoch Flow Matching Loss & F1@5 Progression", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Bridge Epoch", fontsize=11)
    ax1.set_ylabel("Flow Matching Vector Loss", color=color_bloss, fontsize=11, fontweight='bold')
    line1 = ax1.plot(bridge_sub_epochs, bridge_loss, color=color_bloss, linewidth=2.5, marker='o', label='CFM Vector Loss')
    ax1.tick_params(axis='y', labelcolor=color_bloss)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Cross-Modal F1@5 Accuracy (%)", color=color_f1, fontsize=11, fontweight='bold')
    line2 = ax2.plot(bridge_sub_epochs, bridge_f1_step1, color='#27ae60', linewidth=2, linestyle='-', marker='s', label='1-Step Euler F1@5')
    line3 = ax2.plot(bridge_sub_epochs, bridge_f1_step10, color='#2980b9', linewidth=2, linestyle='--', marker='^', label='10-Step ODE F1@5')
    ax2.tick_params(axis='y', labelcolor=color_f1)
    ax2.set_ylim(70, 78)
    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True)
    
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "cfm_bridge_training_curves.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig2_path}")

    # ── 3. Benchmark Comparison Bar Chart ─────────────────────────────
    metrics_names = ['Precision@5', 'Recall@5', 'F1@5', 'mAP@5', 'Precision@10', 'Recall@10', 'F1@10', 'mAP@10']
    cross_modal = [85.34, 73.73, 76.72, 94.02, 76.42, 75.38, 73.13, 94.02]
    same_modal  = [86.55, 75.30, 78.30, 93.98, 77.96, 76.98, 74.88, 93.98]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    rects1 = ax.bar(x - width/2, cross_modal, width, label='Cross-Modal (SAR S1 -> Optical S2)', color='#e67e22', alpha=0.9)
    rects2 = ax.bar(x + width/2, same_modal, width, label='Same-Modal (Optical S2 -> Optical S2)', color='#27ae60', alpha=0.9)

    ax.set_title("SABER Final Retrieval Metric Benchmark (Held-out Test Split)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=15, fontsize=10)
    ax.set_ylim(65, 100)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, axis='y', alpha=0.3)

    # Add values on top of bars
    for bar in rects1:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar in rects2:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "benchmark_evaluation_comparison.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig3_path}")

if __name__ == '__main__':
    plot_all_curves()
