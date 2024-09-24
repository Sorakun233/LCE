import matplotlib.pyplot as plt
import numpy as np
def total_concept_mask(concept_masks,shap_list,num_rows,save_path):
    num_cols = int(np.ceil(len(concept_masks) / num_rows))

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))

    for i, mask in enumerate(concept_masks):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].imshow(mask, cmap='gray', interpolation='none')
        axes[row, col].set_title(f'Mask {shap_list[i]:.4f}')

    for i in range(len(concept_masks), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    