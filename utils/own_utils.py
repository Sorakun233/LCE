from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.ioff()
from PIL import Image
import numpy as np
import csv
from torchvision import datasets, models, transforms
import os
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

def get_weighted_result(save_name,image_reshape,file_name,insertion,deletion,ori_img,mask_list):
    processed_img = np.array(image_reshape(ori_img))
    w,h,_ = processed_img.shape
    mask_num,_,_,_ = mask_list.shape
    p_o = w*h
    p_s = np.sum(mask_list)/mask_num
    Insertion_w = (1-p_s/p_o)*insertion
    Deletion_w = (1-p_s/p_o)*(100-deletion)
    EffectScore = (Insertion_w+Deletion_w)/2

    print(f"p_s: {p_s}")
    print(f"p_o: {p_o}")
    print(f"1-p_s/p_o: {1-p_s/p_o:.4f}")
    print(f"Insertion_w: {Insertion_w:.4f}")
    print(f"Deletion_w: {Deletion_w:.4f}")
    print(f"Score: {EffectScore:.4f}")
    
def get_concept_img(concept_masks,ori_img,save_path,file_name):
    input_img  = np.expand_dims(ori_img,0)
    concept_masks= np.expand_dims(concept_masks,-1)
    val_img_numpy = (input_img  * concept_masks).astype(np.uint8)

    for i in range(val_img_numpy.shape[0]): 
        img_data = val_img_numpy[i]
        img = Image.fromarray(np.uint8(img_data))
        save_path_tmp = f'result/{save_path}/{file_name}/concept_mask/{i}.png'
        os.makedirs(os.path.dirname(save_path_tmp), exist_ok=True)
        img.save(save_path_tmp)

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
    
def replace_zeros_with_random(image):
    random_values = np.random.randint(0, 256, size=image.shape, dtype=np.uint8)
    replaced_image = np.where(image == 0, random_values, image)
    return replaced_image
def get_batch(mask,image_norm,metric_type,save_path,file_name,input_image_copy):
    if metric_type == "Deletion":
        auc_mask = 1 - mask
    else:
        auc_mask = mask
    val_img_numpy = np.expand_dims(input_image_copy,0)
    val_img_numpy = (val_img_numpy * auc_mask).astype(np.uint8)
    batch_img = []
    for i in range(val_img_numpy.shape[0]): 
        img_data = val_img_numpy[i]
        img = Image.fromarray(np.uint8(img_data))
        save_path_tmp = f'result/{save_path}/{file_name}/{metric_type}/{i}.png'
        os.makedirs(os.path.dirname(save_path_tmp), exist_ok=True)
        img.save(save_path_tmp)
        batch_img.append(image_norm(val_img_numpy[i,:,:,:]))
    batch_img = torch.stack(batch_img).cuda()
    return batch_img

def get_auc(batch_img,cvmodel,save_path,file_name,probabilitie_org,image_class,y,metric_type):
    yes_total = 0
    auc_total = 0.0
    with torch.no_grad():
        new_out = torch.nn.functional.softmax(cvmodel(batch_img[0].unsqueeze(0).cuda()),dim=1)
        now_class = int(torch.argmax(new_out))
        out = torch.nn.functional.softmax(cvmodel(batch_img),dim=1)[:,image_class]
    out = out.cpu().numpy()
    out[out>= probabilitie_org] = probabilitie_org ### norm the upper bound of output to the original acc
    out = out/probabilitie_org
    x_axis = np.linspace(0, 1, out.shape[0]) *100
    if x_axis.shape[0] == 1:
        auc_tmp = float(out)
    else:
        auc_tmp = float(metrics.auc(x_axis, out))
    auc_total = auc_total + auc_tmp
    if image_class == y:
        yes_total += 1
        

    x_axis = x_axis/100
    plt.plot(x_axis, out, label=f'ROC curve (area = {auc_tmp:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(metric_type)
    plt.legend(loc='lower right')
    plt.savefig(f'result/{save_path}/{file_name}/{metric_type}/{metric_type}.png')
    plt.close()
    return auc_tmp