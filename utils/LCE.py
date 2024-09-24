from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from torchcam.utils import overlay_mask
from itertools import product
import torch.nn.functional as F
from lime import lime_image


def gen_concept_masks(gen_model, target_img):
    return gen_model.generate(target_img)


def get_points_weight(main_model, target_img, point_p, transform_norm, device, save_dir):
    input_tensor = transform_norm(target_img).unsqueeze(0).to(device)
    targets = None
    target_layers = [main_model.layer4[-1]]
    cam = GradCAM(model=main_model, target_layers=target_layers)
    cam_map = cam(input_tensor=input_tensor, targets=targets)[0]
    pil_target_img = Image.fromarray(target_img)
    pil_target_img.save(save_dir.replace('_with_cam', ''))
    result = overlay_mask(pil_target_img, Image.fromarray(cam_map), alpha=0.6)
    result.save(save_dir)
    img_map_array = np.array(cam_map)
    flattened_indices = np.argsort(img_map_array.flatten())[::-1]
    sorted_coordinates = np.column_stack(np.unravel_index(flattened_indices, img_map_array.shape))
    indices = (np.array(point_p) * 0.01 * sorted_coordinates.shape[0]).astype(int)
    coordinates = sorted_coordinates[indices]
    return coordinates


def batch_predict(images, image_norm, device, cls_model):
    batch = torch.stack(tuple(image_norm(i) for i in images), dim=0)
    batch = batch.to(device)
    with torch.no_grad():
        logits = cls_model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


def get_LIME_mask(img_pil, image_norm, image_reshape, device, cls_model, top_labels, mask_num):
    batch_predict_with_param = lambda image: batch_predict(image, image_norm, device, cls_model=cls_model)
    test_pred = batch_predict([image_reshape(img_pil)], image_norm, device, cls_model)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(image_reshape(img_pil)),
                                             batch_predict_with_param,
                                             top_labels=top_labels,
                                             hide_color=None,
                                             num_features=15,
                                             num_samples=5000)
    Lime_mask_list = []
    temp_mask = None
    for i in range(1, mask_num + 1):
        temp, mask = explanation.get_image_and_mask(test_pred.squeeze().argmax(), positive_only=True, num_features=i,
                                                    hide_rest=False)
        if i == 1:
            concept_mask = mask
        else:
            concept_mask = mask - temp_mask
        Lime_mask_list.append(concept_mask)
        temp_mask = mask
    return Lime_mask_list


def get_rectangle_point(Lime_mask_list):
    coorinates_rectangle_list = []
    for mask in Lime_mask_list:
        nonzero_indices = np.argwhere(mask)
        if len(nonzero_indices) == 0:
            continue
        min_row, min_col = np.min(nonzero_indices, axis=0)
        max_row, max_col = np.max(nonzero_indices, axis=0)
        coorinates_rectangle_list.append([min_col, min_row, max_col, max_row])
    return coorinates_rectangle_list


def gen_concept_masks_LIME(gen_model, image_norm, image_reshape, device, cls_model, target_img, img_pil, top_labels,
                           mask_num):
    lime_mask_list = get_LIME_mask(img_pil, image_norm, image_reshape, device, cls_model, top_labels, mask_num)
    coorinates_rectangle_list = get_rectangle_point(lime_mask_list)
    rectangle_mask_list = []
    gen_model.set_image(target_img)
    for coor in coorinates_rectangle_list:
        input_box = np.array(coor)
        masks, _, _ = gen_model.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        rectangle_mask_list.append(masks[0])
    return rectangle_mask_list


def gen_concept_masks_GradCAM(main_model, gen_model, target_img, point_p, transform_norm, device, save_dir):
    coordinates_chosen = get_points_weight(main_model, target_img, point_p, transform_norm, device, save_dir)
    mask_list = []
    gen_model.set_image(target_img)
    for i in range(0, coordinates_chosen.shape[0]):
        x = coordinates_chosen[i][1]
        y = coordinates_chosen[i][0]
        input_point = np.array([[x, y]])
        input_label = np.array([1])
        masks, scores, logit = gen_model.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        max_score_index = np.argmax(scores)
        mask_list.append(masks[max_score_index])
    return mask_list


def calculate_intersection(mask1, mask2):
    intersection = mask1 & mask2
    intersection_area = intersection.sum()
    return intersection_area / mask1.sum()


def merge_masks(mask1, mask2):
    return mask1 | mask2


def save_img(process_mask_list, save_name, file_name):
    plt.figure(figsize=(10, 5))
    for i in range(len(process_mask_list)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(process_mask_list[i], cmap='gray')
        plt.title(f'Mask {i + 1}')
    plt.tight_layout()
    plt.savefig(f'result/{save_name}/{file_name}/no_processed_masks.png')
    plt.close()


def process_masks(rectangle_mask_list, point_mask_list, save_name, file_name):
    process_mask_list = rectangle_mask_list + point_mask_list
    save_img(process_mask_list, save_name, file_name)
    i = 0
    while i < len(process_mask_list):
        j = i + 1
        while j < len(process_mask_list):
            intersection_ratio = calculate_intersection(process_mask_list[i], process_mask_list[j])
            if intersection_ratio >= 0.9:
                merged_mask = merge_masks(process_mask_list[i], process_mask_list[j])
                process_mask_list[i] = merged_mask
                del process_mask_list[j]
            else:
                j += 1
        i += 1
    return process_mask_list


def replace_zeros_with_random(image):
    random_values = np.random.randint(0, 256, size=image.shape, dtype=np.uint8)
    replaced_image = np.where(image == 0, random_values, image)
    return replaced_image


def LCE(model, img_numpy, image_class, concept_masks, image_norm=None):
    feat_num = len(concept_masks)
    shap_val = []
    permutations = np.array(list(product([0, 1], repeat=feat_num)))
    for i in range(feat_num):
        masks_tmp = concept_masks.copy()

        bin_x_tmp = permutations.copy()
        bin_x_tmp_sec = permutations.copy()

        bin_x_tmp[:, i] = 1
        bin_x_tmp_sec[:, i] = 0

        new_mask = np.array([masks_tmp[np.where(i)[0]].sum(0) for i in bin_x_tmp]).astype(bool)
        new_mask_sec = np.array([masks_tmp[np.where(i)[0]].sum(0) for i in bin_x_tmp_sec]).astype(bool)

        target_img = np.expand_dims(img_numpy, 0)

        new_mask = np.expand_dims(new_mask, 3)
        new_mask_sec = np.expand_dims(new_mask_sec, 3)

        masked_image_1 = target_img * new_mask
        masked_image_2 = target_img * new_mask_sec


        batch_img_1 = []
        batch_img_2 = []
        for i in range(masked_image_1.shape[0]):
            input_tensor_1 = image_norm(masked_image_1[i])
            input_tensor_2 = image_norm(masked_image_2[i])
            batch_img_1.append((input_tensor_1.cpu()))
            batch_img_2.append((input_tensor_2.cpu()))
        tmp_dl_1 = DataLoader(dataset=batch_img_1, batch_size=feat_num, shuffle=False)
        tmp_dl_2 = DataLoader(dataset=batch_img_2, batch_size=feat_num, shuffle=False)

        batch_results_1 = []
        batch_results_2 = []
        for x in tmp_dl_1:
            with torch.no_grad():
                pre_shap = nn.functional.softmax(model(x.cuda()), dim=1)[:, image_class]
                batch_results_1.append(pre_shap)
        for x in tmp_dl_2:
            with torch.no_grad():
                pre_shap = nn.functional.softmax(model(x.cuda()), dim=1)[:, image_class]
                batch_results_2.append(pre_shap)

        batch_results_arra_1 = torch.cat(batch_results_1, dim=0)
        batch_results_arra_2 = torch.cat(batch_results_2, dim=0)

        pre_shap = (batch_results_arra_1 - batch_results_arra_2).detach().cpu().numpy()
        shap_val.append(pre_shap.sum() / feat_num)

    ans = shap_val.index(max(shap_val))
    shap_list = shap_val
    shap_list = np.array(shap_list)
    shap_arg = np.argsort(-shap_list)
    auc_mask = np.expand_dims(
        np.array([concept_masks[shap_arg[:i + 1]].sum(0) for i in range(len(shap_arg))]).astype(bool), 3)
    return auc_mask, shap_list, shap_arg
