import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

def crop_3d_image_with_mask(mask_img, image_img, crop_size=(256, 256)):
    mask_array = sitk.GetArrayFromImage(mask_img)  # [z, y, x]
    image_array = sitk.GetArrayFromImage(image_img)

    z_indices = np.any(mask_array > 0, axis=(1, 2))
    non_zero_z = np.where(z_indices)[0]
    if len(non_zero_z) == 0:
        return None

    z_min, z_max = non_zero_z[0], non_zero_z[-1] + 1
    mask_array = mask_array[z_min:z_max]
    image_array = image_array[z_min:z_max]

    nonzero = np.argwhere(mask_array > 0)
    center_z, center_y, center_x = np.mean(nonzero, axis=0).astype(int)
    _, y, x = image_array.shape
    h, w = crop_size

    y1 = max(center_y - h // 2, 0)
    y2 = min(y1 + h, y)
    x1 = max(center_x - w // 2, 0)
    x2 = min(x1 + w, x)

    cropped = image_array[:, y1:y2, x1:x2]

    pad_y = h - cropped.shape[1]
    pad_x = w - cropped.shape[2]
    if pad_y > 0 or pad_x > 0:
        cropped = np.pad(cropped,
                         ((0, 0), (0, pad_y), (0, pad_x)),
                         mode='constant', constant_values=0)

    cropped_img = sitk.GetImageFromArray(cropped)
    cropped_img.SetSpacing(image_img.GetSpacing())
    cropped_img.SetOrigin(image_img.GetOrigin())
    cropped_img.SetDirection(image_img.GetDirection())
    return cropped_img

def batch_crop_images(mask_root, image_root, output_root, crop_size=(256, 256)):
    os.makedirs(output_root, exist_ok=True)
    t2_image_infos = []

    # Step 1: Find all T2 images and extract case_id AND folder_name
    for folder in sorted(os.listdir(image_root)):
        folder_path = os.path.join(image_root, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith("_t2w.nii.gz"):
                case_id = file.replace("_t2w.nii.gz", "")
                t2_image_infos.append((case_id, folder, os.path.join(folder_path, file)))

    # Step 2: Loop through each case
    for case_id, folder_name, t2_image_path in tqdm(t2_image_infos, desc="Cropping cases"):
        image_dir = os.path.dirname(t2_image_path)
        output_dir = os.path.join(output_root, folder_name)  # <-- use folder_name instead of case_id
        os.makedirs(output_dir, exist_ok=True)

        mask_path = os.path.join(mask_root, f"{case_id}.nii.gz")
        if not os.path.exists(mask_path):
            print(f"[Skip] Mask not found for {case_id}")
            continue

        try:
            mask_img = sitk.ReadImage(mask_path)
        except Exception as e:
            print(f"[Error] Reading mask for {case_id}: {e}")
            continue

        # Step 3: Process all modalities under the folder
        for file in os.listdir(image_dir):
            if file.endswith(".nii.gz"):
                input_path = os.path.join(image_dir, file)
                try:
                    image_img = sitk.ReadImage(input_path)
                    cropped_img = crop_3d_image_with_mask(mask_img, image_img, crop_size)
                    if cropped_img is not None:
                        sitk.WriteImage(cropped_img, os.path.join(output_dir, file))
                except Exception as e:
                    print(f"[Error] Failed processing {input_path}: {e}")


batch_crop_images(
    mask_root="data/label/whole_gland/AI/Bosma22b",
    image_root="data/aligned_image",
    output_root="data/crop_image",
    crop_size=(256, 256)
)
