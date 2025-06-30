import os
import nibabel as nib
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage.transform import resize

# Define root directories
t2_root = "data/aligned_image"
bosma_mask_root = "data/label/AI/Bosma22a"
output_root = "data/label/AI/refine_sam"
os.makedirs(output_root, exist_ok=True)

# Find all Bosma mask files
bosma_mask_paths = sorted(glob(os.path.join(bosma_mask_root, "*.nii.gz")))

# Dummy SAM predictor function (to be replaced with actual MedSAM inference)
def dummy_sam_predict(image_resized, box):
    # This is a placeholder; replace it with actual MedSAM output
    return image_resized > np.mean(image_resized)

# Loop through each Bosma mask
for bosma_mask_path in tqdm(bosma_mask_paths):
    try:
        case_id = os.path.basename(bosma_mask_path).replace(".nii.gz", "")
        t2_search_path = os.path.join(t2_root, case_id, "*_t2w.nii.gz")
        t2_files = glob(t2_search_path)

        if len(t2_files) == 0:
            print(f"⚠️ No T2 file found for {case_id}, skipping.")
            continue

        t2_path = t2_files[0]

        # Load T2 image and Bosma mask
        t2_nii = nib.load(t2_path)
        t2_img = t2_nii.get_fdata()
        t2_affine = t2_nii.affine

        mask_nii = nib.load(bosma_mask_path)
        bosma_mask = mask_nii.get_fdata()
        bosma_mask = (bosma_mask > 0).astype(np.uint8)

        # Check shape consistency
        assert t2_img.shape == bosma_mask.shape, f"Shape mismatch: {t2_img.shape} vs {bosma_mask.shape}"

        H, W, D = t2_img.shape
        refined_volume = np.zeros_like(bosma_mask)

        for i in range(D):
            img_slice = t2_img[:, :, i]
            mask_slice = bosma_mask[:, :, i]

            if np.sum(mask_slice) == 0:
                continue

            # Resize image and mask to 256x256
            img_resized = resize(img_slice, (256, 256), preserve_range=True, anti_aliasing=True)
            mask_resized = resize(mask_slice, (256, 256), order=0, preserve_range=True, anti_aliasing=False)
            mask_resized = (mask_resized > 0.5).astype(np.uint8)

            # Generate bounding box from mask
            pos = np.argwhere(mask_resized)
            y1, x1 = pos.min(axis=0)
            y2, x2 = pos.max(axis=0)
            box_prompt = [x1, y1, x2, y2]

            # Predict using MedSAM (replace dummy function when ready)
            sam_mask = dummy_sam_predict(img_resized, box=box_prompt)
            sam_mask = sam_mask.astype(np.uint8)

            # Resize SAM mask back to original size
            sam_mask_restored = resize(sam_mask, (H, W), order=0, preserve_range=True, anti_aliasing=False)
            sam_mask_restored = (sam_mask_restored > 0.5).astype(np.uint8)

            # Combine SAM mask with Bosma mask
            final_mask = np.logical_or(sam_mask_restored, mask_slice).astype(np.uint8)
            refined_volume[:, :, i] = final_mask

        # Save refined mask
        output_path = os.path.join(output_root, f"{case_id}.nii.gz")
        refined_nii = nib.Nifti1Image(refined_volume.astype(np.uint8), affine=t2_affine)
        nib.save(refined_nii, output_path)
        print(f"✅ Refined mask saved for {case_id}")

    except Exception as e:
        print(f"❌ Error processing {bosma_mask_path}: {e}")
