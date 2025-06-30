import os
import nibabel as nib

def check_3d_image_size_consistency(root_dir):
    for case_name in os.listdir(root_dir):
        case_path = os.path.join(root_dir, case_name)
        if not os.path.isdir(case_path):
            continue

        nii_shapes = {}
        for file_name in os.listdir(case_path):
            if file_name.endswith('.nii.gz'):
                file_path = os.path.join(case_path, file_name)
                try:
                    img = nib.load(file_path)
                    nii_shapes[file_name] = img.shape
                except Exception as e:
                    print(f"[ERROR] Failed to load {file_path}: {e}")
                    continue

    
        if len(nii_shapes) >= 2:
            shapes = list(nii_shapes.values())
            if not all(s == shapes[0] for s in shapes):
                print(f"[Shape Mismatch] Case: {case_name}")
                for f, s in nii_shapes.items():
                    print(f"  {f}: {s}")

# 使用方法
check_3d_image_size_consistency('dataset/train/')


