import os
import shutil
import random

def prepare_dataset(root_dir, train_ratio=0.8):
    case_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    random.shuffle(case_dirs)

    split_idx = int(len(case_dirs) * train_ratio)
    train_cases = case_dirs[:split_idx]
    test_cases = case_dirs[split_idx:]

    train_dir = os.path.join("data/dataset", 'train')
    test_dir = os.path.join("data/dataset", 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    def move_and_rename_cases(cases, target_dir):
        for case in cases:
            src_case_path = os.path.join(root_dir, case)
            dst_case_path = os.path.join(target_dir, case)
            shutil.move(src_case_path, dst_case_path)

            for filename in os.listdir(dst_case_path):
                src_file = os.path.join(dst_case_path, filename)

                if filename.endswith('_t2w.nii.gz'):
                    dst_file = os.path.join(dst_case_path, 't2w.nii.gz')
                    os.rename(src_file, dst_file)
                elif filename.endswith('_hbv.nii.gz'):
                    dst_file = os.path.join(dst_case_path, 'dwi.nii.gz')
                    os.rename(src_file, dst_file)
                elif filename.endswith('_adc.nii.gz'):
                    dst_file = os.path.join(dst_case_path, 'adc.nii.gz')
                    os.rename(src_file, dst_file)

    move_and_rename_cases(train_cases, train_dir)
    move_and_rename_cases(test_cases, test_dir)

    print(f'Total cases: {len(case_dirs)}')
    print(f'Train cases: {len(train_cases)}')
    print(f'Test cases: {len(test_cases)}')
    print('Dataset preparation completed.')

# Example usage:
# Replace '/path/to/root_dir' with the actual root directory
prepare_dataset('data/crop_image/', train_ratio=0.8)


# import os

# def rename_t2w_files(root_dir):
#     for case in os.listdir(root_dir):
#         case_path = os.path.join(root_dir, case)
#         if not os.path.isdir(case_path):
#             continue  # Skip non-directory files

#         for file in os.listdir(case_path):
#             if file.endswith('_t2w.nii.gz'):
#                 old_path = os.path.join(case_path, file)
#                 new_path = os.path.join(case_path, 't2w.nii.gz')
#                 os.rename(old_path, new_path)
#                 print(f'Renamed: {old_path} -> {new_path}')

# # 示例调用
# # replace '/path/to/root_dir' with your actual path
# rename_t2w_files("data/dataset/test")


# import os
# import shutil

# def move_labels_to_train_cases(train_dir, label_dir):
#     for case_id in os.listdir(train_dir):
#         case_path = os.path.join(train_dir, case_id)
#         if not os.path.isdir(case_path):
#             continue

#         # 在 label 目录中查找以 case_id 开头的文件
#         for file in os.listdir(label_dir):
#             if file.startswith(case_id):
#                 src_label_path = os.path.join(label_dir, file)
#                 dst_label_path = os.path.join(case_path, 'whole_gland.nii.gz')

#                 shutil.copy(src_label_path, dst_label_path)
#                 print(f"Copied {src_label_path} -> {dst_label_path}")
#                 break  # 一旦找到一个匹配就退出当前 case_id 的搜索
#         else:
#             print(f"Warning: No label found for case {case_id}")

# # 示例调用
# # 替换为你真实的路径
# move_labels_to_train_cases("data/dataset/train", 'data/crop_label/whole_gland')



# import os
# import shutil

# def copy_labels_merged(train_dir, test_dir, manual_label_root, ai_label_root, output_txt_path):
#     all_cases = []

#     # Collect train and test cases
#     for subset_dir in [train_dir, test_dir]:
#         for case_name in os.listdir(subset_dir):
#             case_path = os.path.join(subset_dir, case_name)
#             if not os.path.isdir(case_path):
#                 continue

#             # Try manual label first
#             label_file = None
#             label_type = -1  # -1 means not found

#             # Search in manual labels
#             for fname in os.listdir(manual_label_root):
#                 if fname.startswith(case_name + "_") and fname.endswith(".nii.gz"):
#                     label_file = os.path.join(manual_label_root, fname)
#                     label_type = 1
#                     break

#             # If not found in manual, search in AI labels
#             if label_file is None:
#                 for fname in os.listdir(ai_label_root):
#                     if fname.startswith(case_name + "_") and fname.endswith(".nii.gz"):
#                         label_file = os.path.join(ai_label_root, fname)
#                         label_type = 0
#                         break

#             # Skip if still not found
#             if label_file is None:
#                 print(f"[Warning] No label found for case {case_name}")
#                 continue

#             # Copy and rename to lesion.nii.gz
#             dst_path = os.path.join(case_path, "lesion.nii.gz")
#             shutil.copy(label_file, dst_path)

#             # Record result
#             all_cases.append(f"{case_name} {label_type}")

#     # Save summary file
#     with open(output_txt_path, 'w') as f:
#         for entry in all_cases:
#             f.write(entry + '\n')

#     print(f"[Info] Finished copying labels. Total cases processed: {len(all_cases)}.")
#     print(f"[Info] Label source record saved to: {output_txt_path}")


# copy_labels_merged(
#     train_dir="data/dataset/train",
#     test_dir="data/dataset/test",
#     manual_label_root='data/crop_label/lesion/AI',
#     ai_label_root='data/crop_label/lesion/human_expert',
#     output_txt_path='label_human_or_AI.txt'
# )



