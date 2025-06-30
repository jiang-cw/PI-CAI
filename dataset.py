import os
import numpy as np
import torch
from torch.utils.data import Dataset

import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage
from scipy.ndimage import zoom, label  
from typing import Optional, List, Dict


# Default interpolation types
_INTERPOLATOR_IMAGE = 'linear'
_INTERPOLATOR_LABEL = 'linear'

def eliminate_false_positives(volume: np.ndarray) -> Optional[np.ndarray]:
    """Keep only the largest connected component in the volume."""
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels_out, num_labels = label(volume, structure=structure)

    if num_labels == 0:
        return np.zeros_like(volume)

    region_sizes = [np.sum(labels_out == i) for i in range(1, num_labels + 1)]
    max_region_id = np.argmax(region_sizes) + 1
    return (labels_out == max_region_id).astype(np.uint8)



def txt2list(path: str) -> List[str]:
    """Read a list of file paths from a text file."""
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines()]

def resize_image_itk(itk_image: sitk.Image, new_size: List[int], interpolator: str) -> sitk.Image:
    """Resize an ITK image to a new size using the specified interpolation method."""
    interpolator_dict = {
        'nearest': sitk.sitkNearestNeighbor,
        'linear': sitk.sitkLinear,
        'gaussian': sitk.sitkGaussian,
        'label_gaussian': sitk.sitkLabelGaussian,
        'bspline': sitk.sitkBSpline,
        'hamming_sinc': sitk.sitkHammingWindowedSinc,
        'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
        'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
        'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
    }

    original_size = np.array(itk_image.GetSize(), dtype=float)
    original_spacing = np.array(itk_image.GetSpacing(), dtype=float)

    if len(original_size) == 2:
        target_size = np.array([new_size[0], new_size[1]], dtype=float)
    else:
        target_size = np.array([new_size[0], new_size[1], original_size[2]], dtype=float)

    scale_factor = original_size / target_size
    new_spacing = original_spacing * scale_factor
    target_size = target_size.astype(int)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(itk_image)
    resampler.SetSize(target_size.tolist())
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(interpolator_dict[interpolator])

    return resampler.Execute(itk_image)

class ResizeTrain:
    """Resize image and label during training."""

    def __init__(self, new_size: List[int], check: bool = True):
        self.name = 'ResizeTrain'
        self.new_size = new_size
        self.check = check

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        image = sample['image']
        label = sample['label']

        image_itk = sitk.GetImageFromArray(image)
        label_itk = sitk.GetImageFromArray(label)

        if self.check:
            image_itk = resize_image_itk(image_itk, self.new_size, _INTERPOLATOR_IMAGE)
            label_itk = resize_image_itk(label_itk, self.new_size, _INTERPOLATOR_LABEL)

        return {
            'image': sitk.GetArrayFromImage(image_itk),
            'label': sitk.GetArrayFromImage(label_itk)
        }

class Resize:
    """Resize T2W, DWI, ADC and lesion images to the target size."""

    def __init__(self, new_size: List[int], check: bool = True):
        self.name = 'Resize'
        self.new_size = new_size
        self.check = check

    def __call__(self, sample: Dict[str, sitk.Image]) -> Dict[str, sitk.Image]:
        if self.check:
            return {
                't2w': resize_image_itk(sample['t2w'], self.new_size, _INTERPOLATOR_IMAGE),
                'dwi': resize_image_itk(sample['dwi'], self.new_size, _INTERPOLATOR_IMAGE),
                'adc': resize_image_itk(sample['adc'], self.new_size, _INTERPOLATOR_IMAGE),
                'lesion': resize_image_itk(sample['lesion'], self.new_size, _INTERPOLATOR_LABEL)
            }
        else:
            return sample



def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # if random.random() > 0.5:
        #     image, label = random_rot_flip(image, label)
        # elif random.random() > 0.5:
        #     image, label = random_rotate(image, label)
        res,x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


def Normalization(image):
    """
    Normalize an image (either NumPy array or SimpleITK Image) to 0 - 255 (8bits)
    """
    if isinstance(image, np.ndarray):
        image = sitk.GetImageFromArray(image)

    normalizeFilter = sitk.NormalizeImageFilter()
    rescaleFilter = sitk.RescaleIntensityImageFilter()
    rescaleFilter.SetOutputMaximum(255)
    rescaleFilter.SetOutputMinimum(0)

    image = normalizeFilter.Execute(image)
    image = rescaleFilter.Execute(image)

    return image





class test_dataset(Dataset):
    def __init__(self, data_dir, transform=[Resize([128, 128], check=True)]):
        self.case_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.bit = sitk.sitkFloat32
        self.transform = transform

    def __len__(self):
        return len(self.case_dirs)

    def read_image(self, path):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        return reader.Execute()

    def __getitem__(self, idx):
        case_path = self.case_dirs[idx]
        name = os.path.basename(case_path)

        t2w = self.read_image(os.path.join(case_path, 't2w.nii.gz'))
        dwi = self.read_image(os.path.join(case_path, 'dwi.nii.gz'))
        adc = self.read_image(os.path.join(case_path, 'adc.nii.gz'))
        lesion = self.read_image(os.path.join(case_path, 'lesion.nii.gz'))

        Direction, Origin, Spacing = t2w.GetDirection(), t2w.GetOrigin(), t2w.GetSpacing()

        # Normalize images
        t2w = Normalization(t2w)
        dwi = Normalization(dwi)
        adc = Normalization(adc)

        # Cast all images to float
        cast = sitk.CastImageFilter()
        cast.SetOutputPixelType(self.bit)
        t2w = cast.Execute(t2w)
        dwi = cast.Execute(dwi)
        adc = cast.Execute(adc)
        lesion = cast.Execute(lesion)

        # Call transforms on sitk.Image BEFORE converting to NumPy
        sample = {
            't2w': t2w,
            'dwi': dwi,
            'adc': adc,
            'lesion': lesion
        }

        if self.transform:
            for t in self.transform:
                sample = t(sample)

        # Convert to NumPy
        t2w_np = sitk.GetArrayFromImage(sample['t2w'])
        dwi_np = sitk.GetArrayFromImage(sample['dwi'])
        adc_np = sitk.GetArrayFromImage(sample['adc'])
        lesion_np = np.abs(np.around(sitk.GetArrayFromImage(sample['lesion'])))

        # Transpose and add channel dim
        t2w_np = np.transpose(t2w_np, (2, 1, 0))[np.newaxis, ...]
        dwi_np = np.transpose(dwi_np, (2, 1, 0))[np.newaxis, ...]
        adc_np = np.transpose(adc_np, (2, 1, 0))[np.newaxis, ...]
        lesion_np = np.transpose(lesion_np, (2, 1, 0))[np.newaxis, ...]

        img = np.concatenate([t2w_np, dwi_np, adc_np], axis=0)
        info = [Direction, Origin, Spacing, name]

        return torch.from_numpy(img), torch.from_numpy(lesion_np), info

    
class train_dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.slice_infos = []

        self._build_index()

    
    def read_image(self, path):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        return reader.Execute()

    def _build_index(self):
        for case_name in os.listdir(self.data_dir):
            case_path = os.path.join(self.data_dir, case_name)
            if not os.path.isdir(case_path):
                continue
            try:
                t2w_path = os.path.join(case_path, 't2w.nii.gz')
                t2w = nib.load(t2w_path).get_fdata()
                num_slices = t2w.shape[2]
                for idx in range(num_slices):
                    self.slice_infos.append((case_path, idx))
            except Exception as e:
                print(f"Failed to process {case_path}: {e}")

    def __len__(self):
        return len(self.slice_infos)

    def __getitem__(self, index):
        case_path, slice_idx = self.slice_infos[index]

        t2w_img = self.read_image(os.path.join(case_path, 't2w.nii.gz'))
        dwi_img = self.read_image(os.path.join(case_path, 'dwi.nii.gz'))
        adc_img = self.read_image(os.path.join(case_path, 'adc.nii.gz'))
        lesion_img = self.read_image(os.path.join(case_path, 'lesion.nii.gz'))


     
        t2w_slice_img = t2w_img[:, :, slice_idx]
        dwi_slice_img = dwi_img[:, :, slice_idx]
        adc_slice_img = adc_img[:, :, slice_idx]
        lesion_slice_img = lesion_img[:, :, slice_idx]

        t2w_slice_img = Normalization(t2w_slice_img)
        dwi_slice_img = Normalization(dwi_slice_img)
        adc_slice_img = Normalization(adc_slice_img)

        lesion_slice = sitk.GetArrayFromImage(lesion_slice_img)
        lesion_slice = np.around(lesion_slice).astype(np.float32)


        t2w_slice = sitk.GetArrayFromImage(t2w_slice_img)
        dwi_slice = sitk.GetArrayFromImage(dwi_slice_img)
        adc_slice = sitk.GetArrayFromImage(adc_slice_img)

        # Stack modalities along channel dimension
        data = np.stack([t2w_slice, dwi_slice, adc_slice], axis=0).astype(np.float32)

        # print(f"[DEBUG] data range: min={data.min()}, max={data.max()}") 
        # print(f"[DEBUG] label range: min={lesion_slice.min()}, max={lesion_slice.max()}") 


        sample = {
            'image': data,
            'label': lesion_slice,
            'case_name': os.path.basename(case_path),
            'slice_idx': slice_idx
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
