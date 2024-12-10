import os
import torch
import nibabel
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
from monai.data import MetaTensor
from multiprocessing import Process, Pool
from sklearn.preprocessing import MinMaxScaler 
from monai.transforms import (
    Orientation, 
    EnsureType,
    ConvertToMultiChannelBasedOnBratsClasses,
)
class Brats2021Task1Dataset:
    def __init__(
        self,
        root_dir,
        case_name,
        D,H,W
    ):
        """
        root_dir: path to the data folder where the raw data
        """
        self.root_dir = root_dir
        assert os.path.exists(self.root_dir)
        self.case_name = case_name
        self.MRI_TYPE = ["flair", "t1", "t1ce", "t2", "seg"]
        self.D = D
        self.H = H
        self.W = W
        
    def __len__(self):
        return self.case_name.__len__()

    def get_modality_fp(self, case_name: str, mri_type: str)->str:
        """
        return the modality file path
        case_name: patient ID
        mri_type: any of the ["flair", "t1", "t1ce", "t2", "seg"]
        """
        modality_fp = os.path.join(
            self.root_dir,
            case_name,
            case_name + f"_{mri_type}.nii",
        )
        return modality_fp

    def load_nifti(self, fp)->list:
        """
        load a nifti file
        fp: path to the nifti file with (nii or nii.gz) extension
        """
        nifti_data = nibabel.load(fp)
        # get the floating point array
        nifti_scan = nifti_data.get_fdata()
        # get affine matrix
        affine = nifti_data.affine
        return nifti_scan, affine

    def normalize(self, x:np.ndarray)->np.ndarray:
        # Transform features by scaling each feature to a given range.
        scaler = MinMaxScaler(feature_range=(0, 1))
        # (H, W, D) -> (H * W, D)
        normalized_1D_array = scaler.fit_transform(x.reshape(-1, x.shape[-1]))
        normalized_data = normalized_1D_array.reshape(x.shape)
        return normalized_data

    def orient(self, x: MetaTensor) -> MetaTensor:
        # orient the array to be in (Right, Anterior, Superior) scanner coordinate systems
        assert type(x) == MetaTensor
        return Orientation(axcodes="RAS")(x)

    def detach_meta(self, x: MetaTensor) -> np.ndarray:
        assert type(x) == MetaTensor
        return EnsureType(data_type="numpy", track_meta=False)(x)

    def crop_brats2021_zero_pixels(self, x: np.ndarray)->np.ndarray:
        # get rid of the zero pixels around mri scan and cut it so that the region is useful
        # crop (1, 240, 240, 155) to (1, 128, 128, 128)
        H_start = (x.shape[1] - self.H)//2
        W_start = (x.shape[2] - self.W)//2
        D_start = (x.shape[3] - self.D)//2
        return x[:, H_start:H_start+self.H, W_start:W_start+self.W, D_start:D_start+self.D]

    def preprocess_brats_modality(self, data_fp: str, is_label: bool = False)->np.ndarray:
        """
        apply preprocess stage to the modality
        data_fp: directory to the modality
        """
        data, affine = self.load_nifti(data_fp)
        # label do not the be normalized 
        if is_label:
            # Binary mask does not need to be float64! For saving storage purposes!
            data = data.astype(np.uint8)
            # categorical -> one-hot-encoded 
            # (240, 240, 155) -> (3, 240, 240, 155)
            data = ConvertToMultiChannelBasedOnBratsClasses()(data)
        else:
            data = self.normalize(x=data)
            # (240, 240, 155) -> (1, 240, 240, 155)
            data = data[np.newaxis, ...]
        
        data = MetaTensor(x=data, affine=affine)
        # for oreinting the coordinate system we need the affine matrix
        data = self.orient(data)
        # detaching the meta values from the oriented array
        data = self.detach_meta(data)
        # (240, 240, 155) -> (128, 128, 128)
        data = self.crop_brats2021_zero_pixels(data)
        return data

    def __getitem__(self, idx):
        case_name = self.case_name[idx]
        # e.g: train/BraTS2021_00000/BraTS2021_00000_flair.nii.gz
        
        # preprocess Flair modality
        FLAIR = self.get_modality_fp(case_name, self.MRI_TYPE[0])
        flair = self.preprocess_brats_modality(data_fp=FLAIR, is_label=False)
        flair_transv = flair.swapaxes(1, 3) # transverse plane
        
        # preprocess T1 modality
        T1 = self.get_modality_fp(case_name, self.MRI_TYPE[1])
        t1 = self.preprocess_brats_modality(data_fp=T1, is_label=False)
        t1_transv = t1.swapaxes(1, 3) # transverse plane
        
        # preprocess T1ce modality
        T1ce = self.get_modality_fp(case_name, self.MRI_TYPE[2])
        t1ce = self.preprocess_brats_modality(data_fp=T1ce, is_label=False)
        t1ce_transv = t1ce.swapaxes(1, 3) # transverse plane
        
        # preprocess T2
        T2 = self.get_modality_fp(case_name, self.MRI_TYPE[3])
        t2 = self.preprocess_brats_modality(data_fp=T2, is_label=False)
        t2_transv = t2.swapaxes(1, 3) # transverse plane
        
        # preprocess segmentation label
        Label = self.get_modality_fp(case_name, self.MRI_TYPE[4])
        label = self.preprocess_brats_modality(data_fp=Label, is_label=True)
        label_transv = label.swapaxes(1, 3) # transverse plane

        # stack modalities along the first dimension 
        modalities = np.concatenate(
            (flair_transv,t1_transv, t1ce_transv, t2_transv),
            axis=0,
        )
        label = label_transv
        return modalities, label