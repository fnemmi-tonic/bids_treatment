# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:07:50 2020

@author: federico.nemmi
"""

from glob import glob
from os.path import isdir
import pandas as pd
from pathlib import PurePath
from os.path import splitext
from dipy.segment.mask import median_otsu
import nibabel as nb
import numpy as np
import os

def create_tsv(bids_dir):
    """Create a tsv file with one line per subject per visit with True/False for several modalites        
        Parameters
        ----------
        bids_dir : absolute path pointing at the root of the bids repository
        
        """
    subjects = [el for el in glob("{}/*".format(bids_dir)) if isdir(el) and "derivatives" not in el]
    info_df = pd.DataFrame(columns = ["subjects", "visit", "t1", "t2", "diff", "pd", "t2star"])
    for s in subjects:
        s = PurePath(s)
        subject_id = s.parts[-1]
        visits = glob("{}/*".format(s))
        for v in visits:
            v = PurePath(v)
            visit = int(v.parts[-1].split("-")[1])
            if len(glob("{}/anat/*T1w*".format(v))) != 0:
                t1_presence = True
            else:
                t1_presence = False
            if len(glob("{}/anat/*T2w*".format(v))) != 0:
                t2_presence = True
            else:
                t2_presence = False
            if len(glob("{}/anat/*PD*".format(v))) != 0:
                pd_presence = True
            else:
                pd_presence = False
            if len(glob("{}/anat/*T2star*".format(v))) != 0:
                t2star_presence = True
            else:
                t2star_presence = False
            if len(glob("{}/dwi/*dwi*".format(v))) != 0:
                diff_presence = True
            else:
                diff_presence = False
            line = [subject_id, visit, t1_presence, t2_presence, diff_presence, pd_presence, t2star_presence]
            sub_df = pd.DataFrame([line], columns = ["subjects", "visit", "t1", "t2", "diff", "pd", "t2star"])
            info_df = pd.concat([info_df, sub_df])
            
    info_df.to_csv("{}/participants.tsv".format(bids_dir), sep = "\t", index = False)
        
# def skip_if_subject_done(done_dict, subject_visit):
#     if subject_visit in done_dict.keys():
#         otp = False
#     else:
#         otp = True
#     return(otp)

def get_base_name_all_type(filename):
    """Get the basename (i.e. the filename WITHOUT the extension) of both nii and nii.gz nifti file        
        Parameters
        ----------
        filename : path or pathlike obect to the nifti file
        
        """
    base_name = splitext(PurePath(filename).parts[-1])[0]
    if "gz" in filename:
        base_name = splitext(PurePath(base_name).parts[-1])[0]
    return(base_name)

def get_mask_crop_and_indexes(img,
                              median_radius = 3, 
                              numpass = 1,
                              autocrop = False,
                              dilate = 2):
    """This function use the dipy.segmentation.median_otsu function to derive a brain mask and mask input data, and calculate a
        cropped version of both the mask and the masked data. It also return indexes of the cropped mask in original array coordinates.
        This is usually used to denoise or fit data in the cropped version (that is faster) and write them in an array of dimension
        equal to the original
    Parameters
        ----------
        img: nifti object from nibabel, nilearn or nipy
        ...: options from median_otsu
        
    Returns
        ----------
        x_ix, y_ix, z_ix: arrays of indexes of the cropped mask IN THE COORDINATES OF THE ORIGINAL ARRAY
        mask: boolean array of the otsu derived mask
        maskdata: array, the masked original array
        mask_crop: boolean array of the cropped otsu derived mask
        maskdata_crop: array, the cropped masked original array
        
        
        """
    img_data = img.get_fdata()
    maskdata, mask = median_otsu(img_data, 
                                        vol_idx = range(0, img_data.shape[3]), 
                                        median_radius = median_radius,
                                        numpass = numpass, 
                                        autocrop = autocrop,
                                        dilate = dilate)
    
    z_slices = np.any(mask, (0,1))
    y_slices = np.any(mask, (0,2))
    x_slices = np.any(mask, (1,2))
    x_ix, y_ix, z_ix = np.ix_(x_slices, y_slices, z_slices)
    mask_crop = mask[x_ix, y_ix, z_ix]
    maskdata_crop = maskdata[x_ix, y_ix, z_ix, :]
    return (x_ix, y_ix, z_ix, mask, maskdata, mask_crop,  maskdata_crop)


def separate_crossectional_longitudinal(participants_df, modalities):
    df = pd.read_table(participants_df)
    if len(modalities) == 1:
        mod_selected_df = df[df[modalities[0]]]
    else:
        modalities_series = df[modalities].apply(np.all, axis = 1)
        mod_selected_df = df[modalities_series]
        long_subjects = mod_selected_df["subjects"][mod_selected_df["subjects"].duplicated()]
        long_df = mod_selected_df[mod_selected_df["subjects"].isin(long_subjects)]
        cross_df = mod_selected_df[~mod_selected_df["subjects"].isin(long_subjects)]
        return(long_df, cross_df)
    