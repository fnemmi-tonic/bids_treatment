# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:07:50 2020

@author: federico.nemmi
"""

from glob import glob
from os.path import isdir
import pandas as pd
from pathlib import PurePath
from os.path import splitext, isfile
from dipy.segment.mask import median_otsu
import nibabel as nb
from skimage.morphology import reconstruction, binary_dilation, square
from nipype.interfaces.fsl.preprocess import BET
import numpy as np
from os import remove

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


def crop_and_indexes(mask, data):
    """This function crop the inout image and returns the indexes of the cropped image in original array coordinates.
        This is usually used to denoise or fit data in the cropped version (that is faster) and write them in an array of dimension
        equal to the original
    Parameters
        ----------
        mask: nifti object from nibabel, nilearn or nipy of the mask
        data: nifti object of the 4d diffusion data
    Returns
        ----------
        x_ix, y_ix, z_ix: arrays of indexes of the cropped mask IN THE COORDINATES OF THE ORIGINAL ARRAY
        mask_crop: boolean array of the cropped otsu derived mask
        maskdata_crop: array, the cropped masked original array
        
        
        """
    mask = mask.get_fdata()
    data = data.get_fdata()
    z_slices = np.any(mask, (0,1))
    y_slices = np.any(mask, (0,2))
    x_slices = np.any(mask, (1,2))
    x_ix, y_ix, z_ix = np.ix_(x_slices, y_slices, z_slices)
    mask_crop = mask[x_ix, y_ix, z_ix]
    maskdata_crop = data[x_ix, y_ix, z_ix, :]
    return (x_ix, y_ix, z_ix, mask_crop,  maskdata_crop)


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
    
def fill_mask(mask_array):
    filled_mask = np.empty_like(mask_array)
    for s in np.arange(mask_array.shape[2]):
        ax_slice = np.copy(mask_array[:,:, s])
        seed = np.copy(ax_slice)
        seed[1:-1, 1:-1] = ax_slice.max()
        filled = reconstruction(seed, ax_slice, method = "erosion")
        filled_mask[:,:, s] = filled
    return(filled_mask)

def dilate_mask(mask_array, selem = 10):
    new_array = np.empty_like(mask_array)
    for slc in range(new_array.shape[2]):
        new_array[:,:,slc] = binary_dilation(mask_array[:,:,slc], selem = square(selem))
    return(new_array)

def custom_brain_extraction(output_base_name,
                            in_file, 
                            output_dir,
                            output_type,
                            median_radius = 3, 
                            numpass = 1,
                            autocrop = False,
                            dilate = 2,
                            selem = 10):
    """This function create a brain mask for the diffusion data. It calculate the average of the 4D diffusion image
        and use this average to create a rough brain mask with FSL bet. This mask is then dilated, used to mask the average 
        diffusion image and the result is fed to the dipy.median_oshu function. Possible hole in the mask caused by failure of the
        oshu method are then filled with a morphological operation.
    Parameters
        ----------
        output_base_name: string, a bids_compatible sub-name_ses-name
        in_file: string or path-like object, the original 4d diffusion data
        output_dir: a valid directory in the form of /base_bids/derivatives/fsl-dipy_dti-preproc/sub-name_ses-name
        output_type: ["NIFTI", "NIFTI_GZ"], the type of nifti output desired,
        ...: options from median_otsu and dilate
        
    Returns
        ----------
        
        mask: brain mask
        masked_data: masked 4d diffusion data
        
        
        """
    if output_type == "NIFTI":
        ext = ".nii"
    elif output_type == "NIFTI_GZ":
        ext = ".nii.gz"
    else:
        raise ValueError("Output file type {} not recognized".format(output_type))
    mask_output_filename = "{}/{}_mask{}".format(output_dir, output_base_name, ext)
    masked_output_filename = "{}/{}_masked{}".format(output_dir, output_base_name, ext)
    if isfile(masked_output_filename):
        print("skipping masking")
        return(None)
    diff_image = nb.load(in_file)
    diff_image_array = diff_image.get_fdata()
    diff_mean = diff_image_array.mean(axis = 3)
    diff_mean_filename = "{}/{}_mean_temp{}".format(output_dir, output_base_name, ext)
    nb.save(nb.Nifti1Image(diff_mean, diff_image.affine, diff_image.header), diff_mean_filename)
    bet_out_file = "{}/{}_ec_bet{}".format(output_dir, output_base_name, ext)
    bet_out_file_mask = "{}/{}_ec_bet_mask{}".format(output_dir, output_base_name, ext)
    bet = BET(in_file = diff_mean_filename, out_file = bet_out_file, 
                  frac = .3,
                  mask = True, output_type = output_type)
    bet.run()
    b0_mask = nb.load(bet_out_file_mask)
    b0_mask_array = b0_mask.get_fdata()
    b0_mask_dilate = dilate_mask(b0_mask_array, selem = selem)
    diff_mean_dilate_mask = diff_mean * b0_mask_dilate
    _, mask = median_otsu(diff_mean_dilate_mask, 
                                        vol_idx = None, 
                                        median_radius = median_radius,
                                        numpass = numpass, 
                                        autocrop = autocrop,
                                        dilate = dilate)
    mask_filled = fill_mask(mask)
    masked_diff = diff_image_array * np.expand_dims(mask_filled,3)
    nb.save(nb.Nifti1Image(mask_filled, diff_image.affine, diff_image.header), mask_output_filename)
    nb.save(nb.Nifti1Image(masked_diff, diff_image.affine, diff_image.header), masked_output_filename)
    remove(diff_mean_filename)
    remove(bet_out_file)
    remove(bet_out_file_mask)
    return(mask_output_filename, masked_output_filename)