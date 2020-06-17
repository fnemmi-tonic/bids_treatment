# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:23:04 2020

@author: federico.nemmi
"""

import nibabel as nb
from nipype.interfaces.fsl.epi import EddyCorrect
from dipy.io import read_bvals_bvecs
from glob import glob
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy, mean_diffusivity, radial_diffusivity, axial_diffusivity
import numpy as np
from pathlib import PurePath
from os.path import splitext, isfile
import dipy.reconst.fwdti as fwdti
from dipy.denoise.localpca import localpca
from dipy.denoise.pca_noise_estimate import pca_noise_estimate
from nipype.interfaces.fsl.preprocess import FLIRT, FNIRT, ApplyWarp
from nipype.interfaces.fsl.utils import ConvertXFM
import subprocess
from utils import get_base_name_all_type 
from scipy.ndimage.morphology import binary_erosion
from utils import crop_and_indexes, custom_brain_extraction, get_mask_crop_and_indexes




def dti_preprocessing(diff_dir,
                   output_dir,
                   denoise = True,
                   ref_slice = 0,
                   median_radius = 3,
                   numpass = 1,
                   autocrop = False,
                   dilate = 2,
                   selem = 10,
                   output_type = "NIFTI"):
    """Function to realign, eddy current correct and eventually denoise diffusion data        
        Parameters
        ----------
        diff_dir : absolute path pointing at directory sub-id_visit-id/dwi where dwi acquisition 
                    (as a single 4D nii or nii file), bvec and bval files are stored
        output_dir: absolute path of the directories were the indexes will be written
        denoise : boolean, should denoise using Marcenko-Pastur PCA algorithm be performed
        fill : boolean, sometimes oshu method create hole in the mask, especially if you are working with populations
                that may have very low values at B0 in certain structures. Fill use erosion based reconstruction methods
                to fill in-mask holes.
        output_type: ["NIFTI", "NIFTI_GZ"], the type of nifti output desired,
        ... : other parameters related to the nipype and dipy functions used
        """
    fdwi = glob("{}/*nii*".format(diff_dir))[0]
    output_base_name = get_base_name_all_type(fdwi)
    if output_type == "NIFTI":
        ext = ".nii"
    elif output_type == "NIFTI_GZ":
        ext = ".nii.gz"
    else:
        raise ValueError("Output file type {} not recognized".format(output_type))
    ec_out_file = "{}/{}_ec{}".format(output_dir, output_base_name, ext)
    if not(isfile(ec_out_file)):
        ec = EddyCorrect(in_file = fdwi, out_file = ec_out_file)
        ec.inputs.output_type = output_type
        print("Eddy Correction")
        ec.run()
    else:
        print("Skipping EC")
    mask_name, masked_name = custom_brain_extraction(output_base_name, 
                                                     ec_out_file, 
                                                     output_dir, 
                                                     output_type, 
                                                     median_radius, 
                                                     numpass, 
                                                     autocrop, 
                                                     dilate, 
                                                     selem)
    if denoise:
        denoised_out_file = "{}/{}_denoised{}".format(output_dir, output_base_name, ext)
        if not(isfile(denoised_out_file)):
            print("Denoising")
            fbval = glob("{}/*bval".format(diff_dir))[0]
            fbvec = glob("{}/*bvec".format(diff_dir))[0]
            bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
            gtab = gradient_table(bvals, bvecs)
            img = nb.load(masked_name)
            mask = nb.load(mask_name)
            x_ix, y_ix, z_ix, mask_crop, maskdata_crop = crop_and_indexes(mask,
                                                                          img)
            sigma = pca_noise_estimate(maskdata_crop, gtab, correct_bias=True, smooth=3)
            denoised_arr = localpca(maskdata_crop, 
                                    sigma, 
                                    tau_factor=2.3, 
                                    patch_radius=2)
            opt = np.zeros(img.shape)
            opt[x_ix, y_ix, z_ix, :] = denoised_arr
            den_img = nb.Nifti1Image(opt.astype(np.float32), img.affine)
            nb.save(den_img, denoised_out_file)
    
            
        
    
def dti_processing(base_dir,
                   subject_id,
                   visit_id,
                   output_dir,
                   denoised_data = True,
                   indexes = ["FA", "MD"], 
                   output_type = "NIFTI"):
    """Function to derive dti indexes from dwi data using dipy functions        
        Parameters
        ----------
        base_dir : absolute path pointing at the root directory of the bids repository
        subject_id: string, the tag subject (i.e. sub-sublabel) of the subject to treat
        visit_id: the VALUE of the session tag to be treated (i.e. for ses-02, only "02" need to be entered)
        output_dir: absolute path of the directories were the indexes will be written
        denoised_data: boolean, should the denoised (True) or only the eddy current correct and realigned data (False) be used for fitting ?
        indexes : list, a list of the indexes that will be written, accepted values are ["FA", "MD", "RD", "AD"]
        ... : other parameters related to the dipy functions used
        """
    if output_type == "NIFTI":
        ext = ".nii"
    elif output_type == "NIFTI_GZ":
        ext = ".nii.gz"
    else:
        raise ValueError("Output file type {} not recognized".format(output_type))
    diff_dir = "{}/{}/ses-{}/dwi".format(base_dir, subject_id, visit_id)
    fbval = glob("{}/*bval".format(diff_dir))[0]
    fbvec = glob("{}/*bvec".format(diff_dir))[0]
    dti_preroc_dir = "{}/derivatives/fsl-dipy_dti-preproc/{}/ses-{}".format(base_dir, subject_id, visit_id)
    if denoised_data:
        fdwi = glob("{}/*_denoised*".format(dti_preroc_dir))[0]
    else:
        fdwi = glob("{}/*_masked.nii*".format(dti_preroc_dir))[0]
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)
    img = nb.load(fdwi)
    mask = nb.load(glob("{}/*_mask.nii*".format(dti_preroc_dir))[0])
    x_ix, y_ix, z_ix, mask_crop, maskdata_crop = crop_and_indexes(mask,
                                                                  img)
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata_crop)
    output_base_name = get_base_name_all_type(fbval)
    if "FA" in indexes:
        print("Computing FA")
        FA = fractional_anisotropy(tenfit.evals)
        FA[np.isnan(FA)] = 0
        opt = np.zeros(mask.shape)
        opt[x_ix, y_ix, z_ix] = FA * mask_crop
        fa_img = nb.Nifti1Image(opt.astype(np.float32), img.affine)
        nb.save(fa_img, "{}/{}_FA{}".format(output_dir, output_base_name, ext))
    if "MD" in indexes:
        print("Computing MD")
        MD = mean_diffusivity(tenfit.evals)
        MD[np.isnan(MD)] = 0
        opt = np.zeros(mask.shape)
        opt[x_ix, y_ix, z_ix] = MD * mask_crop
        md_img = nb.Nifti1Image(opt.astype(np.float32), img.affine)
        nb.save(md_img, "{}/{}_MD{}".format(output_dir, output_base_name, ext))
    if "RD" in indexes:
        print("Computing RD")
        RD = radial_diffusivity(tenfit.evals)
        RD[np.isnan(RD)] = 0
        opt = np.zeros(mask.shape)
        opt[x_ix, y_ix, z_ix] = RD * mask_crop
        rd_img = nb.Nifti1Image(opt.astype(np.float32), img.affine)
        nb.save(rd_img, "{}/{}_RD{}".format(output_dir, output_base_name, ext))
    if "AD" in indexes:
        print("Computing AD")
        AD = axial_diffusivity(tenfit.evals)
        AD[np.isnan(AD)] = 0
        opt = np.zeros(mask.shape)
        opt[x_ix, y_ix, z_ix] = AD * mask_crop
        ad_img = nb.Nifti1Image(opt.astype(np.float32), img.affine)
        nb.save(ad_img, "{}/{}_AD{}".format(output_dir, output_base_name, ext))
        
def hybrid_rois_registration(base_dir,
                             subject_id,
                             visit_id,
                             output_dir,
                             csf_roi,
                             wm_roi,
                             template = "MNI152_T1_2mm.nii.gz",
                             dof = 6,
                             output_type = "NIFTI"):
    """Function to register csf and wm roi from template to diffusion space and output the MEDIAN signal within the roi. Needed for hybrid Beltrami FW fitting
        Parameters
        ----------
        base_dir : absolute path pointing at the root directory of the bids repository
        subject_id: string, the tag subject (i.e. sub-sublabel) of the subject to treat
        visit_id: the VALUE of the session tag to be treated (i.e. for ses-02, only "02" need to be entered)
        output_dir: absolute path of the directories were the indexes will be written
        csf_roi: absolute path of the roi representative of the csf. It should be in the same space as "template" and binary
        wm_roi: absolute path of the roi representative of the wm. It should be in in the same space as "template" and binary
        template: name of the file with extension of the template to be used for inverse registation of the rois. 
        output_type: ["NIFTI", "NIFTI_GZ"], the type of nifti output desired,
        ... : other parameters related to the dipy functions used
        Returns
        ----------
        csf_mean, wm_mean: floats. The median csf and wm values from the registered ROIs
        
        """
    
    diff_dir = "{}/{}/ses-{}/dwi".format(base_dir, subject_id, visit_id)
    anat_dir = "{}/{}/ses-{}/anat".format(base_dir, subject_id, visit_id)
    if output_type == "NIFTI":
        ext = ".nii"
    elif output_type == "NIFTI_GZ":
        ext = ".nii.gz"
    else:
        raise ValueError("Output file type {} not recognized".format(output_type))
    t1_file = glob("{}/*T1w.nii*".format(anat_dir))[0]
    t1_base_name = get_base_name_all_type(t1_file)
    diff_file = glob("{}/*dwi.nii*".format(diff_dir))[0]
    diff_base_name = get_base_name_all_type(diff_file)
    fsldir_system = subprocess.check_output("echo $FSLDIR", shell = True)
    if type(fsldir_system) == bytes:
        fsl_dir = fsldir_system.decode("utf-8").rstrip()
    else:
        fsl_dir = fsldir_system.rstrip()
    template_base_name = get_base_name_all_type(template)
    template_no_underscore = "x".join(template_base_name.split("_"))
    template_abs = "{}/data/standard/{}".format(fsl_dir, template)
    flirt = FLIRT()
    #T1 to template
    print("Registering T1 onto template")
    flirt.inputs.in_file = t1_file
    flirt.inputs.reference = template_abs
    flirt.inputs.out_file = "{}/{}_ref-{}_dof-12_reg-flirt{}".format(output_dir, t1_base_name,
                                                                      template_no_underscore, ext)
    flirt.inputs.out_matrix_file = "{}/{}_ref-{}_dof-12_reg-flirt.mat".format(output_dir, t1_base_name,
                                                                      template_no_underscore)
    flirt.inputs.output_type = output_type
    flirt.run()
    #diff to t1
    print("Registering diffusion onto T1")
    flirt.inputs.in_file = diff_file
    flirt.inputs.reference = t1_file
    flirt.inputs.dof = dof
    flirt.inputs.out_file = "{}/{}_ref-t1_dof-{}_reg-flirt{}".format(output_dir, diff_base_name,
                                                                     str(dof), ext)
    flirt.inputs.out_matrix_file = "{}/{}_ref-t1_dof-{}_reg-flirt.mat".format(output_dir, diff_base_name,
                                                                     str(dof))
    flirt.inputs.output_type = output_type
    flirt.run()
    #concat and inverse matrices
    print("Concat and inverse matrix, send rois to diff")
    mat_transform = ConvertXFM()
    #concat diff to t1 and t1 to template
    mat_transform.inputs.in_file = "{}/{}_ref-{}_dof-12_reg-flirt.mat".format(output_dir, t1_base_name,
                                                                      template_no_underscore)
    mat_transform.inputs.in_file2 = "{}/{}_ref-t1_dof-{}_reg-flirt.mat".format(output_dir, diff_base_name,
                                                                     str(dof))
    mat_transform.inputs.concat_xfm = True
    mat_transform.inputs.out_file = "{}/diff_to_{}.mat".format(output_dir, template_no_underscore)
    mat_transform.run()
    #inverse
    mat_transform = ConvertXFM()
    mat_transform.inputs.in_file = "{}/diff_to_{}.mat".format(output_dir, template_no_underscore)
    mat_transform.inputs.invert_xfm = True
    mat_transform.inputs.out_file = "{}/{}_to_diff.mat".format(output_dir, template_no_underscore)
    mat_transform.run()
    #send csf and wm roi to diff
    csf_roi_base_name = get_base_name_all_type(csf_roi)
    wm_roi_base_name = get_base_name_all_type(wm_roi)
    #csf
    flirt.inputs.in_file = csf_roi
    flirt.inputs.reference = diff_file
    flirt.inputs.apply_xfm = True
    flirt.inputs.in_matrix_file = "{}/{}_to_diff.mat".format(output_dir, template_no_underscore)
    flirt.inputs.interp = "nearestneighbour"
    flirt.inputs.output_type = output_type
    flirt.inputs.out_file = "{}/{}_in_diff{}".format(output_dir, csf_roi_base_name, ext)
    flirt.run()
    #wm
    flirt.inputs.in_file = wm_roi
    flirt.inputs.reference = diff_file
    flirt.inputs.apply_xfm = True
    flirt.inputs.in_matrix_file = "{}/{}_to_diff.mat".format(output_dir, template_no_underscore)
    flirt.inputs.interp = "nearestneighbour"
    flirt.inputs.output_type = output_type
    flirt.inputs.out_file = "{}/{}_in_diff{}".format(output_dir, wm_roi_base_name, ext)
    flirt.run()
    
    #extract values from roi
    csf_image = nb.load("{}/{}_in_diff{}".format(output_dir, csf_roi_base_name, ext))
    csf_array = csf_image.get_fdata()
    csf_array_mask = np.where(csf_array == 1)
    wm_image = nb.load("{}/{}_in_diff{}".format(output_dir, wm_roi_base_name, ext))
    wm_array = wm_image.get_fdata()
    wm_array_mask = np.where(wm_array == 1)
    diff_image = nb.load(diff_file)
    b0_array = diff_image.get_fdata()[:,:,:,0]
    csf_mean = np.median(b0_array[csf_array_mask])
    wm_mean = np.median(b0_array[wm_array_mask])
    return(csf_mean, wm_mean)
    

def hybrid_intensity_threshold_based(base_dir,
                                     subject_id,
                                     visit_id,
                                     output_dir,
                                     bg_wm_bound = 1,
                                     csf_bound = 75,
                                     denoised_data = True,
                                     write_masks = True,
                                     output_type = "NIFTI",
                                     median_radius = 3,
                                     numpass = 1,
                                     autocrop = False,
                                     dilate = 1):
    """Function to extract median csf and wm values from b0 image using an intensity based segmentation of the b0 image
        Parameters
        ----------
        base_dir : absolute path pointing at the root directory of the bids repository
        subject_id: string, the tag subject (i.e. sub-sublabel) of the subject to treat
        visit_id: the VALUE of the session tag to be treated (i.e. for ses-02, only "02" need to be entered)
        output_dir: absolute path of the directories were the indexes will be written
        bg_wm_bound: int. Percentile of the boundary between background and white matter
        csf_bound: Percentile of the boundary between wm and csf
        write_masks: boolean. If intensity threshold method has been choses, should the masks be written ? This is useful for debugging. Default to True
        denoised_data: boolean, should the denoised (True) or only the eddy current correct and realigned data (False) be used for fitting ?
        output_type: ["NIFTI", "NIFTI_GZ"], the type of nifti output desired,
        ... : other parameters related to the dipy functions used
        Returns
        ----------
        csf_mean, wm_mean: floats. The median csf and wm values from the intensity based masks
        
        """
    dti_preroc_dir = "{}/derivatives/fsl-dipy_dti-preproc/{}/ses-{}".format(base_dir, subject_id, visit_id)
    if denoised_data:
        fdwi = glob("{}/*_denoised*".format(dti_preroc_dir))[0]
    else:
        fdwi = glob("{}/*_ec.nii*".format(dti_preroc_dir))[0]
    if output_type == "NIFTI":
        ext = ".nii"
    elif output_type == "NIFTI_GZ":
        ext = ".nii.gz"
    else:
        raise ValueError("Output file type {} not recognized".format(output_type))
    output_base_name = get_base_name_all_type(fdwi)
    img = nb.load(fdwi)
    img_data = img.get_fdata()
    maskdata, mask = median_otsu(img_data, 
                             vol_idx = range(0, img_data.shape[3]), 
                             median_radius = 3,
                             numpass = 1, 
                             autocrop = False,
                             dilate = None)
    b0 = maskdata[:,:,:,0]
    values = b0[mask == True]
    bg_wm_bound = np.percentile(values, 1)
    csf_bound = np.percentile(values, 75)
    new_array = np.copy(b0)
    new_array[new_array < bg_wm_bound] = 0
    new_array_wm = np.copy(new_array)
    new_array_wm[np.logical_or(b0 < bg_wm_bound,  b0 > csf_bound)] = 0
    new_array_csf = np.copy(new_array)
    new_array_csf[b0 < csf_bound] = 0
    binary_wm = np.zeros_like(new_array_wm)
    binary_wm[new_array_wm != 0] = 1
    eroded_wm_mask = binary_erosion(binary_wm)
    eroded_wm = new_array_wm * (eroded_wm_mask * 1)
    binary_csf = np.zeros_like(new_array_csf)
    binary_csf[new_array_csf != 0] = 1
    eroded_csf_mask = binary_erosion(binary_csf)
    eroded_csf = new_array_csf * (eroded_csf_mask * 1)
    csf_median = np.median(eroded_csf[eroded_csf != 0])
    wm_median = np.median(eroded_wm[eroded_wm != 0])

    if write_masks:
        csf_eroded_nifti = nb.Nifti1Image(eroded_csf.astype(np.float32), img.affine, img.header)
        nb.save(csf_eroded_nifti, "{}/{}_csfxintensityxthrxeroded.{}".format(output_dir, output_base_name, ext))
        wm_eroded_nifti = nb.Nifti1Image(eroded_wm.astype(np.float32), img.affine, img.header)
        nb.save(wm_eroded_nifti, "{}/{}_wmxintensityxthrxeroded.{}".format(output_dir, output_base_name, ext))
        wm_nifti = nb.Nifti1Image(new_array_wm, img.affine, img.header)
        csf_nifti = nb.Nifti1Image(new_array_csf, img.affine, img.header)
        nb.save(csf_nifti, "{}/{}_csfxintensityxthr.{}".format(output_dir, output_base_name, ext))
        nb.save(wm_nifti, "{}/{}_wmxintensityxthr.{}".format(output_dir, output_base_name, ext))
    return(csf_median, wm_median)
    
    
#TO DO: take into account the possibility of not using an fsl template (i.e. maybe take out the ability to automatically derive the fsldir)
def fwdti_processing(base_dir,
                   subject_id,
                   visit_id,
                   output_dir,
                   init_method = "md",
                   init_values_extraction = "intensity",
                   csf_roi = None,
                   wm_roi = None,
                   template = "MNI152_T1_2mm.nii.gz",
                   denoised_data = True,
                   bg_wm_bound = 1,
                   csf_bound = 75,
                   write_masks = True,
                   indexes = ["FA", "MD", "FW"], 
                   dof = 6,
                   output_type = "NIFTI",
                   median_radius = 3,
                   numpass = 1,
                   autocrop = False,
                   dilate = 1,
                   learning_rate = 0.0025,
                   iterations = 50):
    """Function to derive dti indexes from dwi data using dipy functions        
        Parameters
        ----------
        base_dir : absolute path pointing at the root directory of the bids repository
        subject_id: string, the tag subject (i.e. sub-sublabel) of the subject to treat
        visit_id: the VALUE of the session tag to be treated (i.e. for ses-02, only "02" need to be entered)
        output_dir: absolute path of the directories were the indexes will be written
        init_method: string. should the Beltrami optimization be initialized using the mean diffusivity or the hybrid method (see BeltramiModel help). Default to md but hybrid is suggested
            Accepted values are "md" and "hybrid"
        init_values_extraction: string. if hybrid has been chosen as initialization, how the initial values
            for wm and csf should be extracted ? If "registration" is chosen, csf and wm roi will be registered to diffusion image
            and the median value extracted. If "intensity" than an intensity based threshold is applied to b0 image
            to segment wm and csf and median values from these masks are extracted.
        csf_roi: absolute path of the roi representative of the csf. It should be in the same space as "template" and binary. Default to None
        wm_roi: absolute path of the roi representative of the wm. It should be in in the same space as "template" and binary. Default to None
        template: name of the file with extension of the template to be used for inverse registation of the rois. 
        bg_wm_bound: int. Percentile of the boundary between background and white matter
        csf_bound: Percentile of the boundary between wm and csf
        denoised_data: boolean, should the denoised (True) or only the eddy current correct and realigned data (False) be used for fitting ?
        write_masks: boolean. If intensity threshold method has been choses, should the masks be written ? This is useful for debugging. Default to True
        indexes : list, a list of the indexes that will be written, accepted values are ["FA", "MD", "RD", "AD", "FW"]
        output_type: ["NIFTI", "NIFTI_GZ"], the type of nifti output desired,
        ... : other parameters related to the dipy functions used
        """
    if output_type == "NIFTI":
        ext = ".nii"
    elif output_type == "NIFTI_GZ":
        ext = ".nii.gz"
    else:
        raise ValueError("Output file type {} not recognized".format(output_type))
    diff_dir = "{}/{}/ses-{}/dwi".format(base_dir, subject_id, visit_id)
    fbval = glob("{}/*bval".format(diff_dir))[0]
    fbvec = glob("{}/*bvec".format(diff_dir))[0]
    if (init_method == "hybrid" and init_values_extraction == "registration") and (csf_roi == None or wm_roi == None):
        raise ValueError("You chose hybrid initialization method without specifing a csf_roi, a wm_roi or both")
    dti_preroc_dir = "{}/derivatives/fsl-dipy_dti-preproc/{}/ses-{}".format(base_dir, subject_id, visit_id)
    if denoised_data:
        fdwi = glob("{}/*_denoised*".format(dti_preroc_dir))[0]
    else:
        fdwi = glob("{}/*_ec.nii*".format(dti_preroc_dir))[0]
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    bvals = bvals / 1000
    gtab = gradient_table(bvals, bvecs, b0_threshold = 0)
    img = nb.load(fdwi)
    mask = nb.load(glob("{}/*_mask.nii*".format(dti_preroc_dir))[0])
    x_ix, y_ix, z_ix, mask_crop,  maskdata_crop = crop_and_indexes(mask,
                                                                   img)
    if init_method == "md":
        print("Fitting tensor")
        tenmodel = fwdti.BeltramiModel(gtab, 
                                       learning_rate = learning_rate,
                                       iterations = iterations)
    elif init_method == "hybrid":
        if init_values_extraction == "registration":
            print("Registering csf and wm rois")
            try:
                csf_mean, wm_mean = hybrid_rois_registration(base_dir = base_dir,
                                                             subject_id = subject_id,
                                                             visit_id = visit_id,
                                                             output_dir = output_dir,
                                                             csf_roi = csf_roi,
                                                             wm_roi = wm_roi,
                                                             template = template,
                                                             dof = dof,
                                                             output_type = output_type)
            except:
                raise BaseException("Could not perform the necessary registration for hybrid initialization for subject {}_ses-{}".format(subject_id, visit_id))
        elif init_values_extraction == "intensity":
            print("Performing intensity based csf and wm extraction")
            try:
                csf_mean, wm_mean = hybrid_intensity_threshold_based(base_dir = base_dir,
                                                                     subject_id = subject_id,
                                                                     visit_id = visit_id,
                                                                     output_dir = output_dir,
                                                                     bg_wm_bound = bg_wm_bound,
                                                                     csf_bound = csf_bound,
                                                                     denoised_data = denoised_data,
                                                                     write_masks = write_masks,
                                                                     output_type = output_type,
                                                                     median_radius = median_radius,
                                                                     numpass = numpass,
                                                                     autocrop = autocrop,
                                                                     dilate = dilate)
            except:
                raise BaseException("Could not perform the intensity based values extraction for hybrid initialization for subject {}_ses-{}".format(subject_id, visit_id))
        print("Fitting tensor")
        tenmodel = fwdti.BeltramiModel(gtab, init_method = "hybrid",
                                       Stissue = wm_mean,
                                       Swater = csf_mean)
    else:
        raise ValueError("{} is an unknwon initialization method".format(init_method))
    tenfit = tenmodel.fit(maskdata_crop)
    output_base_name = splitext(PurePath(fbval).parts[-1])[0]
    if "FA" in indexes:
        print("Computing fwFA")
        FA = fwdti.fractional_anisotropy(tenfit.evals)
        FA[np.isnan(FA)] = 0
        opt = np.zeros(mask.shape)
        opt[x_ix, y_ix, z_ix] = FA * mask_crop
        fa_img = nb.Nifti1Image(opt.astype(np.float32), img.affine)
        nb.save(fa_img, "{}/{}_fwFA{}".format(output_dir, output_base_name, ext))
    if "MD" in indexes:
        print("Computing fwMD")
        MD = fwdti.mean_diffusivity(tenfit.evals)
        MD[np.isnan(MD)] = 0
        opt = np.zeros(mask.shape)
        opt[x_ix, y_ix, z_ix] = MD * mask_crop
        MD_img = nb.Nifti1Image(opt.astype(np.float32), img.affine)
        nb.save(MD_img, "{}/{}_fwMD{}".format(output_dir, output_base_name, ext))
    if "RD" in indexes:
        print("Computing fwRD")
        RD = radial_diffusivity(tenfit.evals)
        RD[np.isnan(RD)] = 0
        opt = np.zeros(mask.shape)
        opt[x_ix, y_ix, z_ix] = RD * mask_crop
        RD_img = nb.Nifti1Image(opt.astype(np.float32), img.affine)
        nb.save(RD_img, "{}/{}_fwRD{}".format(output_dir, output_base_name, ext))
    if "AD" in indexes:
        print("Computing fwAD")
        AD = axial_diffusivity(tenfit.evals)
        AD[np.isnan(AD)] = 0
        opt = np.zeros(mask.shape)
        opt[x_ix, y_ix, z_ix] = AD * mask_crop
        AD_img = nb.Nifti1Image(opt.astype(np.float32), img.affine)
        nb.save(AD_img, "{}/{}_fwAD{}".format(output_dir, output_base_name, ext))
    if "FW" in indexes:
        print("Computing FW")
        FW = tenfit.fw
        FW[np.isnan(FW)] = 0
        opt = np.zeros(mask.shape)
        opt[x_ix, y_ix, z_ix] = FW * mask_crop
        fw_image = nb.Nifti1Image(opt.astype(np.float32), img.affine)
        nb.save(fw_image, "{}/{}_FW{}".format(output_dir, output_base_name, ext))
        
# def diff_direct_registration_crossectional(base_dir,
#                    subject_id,
#                    visit_id,
#                    output_dir,
#                    tbss_like = True,
#                    output_type = "NIFTI",
#                    template = "/usr/local/fsl/data/standard/FMRIB58_FA_1mm.nii.gz",
#                    config_file = "/usr/local/fsl/etc/flirtsch/FA_2_FMRIB58_1mm.cnf"):
#     dti_dir = "{}/derivatives/dipy_dti/{}/ses-{}".format(base_dir, subject_id, visit_id)
#     fdwi = glob("{}/*FA.nii*".format(dti_dir))[0]
#     output_base_name = get_base_name_all_type(fdwi)
#     other_indexes = glob("{}/*nii*".format(dti_dir))
#     other_indexes.remove(fdwi)
#     if output_type == "NIFTI":
#         ext = ".nii"
#     elif output_type == "NIFTI_GZ":
#         ext = ".nii.gz"
#     else:
#         raise ValueError("Output file type {} not recognized".format(output_type))
#     flirt = FLIRT()
#     fnirt = FNIRT()
#     apply_warp = ApplyWarp()
#     flirt.inputs.reference = template
#     flirt.inputs.output_type = output_type
#     fnirt.inputs.ref_file = template
#     fnirt.inputs.output_type = output_type
#     fnirt.inputs.config_file = config_file
#     fnirt.inputs.log_file = "{}/{}_fnirt.txt".format(output_dir, output_base_name)
#     apply_warp.inputs.ref_file = template
#     apply_warp.inputs.output_type = output_type
#     if tbss_like:
#         img = nb.load(fdwi)
#         img_data = img.get_fdata()
#         binary_wm = np.zeros_like(img_data)
#         binary_wm[img_data != 0] = 1
#         eroded_wm_mask = binary_erosion(binary_wm)
#         eroded_wm = img_data * (eroded_wm_mask * 1)
#         eroded_wm[:,:,0] = 0
#         eroded_wm[:,:,-1] = 0
#         clean_fa = nb.Nifti1Image(eroded_wm, img.affine, img.header)
#         nb.save(clean_fa, "{}/{}_clean{}".format(output_dir, output_base_name, ext))
#         flirt.inputs.in_file = "{}/{}_clean{}".format(output_dir, output_base_name, ext)
#         flirt.inputs.out_file = "{}/{}_clean_linear{}".format(output_dir, output_base_name, ext)
#         flirt.inputs.out_matrix_file = "{}/{}_clean_linear.mat".format(output_dir, output_base_name)
#         print("running linear registration")
#         flirt.run()
#         fnirt.inputs.in_file = "{}/{}_clean_linear{}".format(output_dir, output_base_name, ext)
#         fnirt.inputs.warped_file = "{}/{}_clean_nonlinear{}".format(output_dir, output_base_name, ext)
#         fnirt.inputs.field_file = "{}/{}_clean_nonlinearfield{}".format(output_dir, output_base_name, ext)
#         print("running non linear registration")
#         fnirt.run()
#     else:
#         flirt.inputs.in_file = fdwi
#         flirt.inputs.out_file = "{}/{}_linear{}".format(output_dir, output_base_name, ext)
#         flirt.inputs.out_matrix_file = "{}/{}_linear.mat".format(output_dir, output_base_name)
#         print("running linear registration")
#         flirt.run()
#         fnirt.inputs.in_file = "{}/{}_linear{}".format(output_dir, output_base_name, ext)
#         fnirt.inputs.warped_file = "{}/{}_nonlinear{}".format(output_dir, output_base_name, ext)
#         fnirt.inputs.field_file = "{}/{}_nonlinearfield{}".format(output_dir, output_base_name, ext)
#         print("running non linear registration")
#         fnirt.run()
    
#     apply_warp.inputs.premat = flirt.inputs.out_matrix_file
#     apply_warp.inputs.field_file = fnirt.inputs.field_file    
#     print("applying warp")
#     for f in other_indexes:
#         output_base_name = get_base_name_all_type(f)
#         apply_warp.inputs.in_file = f
#         apply_warp.inputs.out_file = "{}/{}_nonlinear{}".format(output_dir, output_base_name, ext)
#         apply_warp.run()
        
# def diff_direct_registration_longitudinal(base_dir,
#                    subject_id,
#                    output_dir,
#                    tbss_like = True,
#                    output_type = "NIFTI",
#                    template = "/usr/local/fsl/data/standard/FMRIB58_FA_1mm.nii.gz",
#                    config_file = "/usr/local/fsl/etc/flirtsch/FA_2_FMRIB58_1mm.cnf"):
#     if output_type == "NIFTI":
#         ext = ".nii"
#     elif output_type == "NIFTI_GZ":
#         ext = ".nii.gz"
#     else:
#         raise ValueError("Output file type {} not recognized".format(output_type))
    
#     dti_dir = "{}/derivatives/dipy_dti/{}".format(base_dir, subject_id)
#     fdwi_01 = glob("{}/ses-01/*FA.nii*".format(dti_dir))[0]
#     fdwi_02 = glob("{}/ses-02/*FA.nii*".format(dti_dir))[0]
#     # output_base_name = get_base_name_all_type(fdwi)
#     # other_indexes = glob("{}/*nii*".format(dti_dir))
#     # other_indexes.remove(fdwi)
#     if tbss_like:
#         for el in [fdwi_01, fdwi_02]:
#             output_base_name = get_base_name_all_type(el)
#             img = nb.load(el)
#             img_data = img.get_fdata()
#             binary_wm = np.zeros_like(img_data)
#             binary_wm[img_data != 0] = 1
#             eroded_wm_mask = binary_erosion(binary_wm)
#             eroded_wm = img_data * (eroded_wm_mask * 1)
#             eroded_wm[:,:,0] = 0
#             eroded_wm[:,:,-1] = 0
#             clean_fa = nb.Nifti1Image(eroded_wm, img.affine, img.header)
#             nb.save(clean_fa, "{}/{}_clean{}".format(output_dir, output_base_name, ext))
#         #flirt back and forth
#         clean_01 = glob("{}/*ses-01*_clean*".format(output_dir, output_base_name))
#         clean_02 = glob("{}/*ses-02*_clean*".format(output_dir, output_base_name))
#         flirt = FLIRT()
#         flirt.inputs.in_file = clean_01
#         flirt.inputs.reference = clean_02
#         flirt.inputs.output_type = output_type
#         flirt.inputs.out_file = "{}/t1_to_t2{}".format(output_dir, ext)
#         flirt.inputs.out_matrix_file = "{}/t1_to_t2.mat".format(output_dir)
#         flirt.run()
#         flirt.inputs.in_file = clean_02
#         flirt.inputs.reference = clean_01
#         flirt.inputs.output_type = output_type
#         flirt.inputs.out_file = "{}/t2_to_t1{}".format(output_dir, ext)
#         flirt.inputs.out_matrix_file = "{}/t2_to_t1.mat".format(output_dir)
#         mat_transform = ConvertXFM()
#         mat_transform.inputs.in_file = "{}/t2_to_t1.mat".format(output_dir)
#         mat_transform.inputs.inverse_xfm = True
#         mat_transform.inputs.out_file = "{}/t2_to_t1_inverted.mat".format(output_dir)
        
#         fnirt = FNIRT()
#         apply_warp = ApplyWarp()
#         flirt.inputs.reference = template
#         flirt.inputs.output_type = output_type
#         fnirt.inputs.ref_file = template
#         fnirt.inputs.output_type = output_type
#         fnirt.inputs.config_file = config_file
#         fnirt.inputs.log_file = "{}/{}_fnirt.txt".format(output_dir, output_base_name)
#         apply_warp.inputs.ref_file = template
#         apply_warp.inputs.output_type = output_type
#         if tbss_like:
#             img = nb.load(fdwi)
#             img_data = img.get_fdata()
#             binary_wm = np.zeros_like(img_data)
#             binary_wm[img_data != 0] = 1
#             eroded_wm_mask = binary_erosion(binary_wm)
#             eroded_wm = img_data * (eroded_wm_mask * 1)
#             eroded_wm[:,:,0] = 0
#             eroded_wm[:,:,-1] = 0
#             clean_fa = nb.Nifti1Image(eroded_wm, img.affine, img.header)
#             nb.save(clean_fa, "{}/{}_clean{}".format(output_dir, output_base_name, ext))
#             flirt.inputs.in_file = "{}/{}_clean{}".format(output_dir, output_base_name, ext)
#             flirt.inputs.out_file = "{}/{}_clean_linear{}".format(output_dir, output_base_name, ext)
#             flirt.inputs.out_matrix_file = "{}/{}_clean_linear.mat".format(output_dir, output_base_name)
#             print("running linear registration")
#             flirt.run()
#             fnirt.inputs.in_file = "{}/{}_clean_linear{}".format(output_dir, output_base_name, ext)
#             fnirt.inputs.warped_file = "{}/{}_clean_nonlinear{}".format(output_dir, output_base_name, ext)
#             fnirt.inputs.field_file = "{}/{}_clean_nonlinearfield{}".format(output_dir, output_base_name, ext)
#             print("running non linear registration")
#             fnirt.run()
#         else:
#             flirt.inputs.in_file = fdwi
#             flirt.inputs.out_file = "{}/{}_linear{}".format(output_dir, output_base_name, ext)
#             flirt.inputs.out_matrix_file = "{}/{}_linear.mat".format(output_dir, output_base_name)
#             print("running linear registration")
#             flirt.run()
#             fnirt.inputs.in_file = "{}/{}_linear{}".format(output_dir, output_base_name, ext)
#             fnirt.inputs.warped_file = "{}/{}_nonlinear{}".format(output_dir, output_base_name, ext)
#             fnirt.inputs.field_file = "{}/{}_nonlinearfield{}".format(output_dir, output_base_name, ext)
#             print("running non linear registration")
#             fnirt.run()
        
#         apply_warp.inputs.premat = flirt.inputs.out_matrix_file
#         apply_warp.inputs.field_file = fnirt.inputs.field_file    
#         print("applying warp")
#         for f in other_indexes:
#             output_base_name = get_base_name_all_type(f)
#             apply_warp.inputs.in_file = f
#             apply_warp.inputs.out_file = "{}/{}_nonlinear{}".format(output_dir, output_base_name, ext)
#             apply_warp.run()