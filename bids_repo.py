# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:53:00 2020

@author: federico.nemmi
"""

import os
from pathlib import PurePath
import pandas as pd
import treatments as trt
from utils import separate_crossectional_longitudinal


#TO DO: dti direct registration
#TO DO: t1 registration both for cross-sectional and two or more timepoints
class bids_repo(object):
    """Object of class bids_repo useful to manage, get info from and treat a bids conform repository
               
        Parameters
        ----------
        base_dir : absolute path pointing at the root of the bids repository
        
        """
    def __init__(self, base_dir, fsl_dir = "/usr/local/fsl"):
        self.base_dir = str(PurePath(base_dir))
        if not(os.path.isdir(self.base_dir)):
            raise FileNotFoundError("base_dir must be an existing directory but {} does not exist or is not a directory".format(self.main_cat_dir))
        self.tsv_file = "{}/participants.tsv".format(self.base_dir)
        if not(os.path.isfile(self.tsv_file)):
            raise FileNotFoundError("No 'participants.tsv' file found in the bids directory. Is your repo compliant with the bids specification ?")
        self.derivatives_dir = "{}/derivatives".format(self.base_dir)
        self.participants_df = pd.read_table(self.tsv_file)
        self.fsl_dir = fsl_dir
    
    def dti_prepocessing(self, 
                         ignore_done = False,
                         denoise = True,
                         ref_slice = 0,
                         median_radius = 3,
                         numpass = 1,
                         autocrop = False,
                         dilate = 2,
                         output_type = "NIFTI"):
        """This function will preprocess the diffusion data performing eddy current correction with fsl and eventually denoising with local PCA 
        Parameters
        ----------
            ignore_done: boolean. Should the csv reporting information about subjects that have already be processed be ignored ? Default to False
            denoise : boolean, should denoise using Marcenko-Pastur PCA algorithm be performed
        ... : other parameters related to the nipype and dipy functions used
        
        """
        if not(os.path.isdir(self.derivatives_dir)):
            os.mkdir(self.derivatives_dir)
        base_output_dir = "{}/fsl-dipy_dti-preproc".format(self.derivatives_dir)
        if os.path.isfile("{}/subject_status.csv".format(base_output_dir)):
            status_df = pd.read_csv("{}/subject_status.csv".format(base_output_dir))
        if not(os.path.isdir(base_output_dir)):
            os.mkdir(base_output_dir)
        df_to_treat = self.participants_df[self.participants_df["diff"]]
        for _, row in df_to_treat.iterrows():
            subject = row["subjects"]
            visit = str(row["visit"]).zfill(2)
            if not(ignore_done):
                if 'status_df' in locals():
                    done_subjects = status_df["subjects"]
                    if done_subjects.isin(["{}_ses-{}".format(subject, visit)]).any():
                        continue
            print("Processing subject {}, visit {}".format(subject, visit))
            diff_dir = "{}/{}/ses-{}/dwi".format(self.base_dir, subject, visit)
            output_dir = "{}/{}/ses-{}".format(base_output_dir, subject, visit)
            if not(os.path.isdir(output_dir)):
                 os.makedirs(output_dir, exist_ok = True)
            try:
                trt.dti_preprocessing(diff_dir, output_dir,
                                   denoise,
                                   ref_slice,
                                   median_radius = median_radius,
                                   numpass = numpass,
                                   autocrop = autocrop,
                                   dilate = dilate,
                                   output_type = output_type)
                otp = pd.DataFrame(data = [["{}_ses-{}".format(subject, visit), "Done"]], columns = ["subjects", "status"])
                if not(os.path.isfile("{}/subject_status.csv".format(base_output_dir))):
                    otp.to_csv("{}/subject_status.csv".format(base_output_dir), index = False)
                else:
                    otp.to_csv("{}/subject_status.csv".format(base_output_dir), mode = 'a', header = False, index = False)
            except:
                otp = pd.DataFrame(data = [["{}_ses-{}".format(subject, visit), "Error"]], columns = ["subjects", "status"])
                if not(os.path.isfile("{}/subject_status.csv".format(base_output_dir))):
                    otp.to_csv("{}/subject_status.csv".format(base_output_dir), index = False)
                else:
                    otp.to_csv("{}/subject_status.csv".format(base_output_dir), mode = 'a', header = False, index = False)
                continue
    
    def dti_indexes(self,
                    denoised_data = True,
                    indexes = ["FA", "MD"],
                    ignore_done = False,
                    median_radius = 3,
                    numpass = 1, 
                    autocrop = False,
                    dilate = 2):
        """This function will output one or more dti index(es) calculated using canonical dti model in dipy    
        Parameters
        ----------
        denoised_data: boolean, should the denoised (True) or only the eddy current correct and realigned data (False) be used for fitting ?
        indexes : list, a list of the indexes that will be written, accepted values are ["FA", "MD", "RD", "AD"]
        ignore_done: boolean. Should the csv reporting information about subjects that have already be processed be ignored ? Default to False
        ... : other parameters related to the dipy functions used
        
        """
        if not(os.path.isdir(self.derivatives_dir)):
            os.mkdir(self.derivatives_dir)
        base_output_dir = "{}/dipy_dti".format(self.derivatives_dir)
        if os.path.isfile("{}/subject_status.csv".format(base_output_dir)):
            status_df = pd.read_csv("{}/subject_status.csv".format(base_output_dir))
        if not(os.path.isdir(base_output_dir)):
            os.mkdir(base_output_dir)
        if not(os.path.isdir("{}/fsl-dipy_dti-preproc".format(self.derivatives_dir))):
            raise FileNotFoundError("Directory {} does not exist. Have you run the dti_preprocessing method ?".format("{}/fsl-dipy_dti-preproc".format(self.derivatives_dir)))
        df_to_treat = self.participants_df[self.participants_df["diff"]]
        for _, row in df_to_treat.iterrows():
            subject = row["subjects"]
            visit = str(row["visit"]).zfill(2)
            if not(ignore_done):
                if 'status_df' in locals():
                    done_subjects = status_df["subjects"]
                    if done_subjects.isin(["{}_ses-{}".format(subject, visit)]).any():
                        continue
            print("Processing subject {}, visit {}".format(subject, visit))
            diff_dir = "{}/{}/ses-{}/dwi".format(self.base_dir, subject, visit)
            output_dir = "{}/{}/ses-{}".format(base_output_dir, subject, visit)
            if not(os.path.isdir(output_dir)):
                 os.makedirs(output_dir, exist_ok = True)
            try:
                trt.dti_processing(self.base_dir,
                                   subject,
                                   visit,
                                   output_dir,
                                   denoised_data,
                                   indexes, median_radius, numpass,
                                   autocrop,
                                   dilate)
                otp = pd.DataFrame(data = [["{}_ses-{}".format(subject, visit), "Done"]], columns = ["subjects", "status"])
                if not(os.path.isfile("{}/subject_status.csv".format(base_output_dir))):
                    otp.to_csv("{}/subject_status.csv".format(base_output_dir), index = False)
                else:
                    otp.to_csv("{}/subject_status.csv".format(base_output_dir), mode = 'a', header = False, index = False)
            except:
                otp = pd.DataFrame(data = [["{}_ses-{}".format(subject, visit), "Error"]], columns = ["subjects", "status"])
                if not(os.path.isfile("{}/subject_status.csv".format(base_output_dir))):
                    otp.to_csv("{}/subject_status.csv".format(base_output_dir), index = False)
                else:
                    otp.to_csv("{}/subject_status.csv".format(base_output_dir), mode = 'a', header = False, index = False)
                continue
    
    def fwdti_indexes(self,
                     ignore_done = False,
                     csf_roi = None,
                     wm_roi = None,
                     denoised_data = True,
                     init_method = "md",
                     init_values_extraction = "intensity",
                     bg_wm_bound = 1,
                     csf_bound = 75,
                     write_masks = True,
                     template = "MNI152_T1_2mm.nii.gz",
                     indexes = ["FA", "MD"],
                     dof = 6,
                     output_type = "NIFTI",
                     median_radius = 3,
                     numpass = 1,
                     autocrop = False,
                     dilate = 1,
                     learning_rate = 0.0025,
                     iterations = 50):
        """This function will output one or more dti index(es) calculated using BeltramiModel in fwdti module in dipy    
        Parameters
        ----------
        ignore_done: boolean. Should the csv reporting information about subjects that have already be processed be ignored ? Default to False
        csf_roi: absolute path of the roi representative of the csf. It should be in the same space as "template" and binary
        wm_roi: absolute path of the roi representative of the wm. It should be in in the same space as "template" and binary
        denoised_data: boolean, should the denoised (True) or only the eddy current correct and realigned data (False) be used for fitting ?
        init_method: string. should the Beltrami optimization be initialized using the mean diffusivity or the hybrid method (see BeltramiModel help). Default to md but hybrid is suggested
            Accepted values are "md" and "hybrid"
        init_values_extraction: string. if hybrid has been chosen as initialization, how the initial values
            for wm and csf should be extracted ? If "registration" is chosen, csf and wm roi will be registered to diffusion image
            and the median value extracted. If "intensity" than an intensity based threshold is applied to b0 image
            to segment wm and csf and median values from these masks are extracted.
        bg_wm_bound: int. Percentile of the boundary between background and white matter
        csf_bound: Percentile of the boundary between wm and csf
        write_masks: boolean. If intensity threshold method has been choses, should the masks be written ? This is useful for debugging. Default to True
        template: name of the file with extension of the template to be used for inverse registation of the rois. 
        indexes : list, a list of the indexes that will be written, accepted values are ["FA", "MD", "RD", "AD"]
        ... : other parameters related to the dipy functions used
        
        """
        if not(os.path.isdir(self.derivatives_dir)):
            os.mkdir(self.derivatives_dir)
        base_output_dir = "{}/dipy_fwdti".format(self.derivatives_dir)
        if os.path.isfile("{}/subject_status.csv".format(base_output_dir)):
            status_df = pd.read_csv("{}/subject_status.csv".format(base_output_dir))
        if not(os.path.isdir(base_output_dir)):
            os.mkdir(base_output_dir)
        if not(os.path.isdir("{}/fsl-dipy_dti-preproc".format(self.derivatives_dir))):
            raise FileNotFoundError("Directory {} does not exist. Have you run the dti_preprocessing method ?".format("{}/fsl-dipy_dti-preproc".format(self.derivatives_dir)))
        df_to_treat = self.participants_df[self.participants_df["diff"]]
        for _, row in df_to_treat.iterrows():
            subject = row["subjects"]
            visit = str(row["visit"]).zfill(2)
            if not(ignore_done):
                if 'status_df' in locals():
                    done_subjects = status_df["subjects"]
                    if done_subjects.isin(["{}_ses-{}".format(subject, visit)]).any():
                        continue
            print("Processing subject {}, visit {}".format(subject, visit))
            diff_dir = "{}/{}/ses-{}/dwi".format(self.base_dir, subject, visit)
            output_dir = "{}/{}/ses-{}".format(base_output_dir, subject, visit)
            if not(os.path.isdir(output_dir)):
                 os.makedirs(output_dir, exist_ok = True)
            try:
                trt.fwdti_processing(base_dir = self.base_dir,
                                     subject_id = subject,
                                     visit_id = visit,
                                     output_dir = output_dir,
                                     init_values_extraction = init_values_extraction,
                                     csf_roi = csf_roi,
                                     wm_roi = wm_roi,
                                     template = template,
                                     dof = dof,
                                     output_type = output_type,
                                     denoised_data = denoised_data,
                                     bg_wm_bound = bg_wm_bound,
                                     csf_bound = csf_bound,
                                     write_masks = write_masks,
                                     init_method = init_method,
                                     indexes = indexes,
                                     median_radius = median_radius, 
                                     numpass = numpass,
                                     autocrop = autocrop,
                                     dilate = dilate,
                                     learning_rate = learning_rate,
                                     iterations = iterations)
                otp = pd.DataFrame(data = [["{}_ses-{}".format(subject, visit), "Done"]], columns = ["subjects", "status"])
                if not(os.path.isfile("{}/subject_status.csv".format(base_output_dir))):
                    otp.to_csv("{}/subject_status.csv".format(base_output_dir), index = False)
                else:
                    otp.to_csv("{}/subject_status.csv".format(base_output_dir), mode = 'a', header = False, index = False)
            except:
                otp = pd.DataFrame(data = [["{}_ses-{}".format(subject, visit), "Error"]], columns = ["subjects", "status"])
                if not(os.path.isfile("{}/subject_status.csv".format(base_output_dir))):
                    otp.to_csv("{}/subject_status.csv".format(base_output_dir), index = False)
                else:
                    otp.to_csv("{}/subject_status.csv".format(base_output_dir), mode = 'a', header = False, index = False)
                continue
    
    #def dti_on_template_direct(self, template = ")
        
    def diff_direct_registration(self,
                                 ignore_done = False,
                                 tbss_like = True,
                                 longitudinal_when_available = True,
                                 output_type = "NIFTI",
                                 template = "/usr/local/fsl/data/standard/FMRIB58_FA_1mm.nii.gz",
                                 config_file = "/usr/local/fsl/etc/flirtsch/FA_2_FMRIB58_1mm.cnf"):
        """This function will directly register the FA images (canonical tensor fitting) onto a selected template using fsl utils
        Parameters
        ----------
        ignore_done: boolean. Should the csv reporting information about subjects that have already be processed be ignored ? Default to False
        tbss_like: boolean. If True, erode the FA map and zeros first and last slices to avoid edge-related artefact. Default to True
        longitudinal_when_available: boolean. If True, subjects with longitudinal data will be registered using halfway registration.
            NOTE that this is available only for 2 time points, as the moment. Default to True
        ... : other parameters related to the dipy functions used
        
        """
        base_output_dir = "{}/fsl_diff-direct-normalization".format(self.derivatives_dir)
        if os.path.isfile("{}/subject_status.csv".format(base_output_dir)):
            status_df = pd.read_csv("{}/subject_status.csv".format(base_output_dir))
        if not(os.path.isdir(base_output_dir)):
            os.mkdir(base_output_dir)
        if not(os.path.isdir("{}/dipy_dti".format(self.derivatives_dir))):
            raise FileNotFoundError("Directory {} does not exist. Have you run the dti_indexes method ?".format("{}/dipy_dti".format(self.derivatives_dir)))
        #all subjects treated as crossectional
        if not(longitudinal_when_available):
            df_to_treat = self.participants_df[self.participants_df["diff"]]
            for _, row in df_to_treat.iterrows():
                subject = row["subjects"]
                visit = str(row["visit"]).zfill(2)
                if not(ignore_done):
                    if 'status_df' in locals():
                        done_subjects = status_df["subjects"]
                        if done_subjects.isin(["{}_ses-{}".format(subject, visit)]).any():
                            continue
                print("Processing subject {}, visit {}".format(subject, visit))
                output_dir = "{}/{}/ses-{}".format(base_output_dir, subject, visit)
                if not(os.path.isdir(output_dir)):
                     os.makedirs(output_dir, exist_ok = True)
                try:
                    trt.diff_direct_registration_crossectional(base_dir = self.base_dir,
                                         subject_id = subject,
                                         visit_id = visit,
                                         output_dir = output_dir,
                                         tbss_like = tbss_like,
                                         output_type = output_type,
                                         template = template,
                                         config_file = config_file)
                    otp = pd.DataFrame(data = [["{}_ses-{}".format(subject, visit), "Done"]], columns = ["subjects", "status"])
                    if not(os.path.isfile("{}/subject_status.csv".format(base_output_dir))):
                        otp.to_csv("{}/subject_status.csv".format(base_output_dir), index = False)
                    else:
                        otp.to_csv("{}/subject_status.csv".format(base_output_dir), mode = 'a', header = False, index = False)
                except:
                    otp = pd.DataFrame(data = [["{}_ses-{}".format(subject, visit), "Error"]], columns = ["subjects", "status"])
                    if not(os.path.isfile("{}/subject_status.csv".format(base_output_dir))):
                        otp.to_csv("{}/subject_status.csv".format(base_output_dir), index = False)
                    else:
                        otp.to_csv("{}/subject_status.csv".format(base_output_dir), mode = 'a', header = False, index = False)
                    continue
            #dividing crossectional from longitudinal
            else:
                long_df, cross_df = separate_crossectional_longitudinal(self.tsv_file, ["diff"])
                #crossectional subjects
                for _, row in cross_df.iterrows():
                    subject = row["subjects"]
                    visit = str(row["visit"]).zfill(2)
                    if not(ignore_done):
                        if 'status_df' in locals():
                            done_subjects = status_df["subjects"]
                            if done_subjects.isin(["{}_ses-{}".format(subject, visit)]).any():
                                continue
                    print("Processing subject {}, visit {}".format(subject, visit))
                    output_dir = "{}/{}/ses-{}".format(base_output_dir, subject, visit)
                    if not(os.path.isdir(output_dir)):
                         os.makedirs(output_dir, exist_ok = True)
                    try:
                        trt.diff_direct_registration_crossectional(base_dir = self.base_dir,
                                             subject_id = subject,
                                             visit_id = visit,
                                             output_dir = output_dir,
                                             tbss_like = tbss_like,
                                             output_type = output_type,
                                             template = template,
                                             config_file = config_file)
                        otp = pd.DataFrame(data = [["{}_ses-{}".format(subject, visit), "Done"]], columns = ["subjects", "status"])
                        if not(os.path.isfile("{}/subject_status.csv".format(base_output_dir))):
                            otp.to_csv("{}/subject_status.csv".format(base_output_dir), index = False)
                        else:
                            otp.to_csv("{}/subject_status.csv".format(base_output_dir), mode = 'a', header = False, index = False)
                    except:
                        otp = pd.DataFrame(data = [["{}_ses-{}".format(subject, visit), "Error"]], columns = ["subjects", "status"])
                        if not(os.path.isfile("{}/subject_status.csv".format(base_output_dir))):
                            otp.to_csv("{}/subject_status.csv".format(base_output_dir), index = False)
                        else:
                            otp.to_csv("{}/subject_status.csv".format(base_output_dir), mode = 'a', header = False, index = False)
                        continue
                    #longitudinal subjects
                    for _, row in long_df.iterrows():
                        subject = row["subjects"]
                        visit = str(row["visit"]).zfill(2)
                        if not(ignore_done):
                            if 'status_df' in locals():
                                done_subjects = status_df["subjects"]
                                if done_subjects.isin(["{}_ses-{}".format(subject, visit)]).any():
                                    continue
                        print("Processing subject {}, visit {}".format(subject, visit))
                        output_dir = "{}/{}/ses-{}".format(base_output_dir, subject, visit)
                        if not(os.path.isdir(output_dir)):
                             os.makedirs(output_dir, exist_ok = True)
                        try:
                            trt.diff_direct_registration_on_temp(base_dir = self.base_dir,
                                                 subject_id = subject,
                                                 output_dir = output_dir,
                                                 tbss_like = tbss_like,
                                                 output_type = output_type,
                                                 template = template,
                                                 config_file = config_file)
                            otp = pd.DataFrame(data = [["{}_ses-{}".format(subject, visit), "Done"]], columns = ["subjects", "status"])
                            if not(os.path.isfile("{}/subject_status.csv".format(base_output_dir))):
                                otp.to_csv("{}/subject_status.csv".format(base_output_dir), index = False)
                            else:
                                otp.to_csv("{}/subject_status.csv".format(base_output_dir), mode = 'a', header = False, index = False)
                        except:
                            otp = pd.DataFrame(data = [["{}_ses-{}".format(subject, visit), "Error"]], columns = ["subjects", "status"])
                            if not(os.path.isfile("{}/subject_status.csv".format(base_output_dir))):
                                otp.to_csv("{}/subject_status.csv".format(base_output_dir), index = False)
                            else:
                                otp.to_csv("{}/subject_status.csv".format(base_output_dir), mode = 'a', header = False, index = False)
                            continue
                        
        