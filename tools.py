import scipy.io 
import h5py # For loading hf5 files
import mne # For loading BrainVision files (EEG)
from mne import io
import numpy as np
from numpy.polynomial.polynomial import polyfit
# from audio_tools import spectools, fbtools, phn_tools #use custom functions for linguistic/acoustic alignment
from scipy.io import wavfile
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import os
import re
import pingouin as pg #stats package 
import pandas as pd
import traceback
import textgrid as tg

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
from matplotlib_venn import venn3, venn3_circles, venn2
from scipy.stats import wilcoxon

# from ridge.utils import make_delayed, counter, save_table_file
# from ridge.ridge import ridge_corr, bootstrap_ridge, bootstrap_ridge_shuffle, eigridge_corr

import random
import itertools as itools
np.random.seed(0)
random.seed(0)

from scipy import stats
import scipy.optimize

import logging
import math
import glob


def loadEEGh5(subject, stimulus_class, data_dir,
	eeg_epochs=True, resp_mean = True, binarymat=False, binaryfeatmat = True, envelope=True, pitch=True, gabor_pc10=False, 
	spectrogram=True, binned_pitches=True, spectrogram_scaled=True, scene_cut=True):
	"""
	Load contents saved per subject from large .h5 created, which contains EEG epochs based on stimulus type 
	and corresponding speech features. 
	
	Parameters
	----------
	subject : string 
		subject ID (i.e. MT0002)
	stimulus_class : string 
		MovieTrailers or TIMIT 
	data_dir : string 
		-change this to match where .h5 is along with EEG data 
	eeg_epochs : bool
		determines whether or not to load EEG epochs per stimulus type per participant
		(default : True)
	resp_mean : bool
		takes the mean across epochs for stimuli played more than once 
		(default : True)
	binarymat : bool
		determines whether or not to load 52 unique individual phoneme types 
		(deafult : False)
	binaryfeatmat : bool
		determines whether or not to load 14 unique phonological features 
		(default : True)
	envelope : bool
		determines whether or not to load the acoustic envelope of each stimulus type 
		(default : True)
	pitch : bool
		determines whether or not to load the pitch of each stimulus type 
	binned_pitches: bool
		load pitch which are binned base on frequency 
	gabor_pc10 : bool
		inclusion of visual weights 
		(default : False)
	spectrogram : bool
		load the spectrogram of a sound 
		(default : False)

	Returns
	-------
	stim_dict : dict
		generates all features for the desired stimulus_class for a given subject as a array within the dict
		the shape of all features are resampled to the shape of phnfeat (phonological features)

	resp_dict : dict
		generates all epochs of features for the desired stimulus_class for a given subject as a array within the dict
		the shape of all epochs are resampled to the shape of phnfeat (phonological features)
	"""	 

	stim_dict = dict()
	resp_dict = dict()
	with h5py.File('%s/fullEEGmatrix.hf5'%(data_dir),'r') as fh:
		print(stimulus_class)
		all_stim = [k for k in fh['/%s' %(stimulus_class)].keys()]
		print(all_stim)
			
		for idx, wav_name in enumerate(all_stim): 
			print(wav_name)
			stim_dict[wav_name] = []
			resp_dict[wav_name] = []
			try:
				epochs_data = fh['/%s/%s/resp/%s/epochs' %(stimulus_class, wav_name, subject)][:]
				phnfeatmat = fh['/%s/%s/stim/phn_feat_timings' %(stimulus_class, wav_name)][:]
				ntimes = phnfeatmat.shape[1] #always resample to the size of phnfeat 
				if binarymat:
					phnmat = fh['/%s/%s/stim/phn_timings' %(stimulus_class, wav_name)][:] 
					stim_dict[wav_name].append(phnmat)
					ntimes = phnmat.shape[1]
					print('phnmat shape is:')
					print(phnmat.shape)
				if binaryfeatmat:
					stim_dict[wav_name].append(phnfeatmat)
					print('phnfeatmat shape is:')
					print(phnfeatmat.shape)
				if envelope:
					envs = fh['/%s/%s/stim/envelope' %(stimulus_class, wav_name)][:] 
					envs = scipy.signal.resample(envs, ntimes) #resampling to size of phnfeat
					stim_dict[wav_name].append(envs.T)
					print('envs shape is:')
					print(envs.shape)
				if pitch:
					pitch_mat = fh['/%s/%s/stim/pitches' %(stimulus_class, wav_name)][:] 
					pitch_mat = scipy.signal.resample(pitch_mat, ntimes) #resample to size of phnfeat
					pitch_mat = np.atleast_2d(pitch_mat)
					stim_dict[wav_name].append(pitch_mat)
					print('pitch_mat shape is:')
					print(pitch_mat.shape)	
				if binned_pitches:
					binned_p = fh['/%s/%s/stim/binned_pitches' %(stimulus_class, wav_name)][:] 
					#binned_p = scipy.signal.resample(binned_p, ntimes) #resample to size of phnfeat
					binned_p = np.atleast_2d(binned_p)
					stim_dict[wav_name].append(binned_p.T)
					print('binned pitch shape is:')
					print(binned_p.shape)				
				if gabor_pc10:
					gabor_pc10_mat = fh['/%s/%s/stim/gabor_pc10' %(stimulus_class, wav_name)][:]
					stim_dict[wav_name].append(gabor_pc10_mat.T)
					print('gabor_mat shape is:')
					print(gabor_pc10_mat.shape)  
				if spectrogram:
					specs = fh['/%s/%s/stim/spec' %(stimulus_class, wav_name)][:] 
					specs = scipy.signal.resample(specs, ntimes, axis=1)
					new_freq = 15 #create new feature size, from 80 to 15. Easier to fit STRF with the specified time delay
					specs = scipy.signal.resample(specs, new_freq, axis=0)
					stim_dict[wav_name].append(specs)
					print('specs shape is:')
					print(specs.shape)
					freqs = fh['/%s/%s/stim/freqs' %(stimulus_class, wav_name)][:]
				if spectrogram_scaled:
					specs = fh['/%s/%s/stim/spec' %(stimulus_class, wav_name)][:] 
					specs = scipy.signal.resample(specs, ntimes, axis=1)
					new_freq = 15 #create new feature size, from 80 to 15. Easier to fit STRF with the specified time delay
					specs = scipy.signal.resample(specs, new_freq, axis=0)
					specs  = specs/np.abs(specs).max()
					stim_dict[wav_name].append(specs)
					print('specs shape is:')
					print(specs.shape)
				if scene_cut:
					s_cuts = fh['/%s/%s/stim/scene_cut' %(stimulus_class, wav_name)][:] 
					s_cuts = scipy.signal.resample(s_cuts, ntimes, axis=1)
					stim_dict[wav_name].append(s_cuts)
					print('scene cut shape is:')
					print(s_cuts.shape)
			
					#return freqs once
					freqs = fh['/%s/%s/stim/freqs' %(stimulus_class, wav_name)][:]
			except Exception:
				traceback.print_exc()
				
			if eeg_epochs:
				try: 
					epochs_data = fh['/%s/%s/resp/%s/epochs' %(stimulus_class, wav_name, subject)][:]
					if resp_mean:
						print('taking the mean across repeats')
						epochs_data = epochs_data.mean(0)
						epochs_data = scipy.signal.resample(epochs_data.T, ntimes).T #resample to size of phnfeat
					else:
						epochs_data = scipy.signal.resample(epochs_data, ntimes, axis=2)
					print(epochs_data.shape)
					resp_dict[wav_name].append(epochs_data)
					
				except Exception:
					traceback.print_exc()
					# print('%s does not have neural data for %s'%(subject, wav_name))

					# epochs_data = []

	if spectrogram:
		return resp_dict, stim_dict, freqs

	if spectrogram_scaled:
		return resp_dict, stim_dict, freqs
		
	else:
		return resp_dict, stim_dict


def predict_response(wt, vStim, vResp):
	''' Predict the response to [vStim] given STRF weights [wt],
	compare to the actual response [vResp], and return the correlation
	between predicted and actual response.

	Inputs:
		wt: [features x delays] x electrodes, your STRF weights
		vStim: time x [features x delays], your delayed stimulus matrix
		vResp: time x electrodes, your true response to vStim
	Outputs:
		corr: correlation between predicted and actual response
		pred: prediction for each electrode [time x electrodes]
	'''
	nchans = wt.shape[1]
	print('Calculating prediction...')
	pred = np.dot(vStim, wt)

	print('Calculating correlation')
	corr = np.array([np.corrcoef(vResp[:,i], pred[:,i])[0,1] for i in np.arange(nchans)])

	return corr, pred

