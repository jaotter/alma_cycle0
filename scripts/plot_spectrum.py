from astropy.io import fits
from scipy.optimize import curve_fit
from scipy import odr
from astropy.table import Table
from astropy.wcs import WCS, utils
from astropy.coordinates import SkyCoord
from regions import CirclePixelRegion, CircleSkyRegion, PixCoord, CircleAnnulusPixelRegion
from spectral_cube import SpectralCube
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import ConnectionPatch, Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib import ticker
from scipy.ndimage import gaussian_filter
from astropy.stats import bayesian_info_criterion_lsq

import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import os

from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
z = 0.007214
galv = np.log(z+1)*const.c.to(u.km/u.s)

color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red', 'tab:pink', 'tab:olive', 'k']

#line name:rest frame freq in GHz
line_dict = {'SiO(2-1)':86.84696, 'SiO(5-4)':217.10498, 'SiO(6-5)':260.51802, 'SiO(7-6)':303.92696, 'SiO(8-7)':347.330631,
				'H13CO+(1-0)':86.7542884, 'H13CO+(3-2)':260.255339, 'H13CO+(4-3)':346.998344, '13CO(2-1)':220.3986842, 'HC18O+(1-0)':85.1622231, 
				'HCN(1-0)':88.6316023, 'HCN(3-2)':265.8864343, 'HCN(4-3)':354.5054779,
				'CH3OH 5(2,3)-5(-1,5)':347.507451, 'CH3OH 14(-1,14)-13(1,12)':346.783, 'CH3OH 2(1,1)-2(0,2)-+':304.208,
				'CH3OH 1(1,0)-1(0,1)-+':303.367, 'CH3OH 3(3,0)-4(-2,3)':260.968, 'NH2CHO 12(1,11)-11(1,10)':261.327, 'HC17O+(3-2)':261.165,
				'HC15N(3-2)':258.157, 'SO 6(6)-5(5)':258.256, 'H13CN(3-2)':259.012, 'H2S 2(2,0)-2(1,1)':216.71, 'HN13C(3-2)':261.263,
				'CH3OH 16(1) -15(3) E1':217.525, 'CH3OH 39(-3) -39(3) E2':217.526028, 'CH3OH 6(1,5)-7(2,6)--':217.299, 'CO(1-0)':115.271,
				'CO(2-1)':230.538, 'CO(3-2)':345.796
				}
spw_dict = {'0':'5~25', '1':'', '2':'5~30;100~110',
			'3':'35~75', '4':'65~95; 145~170', '5':'',
			'6':'', '7':'', '8':'5~145',
			'9':'5~40; 85~105', '10':''}

def extract_spectrum(spw, line_name, ap_radius=3*u.arcsecond, pix_center=None, contsub=False, return_type='freq', robust=0.5):
	#pix_center - center of aperture in pixel coordinates, if None use galaxy center

	if line_name == 'CO(1-0)':
		cube_path = '/Users/jotter/highres_PSBs/ngc1266_data/co.fits'
	elif line_name == 'CO(2-1)':
		cube_path = '/Users/jotter/highres_PSBs/ngc1266_data/co21.fits'
	elif line_name == 'CO(3-2)':
		cube_path = '/Users/jotter/highres_PSBs/ngc1266_data/co32.fits'
	else:
		if contsub == True:
			if spw == '9' and robust == '2':
				cube_path = '/Users/jotter/highres_PSBs/alma_cycle0/fitsimages/contsub_6-12/N1266_spw9_r2_0.8mJy_contsub_6-12_taper2as_g30m.pbcor.fits'
			elif spw == '6' and robust == '2':
				cube_path = '/Users/jotter/highres_PSBs/alma_cycle0/fitsimages/contsub_6-12/N1266_spw6_r2_1.5mJy_contsub_6-12_taper3as.fits'
			else:
				cube_path = glob.glob(f'/Users/jotter/highres_PSBs/alma_cycle0/fitsimages/contsub_6-12/N1266_spw{spw}_r{robust}_*_contsub*.pbcor.fits')[0]
		else:
			cube_path = glob.glob(f'/Users/jotter/highres_PSBs/alma_cycle0/fitsimages/no_contsub/N1266_spw{spw}_r{robust}_*.pbcor.fits')[0]

	print('Spectrum from '+cube_path)

	#cube_fl = fits.open(cube_path)	
	#cube_data = cube_fl[1].data
	#s_wcs = WCS(cube_fl[1].header)
	#cube_fl.close()

	cube = SpectralCube.read(cube_path)

	cube_shape = cube.shape

	#convert dimensions from kpc to arcsec
	z = 0.007214
	D_L = cosmo.luminosity_distance(z).to(u.Mpc)
	as_per_kpc = cosmo.arcsec_per_kpc_comoving(z)

	as_per_pix = utils.proj_plane_pixel_scales(cube.wcs.celestial)[0] * u.degree

	if u.get_physical_type(ap_radius.unit) == 'length':
		ap_radius_as = (ap_radius * as_per_kpc).decompose()
		ap_radius_pix = (ap_radius / as_per_pix.decompose().value)
	elif u.get_physical_type(ap_radius.unit) == 'angle':
		ap_radius_as = ap_radius.to(u.arcsecond)
		ap_radius_pix = (ap_radius / as_per_pix).decompose().value
	elif u.get_physical_type(ap_radius.unit) == 'dimensionless': #assume pixel units
		ap_radius_pix = ap_radius

	#print(f'Aperture radius in pixel: {ap_radius_pix}')

	if pix_center == None:
		gal_cen = SkyCoord(ra='3:16:00.74576', dec='-2:25:38.70151', unit=(u.hourangle, u.degree),
		                  frame='icrs') #from HST H band image, by eye

		center_pix = cube.wcs.celestial.all_world2pix(gal_cen.ra, gal_cen.dec, 0)
		print('center in pix', center_pix)

		center_pixcoord = PixCoord(center_pix[0], center_pix[1])

	else:
		center_pixcoord = PixCoord(pix_center[0], pix_center[1])

	ap = CirclePixelRegion(center=center_pixcoord, radius=ap_radius_pix)
	print(ap_radius_pix, center_pixcoord)
	mask_cube = cube.subcube_from_regions([ap])

	spectrum = mask_cube.sum(axis=(1,2))

	if spw != 'na':
		freq = cube.spectral_axis.to(u.GHz)

		line_freq = line_dict[line_name] * u.GHz
		cube_vel = mask_cube.with_spectral_unit(u.km/u.s, rest_value=line_freq, velocity_convention='optical')
		vel = cube_vel.spectral_axis.to(u.km/u.s)
	else:
		vel = cube.spectral_axis.to(u.km/u.s)

	if return_type == 'freq':
		return spectrum, freq
	if return_type == 'vel':
		print(f'spec velocity range: {vel[0] - galv, vel[-1] - galv}')
		return spectrum, vel


def fit_continuum(spectrum, wave_vel, line_name, degree=1, mask_ind=None):
	## function to do continuum fit for a given line, with hand-selected continuum bands for each line

	line_vel = wave_vel - galv.value

	#if line_name == 'CO(1-0)':

	#for 13CO(2-1)
	if line_name == '13CO(2-1)':
		lower_band_kms = [-550,-300]
		upper_band_kms = [300,550]
		#lower_band_kms = [200, 250]
		#upper_band_kms = [250,450]

	#for HCN(1-0)
	if line_name == 'HCN(1-0)':
		lower_band_kms = [-750,-500]
		upper_band_kms = [500,750]

	else:
		lower_band_kms = [-750,-500]
		upper_band_kms = [500,750]

	#for H13CN(3-2)
	#lower_band_kms = [-495, -350]
	#upper_band_kms = [350,1000]

	#for HN13C(3-2)
	#lower_band_kms = [350,650]
	#upper_band_kms = [1500,1750]


	lower_ind1 = np.where(line_vel > lower_band_kms[0])
	lower_ind2 = np.where(line_vel < lower_band_kms[1])
	lower_ind = np.intersect1d(lower_ind1, lower_ind2)

	upper_ind1 = np.where(line_vel > upper_band_kms[0])
	upper_ind2 = np.where(line_vel < upper_band_kms[1])
	upper_ind = np.intersect1d(upper_ind1, upper_ind2)

	if mask_ind is not None:
		lower_ind = np.setdiff1d(lower_ind, mask_ind)
		upper_ind = np.setdiff1d(upper_ind, mask_ind)

	fit_spec = np.concatenate((spectrum[lower_ind], spectrum[upper_ind]))
	fit_wave = np.concatenate((wave_vel[lower_ind], wave_vel[upper_ind]))

	if degree >= 0:
		cont_model = odr.polynomial(degree)
		data = odr.Data(fit_wave, fit_spec)
		odr_obj = odr.ODR(data, cont_model)

		output = odr_obj.run()

		cont_params = output.beta
		#cont_params is [intercept, slope] with velocity as unit

	else:
		cont_params = [0]


	if degree <= 0:
		cont_fit = wave_vel * 0 + cont_params[0]
	if degree == 1:
		cont_fit = wave_vel * cont_params[0] + cont_params[1]
	if degree == 2:
		cont_fit = (wave_vel**2) * cont_params[2] + wave_vel * cont_params[1] + cont_params[0]
	if degree == 3:
		cont_fit = (wave_vel**3) * cont_params[0] + (wave_vel**2) * cont_params[1] + wave_vel * cont_params[2] + cont_params[3]

	spectrum_contsub = spectrum - cont_fit

	#fit_residuals = fit_spec - (fit_wave * cont_params[1] + cont_params[0])

	if degree <= 0:
		fit_residuals = fit_spec - (fit_wave * 0 + cont_params[0])
	if degree == 1:
		fit_residuals = fit_spec - (fit_wave * cont_params[0] + cont_params[1])
	if degree == 2:
		fit_residuals = fit_spec - ((fit_wave**2) * cont_params[2] + fit_wave * cont_params[1] + cont_params[0])
	if degree == 3:
		fit_residuals = fit_spec - ((fit_wave**3) * cont_params[3] + (fit_wave**2) * cont_params[2] + fit_wave * cont_params[1] + cont_params[0])

	residuals_std = np.nanstd(fit_residuals)
	sclip_keep_ind = np.arange(len(fit_residuals))# np.where(fit_residuals < 3*residuals_std)[0]


	residuals_clip_std = np.nanstd(fit_residuals[sclip_keep_ind])
	cont_serr = residuals_clip_std / np.sqrt(len(sclip_keep_ind))

	return cont_params, residuals_clip_std, cont_serr, mask_ind


def fit_line(spectrum_fit, wave_vel_fit, spectrum_err_fit, cont_params, line_name, start, bounds, ncomp=2, plot=False, mask_ind=None, constrain=False, robust='2'):
	## function to fit line to continuum-subtracted spectrum, with start values, bounds, and varying number of gaussian components
	## spectrum_fit is the non-subtracted spectrum to fit

	#cont_params = [0, 0]

	if len(cont_params) == 1:
		cont_fit = wave_vel_fit * 0 + cont_params[0]
	if len(cont_params) == 2:
		cont_fit = wave_vel_fit * cont_params[1] + cont_params[0]
	if len(cont_params) == 4:
		cont_fit = (wave_vel_fit**3) * cont_params[3] + (wave_vel_fit**2) * cont_params[2] + wave_vel_fit * cont_params[1] + cont_params[0]


	contsub_spec = spectrum_fit - cont_fit

	print(f'fitting {line_name}')
	initial_guess = gauss_sum(wave_vel_fit, *start)

	popt, pcov = curve_fit(gauss_sum, wave_vel_fit, contsub_spec, sigma=spectrum_err_fit, p0=start, bounds=bounds, absolute_sigma=True, maxfev=5000)


	if plot == True:
		fig = plt.figure(figsize=(6,8))
		
		gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=(3,1), hspace=0)
		ax0 = fig.add_subplot(gs[0,0])
		ax1 = fig.add_subplot(gs[1,0], sharex=ax0)

		amp1 = np.round(popt[1])

		sig1 = int(np.round(popt[2]))


		
		ax0.axhline(0, color='k')
		ax0.plot(wave_vel_fit-galv.value, spectrum_fit, color='tab:blue', linestyle='-', marker='.', label='Data')
		ax0.plot(wave_vel_fit-galv.value, gauss_sum(wave_vel_fit, *popt[0:3]) + cont_fit, linestyle='--', color='tab:purple', label=fr'Comp 1 $\sigma$={sig1}')
		if ncomp == 2:
			amp2 = np.round(popt[4])
			sig2 = int(np.round(popt[5]))
			ax0.plot(wave_vel_fit-galv.value, gauss_sum(wave_vel_fit, *popt[3:6]) + cont_fit, linestyle='-', color='tab:purple', label=fr'Comp 2 $\sigma$={sig2}')

		if np.abs(cont_params[0]) > 1e-3:
			ax0.plot(wave_vel_fit-galv.value, cont_fit, linestyle='-', color='tab:orange', label='Continuum')

		ax0.plot(wave_vel_fit-galv.value, cont_fit + gauss_sum(wave_vel_fit, *popt), linestyle='-', color='k', label='Full Fit')

		if mask_ind is not None:
			ax0.plot(wave_vel_fit[mask_ind]-galv.value, spectrum_fit[mask_ind], color='tab:red', linestyle='-', marker='x')

		#ax0.text(0, 1.1, start, transform=ax0.transAxes)

		#ax0.axvspan(-1700, -1000, color='tab:red', alpha=0.1)
		#ax0.axvspan(1000, 1700, color='tab:red', alpha=0.1)

		ax0.legend()

		#ax0.grid()
		#ax0.text()

		residuals = spectrum_fit - gauss_sum(wave_vel_fit, *popt) - cont_fit

		ax1.plot(wave_vel_fit-galv.value, residuals, color='tab:red', marker='p', linestyle='-')
		ax1.axhline(0)

		ax0.set_ylabel('Flux (mJy/beam)')
		ax1.set_ylabel('Residuals (mJy/beam)')

		ax1.set_xlabel('Velocity (km/s)')

		ax0.set_title(f'{line_name} fit' )
		plt.subplots_adjust(hspace=0)

		savename = f'{line_name}_fit_n{ncomp}{"_constr" if constrain == True else ""}_r{robust}.png'
		savepath = f'/Users/jotter/highres_PSBs/alma_cycle0/plots/'
		plt.savefig(f'{savepath}{savename}')
		print(f'Figure saved as {savename}')

		plt.close()

	return popt, pcov


def compute_flux(popt, pcov, line_wave, ncomp=2):
	#take fit parameter output and compute flux and error

	width1 = popt[2] * u.km/u.s
	amp1 = popt[1] * u.mJy#/beam
	wave_width1 = ((width1 / const.c) * line_wave).to(u.angstrom).value

	amp1_err = np.sqrt(pcov[1,1]) *u.mJy #wrong with unit conversion
	width1_err = np.sqrt(pcov[2,2]) * u.km/u.s
	wave_width1_err = ((width1_err / const.c) * line_wave).to(u.angstrom).value


	if ncomp == 2:
		width2 = popt[5] * u.km/u.s
		amp2 = popt[4] * u.mJy #/beam
		wave_width2 = ((width2 / const.c) * line_wave).to(u.angstrom).value

		amp2_err = np.sqrt(pcov[4,4]) * u.mJy
		width2_err = np.sqrt(pcov[5,5]) * u.km/u.s
		wave_width2_err = ((width2_err / const.c) * line_wave).to(u.angstrom).value

		total_flux = np.sqrt(2 * np.pi) * ((width1 * amp1) + (width2 * amp2))
		total_flux_err = np.sqrt(2*np.pi) * np.sqrt((width1*amp1_err)**2 + (amp1*width1_err)**2 + (width2*amp2_err)**2 + (amp2*width2_err)**2)

	else:
		total_flux = np.sqrt(2 * np.pi) * ((width1 * amp1))
		total_flux_err = np.sqrt(2*np.pi) * np.sqrt((width1*amp1_err)**2 + (amp1*width1_err)**2)


	return total_flux.to(u.Jy * u.km / u.s), total_flux_err.to(u.Jy * u.km / u.s)

def gauss_sum(velocity, *params):
	#velocity is velocity array for given line
	#params must be divisible by 3, should be center (velocity), amplitude, width

	y = np.zeros_like(velocity)

	for i in range(0, len(params)-1, 3):
		ctr = params[i]
		amp = params[i+1]
		wid = params[i+2]
		y = y + amp * np.exp( -(velocity - ctr)**2/(2*wid)**2)
	return y


def fit_spectrum(spectrum, velocity, line_name, cont_deg=-1, fit_ncomp=2, constrain=False, robust='2'):
	#given aperture spectrum and list of lines, fit each one, return dictionary with fit parameters for each
	#set cont_deg = -1 for no continuum fitting
	return_dict = {}

	line_freq = line_dict[line_name] * u.GHz
	line_wave = line_freq.to(u.angstrom, equivalencies=u.spectral())

	fit_spec = spectrum.value
	fit_vel = velocity.value

	cont_params, cont_std, cont_serr, mask_ind = fit_continuum(spectrum.value, velocity.value, line_name, degree=cont_deg, mask_ind=None)
	#cont_fit = fit_spec_vel * cont_params[1] + cont_params[0]
	cont_std_err = np.repeat(cont_std, len(fit_spec))

	#fit_ind3 = np.where(velocity > galv-1000* u.km/u.s)
	#fit_ind4 = np.where(velocity < galv+1000* u.km/u.s)
	#fit_ind_line = np.intersect1d(fit_ind3, fit_ind4)
	#fit_spec_line = spectrum[fit_ind_line].value

	peak_flux = np.nanmax(fit_spec)
	peak_ind = np.where(fit_spec == peak_flux)[0][0]
	peak_vel = fit_vel[peak_ind]

	if np.abs(peak_vel - galv.value) > 500:
		peak_vel = galv.value

	if fit_ncomp == 1:
		#start = [peak_vel, peak_flux, 150]
		start = [galv.value, peak_flux, 30]
		bounds = ([galv.value - 500, peak_flux * 0.01, 10], [galv.value + 500, peak_flux*3, 300])

	if fit_ncomp == 2:
		start = [peak_vel, peak_flux, 50, peak_vel, (peak_flux)*0.25, 150]
		bounds = ([galv.value - 500, peak_flux * 0.01, 20, galv.value - 500, 0, 10], [galv.value + 500, peak_flux*3, 600, galv.value + 500, peak_flux*10, 600])
		if constrain == True:
			start = [peak_vel, peak_flux, 50, peak_vel, (peak_flux)*0.25, 150]
			bounds = ([galv.value - 500, peak_flux * 0.01, 20, galv.value - 500, 0, 145], [galv.value + 500, peak_flux*3, 600, galv.value + 500, peak_flux*10, 151])
		
	popt, pcov = fit_line(spectrum.value, velocity.value, cont_std_err, cont_params, line_name, start, bounds, ncomp=fit_ncomp, plot=True, mask_ind=mask_ind,
							constrain=constrain, robust=robust)

	return_dict[line_name] = [popt, pcov, cont_params]

	residuals = spectrum.value - gauss_sum(velocity.value, *popt)
	ssr = np.nansum(np.power(residuals,2))

	bic = bayesian_info_criterion_lsq(ssr, 3 * fit_ncomp, spectrum.shape[0])

	return_dict['bic'] = bic

	sigma_lw = 20
	err_s = cont_std * np.sqrt(np.pi*2)*sigma_lw
	tot_flux, tot_flux_err = compute_flux(popt, pcov, line_wave, ncomp=fit_ncomp)
	snr = tot_flux / err_s

	#return_dict['peak_snr'] = peak_snr
	return_dict['snr'] = snr

	return return_dict



def plot_spectrum(spectrum, freq, spw, smooth_sigma=3, contsub=True):
	fig = plt.figure(figsize=(8,6))

	ax = fig.add_subplot(111)

	freq = freq.value * u.GHz

	freqwid = freq[1:] - freq[0:-1]

	smooth_spec = gaussian_filter(spectrum.value, smooth_sigma)

	ax.bar(freq.value, smooth_spec, freqwid[0].value, ec='k')

	i = 0
	for line in line_dict.keys():
		rest_freq = line_dict[line] * u.GHz
		obs_freq = rest_freq / (1+z)
		if obs_freq > np.min(freq) and obs_freq < np.max(freq):
			i += 1
			vel_axis = const.c * (obs_freq - freq)/freq
			vel_ind1 = np.where(vel_axis > -50*u.km/u.s)[0]
			freq_start = freq[vel_ind1[0]].value

			vel_ind2 = np.where(vel_axis < 50*u.km/u.s)[0]
			freq_end = freq[vel_ind2[-1]].value

			#print(vel_axis)

			#ax.axvspan(freq_start, freq_end, color=color_list[i], alpha=0.2)
			ax.axvline(obs_freq.value, color=color_list[i], label=line)


	ax.set_ylabel('Flux (mJy/beam)', fontsize=16)
	ax.set_xlabel('Frequency (GHz)', fontsize=16)
	ax.tick_params(labelsize=14)

	plt.legend()

	if contsub == True:
		plt.savefig(f'../plots/spw{spw}_spectrum_contsub_6-12.png')
	else:
		#plot continuum subtraction channels

		sub_spws = spw.split(',')
		for i, spw_sub in enumerate(sub_spws):
			chans = spw_dict[spw_sub]
			sub_chans = chans.split(';')
			for rng in sub_chans:
				if len(rng) > 0:
					rng_start, rng_end = rng.split('~')
					ax.axvspan(freq[int(rng_start)].value, freq[int(rng_end)].value, color=color_list[i], alpha=0.2)

		plt.savefig(f'../plots/spw{spw}_spectrum_sigma{smooth_sigma}.png')
	plt.close()


def measure_fluxes(constrain=True, robust='2'):
	#line_list = ['H13CO+(1-0)', 'H13CO+(3-2)', 'H13CO+(4-3)', 'HCN(1-0)', 'H13CN(3-2)', '13CO(2-1)', 'HN13C(3-2)', 
	#			'SiO(2-1)', 'SiO(5-4)', 'SiO(6-5)', 'SiO(7-6)', 'SiO(8-7)']
	#spw_list = ['5,7', '3', '0,1', '6', '4', '9', '3',
	#			'5,7', '10', '3', '2', '0,1']
	
	line_list = ['CO(1-0)', 'CO(2-1)', 'CO(3-2)', '13CO(2-1)', 'HCN(1-0)']
	spw_list = ['na', 'na', 'na', '9', '6']

	flux_comp1_list = []
	flux_comp2_list = []
	flux_tot_list = []
	e_flux_comp1_list = []
	e_flux_comp2_list = []
	e_flux_tot_list = []

	comp1_sigma_list = []
	comp2_sigma_list = []
	e_comp1_sigma_list = []
	e_comp2_sigma_list = []

	comp1_vel_list = []
	comp2_vel_list = []
	e_comp1_vel_list = []
	e_comp2_vel_list = []

	comp1_amp_list = []
	comp2_amp_list = []
	e_comp1_amp_list = []
	e_comp2_amp_list = []

	snr_list = []
	delta_bic_list = []

	for i in np.arange(len(line_list)):
		spw = spw_list[i]
		line_name = line_list[i]

		#if spw == 'na':
		#	cont_deg = -1
		#else:
		#	cont_deg = -1
		cont_deg = -1

		spectrum, vel = extract_spectrum(spw, line_name, contsub=True, return_type='vel', ap_radius=5*u.arcsecond, robust=robust)

		fit_dict = fit_spectrum(spectrum, vel, line_name, fit_ncomp=1, cont_deg=cont_deg, robust=robust)
		bic_1comp = fit_dict['bic']
		fit_dict2 = fit_spectrum(spectrum, vel, line_name, fit_ncomp=2, cont_deg=cont_deg, constrain=constrain, robust=robust)
		bic_2comp = fit_dict2['bic']

		delta_bic = bic_1comp - bic_2comp
		
		line_freq = line_dict[line_name] * u.GHz
		line_wave = line_freq.to(u.angstrom, equivalencies=u.spectral())

		popt, pcov, cont = fit_dict2[line_name]
		comp1_flux, comp1_flux_err = compute_flux(popt[0:3], pcov[0:3], line_wave, ncomp=1)
		comp2_flux, comp2_flux_err = compute_flux(popt[3:6], pcov[3:6], line_wave, ncomp=1)
		tot_flux, tot_flux_err = compute_flux(popt, pcov, line_wave, ncomp=2)

		print(line_name)
		print(f'Total Flux {tot_flux} +- {tot_flux_err}')


		flux_comp1_list.append(comp1_flux)
		flux_comp2_list.append(comp2_flux)
		flux_tot_list.append(tot_flux)
		e_flux_comp1_list.append(comp1_flux_err)
		e_flux_comp2_list.append(comp2_flux_err)
		e_flux_tot_list.append(tot_flux_err)

		comp1_sigma_list.append(popt[2])
		comp2_sigma_list.append(popt[5])
		e_comp1_sigma_list.append(np.sqrt(pcov[2,2]))
		e_comp2_sigma_list.append(np.sqrt(pcov[5,5]))

		comp1_vel_list.append(popt[0])
		comp2_vel_list.append(popt[3])
		e_comp1_vel_list.append(np.sqrt(pcov[0,0]))
		e_comp2_vel_list.append(np.sqrt(pcov[3,3]))

		comp1_amp_list.append(popt[1])
		comp2_amp_list.append(popt[4])
		e_comp1_amp_list.append(np.sqrt(pcov[1,1]))
		e_comp2_amp_list.append(np.sqrt(pcov[4,4]))

		snr_list.append(fit_dict2['snr'])
		delta_bic_list.append(delta_bic)

	tab = Table(names=('line', 'spw', 'comp1_flux', 'comp2_flux', 'total_flux', 'e_comp1_flux', 'e_comp2_flux', 'e_total_flux', 'comp1_sigma', 'comp2_sigma', 'e_comp1_sigma', 'e_comp2_sigma',
						'comp1_vel', 'comp2_vel', 'e_comp1_vel', 'e_comp2_vel', 'comp1_amp', 'comp2_amp', 'e_comp1_amp', 'e_comp2_amp', 'delta_bic', 'snr'),
				data=(line_list, spw_list, flux_comp1_list, flux_comp2_list, flux_tot_list, e_flux_comp1_list, e_flux_comp2_list, e_flux_tot_list, comp1_sigma_list, comp2_sigma_list,
				 		e_comp1_sigma_list, e_comp2_sigma_list, comp1_vel_list, comp2_vel_list, e_comp1_vel_list, e_comp2_vel_list, comp1_amp_list, comp2_amp_list, e_comp1_amp_list, e_comp2_amp_list, 
				 		delta_bic_list, snr_list))
	tab.write(f'/Users/jotter/highres_PSBs/alma_cycle0/twocomp_fluxes{"_constr" if constrain == True else ""}_r{robust}.csv', format='csv', overwrite=True)


def spectrum_fit_plot(fit_table_path, constr_fit_path, robust='2'):

	HCN_spec, HCN_vel = extract_spectrum('6', 'HCN(1-0)', contsub=True, return_type='vel', ap_radius=5*u.arcsecond, robust=robust)
	CO13_spec, CO13_vel = extract_spectrum('9', '13CO(2-1)', contsub=True, return_type='vel', ap_radius=5*u.arcsecond, robust=robust)

	fit_table = Table.read(fit_table_path, format='csv')
	fit_table_constr = Table.read(constr_fit_path, format='csv')
	
	HCN_row = np.where(fit_table['line'] == 'HCN(1-0)')[0][0]
	HCN_params = [fit_table['comp1_vel'][HCN_row], fit_table['comp1_amp'][HCN_row], fit_table['comp1_sigma'][HCN_row],
			fit_table['comp2_vel'][HCN_row], fit_table['comp2_amp'][HCN_row], fit_table['comp2_sigma'][HCN_row]]
	
	CO13_row = np.where(fit_table['line'] == '13CO(2-1)')[0][0]
	CO13_params = [fit_table['comp1_vel'][CO13_row], fit_table['comp1_amp'][CO13_row], fit_table['comp1_sigma'][CO13_row],
			fit_table['comp2_vel'][CO13_row], fit_table['comp2_amp'][CO13_row], fit_table['comp2_sigma'][CO13_row]]

	HCN_row_c = np.where(fit_table_constr['line'] == 'HCN(1-0)')[0][0]
	HCN_params_c = [fit_table_constr['comp1_vel'][HCN_row], fit_table_constr['comp1_amp'][HCN_row], fit_table_constr['comp1_sigma'][HCN_row],
			fit_table_constr['comp2_vel'][HCN_row], fit_table_constr['comp2_amp'][HCN_row], fit_table_constr['comp2_sigma'][HCN_row]]
	
	CO13_row_c = np.where(fit_table_constr['line'] == '13CO(2-1)')[0][0]
	CO13_params_c = [fit_table_constr['comp1_vel'][CO13_row], fit_table_constr['comp1_amp'][CO13_row], fit_table_constr['comp1_sigma'][CO13_row],
			fit_table_constr['comp2_vel'][CO13_row], fit_table_constr['comp2_amp'][CO13_row], fit_table_constr['comp2_sigma'][CO13_row]]


	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, gridspec_kw={'height_ratios': [1,0.2]}, figsize=(10,4))

	### ax1 
	HCN_velwid = HCN_vel[1:] - HCN_vel[:-1]
	ax1.bar(HCN_vel.value - galv.value, HCN_spec.value, HCN_velwid.value[0], ec=None, color='gray', alpha=0.25, label='HCN(1-0) spectrum')

	ax1.plot(HCN_vel.value - galv.value, gauss_sum(HCN_vel.value, *HCN_params[0:3]), linestyle='dotted', linewidth=1, color='tab:red', label=rf'Fit 1 $\sigma=${int(np.round(HCN_params[2]))}')
	ax1.plot(HCN_vel.value - galv.value, gauss_sum(HCN_vel.value, *HCN_params[3:6]), linestyle='dotted', linewidth=1, color='tab:red', label=rf'Fit 1 $\sigma=${int(np.round(HCN_params[5]))}')
	ax1.plot(HCN_vel.value - galv.value, gauss_sum(HCN_vel.value, *HCN_params), linestyle='-', linewidth=2, color='tab:red', label='Fit 1 Total')

	ax1.plot(HCN_vel.value - galv.value, gauss_sum(HCN_vel.value, *HCN_params_c[0:3]), linestyle=(0,(5,1)), linewidth=1, color='tab:purple', label=rf'Fit 2 $\sigma=${int(np.round(HCN_params_c[2]))}')
	ax1.plot(HCN_vel.value - galv.value, gauss_sum(HCN_vel.value, *HCN_params_c[3:6]), linestyle=(0,(5,1)), linewidth=1, color='tab:purple', label=rf'Fit 2 $\sigma=${int(np.round(HCN_params_c[5]))}')
	ax1.plot(HCN_vel.value - galv.value, gauss_sum(HCN_vel.value, *HCN_params_c), linestyle='dashed', linewidth=1.5, color='tab:purple', label='Fit 2 Total')

	ax1.set_title('HCN(1-0)', fontsize=12)
	ax1.set_xlabel('Velocity (km/s)', fontsize=12)
	ax1.set_ylabel('Flux (Jy/beam km/s)', fontsize=12)

	ax1.legend(fontsize=8)
	### ax2
	CO13_velwid = CO13_vel[1:] - CO13_vel[:-1]
	ax2.bar(CO13_vel.value - galv.value, CO13_spec.value, CO13_velwid.value[0], ec=None, color='gray', alpha=0.25, label='13CO(2-1) spectrum')

	ax2.plot(CO13_vel.value - galv.value, gauss_sum(CO13_vel.value, *CO13_params[0:3]), linestyle='dotted', linewidth=1, color='tab:red', label=rf'Fit 1 $\sigma=${int(np.round(CO13_params[2]))}')
	ax2.plot(CO13_vel.value - galv.value, gauss_sum(CO13_vel.value, *CO13_params[3:6]), linestyle='dotted', linewidth=1, color='tab:red', label=rf'Fit 1 $\sigma=${int(np.round(CO13_params[5]))}')
	ax2.plot(CO13_vel.value - galv.value, gauss_sum(CO13_vel.value, *CO13_params), linestyle='-', linewidth=2, color='tab:red', label='Fit 1 Total')

	ax2.plot(CO13_vel.value - galv.value, gauss_sum(CO13_vel.value, *CO13_params_c[0:3]), linestyle=(0,(5,1)), linewidth=1, color='tab:purple', label=rf'Fit 2 $\sigma=${int(np.round(CO13_params_c[2]))}')
	ax2.plot(CO13_vel.value - galv.value, gauss_sum(CO13_vel.value, *CO13_params_c[3:6]), linestyle=(0,(5,1)), linewidth=1, color='tab:purple', label=rf'Fit 2 $\sigma=${int(np.round(CO13_params_c[5]))}')
	ax2.plot(CO13_vel.value - galv.value, gauss_sum(CO13_vel.value, *CO13_params_c), linestyle='dashed', linewidth=1.5, color='tab:purple', label='Fit 2 Total')

	ax2.set_title(r'$^{13}$CO(2-1)', fontsize=12)
	ax2.set_xlabel('Velocity (km/s)', fontsize=12)

	ax2.legend(fontsize=8)

	residuals_HCN = HCN_spec.value - gauss_sum(HCN_vel.value, *HCN_params)
	ax3.bar(HCN_vel.value - galv.value, residuals_HCN, HCN_velwid.value[0], color='tab:red', alpha=0.25, ec=None)
	residuals_HCN_c = HCN_spec.value - gauss_sum(HCN_vel.value, *HCN_params_c)
	ax3.bar(HCN_vel.value - galv.value, residuals_HCN_c, HCN_velwid.value[0], color='tab:purple', alpha=0.25, ec=None)

	residuals_CO13 = CO13_spec.value - gauss_sum(CO13_vel.value, *CO13_params)
	ax4.bar(CO13_vel.value - galv.value, residuals_CO13, CO13_velwid.value[0], color='tab:red', alpha=0.25, ec=None)
	residuals_CO13_c = CO13_spec.value - gauss_sum(CO13_vel.value, *CO13_params_c)
	ax4.bar(CO13_vel.value - galv.value, residuals_CO13_c, CO13_velwid.value[0], color='tab:purple', alpha=0.25, ec=None)

	plt.savefig('../plots/HCN_13CO_fit_spectra.png', dpi=300, bbox_inches='tight')


#measure_fluxes(constrain=False, robust='2')
#measure_fluxes(constrain=True, robust='2')
#measure_fluxes(constrain=False, robust='0.5')
#measure_fluxes(constrain=True, robust='0.5')

spectrum_fit_plot('../twocomp_fluxes_r2.csv', '../twocomp_fluxes_constr_r2.csv')

'''spw = 'na'
line_name = 'CO(1-0)'
spectrum, vel = extract_spectrum(spw, line_name, contsub=False, return_type='vel', ap_radius=10*u.arcsecond)
#fit_dict = fit_spectrum(spectrum, vel, line_name, fit_ncomp=1, cont_deg=0)
#bic_1comp = fit_dict['bic']
fit_dict2 = fit_spectrum(spectrum, vel, line_name, fit_ncomp=2, cont_deg=-1, constrain=True)
bic_2comp = fit_dict2['bic']

#delta_bic = bic_1comp - bic_2comp
#print(line_name)
#print(f'BIC 1comp - BIC 2comp = {delta_bic}')

line_freq = line_dict[line_name] * u.GHz
line_wave = line_freq.to(u.angstrom, equivalencies=u.spectral())
popt, pcov, cont = fit_dict2[line_name]
comp1_flux, comp1_flux_err = compute_flux(popt[0:3], pcov[0:3], line_wave, ncomp=1)
comp2_flux, comp2_flux_err = compute_flux(popt[3:6], pcov[3:6], line_wave, ncomp=1)
tot_flux, tot_flux_err = compute_flux(popt, pcov, line_wave, ncomp=2)

print(f'Component 1 Flux {comp1_flux} +- {comp1_flux_err}, fraction {comp1_flux/tot_flux}')
print(f'Component 2 Flux {comp2_flux} +- {comp2_flux_err}, fraction {comp2_flux/tot_flux}')
print(f'Total Flux {tot_flux} +- {tot_flux_err}')'''




#spw = '9'
#line_name = '13CO(2-1)'
#spw = '6'
#line_name = 'HCN(1-0)'
#spectrum, vel = extract_spectrum(spw, line_name, contsub=True, return_type='vel', ap_radius=3*u.arcsecond, robust='2')
#fit_dict = fit_spectrum(spectrum, vel, line_name, fit_ncomp=1)
#bic_1comp = fit_dict['bic']
#fit_dict2 = fit_spectrum(spectrum, vel, line_name, fit_ncomp=2, constrain=True)
#bic_2comp = fit_dict2['bic']

#delta_bic = bic_1comp - bic_2comp
#print(line_name)
#print(f'BIC 1comp - BIC 2comp = {delta_bic}')

#line_freq = line_dict[line_name] * u.GHz
#line_wave = line_freq.to(u.angstrom, equivalencies=u.spectral())
#popt, pcov, cont = fit_dict2[line_name]
#comp1_flux, comp1_flux_err = compute_flux(popt[0:3], pcov[0:3], line_wave, ncomp=1)
#comp2_flux, comp2_flux_err = compute_flux(popt[3:6], pcov[3:6], line_wave, ncomp=1)
#tot_flux, tot_flux_err = compute_flux(popt, pcov, line_wave, ncomp=2)

#print(f'Component 1 Flux {comp1_flux} +- {comp1_flux_err}, fraction {comp1_flux/tot_flux}')
#print(f'Component 2 Flux {comp2_flux} +- {comp2_flux_err}, fraction {comp2_flux/tot_flux}')
#print(f'Total Flux {tot_flux} +- {tot_flux_err}')

#print(comp1_flux + comp2_flux, tot_flux)

'''spw = '6'
line_name = 'HCN(1-0)'
spectrum, vel = extract_spectrum(spw, line_name, contsub=True, return_type='vel', ap_radius=3*u.arcsecond, robust=0.5)
fit_dict = fit_spectrum(spectrum, vel, line_name, fit_ncomp=1)
bic_1comp = fit_dict['bic']
fit_dict2 = fit_spectrum(spectrum, vel, line_name, fit_ncomp=2)
bic_2comp = fit_dict2['bic']

delta_bic = bic_1comp - bic_2comp
print(line_name)
print(f'BIC 1comp - BIC 2comp = {delta_bic}')

line_freq = line_dict[line_name] * u.GHz
line_wave = line_freq.to(u.angstrom, equivalencies=u.spectral())
popt, pcov, cont = fit_dict2[line_name]
comp1_flux, comp1_flux_err = compute_flux(popt[0:3], pcov[0:3], line_wave, ncomp=1)
comp2_flux, comp2_flux_err = compute_flux(popt[3:6], pcov[3:6], line_wave, ncomp=1)
tot_flux, tot_flux_err = compute_flux(popt, pcov, line_wave, ncomp=2)

print(f'Component 1 Flux {comp1_flux} +- {comp1_flux_err}, fraction {comp1_flux/tot_flux}')
print(f'Component 2 Flux {comp2_flux} +- {comp2_flux_err}, fraction {comp2_flux/tot_flux}')
print(f'Total Flux {tot_flux} +- {tot_flux_err}')

print(comp1_flux + comp2_flux, tot_flux)'''

'''spw='4'
#spw = '3'
line_name = 'H13CN(3-2)'
#line_name = 'HN13C(3-2)'
spectrum, vel = extract_spectrum(spw, line_name, contsub=False, return_type='vel', ap_radius=3*u.arcsecond, robust=0.5)
fit_dict = fit_spectrum(spectrum, vel, line_name, fit_ncomp=1)
bic_1comp = fit_dict['bic']
fit_dict2 = fit_spectrum(spectrum, vel, line_name, fit_ncomp=2)
bic_2comp = fit_dict2['bic']

delta_bic = bic_1comp - bic_2comp
print(line_name)
print(f'BIC 1comp - BIC 2comp = {delta_bic}')

line_freq = line_dict[line_name] * u.GHz
line_wave = line_freq.to(u.angstrom, equivalencies=u.spectral())
popt, pcov, cont = fit_dict2[line_name]
comp1_flux, comp1_flux_err = compute_flux(popt[0:3], pcov[0:3], line_wave, ncomp=1)
comp2_flux, comp2_flux_err = compute_flux(popt[3:6], pcov[3:6], line_wave, ncomp=1)
tot_flux, tot_flux_err = compute_flux(popt, pcov, line_wave, ncomp=2)

print(f'Component 1 Flux {comp1_flux} +- {comp1_flux_err}, fraction {comp1_flux/tot_flux}')
print(f'Component 2 Flux {comp2_flux} +- {comp2_flux_err}, fraction {comp2_flux/tot_flux}')
print(f'Total Flux {tot_flux} +- {tot_flux_err}')

print(comp1_flux + comp2_flux, tot_flux)'''


## constraining width of broad component to be consistent with HCN and 12CO from Alatalo11

'''spw = '9'
line_name = '13CO(2-1)'
spectrum, vel = extract_spectrum(spw, line_name, contsub=False, return_type='vel', ap_radius=4*u.arcsecond)
fit_dict2 = fit_spectrum(spectrum, vel, line_name, fit_ncomp=2, constrain=True)

line_freq = line_dict[line_name] * u.GHz
line_wave = line_freq.to(u.angstrom, equivalencies=u.spectral())
popt, pcov, cont = fit_dict2[line_name]
comp1_flux, comp1_flux_err = compute_flux(popt[0:3], pcov[0:3], line_wave, ncomp=1)
comp2_flux, comp2_flux_err = compute_flux(popt[3:6], pcov[3:6], line_wave, ncomp=1)
tot_flux, tot_flux_err = compute_flux(popt, pcov, line_wave, ncomp=2)

print(f'Component 1 Flux {comp1_flux} +- {comp1_flux_err}, fraction {comp1_flux/tot_flux}')
print(f'Component 2 Flux {comp2_flux} +- {comp2_flux_err}, fraction {comp2_flux/tot_flux}')
print(f'Total Flux {tot_flux} +- {tot_flux_err}')

print(f'Component 1 velocity {popt[1]}')
print(f'Component 2 velocity {popt[4]}')
'''

#newz = np.exp(((galv + popt[1]*u.km/u.s) / const.c).decompose()) - 1 
#print(f'new z {newz}')


'''spw = '4'
line_name = 'H13CN(3-2)'
spectrum, vel = extract_spectrum(spw, line_name, contsub=False, return_type='vel', ap_radius=4*u.arcsecond)
fit_dict2 = fit_spectrum(spectrum, vel, line_name, fit_ncomp=2, constrain=True)

line_freq = line_dict[line_name] * u.GHz
line_wave = line_freq.to(u.angstrom, equivalencies=u.spectral())
popt, pcov, cont = fit_dict2[line_name]
comp1_flux, comp1_flux_err = compute_flux(popt[0:3], pcov[0:3], line_wave, ncomp=1)
comp2_flux, comp2_flux_err = compute_flux(popt[3:6], pcov[3:6], line_wave, ncomp=1)
tot_flux, tot_flux_err = compute_flux(popt, pcov, line_wave, ncomp=2)

print(f'Component 1 Flux {comp1_flux} +- {comp1_flux_err}, fraction {comp1_flux/tot_flux}')
print(f'Component 2 Flux {comp2_flux} +- {comp2_flux_err}, fraction {comp2_flux/tot_flux}')
print(f'Total Flux {tot_flux} +- {tot_flux_err}')

print(f'Component 1 velocity {popt[1]}')
print(f'Component 2 velocity {popt[4]}')
print(f'recessional velocity {galv.value + popt[1]}')'''


