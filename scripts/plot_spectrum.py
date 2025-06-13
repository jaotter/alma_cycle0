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
from spectres import spectres

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

cont_lims_dict = {'HCN(1-0)':'-750:-500,500:750', '13CO(2-1)':'-550:-300,300:550', 'H13CO+(4-3)':'-600:-200,200:400', 'SiO(8-7)':'-350:-150,150:600', 'SiO(7-6)':'-850:-500,750:850', 'CH3OH 2(1,1)-2(0,2)-+':'-500:-400,400:700', 'CH3OH 1(1,0)-1(0,1)-+':'-1300:-900,-500:-250', 
				'H13CO+(3-2)':'-1000:-500,300:700', 'SiO(6-5)':'-650:-300,500:1000', 'HN13C(3-2)':'250:750,1500:1750',
				'HC15N(3-2)':'-600:-400,400:750', 'H13CN(3-2)':'-300:-200,250:750', '13CO(2-1)':'-550:-300,300:550', 'SiO(5-4)':'-450:-300,250:450', 'H2S 2(2,0)-2(1,1)':'-1000:-800,-400:-200',
				'SiO(2-1)':'-500:-250,500:1000', 'HCN(1-0)':'-750:-500,500:750'}

def extract_spectrum(spw, line_name, ap_radius=3*u.arcsecond, pix_center=None, contsub=False, return_type='freq', robust=0.5, rebin_factor=1):
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

	spectrum_pbeam = mask_cube.sum(axis=(1,2))

	pixscale = utils.proj_plane_pixel_area(cube.wcs.celestial)**0.5 * u.deg
	ppbeam = (cube.beam.sr/(pixscale**2)).decompose().value
	
	spectrum = spectrum_pbeam / ppbeam

	if spw != 'na':
		freq = cube.spectral_axis.to(u.GHz)

		line_freq = line_dict[line_name] * u.GHz
		cube_vel = mask_cube.with_spectral_unit(u.km/u.s, rest_value=line_freq, velocity_convention='optical')
		vel = cube_vel.spectral_axis.to(u.km/u.s)
	else:
		vel = cube.spectral_axis.to(u.km/u.s)

	if rebin_factor > 1:
		wave_arr = cube.spectral_axis.to(u.mm, equivalencies=u.spectral())
		wave_inds = [i for i in np.arange(1,len(wave_arr)-1) if i%rebin_factor == 0]

		wave_arr_rebin = wave_arr[wave_inds]

		print(wave_arr_rebin)
		print(wave_arr)

		print(spectrum.value)

		rebin_spec = spectres(new_wavs=wave_arr_rebin.value, spec_wavs=wave_arr.value, spec_fluxes=spectrum.value)

		print(len(rebin_spec))

		orig_unit = spectrum.unit
		spectrum = rebin_spec * orig_unit

		freq = freq[wave_inds]
		vel =  vel[wave_inds]


	if return_type == 'freq':
		return spectrum, freq
	if return_type == 'vel':
		print(f'spec velocity range: {vel[0] - galv, vel[-1] - galv}')
		return spectrum, vel



def generate_spectrum(peak_snr, amp_ratio):

	vel_arr = np.arange(-2000,2000,10) #kms

	amp1 = 1
	amp2 = amp_ratio
	wid1 = 40 #km/s
	wid2 = 145 #km/s
	vel1 = 0
	vel2 = 0

	gaussian_params = [vel1, amp1, wid1, vel2, amp2, wid2]
	gauss_spec = gauss_sum(vel_arr, *gaussian_params)

	total_flux = np.sqrt(2 * np.pi) * ((wid1 * amp1) + (wid2 * amp2))

	sigma_lw = 20

	noise_sigma = np.max(gauss_spec) / peak_snr
	noise_arr = np.random.normal(loc=0, scale=noise_sigma, size=vel_arr.shape)

	print(noise_sigma)
	print(np.max(gauss_spec))


	final_spec = gauss_spec + noise_arr

	return final_spec, vel_arr


def fit_continuum(spectrum, wave_vel, line_name, degree=1, mask_ind=None, cont_type='fit', cont_value=0):
	## function to do continuum fit for a given line, with hand-selected continuum bands for each line
	#cont_type is 'fit' to do a fit or 'manual' to enter value


	line_vel = wave_vel - galv.value

	lims_str = cont_lims_dict[line_name]
	lower_str, upper_str = lims_str.split(',')
	lower_band_kms = np.array(lower_str.split(':'), dtype=np.float64)
	upper_band_kms = np.array(upper_str.split(':'), dtype=np.float64)

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

	if cont_type == 'fit':
		cont_model = odr.polynomial(degree)
		data = odr.Data(fit_wave, fit_spec)
		odr_obj = odr.ODR(data, cont_model)

		output = odr_obj.run()

		cont_params = output.beta
		#cont_params is [intercept, slope] with velocity as unit

	elif cont_type == 'manual':
		cont_params = [cont_value]
		degree = -1


		'''if degree <= 0:
									cont_fit = wave_vel * 0 + cont_params[0]
								if degree == 1:
									cont_fit = wave_vel * cont_params[0] + cont_params[1]
								if degree == 2:
									cont_fit = (wave_vel**2) * cont_params[2] + wave_vel * cont_params[1] + cont_params[0]
								if degree == 3:
									cont_fit = (wave_vel**3) * cont_params[0] + (wave_vel**2) * cont_params[1] + wave_vel * cont_params[2] + cont_params[3]
						
								spectrum_contsub = spectrum - cont_fit'''

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

		wave_lin = np.linspace(wave_vel_fit[0], wave_vel_fit[-1], 100)
		cont_lin_fit = wave_lin * 0 + cont_params[0]
		
		ax0.axhline(0, color='k')
		ax0.plot(wave_vel_fit-galv.value, spectrum_fit, color='tab:blue', linestyle='-', marker='.', label='Data')
		ax0.plot(wave_lin-galv.value, gauss_sum(wave_lin, *popt[0:3]) + cont_lin_fit, linestyle='--', color='tab:purple', label=fr'Comp 1 $\sigma$={sig1}')
		if ncomp == 2:
			amp2 = np.round(popt[4])
			sig2 = int(np.round(popt[5]))
			ax0.plot(wave_lin-galv.value, gauss_sum(wave_lin, *popt[3:6]) + cont_lin_fit, linestyle='-', color='tab:purple', label=fr'Comp 2 $\sigma$={sig2}')

		if np.abs(cont_params[0]) > 1e-3:
			ax0.plot(wave_vel_fit-galv.value, cont_fit, linestyle='-', color='tab:orange', label='Continuum')

		ax0.plot(wave_vel_fit-galv.value, cont_fit + gauss_sum(wave_vel_fit, *popt), linestyle='-', color='k', label='Full Fit')

		#ax0_range = ax0.get_ylim()
		#ax0.fill_between([-300,300],ax0_range[0], ax0_range[1], alpha=0.3, color='tab:purple')

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
	amp1 = popt[1] * u.Jy#/beam
	wave_width1 = ((width1 / const.c) * line_wave).to(u.angstrom).value

	amp1_err = np.sqrt(pcov[1,1]) * u.Jy #wrong with unit conversion
	width1_err = np.sqrt(pcov[2,2]) * u.km/u.s
	wave_width1_err = ((width1_err / const.c) * line_wave).to(u.angstrom).value


	if ncomp == 2:
		width2 = popt[5] * u.km/u.s
		amp2 = popt[4] * u.Jy #/beam
		wave_width2 = ((width2 / const.c) * line_wave).to(u.angstrom).value

		amp2_err = np.sqrt(pcov[4,4]) * u.Jy
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
		y = y + amp * np.exp( -(velocity - ctr)**2/(2*wid**2))
	return y


def fit_spectrum(spectrum, velocity, line_name, cont_deg=-1, fit_ncomp=2, constrain=False, robust='2', 
					cont_type='fit', cont_value=0, start_bounds=None):
	#given aperture spectrum and list of lines, fit each one, return dictionary with fit parameters for each
	#set cont_deg = -1 for no continuum fitting
	#start_bounds - set to initialize these

	return_dict = {}

	line_freq = line_dict[line_name] * u.GHz
	line_wave = line_freq.to(u.angstrom, equivalencies=u.spectral())

	fit_spec = spectrum.value
	fit_vel = velocity.value

	cont_params, cont_std, cont_serr, mask_ind = fit_continuum(spectrum.value, velocity.value, line_name, 
																degree=cont_deg, mask_ind=None, cont_type=cont_type, cont_value=cont_value)
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
		bounds = ([galv.value - 500, peak_flux * 0.01, 20, galv.value - 500, 0, 10], 
					[galv.value + 500, peak_flux*3, 600, galv.value + 500, peak_flux*10, 600])
		if constrain == True:
			start = [peak_vel, peak_flux, 50, peak_vel, (peak_flux)*0.25, 150]
			bounds = ([galv.value - 500, peak_flux * 0.01, 20, galv.value - 500, 0, 145], [galv.value + 500, peak_flux*3, 600, galv.value + 500, peak_flux*10, 151])
	
	if start_bounds is not None:
		start, bounds = start_bounds

	popt, pcov = fit_line(spectrum.value, velocity.value, cont_std_err, cont_params, line_name, start, bounds, ncomp=fit_ncomp, plot=True, mask_ind=mask_ind,
							constrain=constrain, robust=robust)

	return_dict[line_name] = [popt, pcov, cont_params]

	residuals = spectrum.value - gauss_sum(velocity.value, *popt)
	ssr = np.nansum(np.power(residuals,2))

	bic = bayesian_info_criterion_lsq(ssr, 3 * fit_ncomp, spectrum.shape[0])

	return_dict['bic'] = bic

	#calculate flux from summing channels instead of fitting
	chan_range = [galv.value-300, galv.value+300]
	ind1 = np.where(velocity.value > chan_range[0])[0][0]
	ind2 = np.where(velocity.value < chan_range[1])[0][-1]


	if len(cont_params) == 1:
		cont_fit = velocity.value * 0 + cont_params[0]
	if len(cont_params) == 2:
		cont_fit = velocity.value * cont_params[1] + cont_params[0]

	contsub_spec = spectrum.value - cont_fit

	sum_spectrum = contsub_spec[ind1:ind2] * u.Jy * u.km / u.s
	sum_flux = np.nansum(sum_spectrum) * np.average(velocity[1:]-velocity[:-1]) #multiply by velocity bin size

	#sum_flux_contsub = np.nansum(spectrum.value[ind1:ind2] - (velocity.value[ind1:ind2] * cont_params[1] + cont_params[0]))

	return_dict['sum_flux'] = sum_flux

	sigma_lw = 20
	err_s = cont_std * np.sqrt(np.pi*2)*sigma_lw
	tot_flux, tot_flux_err = compute_flux(popt, pcov, line_wave, ncomp=fit_ncomp)
	snr = tot_flux / err_s

	peak_signal = np.max(gauss_sum(velocity.value, *popt))
	peak_snr = peak_signal / cont_std

	return_dict['peak_snr'] = peak_snr
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
	sum_flux_list = []

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
	peak_snr_list = []
	delta_bic_list = []

	for i in np.arange(len(line_list)):
		spw = spw_list[i]
		line_name = line_list[i]

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
		sum_flux_list.append(fit_dict2['sum_flux'])

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
		peak_snr_list.append(fit_dict2['peak_snr'])
		delta_bic_list.append(delta_bic)

	tab = Table(names=('line', 'spw', 'comp1_flux', 'comp2_flux', 'total_flux', 'sum_flux', 'e_comp1_flux', 'e_comp2_flux', 'e_total_flux', 'comp1_sigma', 'comp2_sigma', 'e_comp1_sigma', 'e_comp2_sigma',
						'comp1_vel', 'comp2_vel', 'e_comp1_vel', 'e_comp2_vel', 'comp1_amp', 'comp2_amp', 'e_comp1_amp', 'e_comp2_amp', 'delta_bic', 'peak_snr', 'snr'),
				data=(line_list, spw_list, flux_comp1_list, flux_comp2_list, flux_tot_list, sum_flux_list, e_flux_comp1_list, e_flux_comp2_list, e_flux_tot_list, comp1_sigma_list, comp2_sigma_list,
				 		e_comp1_sigma_list, e_comp2_sigma_list, comp1_vel_list, comp2_vel_list, e_comp1_vel_list, e_comp2_vel_list, comp1_amp_list, comp2_amp_list, e_comp1_amp_list, e_comp2_amp_list, 
				 		delta_bic_list, peak_snr_list, snr_list))
	tab.write(f'/Users/jotter/highres_PSBs/alma_cycle0/tables/twocomp_fluxes{"_constr" if constrain == True else ""}_r{robust}.csv', format='csv', overwrite=True)


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

	#for plotting bar outline
	HCN_velwid = HCN_vel[1:] - HCN_vel[:-1]
	vel_leftside_HCN = HCN_vel.value - galv.value - HCN_velwid.value[0]/2
	vel_outline_HCN = np.concatenate(([vel_leftside_HCN[0]], np.repeat(vel_leftside_HCN[1:], 2), [vel_leftside_HCN[-1] + HCN_velwid.value[0]/2]))
	bar_tops_HCN = np.repeat(HCN_spec.value, 2)

	### ax1 
	ax1.bar(HCN_vel.value - galv.value, HCN_spec.value, HCN_velwid.value[0], color='gray', alpha=0.25, label='Data')
	ax1.plot(vel_outline_HCN, bar_tops_HCN, marker=None, linestyle='-', linewidth=1, color='k')

	ax1.plot(HCN_vel.value - galv.value, gauss_sum(HCN_vel.value, *HCN_params[0:3]), linestyle='-', linewidth=0.75, color='white')
	ax1.plot(HCN_vel.value - galv.value, gauss_sum(HCN_vel.value, *HCN_params[3:6]), linestyle='-', linewidth=0.75, color='white')
	ax1.plot(HCN_vel.value - galv.value, gauss_sum(HCN_vel.value, *HCN_params[0:3]), linestyle='dotted', linewidth=1, color='tab:red', label=rf'Fit A$_1$ $\sigma=${int(np.round(HCN_params[2]))}')
	ax1.plot(HCN_vel.value - galv.value, gauss_sum(HCN_vel.value, *HCN_params[3:6]), linestyle='dotted', linewidth=1, color='tab:red', label=rf'Fit A$_2$ $\sigma=${int(np.round(HCN_params[5]))}')
	ax1.plot(HCN_vel.value - galv.value, gauss_sum(HCN_vel.value, *HCN_params), linestyle='-', linewidth=2, color='tab:red', label='Fit A Total')

	#ax1.plot(HCN_vel.value - galv.value, gauss_sum(HCN_vel.value, *HCN_params_c[0:3]), linestyle='-', linewidth=0.75, color='white')
	#ax1.plot(HCN_vel.value - galv.value, gauss_sum(HCN_vel.value, *HCN_params_c[3:6]), linestyle='-', linewidth=0.75, color='white')
	#ax1.plot(HCN_vel.value - galv.value, gauss_sum(HCN_vel.value, *HCN_params_c[0:3]), linestyle=(0,(5,1)), linewidth=1, color='tab:purple', label=rf'Fit B $\sigma=${int(np.round(HCN_params_c[2]))}')
	#ax1.plot(HCN_vel.value - galv.value, gauss_sum(HCN_vel.value, *HCN_params_c[3:6]), linestyle=(0,(5,1)), linewidth=1, color='tab:purple', label=rf'Fit B $\sigma=${int(np.round(HCN_params_c[5]))}')
	#ax1.plot(HCN_vel.value - galv.value, gauss_sum(HCN_vel.value, *HCN_params_c), linestyle='dashed', linewidth=1.5, color='tab:purple', label='Fit B Total')

	ax1.set_title('HCN(1-0)', fontsize=12)
	ax1.set_xlabel('Velocity (km/s)', fontsize=12)
	ax1.set_ylabel('Flux (Jy/beam km/s)', fontsize=12)
	ax1.set_xlim(-500,500)

	ax1.legend(fontsize=8)
	### ax2
	CO13_velwid = CO13_vel[1:] - CO13_vel[:-1]
	vel_leftside_CO13 = CO13_vel.value - galv.value - CO13_velwid.value[0]/2
	vel_outline_CO13 = np.concatenate(([vel_leftside_CO13[0]], np.repeat(vel_leftside_CO13[1:], 2), [vel_leftside_CO13[-1] + CO13_velwid.value[0]/2]))
	bar_tops_CO13 = np.repeat(CO13_spec.value, 2)


	ax2.bar(CO13_vel.value - galv.value, CO13_spec.value, CO13_velwid.value[0], color='gray', alpha=0.25, label='Data')
	ax2.plot(vel_outline_CO13, bar_tops_CO13, marker=None, linestyle='-', linewidth=1, color='k')

	ax2.plot(CO13_vel.value - galv.value, gauss_sum(CO13_vel.value, *CO13_params[0:3]), linestyle='-', linewidth=1, color='white')
	ax2.plot(CO13_vel.value - galv.value, gauss_sum(CO13_vel.value, *CO13_params[3:6]), linestyle='-', linewidth=1, color='white')
	ax2.plot(CO13_vel.value - galv.value, gauss_sum(CO13_vel.value, *CO13_params[0:3]), linestyle='dotted', linewidth=1, color='tab:red', label=rf'Fit A$_1$ $\sigma=${int(np.round(CO13_params[2]))}')
	ax2.plot(CO13_vel.value - galv.value, gauss_sum(CO13_vel.value, *CO13_params[3:6]), linestyle='dotted', linewidth=1, color='tab:red', label=rf'Fit A$_2$ $\sigma=${int(np.round(CO13_params[5]))}')
	ax2.plot(CO13_vel.value - galv.value, gauss_sum(CO13_vel.value, *CO13_params), linestyle='-', linewidth=2, color='tab:red', label='Fit A Total')

	ax2.plot(CO13_vel.value - galv.value, gauss_sum(CO13_vel.value, *CO13_params_c[0:3]), linestyle='-', linewidth=0.75, color='white')
	ax2.plot(CO13_vel.value - galv.value, gauss_sum(CO13_vel.value, *CO13_params_c[3:6]), linestyle='-', linewidth=0.75, color='white')
	ax2.plot(CO13_vel.value - galv.value, gauss_sum(CO13_vel.value, *CO13_params_c[0:3]), linestyle=(0,(5,1)), linewidth=1, color='tab:purple', label=rf'Fit B$_1$ $\sigma=${int(np.round(CO13_params_c[2]))}')
	ax2.plot(CO13_vel.value - galv.value, gauss_sum(CO13_vel.value, *CO13_params_c[3:6]), linestyle=(0,(5,1)), linewidth=1, color='tab:purple', label=rf'Fit B$_2$ $\sigma=${int(np.round(CO13_params_c[5]))}')
	ax2.plot(CO13_vel.value - galv.value, gauss_sum(CO13_vel.value, *CO13_params_c), linestyle='dashed', linewidth=1.5, color='tab:purple', label='Fit B Total')

	ax2.set_title(r'$^{13}$CO(2-1)', fontsize=12)
	ax2.set_xlabel('Velocity (km/s)', fontsize=12)
	ax2.set_xlim(-500,500)

	ax2.legend(fontsize=8)

	residuals_HCN = HCN_spec.value - gauss_sum(HCN_vel.value, *HCN_params)
	ax3.bar(HCN_vel.value - galv.value, residuals_HCN, HCN_velwid.value[0], color='tab:red', alpha=0.25, ec=None)
	residuals_HCN_c = HCN_spec.value - gauss_sum(HCN_vel.value, *HCN_params_c)
	ax3.bar(HCN_vel.value - galv.value, residuals_HCN_c, HCN_velwid.value[0], color='tab:purple', alpha=0.25, ec=None)

	residuals_CO13 = CO13_spec.value - gauss_sum(CO13_vel.value, *CO13_params)
	ax4.bar(CO13_vel.value - galv.value, residuals_CO13, CO13_velwid.value[0], color='tab:red', alpha=0.25, ec=None)
	residuals_CO13_c = CO13_spec.value - gauss_sum(CO13_vel.value, *CO13_params_c)
	ax4.bar(CO13_vel.value - galv.value, residuals_CO13_c, CO13_velwid.value[0], color='tab:purple', alpha=0.25, ec=None)

	plt.savefig('../plots/HCN_13CO_fit_spectra.pdf', dpi=300, bbox_inches='tight')
	plt.savefig('../plots/HCN_13CO_fit_spectra.png', bbox_inches='tight')

def twocomp_upperlimit(snr):

	#amp_ratio_list = [1, 0.5, 0.3, 0.1, 0.075, 0.05, 0.03, 0.02, 0.015, 00.01, 0.009, 0.008, 0.007, 0.006, 0.005, 1e-5]
	amp_ratio_list = np.logspace(-4, 0, 100)

	final_input_amp = []
	recov_amp_ratio = []

	for i in range(100):
		for amp in amp_ratio_list:
			amp = np.abs(amp + np.random.normal(loc=0, scale=amp*0.01))

			line_name = f'ampratio_{amp}'

			gen_spec, vel_list = generate_spectrum(snr, amp)

			cont_params, cont_std, cont_serr, mask_ind = fit_continuum(gen_spec, vel_list, line_name, degree=-1, mask_ind=None)
			cont_std_err = np.repeat(cont_std, len(gen_spec))

			peak_flux = np.nanmax(gen_spec)
			peak_ind = np.where(gen_spec == peak_flux)[0][0]
			peak_vel = vel_list[peak_ind]

			start = [peak_vel, peak_flux, 50, peak_vel, (peak_flux)*0.25, 150]
			bounds = ([-500, peak_flux * 0.01, 20, -500, 0, 145], [500, peak_flux*3, 600, 500, peak_flux*10, 151])
		
			popt, pcov = fit_line(gen_spec, vel_list, cont_std_err, cont_params, line_name, start, bounds, ncomp=2, plot=False, mask_ind=None,
									constrain=True, robust='X')

			meas_amp_ratio = popt[4] / popt[1]

			recov_amp_ratio.append(meas_amp_ratio)
			final_input_amp.append(amp)

	recov_amp_ratio = np.array(recov_amp_ratio)
	final_input_amp = np.array(final_input_amp)

	np.save(f'../tables/recov_amp_ratio_snr{snr}.npy', recov_amp_ratio)
	np.save(f'../tables/final_input_amp_snr{snr}.npy', final_input_amp)

	fig = plt.figure(figsize=(6,6))

	plt.plot(final_input_amp, recov_amp_ratio, linestyle='', marker='o')

	plt.ylim(0.0001, 1.05)
	plt.xlim(0.0001, 1.05)
	plt.loglog()
	plt.ylabel('Recovered amplitude ratio', fontsize=14)
	plt.xlabel('True amplitude ratio', fontsize=14)

	plt.title(f'SNR = {snr}', fontsize=14)

	plt.plot([0.0001, 1.05],[0.0001, 1.05], marker='', color='k')

	plt.axhline(0.00488, linestyle='dashed', color='k') #measured amplitude ratio for 13CO(2-1)

	plt.savefig(f'../plots/simulated_amp_recovered_snr{snr}.png')



def measure_fluxes_all(robust='0.5', name='', cont_fit_type='fit', rebin_default=1):
	#line_list = ['H13CO+(1-0)', 'H13CO+(3-2)', 'H13CO+(4-3)', 'HCN(1-0)', 'H13CN(3-2)', '13CO(2-1)', 'HN13C(3-2)', 
	#			'SiO(2-1)', 'SiO(5-4)', 'SiO(6-5)', 'SiO(7-6)', 'SiO(8-7)']
	#spw_list = ['5,7', '3', '0,1', '6', '4', '9', '3',
	#			'5,7', '10', '3', '2', '0,1']

	#cont_fit_type is 'fit' to fit the continuum, 'lowcont' for the low by eye estimate, and 'hicont' for the high by eye estimate
	
	line_list = [
				'H13CO+(4-3)', 'SiO(8-7)', 'SiO(7-6)', 'CH3OH 2(1,1)-2(0,2)-+', 'CH3OH 1(1,0)-1(0,1)-+', 'H13CO+(3-2)', 'SiO(6-5)', 'HN13C(3-2)',
				'HC15N(3-2)', 'H13CN(3-2)', 'SiO(5-4)', 'H2S 2(2,0)-2(1,1)', 'SiO(2-1)']
	spw_list = [
				'0,1', '0,1', '2', '2', '2', '3', '3', '3',
				'4', '4', '10', '10', '5,7']

	flux_comp1_list = []
	flux_comp2_list = []
	sum_flux_list = []

	e_flux_comp1_list = []
	e_flux_comp2_list = []

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
	peak_snr_list = []

	for i in np.arange(len(line_list)):
		spw = spw_list[i]
		line_name = line_list[i]

		if cont_fit_type == 'fit':
			cont_deg = 1
			cont_value = None
			cont_type = 'fit'

		elif cont_fit_type == 'lowcont':
			cont_deg = -1
			cont_value = lowcont_dict[line_name]
			cont_type = 'manual'

		elif cont_fit_type == 'hicont':
			cont_deg = -1
			cont_value = hicont_dict[line_name]
			cont_type = 'manual'

		rebin_factor = rebin_default

		if line_name == 'SiO(8-7)':
			rebin_factor = 9

		spectrum, vel = extract_spectrum(spw, line_name, contsub=False, return_type='vel', ap_radius=5*u.arcsecond, robust=robust,
											rebin_factor=rebin_factor)

		ncomp = 1
		peak_flux = np.max(spectrum).value
		start = [galv.value, peak_flux, 30]
		bounds = ([galv.value - 50, peak_flux * 0.01, 10], [galv.value + 50, peak_flux*3, 150])

		if line_name == 'SiO(6-5)':
			ncomp = 2
			restfreq = line_dict[line_name] * u.GHz
			freq_to_vel = u.doppler_optical(restfreq)
			second_line_vel = (line_dict['H13CO+(3-2)'] * u.GHz).to(u.km / u.s, equivalencies=freq_to_vel) + galv

		if line_name == 'SiO(7-6)':
			ncomp = 2
			restfreq = line_dict[line_name] * u.GHz
			freq_to_vel = u.doppler_optical(restfreq)
			second_line_vel = (line_dict['CH3OH 2(1,1)-2(0,2)-+'] * u.GHz).to(u.km / u.s, equivalencies=freq_to_vel) + galv

		if line_name == 'SiO(8-7)':
			ncomp = 2
			restfreq = line_dict[line_name] * u.GHz
			freq_to_vel = u.doppler_optical(restfreq)
			second_line_vel = (line_dict['H13CO+(4-3)'] * u.GHz).to(u.km / u.s, equivalencies=freq_to_vel) + galv

		if ncomp == 2:
			start = [galv.value, peak_flux, 30, second_line_vel.value, peak_flux, 30]
			bounds = ([galv.value - 50, peak_flux * 0.01, 10, second_line_vel.value - 50, peak_flux * 0.01, 10], [galv.value + 50, peak_flux*3, 150, second_line_vel.value + 50, peak_flux*3, 150])

		start_bounds = [start, bounds]

		fit_dict = fit_spectrum(spectrum, vel, line_name, fit_ncomp=ncomp, cont_deg=cont_deg, robust=robust, cont_type=cont_type,
											cont_value=cont_value, start_bounds=start_bounds)
		
		line_freq = line_dict[line_name] * u.GHz
		line_wave = line_freq.to(u.angstrom, equivalencies=u.spectral())

		popt, pcov, cont = fit_dict[line_name]

		comp1_flux, comp1_flux_err = compute_flux(popt[0:3], pcov[0:3], line_wave, ncomp=1)

		flux_comp1_list.append(comp1_flux)
		e_flux_comp1_list.append(comp1_flux_err)
		sum_flux_list.append(fit_dict['sum_flux'])

		comp1_sigma_list.append(popt[2])
		e_comp1_sigma_list.append(np.sqrt(pcov[2,2]))
		
		comp1_vel_list.append(popt[0])
		e_comp1_vel_list.append(np.sqrt(pcov[0,0]))
		
		comp1_amp_list.append(popt[1])
		e_comp1_amp_list.append(np.sqrt(pcov[1,1]))
		

		if ncomp == 1:
			flux_comp2_list.append(-1)
			e_flux_comp2_list.append(-1)

			comp2_sigma_list.append(-1)
			e_comp2_sigma_list.append(-1)

			comp2_vel_list.append(-1)
			e_comp2_vel_list.append(-1)

			comp2_amp_list.append(-1)
			e_comp2_amp_list.append(-1)

		if ncomp == 2:
			comp2_flux, comp2_flux_err = compute_flux(popt[3:6], pcov[3:6], line_wave, ncomp=1)

			flux_comp2_list.append(comp2_flux)
			e_flux_comp2_list.append(comp2_flux_err)

			comp2_sigma_list.append(popt[5])
			e_comp2_sigma_list.append(np.sqrt(pcov[5,5]))

			comp2_vel_list.append(popt[3])
			e_comp2_vel_list.append(np.sqrt(pcov[3,3]))

			comp2_amp_list.append(popt[4])
			e_comp2_amp_list.append(np.sqrt(pcov[4,4]))


		snr_list.append(fit_dict['snr'])
		peak_snr_list.append(fit_dict['peak_snr'])

	tab = Table(names=('line', 'spw', 'comp1_flux', 'comp2_flux', 'sum_flux', 'e_comp1_flux', 'e_comp2_flux', 'comp1_sigma', 'comp2_sigma',
						'e_comp1_sigma', 'e_comp2_sigma', 'comp1_vel', 'comp2_vel', 'e_comp1_vel', 'e_comp2_vel', 'comp1_amp', 'comp2_amp',
						'e_comp1_amp', 'e_comp2_amp', 'peak_snr', 'snr'),
				data=(line_list, spw_list, flux_comp1_list, flux_comp2_list, sum_flux_list, e_flux_comp1_list, e_flux_comp2_list, comp1_sigma_list,
						comp2_sigma_list, e_comp1_sigma_list, e_comp2_sigma_list, comp1_vel_list, comp2_vel_list, e_comp1_vel_list, e_comp2_vel_list, 
						comp1_amp_list, comp2_amp_list, e_comp1_amp_list, e_comp2_amp_list, peak_snr_list, snr_list))
	tab.write(f'/Users/jotter/highres_PSBs/alma_cycle0/tables/all_line_table_{name}.csv', format='csv', overwrite=True)



#twocomp_upperlimit(snr=88)

#measure_fluxes(constrain=False, robust='2')
#measure_fluxes(constrain=True, robust='2')
#measure_fluxes(constrain=False, robust='0.5')
#measure_fluxes(constrain=True, robust='0.5')

#spectrum_fit_plot('../tables/twocomp_fluxes_r2.csv', '../tables/twocomp_fluxes_constr_r2.csv')

lowcont_dict = {'H13CO+(4-3)':0.01, 'SiO(8-7)':0.01, 'SiO(7-6)':0.01, 'CH3OH 2(1,1)-2(0,2)-+':0.01, 'CH3OH 1(1,0)-1(0,1)-+':0.01, 'H13CO+(3-2)':0.006, 
				'SiO(6-5)':0.006, 'HN13C(3-2)':0.006,'HC15N(3-2)':0.006, 'H13CN(3-2)':0.006, 'SiO(5-4)':0.005, 'H2S 2(2,0)-2(1,1)':0.004, 'SiO(2-1)':0.002}

hicont_dict = {'H13CO+(4-3)':0.015, 'SiO(8-7)':0.015, 'SiO(7-6)':0.012, 'CH3OH 2(1,1)-2(0,2)-+':0.012, 'CH3OH 1(1,0)-1(0,1)-+':0.012, 'H13CO+(3-2)':0.008, 
				'SiO(6-5)':0.008, 'HN13C(3-2)':0.008,'HC15N(3-2)':0.008, 'H13CN(3-2)':0.008, 'SiO(5-4)':0.006, 'H2S 2(2,0)-2(1,1)':0.006, 'SiO(2-1)':0.004}




measure_fluxes_all(name='lowcont_rebin2', cont_fit_type='lowcont', rebin_default=3)
#measure_fluxes_all(name='hicont_sig1', cont_fit_type='hicont', smooth_sigma=5)

