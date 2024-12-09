import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import glob
from spectral_cube import SpectralCube
import astropy.constants as const

from scipy.ndimage import gaussian_filter
from astropy.wcs import WCS, utils
from regions import CirclePixelRegion, CircleSkyRegion, PixCoord, CircleAnnulusPixelRegion
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM

from plot_spectrum import fit_continuum

cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
z = 0.007214
galv = np.log(z+1)*const.c.to(u.km/u.s)

## plot the line profiles of CO fits from alatalo+11, and the HCN 2 component fit and maybe 13CO as well

FWHM_TO_SIGMA = 2*np.sqrt(2 * np.log(2)) #divide FWHM by this to get sigma

alatalo11_fits = {'CO(1-0)':[2162, 0.21/(0.21 + 0.03), 114 / FWHM_TO_SIGMA, 2142, 0.03/(0.21 + 0.03), 353 / FWHM_TO_SIGMA], 'CO(2-1)':[2161, 
					0.53/(0.53 + 0.1), 128 / FWHM_TO_SIGMA, 2151, 0.1/(0.53 + 0.1), 353 / FWHM_TO_SIGMA], 'CO(3-2)':[2166, 4.15/(4.15 + 0.82), 134 / FWHM_TO_SIGMA, 2162, 0.82/(4.15 + 0.82), 353 / FWHM_TO_SIGMA]}


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

def extract_HCN_spectrum(ap_radius=5*u.arcsecond, pix_center=None, contsub=False, robust=2):
	#pix_center - center of aperture in pixel coordinates, if None use galaxy center

	if contsub == True:
		cube_path = '/Users/jotter/highres_PSBs/alma_cycle0/fitsimages/contsub_6-12/N1266_spw6_r2_1.5mJy_contsub_6-12_taper3as.fits'
	else:
		cube_path = glob.glob(f'/Users/jotter/highres_PSBs/alma_cycle0/fitsimages/no_contsub/N1266_spw6_r0.5_*.pbcor.fits')[0]

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
	mask_cube = cube.subcube_from_regions([ap])

	spectrum = mask_cube.sum(axis=(1,2))
	HCN_freq = 88.6316023 * u.GHz
	cube_vel = mask_cube.with_spectral_unit(u.km/u.s, rest_value=HCN_freq, velocity_convention='optical')
	vel = cube_vel.spectral_axis.to(u.km/u.s)

	return spectrum.value, vel.value



def plot_line_profiles(HCN_spectrum, HCN_vel, HCN_params, CO13_params, CO_lines):

	vel_arr = np.linspace(1500,2800,200)

	fig = plt.figure(figsize=(8,8))

	ax = fig.add_subplot(111)

	linestyles = ['dotted', 'dashed', 'dashdot']
	colors = ['tab:blue', 'tab:orange', 'tab:green']

	if HCN_spectrum is not None:
		#smooth_spec = gaussian_filter(HCN_spectrum, 2)
		velwid = HCN_vel[1:] - HCN_vel[:-1]
		ax.bar(HCN_vel, HCN_spectrum, velwid[0], ec='k', color='gray', alpha=0.25, label='HCN(1-0) spectrum')

	for i,line in enumerate(CO_lines):
		line_params = alatalo11_fits[line]
		line_fit = gauss_sum(vel_arr, *line_params)
		ax.plot(vel_arr, line_fit, linestyle=linestyles[i], color=colors[i], marker='', label=f'{line} fit')

	if CO13_params is not None:
		CO13_fit = gauss_sum(vel_arr, *CO13_params)
		ax.plot(vel_arr, CO13_fit, linestyle='dashed', marker='', label='13CO(2-1) fit', linewidth=3, color='tab:purple')

	if HCN_params is not None:
		HCN_fit = gauss_sum(vel_arr, *HCN_params)
		ax.plot(vel_arr, HCN_fit, linestyle='-', marker='', label='HCN(1-0) fit', linewidth=3, color='k')

	ax.set_ylabel('Normalized Flux', size=16)
	ax.set_xlabel('Velocity (km/s)', size=16)

	ax.tick_params(axis='both', labelsize=14)

	ax.set_xlim(1600,2700)

	plt.legend(fontsize=14)
	plt.savefig('../plots/line_profiles.pdf', bbox_inches='tight', dpi=300)
	plt.savefig('../plots/line_profiles.png', bbox_inches='tight')


fit_table = Table.read('../twocomp_fluxes_r2.csv', format='csv')

spectrum, vel = extract_HCN_spectrum()
#cont_params, residuals_clip_std, cont_serr, mask_ind = fit_continuum(spectrum, vel, 'HCN(1-0)', degree=1)
#spectrum = spectrum - (vel * cont_params[1] + cont_params[0])

HCN_row = np.where(fit_table['line'] == 'HCN(1-0)')[0][0]
HCN_params = [fit_table['comp1_vel'][HCN_row], fit_table['comp1_amp'][HCN_row]/(fit_table['comp1_amp'][HCN_row]+fit_table['comp2_amp'][HCN_row]), fit_table['comp1_sigma'][HCN_row],
			fit_table['comp2_vel'][HCN_row], fit_table['comp2_amp'][HCN_row]/(fit_table['comp1_amp'][HCN_row]+fit_table['comp2_amp'][HCN_row]), fit_table['comp2_sigma'][HCN_row]]

spectrum = spectrum / (fit_table['comp1_amp'][HCN_row] + fit_table['comp2_amp'][HCN_row])

CO13_row = np.where(fit_table['line'] == '13CO(2-1)')[0][0]
CO13_params = [fit_table['comp1_vel'][CO13_row], fit_table['comp1_amp'][CO13_row]/(fit_table['comp1_amp'][CO13_row]+fit_table['comp2_amp'][CO13_row]), fit_table['comp1_sigma'][CO13_row],
			fit_table['comp2_vel'][CO13_row], fit_table['comp2_amp'][CO13_row]/(fit_table['comp1_amp'][CO13_row]+fit_table['comp2_amp'][CO13_row]), fit_table['comp2_sigma'][CO13_row]]

plot_line_profiles(spectrum, vel, HCN_params, CO13_params, ['CO(1-0)','CO(2-1)', 'CO(3-2)'])


