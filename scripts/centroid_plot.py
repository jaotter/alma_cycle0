from astropy.io import fits
from spectral_cube import SpectralCube
from photutils.centroids import (centroid_1dg, centroid_2dg,
                                 centroid_com, centroid_quadratic)

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from matplotlib.gridspec import GridSpec

import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const


line_dict = {'SiO(2-1)':86.84696, 'SiO(5-4)':217.10498, 'SiO(6-5)':260.51802, 'SiO(7-6)':303.92696, 'SiO(8-7)':347.330631,
				'H13CO+(1-0)':86.7542884, 'H13CO+(3-2)':260.255339, 'H13CO+(4-3)':346.998344, '13CO(2-1)':220.3986842, 'HC18O+(1-0)':85.1622231, 
				'HCN(1-0)':88.6316023, 'HCN(3-2)':265.8864343, 'HCN(4-3)':354.5054779,
				'CH3OH 5(2,3)-5(-1,5)':347.507451, 'CH3OH 14(-1,14)-13(1,12)':346.783, 'CH3OH 2(1,1)-2(0,2)-+':304.208,
				'CH3OH 1(1,0)-1(0,1)-+':303.367, 'CH3OH 3(3,0)-4(-2,3)':260.968, 'NH2CHO 12(1,11)-11(1,10)':261.327, 'HC17O+(3-2)':261.165,
				'HC15N(3-2)':258.157, 'SO 6(6)-5(5)':258.256, 'H13CN(3-2)':259.012, 'H2S 2(2,0)-2(1,1)':216.71, 'HN13C(3-2)':261.263,
				'CH3OH 16(1) -15(3) E1':217.525, 'CH3OH 39(-3) -39(3) E2':217.526028, 'CH3OH 6(1,5)-7(2,6)--':217.299, 'CO(1-0)':115.271,
				'CO(2-1)':230.538, 'CO(3-2)':345.796
				}

z = 0.007214


def channel_centroids(cube_kms, bin_centers, bin_widths, centroid_method='com'):
	#cube should be line cube to plot
	#channel bins should be edges of channel bins to plot, so len(channel_bins) = n_bins + 1
	#returns list of centroids in pix coords

	centroid_list = []

	fig = plt.figure(figsize=(12,12))

	gs = GridSpec(4,4,figure=fig)

	ygrid, xgrid = np.mgrid[0:4,0:4]
	ygrid_flat = ygrid.flatten()
	xgrid_flat = xgrid.flatten()

	#create channel maps figure

	for i, vel_center in enumerate(bin_centers):
		vel_upper = vel_center + bin_widths[i]/2
		vel_lower = vel_center - bin_widths[i]/2
		
		channel_slab = cube_kms.spectral_slab(vel_lower, vel_upper)
		channel_mom0_full = channel_slab.moment(order=0)
		pix_cent = np.array(channel_mom0_full.shape)/2
		channel_mom0 = Cutout2D(channel_mom0_full, pix_cent, size=10*u.arcsecond, wcs=cube_kms.wcs.celestial).data

		if centroid_method == 'com':
			channel_cent = centroid_com(channel_mom0)
		if centroid_method == '2dg':
			channel_cent = centroid_2dg(channel_mom0)
		if centroid_method == 'quad':
			channel_cent = centroid_quadratic(channel_mom0)

		xpos = xgrid_flat[i]
		ypos = ygrid_flat[i]

		ax = fig.add_subplot(gs[ypos, xpos])
		ax.imshow(channel_mom0.value, origin='lower')
		ax.plot(channel_cent[0], channel_cent[1], marker='+', markersize=20, color='red')
		ax.text(0.1, 0.9, rf'{int(vel_center.value)} $\pm$ {bin_widths[i]/2}', transform=ax.transAxes, color='white')
		#plt.xlim(65, 115)
		#plt.ylim(65, 115)
		

		centroid_list.append(channel_cent)

	plt.savefig(f'../plots/channel_maps_{line_name}.png')

	return 	centroid_list


def plot_centroids(cube_kms, centroid_list, bin_centers, bin_widths, line_name):

	line_slab = cube_kms.spectral_slab(bin_centers[0], bin_centers[-1])
	line_mom0_full = line_slab.moment(order=0)
	pix_cent = np.array(line_mom0_full.shape)/2
	line_mom0 = Cutout2D(line_mom0_full, pix_cent, size=10*u.arcsecond, wcs=cube_kms.wcs.celestial).data

	fig = plt.figure(figsize=(6,6))
	ax = fig.add_subplot(111, projection=cube_kms.wcs.celestial)
	ax.imshow(line_mom0.value, cmap='Greys', origin='lower')
	#ax.set_ylim(65,115)
	#ax.set_xlim(65,115)

	#centroid_world = cube.wcs.celestial.all_pix2world(centroid_list, (0,0))

	centroid_x_vals = [c[0] for c in centroid_list]
	centroid_y_vals = [c[1] for c in centroid_list]

	ax.scatter(centroid_x_vals, centroid_y_vals, c=bin_centers.value, marker='+', cmap='RdBu_r', linestyle='--')

	plt.savefig(f'../plots/{line_name}centroid_plot.png')


#spw = '9'
#line_name = '13CO(2-1)'

#spw = '6'
#line_name = 'HCN(1-0)'
#cube_path = glob.glob(f'/Users/jotter/highres_PSBs/alma_cycle0/fitsimages/no_contsub/N1266_spw{spw}_r0.5_*.pbcor.fits')[0]

#line_name = 'CO(2-1)'
#cube_path = '/Users/jotter/highres_PSBs/ngc1266_data/co21.fits'
#line_name = 'CO(1-0)'
#cube_path = '/Users/jotter/highres_PSBs/ngc1266_data/co.fits'
line_name = 'CO(3-2)'
cube_path = '/Users/jotter/highres_PSBs/ngc1266_data/co32.fits'


cube = SpectralCube.read(cube_path)
rest_freq = line_dict[line_name] * u.GHz
line_freq = rest_freq / (1+z)
cube_vel = cube.with_spectral_unit(u.km/u.s, rest_value=line_freq, velocity_convention='optical')

bin_centers = [-575,-500,-450,-350,-300,-250,-200,-150,150,200,250,300,350,450,500,575] * u.km/u.s
bin_widths = [350,300,300,200,200,200,200,200,200,200,200,200,200,300,300,350] * u.km/u.s

cent_list = channel_centroids(cube_vel, bin_centers, bin_widths, centroid_method='quad')
plot_centroids(cube_vel, cent_list, bin_centers, bin_widths, line_name=line_name, )

