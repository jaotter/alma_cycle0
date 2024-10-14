from astropy.io import fits
from spectral_cube import SpectralCube
from photutils.centroids import (centroid_1dg, centroid_2dg,
                                 centroid_com, centroid_quadratic)

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from astropy.convolution import Gaussian1DKernel
from radio_beam import Beam

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


def highvel_contours(cube_kms, line_name, vel_offset=400*u.km/u.s, bin_width=200*u.km/u.s):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	
	#taking velocity ranges from Alatalo11
	#vel_lower_blue, vel_upper_blue = (-575, -275) * u.km/u.s #-vel_offset - bin_width/2, -vel_offset + bin_width/2
	#vel_lower_red, vel_upper_red = (175, 475) * u.km/u.s #vel_offset - bin_width/2, vel_offset + bin_width/2

	vel_lower_blue, vel_upper_blue = (-400, -175) * u.km/u.s #-vel_offset - bin_width/2, -vel_offset + bin_width/2
	vel_lower_red, vel_upper_red = (125, 400) * u.km/u.s #vel_offset - bin_width/2, vel_offset + bin_width/2
	vel_lower_cent, vel_upper_cent = (-50, 50) * u.km/u.s

	channel_slab_bl = cube_kms.spectral_slab(vel_lower_blue, vel_upper_blue)
	channel_slab_red = cube_kms.spectral_slab(vel_lower_red, vel_upper_red)
	channel_slab_cent = cube_kms.spectral_slab(vel_lower_cent, vel_upper_cent)

	channel_mom0_bl = channel_slab_bl.moment(order=0)
	channel_mom0_red = channel_slab_red.moment(order=0)
	channel_mom0_cent = channel_slab_cent.moment(order=0)

	pix_cent = np.array(channel_mom0_bl.shape)/2
	mom0_blue = Cutout2D(channel_mom0_bl, pix_cent, size=10*u.arcsecond, wcs=cube_kms.wcs.celestial).data
	mom0_red = Cutout2D(channel_mom0_red, pix_cent, size=10*u.arcsecond, wcs=cube_kms.wcs.celestial).data
	mom0_cent = Cutout2D(channel_mom0_cent, pix_cent, size=10*u.arcsecond, wcs=cube_kms.wcs.celestial).data

	cent_cont_lvls = np.nanmax(mom0_cent) * np.array([0.7, 0.9, 0.95, 0.99])
	#red_cont_lvls = np.nanmax(mom0_red) * np.array([0.5, 0.7, 0.9, 0.95, 0.99])
	#blue_cont_lvls = np.nanmax(mom0_blue) * np.array([0.5, 0.7, 0.9, 0.95, 0.99])

	blue_cont_lvls = [3, 4, 5, 6] * u.Jy * u.km / u.s
	red_cont_lvls = blue_cont_lvls
	#cent_cont_lvls = [6.4, 8.6, 10.8, 13, 15.2, 17.4] * u.Jy *u.km / u.s

	print(cent_cont_lvls, red_cont_lvls, blue_cont_lvls)

	print(np.nanstd(mom0_cent), np.nanstd(mom0_blue), np.nanstd(mom0_red))

	ax.contour(mom0_blue, colors='blue', levels=blue_cont_lvls)
	ax.contour(mom0_red, colors='red', levels=red_cont_lvls)
	ax.contour(mom0_cent, colors='k', levels=cent_cont_lvls)

	labels = [rf'{vel_lower_blue.value}km/s : {vel_upper_blue.value}km/s', f'{vel_lower_red.value}km/s : {vel_upper_red.value}km/s', f'{vel_lower_cent.value}km/s : {vel_upper_cent.value}km/s']

	custom_lines = [Line2D([0], [0], color='blue', lw=2),
					Line2D([0], [0], color='red', lw=2),
					Line2D([0], [0], color='k', lw=2)]


	ax.legend(custom_lines, labels)

	ax.set_title(line_name)

	#ax.imshow(channel_mom0.value, origin='lower')
	#ax.text(0.1, 0.9, rf'{int(vel_center.value)} $\pm$ {bin_widths[i]/2}', transform=ax.transAxes, color='white')

	beam = cube_kms.beam
	pix_scale = cube_kms.wcs.celestial.proj_plane_pixel_scales()

	bmwidth = (beam.minor/pix_scale[0]).value
	bmheight = (beam.major/pix_scale[0]).value
	bmpa = beam.pa.value
	bmpatch = Ellipse((10,10), bmwidth, bmheight, bmpa, facecolor='lime',edgecolor='k', alpha=0.5)
	ax.add_patch(bmpatch)
	
	plt.savefig(f'../plots/highv_contours_{line_name}.png')


#spw = '9'
#line_name = '13CO(2-1)'

#spw = '6'
#line_name = 'HCN(1-0)'
#cube_path = glob.glob(f'/Users/jotter/highres_PSBs/alma_cycle0/fitsimages/no_contsub/N1266_spw{spw}_r0.5_*.pbcor.fits')[0]

line_name = 'CO(2-1)'
cube_path = '/Users/jotter/highres_PSBs/ngc1266_data/co21.fits'
#line_name = 'CO(1-0)'
#cube_path = '/Users/jotter/highres_PSBs/ngc1266_data/co.fits'
#line_name = 'CO(3-2)'
#cube_path = '/Users/jotter/highres_PSBs/ngc1266_data/co32.fits'


cube = SpectralCube.read(cube_path)
rest_freq = line_dict[line_name] * u.GHz
line_freq = rest_freq / (1+z)
cube_vel = cube.with_spectral_unit(u.km/u.s, rest_value=line_freq, velocity_convention='optical')

#kern = Gaussian1DKernel(5)
#cube_vel = cube_vel.spectral_smooth(kern)


highvel_contours(cube_vel, line_name)

#bin_centers = [-575,-500,-450,-350,-300,-250,-200,-150,150,200,250,300,350,450,500,575] * u.km/u.s
#bin_widths = [500,400,400,300,300,300,200,100,100,100,100,200,300,300,400,400,500] * u.km/u.s

#cent_list = channel_centroids(cube_vel, bin_centers, bin_widths, centroid_method='quad')
#plot_centroids(cube_vel, cent_list, bin_centers, bin_widths, line_name=line_name, )

