import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad


cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

z = 0.007214
galv = np.log(z+1)*const.c.to(u.km/u.s)

X_HCN = 20 * u.Msun / (u.K * u.km / u.s * u.pc**2)
z = 0.007214
D_L = cosmo.luminosity_distance(z)  
as_to_pc = (cosmo.angular_diameter_distance(z) / u.radian).to(u.pc / u.arcsecond)

tdyn = 2.6 * u.Myr

def gaussfunc(velocity, ctr, amp, wid):
	#velocity is velocity array for given line
	#params must be divisible by 3, should be center (velocity), amplitude, width

	y = amp * np.exp( -(velocity - ctr)**2/(2*wid**2))

	return y

def vesc_mass():
	vesc = 340 #* u.km/u.s

	#fraction of emission beyond this for sigma = 145 km/s
	#for CO(2-1) constrained
	sigma = 145 #* u.km/u.s
	amp = 0.244 #Jy/beam
	cent = 2166 #km/s

	integral = quad(gaussfunc, galv.value - vesc, galv.value + vesc, args=(cent, amp, sigma))
	#full_flux = quad(gaussfunc, -1e5, 1e5, args=(cent, amp, sigma))
	
	full_flux = np.sqrt(2 * np.pi) * sigma * amp

	print('CO(2-1) escape fraction = '+str(1 - integral[0]/full_flux))

	#fraction of emission beyond this for sigma = 145 km/s
	#for HCN(1-0) unconstrained
	sigma = 144.7 #* u.km/u.s
	amp = 0.0171 #Jy/beam
	cent = 2174 #km/s

	integral = quad(gaussfunc, galv.value - vesc, galv.value + vesc, args=(cent, amp, sigma))
	#full_flux = quad(gaussfunc, -1e5, 1e5, args=(cent, amp, sigma))
	
	full_flux = np.sqrt(2 * np.pi) * sigma * amp

	print('HCN(1-0) escape fraction = '+str(1 - integral[0]/full_flux))


def mass_HCN():
	#value from Solomon 98 and Greve 2006
	#the K is from brightness temp, need to convert Jy to brightness temp

	bmaj = (1.106414397558E-03 * u.degree).to(u.radian)
	bmin = (9.102843867408E-04 * u.degree).to(u.radian)
	freq = 88.631 * u.GHz
	beam_area = np.pi * bmaj * bmin
	#HCN_flux_Jy = 2.96 * u.Jy #* u.km / u.s
	HCN_flux_Jy = 6.2 * u.Jy #* u.km / u.s


	theta_n1266_HCN = (3.98*u.arcsecond).to(u.radian) #beam size in radians


	#HCN_flux_K = HCN_flux_Jy / (2 * const.k_B * freq**2 / const.c**2)
	#print(HCN_flux_K.to(u.K * u.km/u.s))
	equiv = u.brightness_temperature(beam_area, freq)
	HCN_flux_K = HCN_flux_Jy.to(u.K, equivalencies=equiv) * u.km / u.s

	#use equation from Gao&Solomon03, eq1
	#HCN_L_K = np.pi/(4 * np.log(2)) * bmaj**2 * HCN_flux_K * D_L**2 / ((1+z)**3)
	#HCN_L_K = 3.25e7 * 

	#simple flux to luminosity
	#HCN_L_K = HCN_flux_K * 4 * np.pi * D_L**2

	#from Gao 2004:
	HCN_L_K = HCN_flux_K * (D_L.to(u.pc))**2 * np.pi / (4 * np.log(2)) * (1+z)**3 * theta_n1266_HCN.value**2
	print(HCN_L_K.to(u.K * u.km / u.s  * u.pc**2))

	mass = HCN_L_K * X_HCN

	print('mass from HCN')
	print(mass.to(u.Msun))

	Mdot_out = (mass / tdyn).to(u.Msun / u.yr)

	print('Outflow mass rate: '+str(Mdot_out))
	print('Outflow escape rate: '+str(Mdot_out * 0.02))




def mass_13CO():
	###13CO outflow mass upper limit
	# this is from Pineda 2010
	freq_13CO = 110.2013543 * u.GHz

	#bmaj_13CO = (2.0 * u.arcsecond).to(u.radian)
	#bmin_13CO = (1.9 * u.arcsecond).to(u.radian)
	#beam_area_13CO = np.pi * bmaj * bmin
	ap_area_13CO = np.pi * (5*u.arcsecond)**2
	ap_area_13CO_phys = ap_area_13CO * as_to_pc**2

	print(as_to_pc)
	print(ap_area_13CO_phys)

	flux_13CO_total_Jy = 20.65 * u.Jy #* u.km / u.s
	flux_13CO_lim_Jy = flux_13CO_total_Jy * 0.02
	flux_13CO_lim_Tmb = flux_13CO_lim_Jy.to(u.K, equivalencies=u.brightness_temperature(freq_13CO, ap_area_13CO)) * u.km / u.s
	print(flux_13CO_lim_Tmb)

	#Aul = 6.33e-8 / u.s #this value is for 13CO(1-0)
	Aul = 3.73e-08 / u.s #from Goorvitch 1994
	Tex = 10 * u.K
	B0 = 5.51e10 / u.s


	Nu = (8 * np.pi * const.k_B * freq_13CO**2) / (const.h * const.c**3 * Aul) * flux_13CO_lim_Tmb

	N13CO = (Nu * (const.k_B * Tex / (const.h * B0)) / (5) * np.exp(const.h * B0 * 5 / (const.k_B * Tex))).to(u.cm**-2)

	#abund_1213CO = 69
	abund_1213CO = 250

	print('H2 column density from 13CO')
	print(N13CO * abund_1213CO / 1e-4)

	n_13CO = (N13CO * ap_area_13CO_phys).decompose()
	n_12CO = n_13CO * abund_1213CO
	n_H2 = n_12CO / 1e-4 #ratio used in Alatalo11
	M_H2_lim = n_H2 * 2 * const.u * 1.36 #atomic mass, then multiply to include He

	print(M_H2_lim.to(u.Msun))

	Mdot_out = (M_H2_lim / tdyn).to(u.Msun/u.yr)

	print('Outflow mass rate: '+str(Mdot_out))
	print('Outflow escape rate: '+str(Mdot_out * 0.02))


def gas_depletion_time():
	total_mass = 1.1e9 * u.Msun

	Mdot_outflow = 1.706 * u.Msun / u.yr
	SFR = 0.7 * u.Msun / u.yr

	tdepletion = total_mass / (Mdot_outflow + SFR)

	print('depletion time')
	print(tdepletion.to(u.Myr))
	


#vesc_mass()

#mass_HCN()
mass_13CO()
#gas_depletion_time()
