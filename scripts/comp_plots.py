from astropy.table import Table

import numpy as np
import matplotlib.pyplot as plt



def convert_T_iram(T_mb, epoch='2008', line='CO(1-0)'):
	#Sv = 3514 * T_SD / (v_a_iram * D_iram**2) -- OLD
	#Sv = fconv * T_mb
	#T_mb is main beam temperature, NOT antenna temp

	if epoch == '2008' and  line == 'CO(1-0)':
		#these parameters are appropriate for Young11 ATLAS3D CO(1-0) using ABCD receiver
		fconv = 4.73 #Jy/K

	if epoch == '2003' and  line == 'CO(1-0)':
		#these parameters are appropriate for Combes07 in Sauron ETGs using ABCD receiver
		fconv = 4.95 #Jy/K

	if epoch == '2009' and line == 'HCN(1-0)':
		#use this value for post-2008 EMIR obs, appropriate for Krips2010, Crocker12 HCN
		fconv = 5.03 #Jy/K
	if epoch == '2009' and line == '13CO(1-0)':
		#use this value for post-2008 EMIR obs, appropriate for Krips2010, Crocker12 HCN
		fconv = 5.06 #Jy/K
	if epoch == '2009' and line == '12CO(1-0)':
		#use this value for post-2008 EMIR obs, appropriate for Krips2010 HCN
		fconv = 5.06 #Jy/K
	if epoch == '2009' and line == '13CO(2-1)':
		#use this value for post-2008 EMIR obs, appropriate for Krips2010, Crocker12 HCN
		fconv = 4.875 #Jy/K
	if epoch == '2013' and line == 'HCN(1-0)':
		#use for Privon 2015 GOALS
		fconv = 5.27 #Jy/K

	Sv = fconv * T_mb

	return Sv

#gao and solomon SFGs

gso_table = Table.read('../tables/gaosolomon04_tab3_Edit_tab4_edit.csv')
#this edited version removed galaxies also in the GOALs sample
#this also removed 1 starburst from the table, so the non lirgs should just be star-forming gals (NGC 1022)

gso_HCN = gso_table['S_HCN_delV']
gso_CO = gso_table['S_CO_delV']

frac_err_I_HCN = gso_table['sigma_HCN'] / gso_table['I_HCN']
frac_err_I_CO = gso_table['sigma_CO'] / gso_table['I_CO']

gso_e_HCN = frac_err_I_HCN * gso_HCN#/np.log(10)
gso_e_CO = frac_err_I_CO * gso_CO#/np.log(10)

gso_sfg = np.where(gso_table['LIR'] < 10)[0]
gso_lirg = np.where(gso_table['LIR'] >= 10)[0]

sfg_HCN = gso_HCN[gso_sfg]
sfg_CO = gso_CO[gso_sfg]
sfg_e_HCN = gso_e_HCN[gso_sfg]
sfg_e_CO = gso_e_CO[gso_sfg]

lirg_HCN = gso_HCN[gso_lirg]
lirg_CO = gso_CO[gso_lirg]
lirg_e_HCN = gso_e_HCN[gso_lirg]
lirg_e_CO = gso_e_CO[gso_lirg]

sfg_table = gso_table[gso_sfg]
lirg_table = gso_table[gso_lirg]

sfg_noulim_CO_HCN = np.intersect1d(np.where(sfg_table['CO_lim'] == 'N')[0], np.where(sfg_table['HCN_lim'] == 'N')[0])
sfg_ulim_HCN = np.where(sfg_table['HCN_lim'] == 'upper')[0]
sfg_llim_HCN = np.where(sfg_table['HCN_lim'] == 'lower')[0]
sfg_llim_CO = np.where(sfg_table['CO_lim'] == 'lower')[0]

lirg_noulim_CO_HCN = np.intersect1d(np.where(lirg_table['CO_lim'] == 'N')[0], np.where(lirg_table['HCN_lim'] == 'N')[0])
lirg_ulim_HCN = np.where(lirg_table['HCN_lim'] == 'upper')[0]
lirg_llim_HCN = np.where(lirg_table['HCN_lim'] == 'lower')[0]
lirg_llim_CO = np.where(lirg_table['CO_lim'] == 'lower')[0]


#crocker12 ATLAS3D ETGs
#error of -1 indicates upper limit

etg_table3 = Table.read('../tables/crocker12_tab3.csv')
etg_HCN_Kkms = etg_table3['HCN(1–0) (K km s−1)']
etg_e_HCN_Kkms = etg_table3['e_HCN(1–0) (K km s−1)']
etg_HCN = convert_T_iram(etg_HCN_Kkms, epoch='2009', line='HCN(1-0)')
etg_e_HCN = convert_T_iram(etg_e_HCN_Kkms, epoch='2009', line='HCN(1-0)')

etg_12CO_Kkms = etg_table3['12CO(1–0) (K km s−1)']
etg_e_12CO_Kkms = etg_table3['e_12CO(1–0) (K km s−1)']
etg_12CO = convert_T_iram(etg_12CO_Kkms, epoch='2009', line='12CO(1-0)')
etg_e_12CO = convert_T_iram(etg_e_12CO_Kkms, epoch='2009', line='12CO(1-0)')

etg_13CO_Kkms = etg_table3['13CO(1–0) (K km s−1)']
etg_13CO = convert_T_iram(etg_13CO_Kkms, epoch='2009', line='13CO(1-0)')

etg_13CO21_Kkms = etg_table3['13CO(2–1) (K km s−1)']
etg_13CO21 = convert_T_iram(etg_13CO21_Kkms, epoch='2009', line='13CO(2-1)')

etg_noulim_t3= np.where(etg_table3['e_HCN(1–0) (K km s−1)'] > 0)[0]
etg_noulim_t3 = np.setdiff1d(etg_noulim_t3, [3]) #remove NGC 1266
etg_ulim_HCN_t3 = np.where(etg_table3['e_HCN(1–0) (K km s−1)'] < 0)[0]

etg_HCN_n1266 = etg_HCN[3]
etg_13CO10_n1266 = etg_13CO[3]
etg_13CO21_n1266 = etg_13CO21[3]


print('Crocker N1266 fluxes')
print(f'HCN: {etg_HCN_n1266}, 13CO(1-0): {etg_13CO10_n1266}, 13CO(2-1): {etg_13CO21_n1266}')

#ratio table 4
etg_table = Table.read('../tables/crocker12_tab4.csv')

etg_hcn_13co = etg_table['HCN/13CO(1-0)']
e_up_etg_hcn_13co = etg_table['e_up_HCN/13CO(1-0)']
e_down_etg_hcn_13co = etg_table['e_down_HCN/13CO(1-0)']

etg_hcn_12co = etg_table['HCN/12CO(1-0)']
e_up_etg_hcn_12co = etg_table['e_up_HCN/12CO(1-0)']
e_down_etg_hcn_12co = etg_table['e_down_HCN/12CO(1-0)']

etg_12co_13co = etg_table['12CO/13CO(1-0)']
e_up_etg_12co_13co = etg_table['e_up_12CO/13CO(1-0)']
e_down_etg_12co_13co = etg_table['e_down_12CO/13CO(1-0)']

etg_13co_12co = 1/etg_12co_13co
e_up_etg_13co_12co = etg_13co_12co * (e_up_etg_12co_13co/etg_12co_13co)
e_down_etg_13co_12co = etg_13co_12co * (e_down_etg_12co_13co/etg_12co_13co)

etg_noulim = np.where(e_up_etg_hcn_13co > 0)[0]
print(etg_noulim)
etg_noulim = np.setdiff1d(etg_noulim, [3]) #remove NGC 1266
print(etg_noulim)
etg_ulim_HCN = np.where(e_up_etg_hcn_13co < 0)[0]



#French22 psbs

psb_table = Table.read('../tables/French22_PSBs.csv')

psb_HCN = psb_table['HCN(1-0)']
psb_e_HCN = psb_table['e_HCN(1-0)']#/(psb_table['HCN(1-0)']*np.log(10))
psb_HCN_ulim_col = psb_table['HCN_ulim']
psb_noulim = np.where(psb_HCN_ulim_col == 'N')[0]
psb_ulim_HCN = np.where(psb_HCN_ulim_col == 'Y')[0]

psb_CO = psb_table['CO(1-0)']
psb_e_CO = psb_table['e_CO(1-0)']#/(psb_table['CO(1-0)']*np.log(10))


#GOALS galaxies - privon15 and herrero-ilana 2019
gls_table = Table.read('../tables/goals_HCN_CO_matched.csv')

gls_HCN_Kkms = gls_table['Sigma T _mb dv (HCN (1-0)) (K km s^-1)']
gls_HCN = convert_T_iram(gls_HCN_Kkms, epoch='2013', line='HCN(1-0)')
e_gls_HCN_Kkms = gls_table['e_Sigma T _mb dv (HCN (1-0)) (K km s^-1)']
gls_e_HCN = convert_T_iram(e_gls_HCN_Kkms, epoch='2013', line='HCN(1-0)')#/(convert_T_iram(gls_HCN_Kkms, epoch='2013', line='HCN(1-0)')*np.log(10))

gls_ulim_HCN = np.where(gls_e_HCN < 0)[0]
gls_noulim = np.where(gls_e_HCN > 0)[0]

gls_CO = gls_table['S12COdeltav (Jy km/s)']
gls_e_CO = gls_table['e_S12COdeltav (Jy km/s)']#/(gls_table['S12COdeltav (Jy km/s)'] * np.log(10))

gls_13CO = gls_table['S13COdeltav (Jy km/s)']
gls_e_13CO = gls_table['e_S13COdeltav (Jy km/s)']#/(gls_table['S12COdeltav (Jy km/s)'] * np.log(10))

gls_1312CO = gls_13CO / gls_CO
e_gls_1312CO = gls_1312CO * np.sqrt((gls_e_13CO/gls_13CO)**2 + (gls_e_CO/gls_CO)**2)

gls_HCN_13CO = gls_HCN / gls_13CO
e_gls_HCN_13CO = gls_HCN_13CO * np.sqrt((gls_e_13CO/gls_13CO)**2 + (gls_e_HCN/gls_CO)**2)

### NGC 1266

n1266_tab = Table.read('../tables/twocomp_fluxes_constr_r2.csv')

n1266_HCN = n1266_tab['total_flux'][4]
e_n1266_HCN = n1266_tab['e_total_flux'][4]#/(n1266_tab['total_flux'][4] * np.log(10))
n1266_CO = n1266_tab['total_flux'][0]
e_n1266_CO = n1266_tab['e_total_flux'][0]#/(n1266_tab['total_flux'][0] * np.log(10))
n1266_13CO21 = n1266_tab['total_flux'][3]
e_n1266_13CO21 = n1266_tab['e_total_flux'][3]

n1266_HCN_sys = n1266_tab['comp1_flux'][4]
e_n1266_HCN_sys = n1266_tab['e_comp1_flux'][4]
n1266_CO_sys = n1266_tab['comp1_flux'][0]
e_n1266_CO_sys = n1266_tab['e_comp1_flux'][0]
n1266_13CO21_sys = n1266_tab['comp1_flux'][3]
e_n1266_13CO21_sys = n1266_tab['e_comp1_flux'][3]

n1266_HCN_out = n1266_tab['comp2_flux'][4]
e_n1266_HCN_out = n1266_tab['e_comp2_flux'][4]
n1266_CO_out = n1266_tab['comp2_flux'][0]
e_n1266_CO_out = n1266_tab['e_comp2_flux'][0]
n1266_13CO21_out = n1266_tab['comp2_flux'][3]
e_n1266_13CO21_out = n1266_tab['e_comp2_flux'][3]

amp_ratio_ulim = 0.02 #figured out from plot_spectrum.py -> twocomp_upperlimit(snr=88)
n1266_13CO21_out_ulim = n1266_13CO21_sys * amp_ratio_ulim

#from Crocker12
#12CO(2-1)/13CO(2-1)
#n1266_R21 = 24.8
#e_n1266_R21 = 3.7
#13CO(2-1)/13CO(1-0)
n1266_13CO_2110 = 4.04 / 1.04
e_n1266_13CO_2110 = n1266_13CO_2110 * np.sqrt((0.19/4.04)**2 + (0.09/1.04)**2) 


n1266_13CO = n1266_13CO21 * n1266_13CO_2110
n1266_13CO_sys = n1266_13CO21_sys * n1266_13CO_2110
e_n1266_13CO_sys = n1266_13CO_sys * np.sqrt((e_n1266_13CO21_sys/n1266_13CO21_sys)**2 + (e_n1266_13CO_2110/n1266_13CO_2110)**2)
n1266_13CO_out = n1266_13CO21_out * n1266_13CO_2110
e_n1266_13CO_out = n1266_13CO_out * np.sqrt((e_n1266_13CO21_out/n1266_13CO21_out)**2 + (e_n1266_13CO_2110/n1266_13CO_2110)**2)
n1266_13CO_out_ulim = n1266_13CO21_out_ulim * n1266_13CO_2110

print(e_n1266_13CO_sys/n1266_13CO_sys)
print(e_n1266_HCN_sys/n1266_HCN_sys)

print('my measured fluxes')
print(f'HCN: {n1266_HCN}, 13CO(2-1): {n1266_13CO}')

#krips 2010 collated lit table
krips_tab = Table.read('../tables/krips_2010.csv')

gal_type = krips_tab['Type']
CO_1312_raw = krips_tab['13CO_12CO(1-0)']
e_CO_1312_raw = krips_tab['e_13CO_12CO(1-0)']
HCN_13CO_raw = krips_tab['HCN_13CO']
e_HCN_13CO_raw = krips_tab['e_HCN_13CO']

CO_1312_ind = []
CO_1312_ulim_ind = []
HCN_13CO_ind = []
HCN_13CO_llim_ind = []
HCN_13CO_ulim_ind = []

for i in range(len(krips_tab)):
	if CO_1312_raw[i].replace('.','',1).isdigit() == True:
		CO_1312_ind.append(i)
	elif CO_1312_raw[i][0] == '<':
		CO_1312_ulim_ind.append(i)

	if HCN_13CO_raw[i].replace('.','',1).isdigit() == True:
		HCN_13CO_ind.append(i)
	elif HCN_13CO_raw[i][0] == '<':
		HCN_13CO_ulim_ind.append(i)
	elif HCN_13CO_raw[i][0] == '>':
		HCN_13CO_llim_ind.append(i)

# Yamaguchi 2012 L1157 shocked region
L1157_HCN = 74.37
e_L1157_HCN = 0.05
L1157_13CO = 6.04
e_L1157_13CO = 0.03
L1157_12CO = 51.17
e_L1157_12CO = 0.06

L1157_HCN_13CO = L1157_HCN / L1157_13CO
e_L1157_HCN_13CO = L1157_HCN_13CO * np.sqrt((e_L1157_HCN/L1157_HCN)**2 + (e_L1157_13CO/L1157_13CO)**2)

L1157_1312_CO = L1157_13CO / L1157_12CO
e_L1157_1312_CO = L1157_1312_CO * np.sqrt((e_L1157_12CO/L1157_12CO)**2 + (e_L1157_13CO/L1157_13CO)**2)

L1157_HCN_12CO = L1157_HCN / L1157_12CO
e_L1157_HCN_12CO = L1157_HCN_12CO * np.sqrt((e_L1157_HCN/L1157_HCN)**2 + (e_L1157_12CO/L1157_12CO)**2)

#### plotting

def CO_ratio_HCN_plot():
	### make plot of 13/12 CO ratio and HCN/13CO ratio
	# get the correct indices
	# convert the arrays to floats, then plot appropriately for Krips2010 table

	both_ind = np.intersect1d(CO_1312_ind, HCN_13CO_ind)

	CO_1312_both = np.array(CO_1312_raw[both_ind], dtype='float64')
	HCN_13CO_both = np.array(HCN_13CO_raw[both_ind], dtype='float64')

	e_CO_1312_raw[e_CO_1312_raw == 'na'] = 0
	e_CO_1312_both = np.array(e_CO_1312_raw[both_ind], dtype='float64')

	e_HCN_13CO_raw[e_HCN_13CO_raw == 'na'] = 0
	e_HCN_13CO_both = np.array(e_HCN_13CO_raw[both_ind], dtype='float64')


	etg = np.concatenate((np.where(gal_type[both_ind] == 'Elliptical')[0], np.where(gal_type[both_ind] == 'Sauron S(0)')[0]))
	seyfert = np.where(gal_type[both_ind] == 'Seyfert')[0]
	starburst = np.where(gal_type[both_ind] == 'Starburst')[0]
	SF = np.where(gal_type[both_ind] == 'SF-spiral')[0]
	dwarf = np.where(gal_type[both_ind] == 'Dwarf')[0]
	pec = np.where(gal_type[both_ind] == 'Peculiar')[0]

	fig = plt.figure(figsize=(8,8))


	#GOALS
	plt.errorbar(gls_1312CO[gls_noulim], gls_HCN_13CO[gls_noulim], xerr=e_gls_1312CO[gls_noulim], yerr=e_gls_HCN_13CO[gls_noulim], linestyle='', marker='d', color='tab:blue', label='GOALS')
	plt.errorbar(gls_1312CO[gls_ulim_HCN], gls_HCN_13CO[gls_ulim_HCN], xerr=e_gls_1312CO[gls_ulim_HCN], yerr=gls_HCN_13CO[gls_ulim_HCN]*0.15, uplims=True, linestyle='', marker='d', color='tab:blue', mfc='white')

	#plt.errorbar(CO_1312_both[etg], HCN_13CO_both[etg], xerr=e_CO_1312_both[etg], yerr=e_HCN_13CO_both[etg], linestyle='', marker='o', color='tab:red', label='Early-type')
	plt.errorbar(CO_1312_both[seyfert], HCN_13CO_both[seyfert], xerr=e_CO_1312_both[seyfert], yerr=e_HCN_13CO_both[seyfert], linestyle='', marker='s', color='k', label='Seyfert')
	plt.errorbar(CO_1312_both[starburst], HCN_13CO_both[starburst], xerr=e_CO_1312_both[starburst], yerr=e_HCN_13CO_both[starburst], linestyle='', marker='*', color='tab:green', label='Starburst')
	#plt.errorbar(CO_1312_both[SF], HCN_13CO_both[SF], xerr=e_CO_1312_both[SF], yerr=e_HCN_13CO_both[SF], linestyle='', marker='o', color='tab:cyan', label='Star-forming')
	#plt.errorbar(CO_1312_both[dwarf], HCN_13CO_both[dwarf], xerr=e_CO_1312_both[dwarf], yerr=e_HCN_13CO_both[dwarf], linestyle='', marker='P', color='tab:purple', label='Dwarf')
	#plt.errorbar(CO_1312_both[pec], HCN_13CO_both[pec], xerr=e_CO_1312_both[pec], yerr=e_HCN_13CO_both[pec], linestyle='', marker='v', color='tab:olive', label='Merger/Peculiar')

	
	#ATLAS3D
	plt.errorbar(etg_13co_12co[etg_noulim], etg_hcn_13co[etg_noulim], xerr=[e_up_etg_13co_12co[etg_noulim], e_down_etg_13co_12co[etg_noulim]], 
					yerr=[e_up_etg_hcn_13co[etg_noulim], e_down_etg_hcn_13co[etg_noulim]], linestyle='', marker='p', color='tab:red', label='ATLAS3D ETGs')
	plt.errorbar(etg_13co_12co[etg_ulim_HCN], etg_hcn_13co[etg_ulim_HCN], xerr=[e_up_etg_13co_12co[etg_ulim_HCN], e_down_etg_13co_12co[etg_ulim_HCN]],
					yerr=etg_hcn_13co[etg_ulim_HCN]*0.15, uplims=True, linestyle='', marker='p', color='tab:red', mfc='white')

	#L1157 Yamaguchi 2012
	plt.errorbar(L1157_1312_CO, L1157_HCN_13CO, xerr=e_L1157_1312_CO, yerr=e_L1157_HCN_13CO, linestyle='', marker='H', color='tab:olive',
					label='L1157 Shock', markersize=8)

	#NGC 1266 this work outflow and sys
	n1266_1312CO_out_ulim = n1266_13CO_out_ulim / n1266_CO_out
	n1266_HCN_13CO_out_llim = n1266_HCN_out / n1266_13CO_out_ulim
	n1266_1312CO_sys = n1266_13CO_sys / n1266_CO_sys
	n1266_HCN_13CO_sys = n1266_HCN_sys / n1266_13CO_sys

	e_n1266_1312CO_sys = n1266_1312CO_sys * np.sqrt((e_n1266_13CO_sys/n1266_13CO_sys)**2 + (e_n1266_CO_sys/n1266_CO_sys)**2)
	e_n1266_HCN_13CO_sys = n1266_HCN_13CO_sys  * np.sqrt((e_n1266_13CO_sys/n1266_13CO_sys)**2 + (e_n1266_HCN_sys/n1266_HCN_sys)**2)

	plt.errorbar(n1266_1312CO_out_ulim, n1266_HCN_13CO_out_llim, xerr=n1266_1312CO_out_ulim*0.15, yerr=n1266_HCN_13CO_out_llim*0.15, lolims=True, xuplims=True, linestyle='', marker='s', color='tab:orange', label='NGC 1266 (outflow)', markersize=10, mec='k')
	plt.errorbar(n1266_1312CO_sys, n1266_HCN_13CO_sys, xerr=e_n1266_1312CO_sys, yerr=e_n1266_HCN_13CO_sys, linestyle='', marker='D', color='tab:orange', label='NGC 1266 (systemic)', markersize=10, mec='k')


	#using Crocker12 NGC 1266 values
	n1266_1312CO = 	etg_13co_12co[3]
	e_up_n1266_1312CO = e_up_etg_13co_12co[3]
	e_down_n1266_1312CO = e_down_etg_13co_12co[3]
	n1266_HCN_13CO = etg_hcn_13co[3]
	e_up_n1266_HCN_13CO = e_up_etg_hcn_13co[3]
	e_down_n1266_HCN_13CO = e_down_etg_hcn_13co[3]

	plt.errorbar(n1266_1312CO, n1266_HCN_13CO, xerr=[[e_up_n1266_1312CO], [e_down_n1266_1312CO]], yerr=[[e_up_n1266_HCN_13CO], [e_down_n1266_HCN_13CO]], linestyle='', marker='H', color='tab:orange', label='NGC 1266 (single dish)', markersize=10, mec='k')


	plt.xlabel(r'$^{13}$CO(1-0) / $^{12}$CO(1-0)  flux ratio', fontsize=16)
	plt.ylabel(r'HCN(1-0) / $^{13}$CO(1-0) flux ratio', fontsize=16)

	plt.tick_params(labelsize=12)

	plt.loglog()

	plt.legend(fontsize=14)
	plt.ylim(0.02,20)

	plt.savefig('../plots/HCN_CO_ratio_plot.pdf', bbox_inches='tight', dpi=300)



def HCN_12CO_ratio_histogram():

	fig, (ax0,ax1) = plt.subplots(2, 1, figsize=(8,8), sharex=True, gridspec_kw={'hspace':0.03})

	sfg_ratio = sfg_HCN[sfg_noulim_CO_HCN]/sfg_CO[sfg_noulim_CO_HCN]
	sfg_ratio_ulim = sfg_HCN[np.concatenate((sfg_ulim_HCN, sfg_llim_CO))]/sfg_CO[np.concatenate((sfg_ulim_HCN, sfg_llim_CO))]
	sfg_ratio_llim = sfg_HCN[sfg_llim_HCN]/sfg_CO[sfg_llim_HCN]

	lirg_ratio = lirg_HCN[lirg_noulim_CO_HCN]/lirg_CO[lirg_noulim_CO_HCN]
	lirg_ratio_ulim = lirg_HCN[np.concatenate((lirg_ulim_HCN, lirg_llim_CO))]/lirg_CO[np.concatenate((lirg_ulim_HCN, lirg_llim_CO))]
	lirg_ratio_llim = lirg_HCN[lirg_llim_HCN]/lirg_CO[lirg_llim_HCN]

	etg_ratio = etg_HCN[etg_noulim_t3]/etg_12CO[etg_noulim_t3]
	etg_ratio_ulim = etg_HCN[etg_ulim_HCN_t3]/etg_12CO[etg_ulim_HCN_t3]

	psb_ratio = psb_HCN[psb_noulim]/psb_CO[psb_noulim]
	psb_ratio_ulim = psb_HCN[psb_ulim_HCN]/psb_CO[psb_ulim_HCN]

	gls_ratio = gls_HCN[gls_noulim]/gls_CO[gls_noulim]
	gls_ratio_ulim = gls_HCN[gls_ulim_HCN]/gls_CO[gls_ulim_HCN]

	#ngc1266_ratio = n1266_HCN/n1266_CO
	n1266_ratio = etg_ratio[3] #using Crocker 2012 value
	n1266_ratio_out = n1266_HCN_out/n1266_CO_out
	n1266_ratio_sys = n1266_HCN_sys/n1266_CO_sys

	all_ratio = np.concatenate((sfg_ratio, lirg_ratio, etg_ratio, psb_ratio, gls_ratio))
	all_hist_log, all_bins_log = np.histogram(np.log10(all_ratio), bins='auto')

	all_bins = np.power(10, all_bins_log)

	hatches = [None,"/"]
	linestyles = ['-', 'dashed']
	
	ax0.hist([sfg_ratio, sfg_ratio_ulim, sfg_ratio_llim], all_bins, color=['tab:blue','tab:blue','tab:blue'], edgecolor=['navy','navy','navy'], alpha=0.25, label='Star-forming', histtype='barstacked', hatch=[None,"/", "\\"])
	ax0.hist([sfg_ratio, sfg_ratio_ulim, sfg_ratio_llim], all_bins, edgecolor=['tab:blue','tab:blue','tab:blue'], histtype='step', stacked=True, linewidth=4, linestyle=['-', 'dashed', 'dotted'])
	#ax0.hist(sfg_ratio_ulim, all_bins, edgecolor='tab:blue', histtype='step', linewidth=3, linestyle='dashed', hatch='//')
	#ax0.hist(sfg_ratio_llim, all_bins, edgecolor='tab:blue', histtype='step', linewidth=3, linestyle='dashed', hatch='\\')

	ax0.hist([lirg_ratio, lirg_ratio_ulim], all_bins, color=['tab:cyan', 'tab:cyan'], edgecolor=['darkcyan', 'darkcyan'], alpha=0.25, label='LIRG (Gao 04)', histtype='barstacked', hatch=hatches)
	ax0.hist([lirg_ratio, lirg_ratio_ulim], all_bins, color=['tab:cyan', 'tab:cyan'], histtype='step', stacked=True, linewidth=3, linestyle=linestyles)


	#ax0.hist([gls_ratio,gls_ratio_ulim], all_bins, color=['tab:pink', 'k'], alpha=0.25, label='LIRG (GOALS)', histtype='bar', stacked=True, hatch=hatches)
	ax0.hist([gls_ratio,gls_ratio_ulim], all_bins, color=['tab:pink', 'tab:pink'], edgecolor=['purple', 'purple'], alpha=0.25, label='LIRG (GOALS)', histtype='barstacked', hatch=hatches)
	#ax0.hist([gls_ratio, gls_ratio_ulim], all_bins, color=['tab:pink', 'tab:pink'], alpha=0.25, label='LIRG (GOALS)')
	ax0.hist([gls_ratio, gls_ratio_ulim], all_bins, edgecolor=['tab:pink', 'tab:pink'], histtype='step', stacked=True, linewidth=2, linestyle=linestyles)
	#ax0.hist([gls_ratio, gls_ratio_ulim], all_bins, color=['tab:pink', 'tab:pink'], histtype='step', stacked=True, linewidth=3, linestyle=linestyles, hatch=hatches)
	#ax0.hist(gls_ratio_ulim, all_bins, edgecolor='tab:pink', histtype='step', linewidth=3, linestyle='dashed', hatch='///')

	
	ax1.hist([psb_ratio, psb_ratio_ulim], all_bins, color=['tab:purple','tab:purple'], edgecolor=['purple','purple'], alpha=0.25, label='PSB', histtype='barstacked', hatch=hatches)
	ax1.hist([psb_ratio, psb_ratio_ulim], all_bins, edgecolor=['tab:purple','tab:purple'], histtype='step', stacked=True, linewidth=4, linestyle=linestyles)
	#ax1.hist(psb_ratio_ulim, all_bins, edgecolor='tab:purple', histtype='step', linewidth=3, linestyle='dashed', hatch='//')

	ax1.hist([etg_ratio, etg_ratio_ulim], all_bins, color=['tab:red','tab:red'], edgecolor=['maroon','maroon'], alpha=0.25, label='Early Type', histtype='barstacked', hatch=hatches)
	ax1.hist([etg_ratio, etg_ratio_ulim], all_bins, edgecolor=['tab:red','tab:red'], histtype='step', stacked=True, linewidth=2, linestyle=linestyles)
	#ax1.hist(etg_ratio_ulim, all_bins, edgecolor='tab:red', histtype='step', linewidth=3, linestyle='dashed', hatch='/')
	
	ax0.axvline(n1266_ratio, color='tab:orange', label='NGC 1266 (single dish)', linewidth=3)
	ax0.axvline(n1266_ratio_out, color='tab:orange', label='NGC 1266 (outflow)', linewidth=3, linestyle='dashed')
	ax0.axvline(n1266_ratio_sys, color='tab:orange', label='NGC 1266 (systemic)', linewidth=3, linestyle='dotted')
	ax1.axvline(n1266_ratio, color='tab:orange', linewidth=3)
	ax1.axvline(n1266_ratio_out, color='tab:orange', linewidth=3, linestyle='dashed')
	ax1.axvline(n1266_ratio_sys, color='tab:orange', linewidth=3, linestyle='dotted')

	ax1.set_xlabel(r'HCN/$^{12}$CO flux ratio', fontsize=16)
	ax0.set_ylabel('Number', fontsize=16)
	ax1.set_ylabel('Number', fontsize=16)

	ax0.tick_params(axis='y', labelsize=14)
	ax0.tick_params(axis='x', labelsize=0)
	ax0.set_yticks(np.arange(0,17,2))
	ax1.tick_params(axis='both', labelsize=14)

	ax0.legend(fontsize=12)
	ax1.legend(fontsize=12)

	plt.semilogx()

	plt.savefig('../plots/HCN_CO_ratio_histogram.pdf')

CO_ratio_HCN_plot()
#HCN_12CO_ratio_histogram()



