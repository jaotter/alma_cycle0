from astropy.table import Table

import numpy as np
import matplotlib.pyplot as plt

#IRAM values
D_iram = 30 #m
v_a_iram = 0.57 #at 110 GHz, see https://web-archives.iram.fr/ARN/jan95/node46.html

def convert_T_iram(T_SD):
	Sv = 3514 * T_SD / (v_a_iram * D_iram**2)
	return Sv

#gao and solomon SFGs

gso_table = Table.read('../tables/gaosolomon04_tab3_Edit_tab4_edit.csv')
#this edited version removed galaxies also in the GOALs sample
#this also removed 1 starburst from the table, so the non lirgs should just be star-forming gals (NGC 1022)

gso_HCN = np.log10(gso_table['S_HCN_delV'])
gso_CO = np.log10(gso_table['S_CO_delV'])

frac_err_I_HCN = gso_table['sigma_HCN'] / gso_table['I_HCN']
frac_err_I_CO = gso_table['sigma_CO'] / gso_table['I_CO']

gso_e_HCN = frac_err_I_HCN/np.log(10)
gso_e_CO = frac_err_I_CO/np.log(10)

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

#crocker12 ETGs
#error of -1 indicates upper limit

etg_table = Table.read('../tables/crocker12_tab3.csv')

etg_HCN_Kkms = etg_table['HCN(1–0) (K km s−1)']
etg_e_HCN_Kkms = etg_table['e_HCN(1–0) (K km s−1)']
etg_HCN = np.log10(convert_T_iram(etg_HCN_Kkms))
etg_e_HCN = convert_T_iram(etg_e_HCN_Kkms)/(convert_T_iram(etg_HCN_Kkms)*np.log(10))

etg_CO_Kkms = etg_table['12CO(1–0) (K km s−1)']
etg_e_CO_Kkms = etg_table['e_12CO(1–0) (K km s−1)']
etg_CO = np.log10(convert_T_iram(etg_CO_Kkms))
etg_e_CO = convert_T_iram(etg_e_CO_Kkms)/(convert_T_iram(etg_CO_Kkms)*np.log(10))

etg_noulim = np.where(etg_table['e_HCN(1–0) (K km s−1)'] > 0)[0]
etg_ulim_HCN = np.where(etg_table['e_HCN(1–0) (K km s−1)'] < 0)[0]

#French22 psbs

psb_table = Table.read('../tables/French22_PSBs.csv')

psb_HCN = np.log10(psb_table['HCN(1-0)'])
psb_e_HCN = psb_table['e_HCN(1-0)']/(psb_table['HCN(1-0)']*np.log(10))
psb_HCN_ulim_col = psb_table['HCN_ulim']
psb_noulim = np.where(psb_HCN_ulim_col == 'N')[0]
psb_ulim_HCN = np.where(psb_HCN_ulim_col == 'Y')[0]

psb_CO = np.log10(psb_table['CO(1-0)'])
psb_e_CO = psb_table['e_CO(1-0)']/(psb_table['CO(1-0)']*np.log(10))


#GOALS galaxies
gls_table = Table.read('../tables/goals_HCN_CO_matched.csv')

gls_HCN_Kkms = gls_table['Sigma T _mb dv (HCN (1-0)) (K km s^-1)']
gls_HCN = np.log10(convert_T_iram(gls_HCN_Kkms))
e_gls_HCN_Kkms = gls_table['e_Sigma T _mb dv (HCN (1-0)) (K km s^-1)']
gls_e_HCN = convert_T_iram(e_gls_HCN_Kkms)/(convert_T_iram(gls_HCN_Kkms)*np.log(10))

gls_ulim_HCN = np.where(gls_e_HCN < 0)[0]
gls_noulim = np.where(gls_e_HCN > 0)[0]

gls_CO = np.log10(gls_table['S12COdeltav (Jy km/s)'])
gls_e_CO = gls_table['e_S12COdeltav (Jy km/s)']/(gls_table['S12COdeltav (Jy km/s)'] * np.log(10))

### NGC 1266

n1266_tab = Table.read('../tables/twocomp_fluxes_constr_r2.csv')

n1266_HCN = np.log10(n1266_tab['total_flux'][4])
e_n1266_HCN = n1266_tab['e_total_flux'][4]/(n1266_tab['total_flux'][4] * np.log(10))
n1266_CO = np.log10(n1266_tab['total_flux'][0])
e_n1266_CO = n1266_tab['e_total_flux'][0]/(n1266_tab['total_flux'][0] * np.log(10))

print(convert_T_iram(34.7))
print(convert_T_iram(2.83))


#krips 2010 collated lit table
krips_tab = Table.read('../tables/krips_2010.csv')


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

both_ind = np.intersect1d(CO_1312_ind, HCN_13CO_ind)




#### plotting

def CO_ratio_HCN_plot():
	### make plot of 13/12 CO ratio and HCN/13CO ratio
	# get the correct indices
	# convert the arrays to floats, then plot appropriately

	print(len(both_ind))

def HCN_CO_plot():

	fig = plt.figure(figsize=(8,8))

	plt.errorbar(n1266_HCN, n1266_CO, xerr=e_n1266_HCN, yerr=e_n1266_CO, marker='X', color='tab:green', mec='k', label='NGC 1266', markersize=10)

	plt.errorbar(sfg_HCN[sfg_noulim_CO_HCN], sfg_CO[sfg_noulim_CO_HCN], xerr=sfg_e_HCN[sfg_noulim_CO_HCN], yerr=sfg_e_CO[sfg_noulim_CO_HCN], marker='*', color='tab:blue', label='Star-forming', linestyle='')
	plt.errorbar(lirg_HCN[lirg_noulim_CO_HCN], lirg_CO[lirg_noulim_CO_HCN], xerr=lirg_e_HCN[lirg_noulim_CO_HCN], yerr=lirg_e_CO[lirg_noulim_CO_HCN], marker='D', color='tab:cyan', label='LIRG (Gao 04)', linestyle='')
	plt.errorbar(etg_HCN[etg_noulim], etg_CO[etg_noulim], xerr=etg_e_HCN[etg_noulim], yerr=etg_e_CO[etg_noulim], marker='o', color='tab:red', label='Early-type', linestyle='')
	plt.errorbar(psb_HCN[psb_noulim], psb_CO[psb_noulim], xerr=psb_e_HCN[psb_noulim], yerr=psb_e_CO[psb_noulim], marker='s', color='tab:purple', label='PSB', linestyle='')
	plt.errorbar(gls_HCN[gls_noulim], gls_CO[gls_noulim], xerr=gls_e_HCN[gls_noulim], yerr=gls_e_CO[gls_noulim],marker='p', color='tab:pink', label='LIRG (GOALS)', linestyle='')

	plt.plot(sfg_HCN[sfg_ulim_HCN], sfg_CO[sfg_ulim_HCN], marker='*', color='tab:blue', linestyle='')
	plt.plot(sfg_HCN[sfg_llim_HCN], sfg_CO[sfg_llim_HCN], marker='*', color='tab:blue', linestyle='')
	plt.plot(sfg_HCN[sfg_llim_CO], sfg_CO[sfg_llim_CO], marker='*', color='tab:blue', linestyle='')
	plt.plot(lirg_HCN[lirg_ulim_HCN], lirg_CO[lirg_ulim_HCN], marker='D', color='tab:cyan', linestyle='')
	plt.plot(lirg_HCN[lirg_llim_HCN], lirg_CO[lirg_llim_HCN], marker='D', color='tab:cyan', linestyle='')
	plt.plot(lirg_HCN[lirg_llim_CO], lirg_CO[lirg_llim_CO], marker='D', color='tab:cyan', linestyle='')

	plt.plot(etg_HCN[etg_ulim_HCN], etg_CO[etg_ulim_HCN], marker='o', color='tab:red', linestyle='')
	plt.plot(psb_HCN[psb_ulim_HCN], psb_CO[psb_ulim_HCN], marker='s', color='tab:purple', linestyle='')
	plt.plot(gls_HCN[gls_ulim_HCN], gls_CO[gls_ulim_HCN], marker='p', color='tab:pink', linestyle='')


	#upper limits
	for ind in sfg_ulim_HCN:
		plt.arrow(sfg_HCN[ind], sfg_CO[ind], dx=-0.1, dy=0, width=0.005, head_width=0.02, head_length=0.02, color='tab:blue')
	for ind2 in sfg_llim_HCN:
		plt.arrow(sfg_HCN[ind2], sfg_CO[ind2], dx=0.1, dy=0, width=0.005, head_width=0.02, head_length=0.02, color='tab:blue')
	for ind3 in sfg_llim_CO:
		plt.arrow(sfg_HCN[ind3], sfg_CO[ind3], dx=0, dy=0.1, width=0.005, head_width=0.02, head_length=0.02, color='tab:blue')
	for ind4 in etg_ulim_HCN:
		plt.arrow(etg_HCN[ind4], etg_CO[ind4], dx=-0.1, dy=0, width=0.005, head_width=0.02, head_length=0.02, color='tab:red')
	for ind5 in psb_ulim_HCN:
		plt.arrow(psb_HCN[ind5], psb_CO[ind5], dx=-0.1, dy=0, width=0.005, head_width=0.02, head_length=0.02, color='tab:purple')
	for ind6 in gls_ulim_HCN:
		plt.arrow(gls_HCN[ind6], gls_CO[ind6], dx=-0.1, dy=0, width=0.005, head_width=0.02, head_length=0.02, color='tab:pink')
	for ind7 in lirg_ulim_HCN:
		plt.arrow(lirg_HCN[ind7], lirg_CO[ind7], dx=-0.1, dy=0, width=0.005, head_width=0.02, head_length=0.02, color='tab:teal')
	for ind8, in lirg_llim_HCN:
		plt.arrow(lirg_HCN[ind8], lirg_CO[ind8], dx=0.1, dy=0, width=0.005, head_width=0.02, head_length=0.02, color='tab:teal')
	for ind9, in lirg_llim_CO:
		plt.arrow(lirg_HCN[ind9], lirg_CO[ind9], dx=0, dy=0.1, width=0.005, head_width=0.02, head_length=0.02, color='tab:teal')


	plt.xlabel('log HCN(1-0) Flux (Jy km/s)', fontsize=16)
	plt.ylabel('log CO(1-0) Flux (Jy km/s)', fontsize=16)

	#plt.loglog()

	plt.legend()

	plt.savefig('../plots/HCN_CO_comparison.png', dpi=300)

def HCN_CO_ratio_histogram():

	fig, (ax0,ax1) = plt.subplots(2, 1, figsize=(8,8), sharex=True, gridspec_kw={'hspace':0.03})

	ngc1266_ratio = n1266_HCN-n1266_CO
 
	sfg_ratio = sfg_HCN[sfg_noulim_CO_HCN]-sfg_CO[sfg_noulim_CO_HCN]
	sfg_ratio_ulim = sfg_HCN[np.concatenate((sfg_ulim_HCN, sfg_llim_CO))]-sfg_CO[np.concatenate((sfg_ulim_HCN, sfg_llim_CO))]
	sfg_ratio_llim = sfg_HCN[sfg_llim_HCN]-sfg_CO[sfg_llim_HCN]

	lirg_ratio = lirg_HCN[lirg_noulim_CO_HCN]-lirg_CO[lirg_noulim_CO_HCN]
	lirg_ratio_ulim = lirg_HCN[np.concatenate((lirg_ulim_HCN, lirg_llim_CO))]-lirg_CO[np.concatenate((lirg_ulim_HCN, lirg_llim_CO))]
	lirg_ratio_llim = lirg_HCN[lirg_llim_HCN]-lirg_CO[lirg_llim_HCN]

	etg_ratio = etg_HCN[etg_noulim]-etg_CO[etg_noulim]
	etg_ratio_ulim = etg_HCN[etg_ulim_HCN]-etg_CO[etg_ulim_HCN]

	psb_ratio = psb_HCN[psb_noulim]-psb_CO[psb_noulim]
	psb_ratio_ulim = psb_HCN[psb_ulim_HCN]-psb_CO[psb_ulim_HCN]

	gls_ratio = gls_HCN[gls_noulim]-gls_CO[gls_noulim]
	gls_ratio_ulim = gls_HCN[gls_ulim_HCN]-gls_CO[gls_ulim_HCN]

	all_ratio = np.concatenate((sfg_ratio, lirg_ratio, etg_ratio, psb_ratio, gls_ratio))
	all_hist, all_bins = np.histogram(all_ratio, bins='auto')

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
	
	ax0.axvline(ngc1266_ratio, color='tab:green', label='NGC 1266', linewidth=3)
	ax1.axvline(ngc1266_ratio, color='tab:green', label='NGC 1266', linewidth=3)

	ax1.set_xlabel('log HCN/CO ratio', fontsize=16)
	ax0.set_ylabel('Number', fontsize=16)
	ax1.set_ylabel('Number', fontsize=16)

	ax0.tick_params(axis='y', labelsize=14)
	ax0.tick_params(axis='x', labelsize=0)
	ax0.set_yticks(np.arange(0,17,2))
	ax1.tick_params(axis='both', labelsize=14)

	ax0.legend(fontsize=14)
	ax1.legend(fontsize=14)

	plt.savefig('../plots/HCN_CO_ratio_histogram.png')

#HCN_CO_ratio_histogram()



