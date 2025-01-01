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
#this INCLUDES 2 starbursts in the star-forming sample

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

#### plotting

fig = plt.figure(figsize=(8,8))

plt.errorbar(n1266_HCN, n1266_CO, xerr=e_n1266_HCN, yerr=e_n1266_CO, marker='X', color='tab:green', mec='k', label='NGC 1266', markersize=10)

plt.errorbar(sfg_HCN[sfg_noulim_CO_HCN], sfg_CO[sfg_noulim_CO_HCN], xerr=sfg_e_HCN[sfg_noulim_CO_HCN], yerr=sfg_e_CO[sfg_noulim_CO_HCN], marker='*', color='tab:blue', label='Star-forming', linestyle='')
plt.errorbar(lirg_HCN[lirg_noulim_CO_HCN], lirg_CO[lirg_noulim_CO_HCN], xerr=lirg_e_HCN[lirg_noulim_CO_HCN], yerr=lirg_e_CO[lirg_noulim_CO_HCN], marker='D', color='tab:cyan', label='LIRGs (Gao 04)', linestyle='')
plt.errorbar(etg_HCN[etg_noulim], etg_CO[etg_noulim], xerr=etg_e_HCN[etg_noulim], yerr=etg_e_CO[etg_noulim], marker='o', color='tab:red', label='Early-type', linestyle='')
plt.errorbar(psb_HCN[psb_noulim], psb_CO[psb_noulim], xerr=psb_e_HCN[psb_noulim], yerr=psb_e_CO[psb_noulim], marker='s', color='tab:purple', label='PSB', linestyle='')
plt.errorbar(gls_HCN[gls_noulim], gls_CO[gls_noulim], xerr=gls_e_HCN[gls_noulim], yerr=gls_e_CO[gls_noulim],marker='p', color='tab:pink', label='LIRGs (GOALS)', linestyle='')

#upper limits
for ind in sfg_ulim_HCN:
	plt.arrow(sfg_HCN[ind], sfg_CO[ind], dx=-0.1, dy=0, width=0.005, head_width=0.02, head_length=0.02, color='tab:blue')
for ind2 in sfg_llim_HCN:
	plt.arrow(sfg_HCN[ind2], sfg_CO[ind2], dx=0.1, dy=0, width=0.005, head_width=0.02, head_length=0.02, color='tab:blue')
for ind3 in sfg_llim_CO:
	plt.arrow(sfg_HCN[ind3], sfg_CO[ind3], dx=0, dy=0.1, width=0.005, head_width=0.02, head_length=0.02, color='tab:blue')
for ind4 in etg_ulim_HCN:
	plt.arrow(etg_HCN[ind4], sfg_CO[ind4], dx=-0.1, dy=0, width=0.005, head_width=0.02, head_length=0.02, color='tab:red')
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


plt.xlabel('log HCN(1-0) Flux (Jy km/s)')
plt.ylabel('log CO(1-0) Flux (Jy km/s)')

#plt.loglog()

plt.legend()

plt.savefig('../plots/HCN_CO_comparison.png', dpi=300)