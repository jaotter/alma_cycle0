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

sfg_table = Table.read('../tables/gaosolomon04_tab3_Edit.csv')

sfg_HCN = np.log10(sfg_table['S_HCN_delV'])
sfg_CO = np.log10(sfg_table['S_CO_delV'])

frac_err_I_HCN = sfg_table['sigma_HCN'] / sfg_table['I_HCN']
frac_err_I_CO = sfg_table['sigma_CO'] / sfg_table['I_CO']

e_sfg_HCN = frac_err_I_HCN/np.log(10)
e_sfg_CO = frac_err_I_CO/np.log(10)

sfg_noulim_CO_HCN = np.intersect1d(np.where(sfg_table['CO_lim'] == 'N')[0], np.where(sfg_table['HCN_lim'] == 'N')[0])

sfg_ulim_HCN = np.where(sfg_table['HCN_lim'] == 'upper')[0]
sfg_llim_HCN = np.where(sfg_table['HCN_lim'] == 'lower')[0]
sfg_llim_CO = np.where(sfg_table['CO_lim'] == 'lower')[0]

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
e_gls_HCN = convert_T_iram(e_gls_HCN_Kkms)/(convert_T_iram(gls_HCN_Kkms)*np.log(10))

gls_ulim_HCN = np.where(e_gls_HCN < 0)[0]
gls_noulim = np.where(e_gls_HCN > 0)[0]

gls_CO = np.log10(gls_table['S12COdeltav (Jy km/s)'])



#### plotting

fig = plt.figure(figsize=(8,8))

plt.plot(sfg_HCN[sfg_noulim_CO_HCN], sfg_CO[sfg_noulim_CO_HCN], marker='*', color='tab:blue', label='Star-forming', linestyle='')
plt.plot(etg_HCN[etg_noulim], sfg_CO[etg_noulim], marker='o', color='tab:red', label='Early-type', linestyle='')
plt.plot(psb_HCN[psb_noulim], psb_CO[psb_noulim], marker='s', color='tab:purple', label='PSB', linestyle='')
plt.plot(gls_HCN[gls_noulim], gls_CO[gls_noulim], marker='p', color='tab:pink', label='LIRGs', linestyle='')

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


plt.xlabel('log HCN(1-0) Flux (Jy km/s)')
plt.ylabel('log CO(1-0) Flux (Jy km/s)')

#plt.loglog()

plt.legend()

plt.savefig('../plots/HCN_CO_comparison.png', dpi=300)