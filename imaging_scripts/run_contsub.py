import os

fullvis = '/users/jotter/DATA/2011.0.00511.S/concat_vis_calib.ms'

#dict = spw : line free chans

spw_dict = {'0':'0:5~25, 1:45~63', '1':'0:5~25, 1:45~63', '2':'2:5~30;100~110',
            '3':'3:35~75, 4:65~95; 145~170', '4':'3:35~75, 4:65~95; 145~170', '5':'8:5~145',
            '6':'8:5~145', '7':'8:5~145', '8':'8:5~145',
            '9':'9:5~40; 85~105', '10':'9:5~40; 85~105'}

#old cont sub channels no longer in use
# '0':'0:5~25;47~54', '1':'1:19~25;45~63', '2':'2:5~40;95~115;150~160', '3':'3:40~80;145~185', '4':'4:5~22; 65~100; 145~180',
# '5':'5:5~35;90~140', '6':'6:5~40;100~140', '7':'7:5~30;100~140', '8':'8:5~50;90~140', '9':'9:5~40; 80~105','10':'10:72~97'}

#spw_dict =  {'9':['220.398GHz', 0.783, 7.344, '13CO21', 48], '5':['86.847GHz', 1.913, 67.495, 'SiO21', 180], '10':['217.105GHz', 0.783, 7.344, 'SiO54', 48],
#             '3':['260.518GHz', 0.585, 5.343, 'SiO65', 48], '2':['303.927GHz', 0.548, 5.147, 'SiO76', 48], '0':['347.330GHz', 0.440, 3.829, 'SiO87', 48],
#             '1':['344.518GHz', 0.440, 3.829, 'SiO87', 48], '4':['256.6GHz', 0.585, 5.343, '', 48], '6':['87.998GHz', 1.913, 67.495, 'HCN10', 180],
#             '7':['86.134GHz', 1.913, 67.495, 'H13CO+10', 180], '8':['84.553GHz', 1.913, 67.495, 'HC18O+10', 180]}

for spw in spw_dict.keys():

    fitspec = spw#spw+':'+spw_dict[spw]
    outputvis = '/users/jotter/DATA/2011.0.00511.S/spw'+spw+'_uvcontsub_vis_6-10.ms'

    if os.path.exists(outputvis) == True:
        continue
    
    uvcontsub(vis=fullvis, outputvis=outputvis, spw=spw, fitspec=fitspec, fitorder=0)
