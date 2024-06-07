import os

fullvis = '/users/jotter/DATA/2011.0.00511.S/concat_vis_calib.ms'

#rest freq, angular resolution ("), fov ("), spw, efficient imsize
#line_dict = {'13CO21':['220.398GHz', 0.783, 7.344, '9', 48], 'SiO21':['86.847GHz', 1.913, 67.495, '5', 180], 'SiO54':['217.105GHz', 0.783, 7.344, '10', 48],
#             'SiO65':['260.518GHz', 0.585, 5.343, '3', 48], 'SiO76':['303.927GHz', 0.548, 5.147, '2', 48], 'SiO87':['347.330GHz', 0.440, 3.829, '0', 48]}

spw_dict =  {'9':['220.398GHz', 0.783, 26.810, '13CO21', 180], '5':['86.847GHz', 1.913, 67.495, 'SiO21', 180], '10':['217.105GHz', 0.783, 26.810, 'SiO54', 180],
             '3':['260.518GHz', 0.585, 22.603, 'SiO65', 180], '2':['303.927GHz', 0.548, 19.298, 'SiO76', 180], '0':['347.330GHz', 0.440, 16.894, 'SiO87', 180],
             '1':['344.518GHz', 0.440, 16.894, 'SiO87', 180], '4':['256.6GHz', 0.585, 22.603, '', 180], '6':['87.998GHz', 1.913, 67.495, 'HCN10', 180],
             '7':['86.134GHz', 1.913, 67.495, 'H13CO+10', 180], '8':['84.553GHz', 1.913, 67.495, 'HC18O+10', 180],
             '0,1':['344.518GHz', 0.440, 16.894, 'SiO87', 180], '5,7':['86.847GHz', 1.913, 67.495, 'SiO21', 180]}

#line_name = 'SiO21'
#line_name = 'SiO54'
#line_name = '13CO21'
#line_name = 'SiO65'
#line_name = 'SiO76'
#line_name = 'SiO87'

contsub=True

clean_spw_list = ['0,1', '5,7']

for spw in clean_spw_list:
    print('CREATING DIRTY CUBE FOR SPW '+spw)

    rest_freq, ang_res, fov, line_name, npix = spw_dict[spw]

    if contsub == True:
        contvis = '/users/jotter/DATA/2011.0.00511.S/spw'+spw+'_uvcontsub_vis_calib.ms'
        fullvis = contvis
        if len(spw.split(',')) > 1:
            spw1, spw2 = spw.split(',')
            contvis1 = '/users/jotter/DATA/2011.0.00511.S/spw'+spw1+'_uvcontsub_vis_calib.ms'
            contvis2 = '/users/jotter/DATA/2011.0.00511.S/spw'+spw2+'_uvcontsub_vis_calib.ms'
            fullvis = [contvis1,contvis2]
        print('USING CONTINUUM SUBTRACTED MS')
        print(fullvis)
    
    ## setting up tclean parameters for dirty cube
    robust_value = 0.5
    restoringbeam = 'common'
    theoutframe = 'LSRK'
    theveltype = 'radio'
    chanchunks = -1
    niter = 0

    pixsize = ang_res / 5
    thecellsize = str(pixsize)+'arcsec'
    if npix == 1:
        npix = int(fov / pixsize)
    theimsize=[npix, npix]
    velwidth='11km/s'

    if contsub == True:
        imgname = '/users/jotter/alma_cycle0/dirty_cubes/N1266_spw'+spw+'_contsub_dirty'
        fitsname = '/users/jotter/alma_cycle0/fitsimages/N1266_spw'+spw+'_contsub_dirty'
    else:
        imgname = '/users/jotter/alma_cycle0/dirty_cubes/N1266_spw'+spw+'_dirty'
        fitsname = '/users/jotter/alma_cycle0/fitsimages/N1266_spw'+spw+'_dirty'
        

    tclean(vis=fullvis,
           imagename=imgname,
           field='NGC1266',
           spw=spw,
           restfreq=rest_freq,
           threshold='1Jy', # high threshold for dirty cube  
           imsize=theimsize,  
           cell=thecellsize,
           outframe=theoutframe,
           restoringbeam=restoringbeam,   
           specmode='cube',
           gridder='standard',
           width=velwidth,
           veltype=theveltype,
           niter=0, # 0 for dirty cube
           pbcor=False,
           deconvolver='hogbom',
           weighting = 'briggs', #'briggs' or 'natural' or 'uniform' 
           robust = robust_value,
           #mask = cleanmask,
           interactive=False)

    exportfits(imgname+'.image', fitsname+'.fits')
