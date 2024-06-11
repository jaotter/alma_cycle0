import os

fullvis = '/users/jotter/DATA/2011.0.00511.S/concat_vis_calib.ms'

#rest freq, angular resolution ("), spw, efficient imsize, flux threshold, clean mask radius
#line_dict = {'13CO21':['220.398GHz', 0.783, 7.344, '9', 48], 'SiO21':['86.847GHz', 1.913, 67.495, '5', 180], 'SiO54':['217.105GHz', 0.783, 7.344, '10', 48],
#             'SiO65':['260.518GHz', 0.585, 5.343, '3', 48], 'SiO76':['303.927GHz', 0.548, 5.147, '2', 48], 'SiO87':['347.330GHz', 0.440, 3.829, '0', 48]}

spw_dict =  {'9':['220.398GHz', 0.783, '13CO21', 180, '0.8mJy', None], '5':['86.847GHz', 1.913, 'SiO21', 180, '1.5mJy', 14],
             '10':['217.105GHz', 0.783, 'SiO54', 180, '0.8mJy', 10], '3':['260.518GHz', 0.585, 'SiO65', 180, '1.2mJy', 12],
             '2':['303.927GHz', 0.548, 'SiO76', 180, '1.2mJy', 12], '0':['347.330GHz', 0.440, 'SiO87', 180, '2mJy', 10],
             '1':['344.518GHz', 0.440, 'H13CO+(4-3)', 180, '2mJy', 9], '4':['256.6GHz', 0.585, '', 180, '1mJy', 12],
             '6':['87.998GHz', 1.913, 'HCN10', 180, '1.5mJy', 14],
             '7':['86.134GHz', 1.913, 'H13CO+10', 180, '1.3mJy', 14], '8':['84.553GHz', 1.913, 'HC18O+10', 180, '1.5mJy', 14],
             '0,1':['347.330GHz', 0.440, 'SiO87', 180, '1.8mJy', 10], '5,7':['86.847GHz', 1.913, 'SiO21', 180, '1.4mJy', 14]}


spw_list = ['0,1', '5,7', '2', '3', '4', '6', '8', '9', '10']
contsub = True


for spw in spw_list:
    print('CREATING CLEAN CLUBE FOR SPW '+spw)
    
    if contsub == True:
        contvis = '/users/jotter/DATA/2011.0.00511.S/spw'+spw+'_uvcontsub_vis_6-10.ms'
        fullvis = contvis
        if len(spw.split(',')) > 1:
            spw1, spw2 = spw.split(',')
            contvis1 = '/users/jotter/DATA/2011.0.00511.S/spw'+spw1+'_uvcontsub_vis_6-10.ms'
            contvis2 = '/users/jotter/DATA/2011.0.00511.S/spw'+spw2+'_uvcontsub_vis_6-10.ms'
            fullvis = [contvis1, contvis2]
        print('USING CONTINUUM SUBTRACTED MS')
        print(fullvis)

        
    
    rest_freq, ang_res, line_name, npix, flux_thresh, mask_rad = spw_dict[spw]

    ## setting up tclean parameters for clean cube
    robust_value = 0.5
    restoringbeam = 'common'
    theoutframe = 'LSRK'
    theveltype = 'radio'
    chanchunks = -1
    niter = 10000

    pixsize = ang_res / 5
    thecellsize = str(pixsize)+'arcsec'
    theimsize=[npix, npix]
    velwidth='11km/s'
    #threshold should be rms of dirty cube

    xpos=str(theimsize[0]/2)
    ypos=str(theimsize[1]/2)
    radius = str(mask_rad)
    if mask_rad == None:
        cleanmask = ''
    else:
        cleanmask = 'circle[['+xpos+'pix,'+ypos+'pix], '+radius+'pix]'

    if contsub == True:
        imgname = '/users/jotter/DATA/cube_reduction/contsub_clean_6-10/N1266_spw'+spw+'_r'+str(robust_value)+'_'+flux_thresh+'_contsub_6-10'
        fitsname = '/users/jotter/alma_cycle0/fitsimages/N1266_spw'+spw+'_r'+str(robust_value)+'_'+flux_thresh+'_contsub_6-10'
    else:
        imgname = '/users/jotter/DATA/cube_reduction/nocontsub_clean/N1266_spw'+spw+'_r'+str(robust_value)+'_'+flux_thresh
        fitsname = '/users/jotter/alma_cycle0/fitsimages/N1266_spw'+spw+'_r'+str(robust_value)+'_'+flux_thresh

    tclean(vis=fullvis,
               imagename=imgname,
               field='NGC1266',
               spw=spw,
               restfreq=rest_freq,
               threshold=flux_thresh,  
               imsize=theimsize,  
               cell=thecellsize,
               outframe=theoutframe,
               restoringbeam=restoringbeam,   
               specmode='cube',
               gridder='standard',
               width=velwidth,
               veltype=theveltype,
               niter=10000, #high number 
               pbcor=False,
               deconvolver='hogbom',
               weighting = 'briggs', #'briggs' or 'natural' or 'uniform' 
               robust = robust_value,
               mask = cleanmask,
               interactive=False)


    exportfits(imgname+'.image', fitsname+'.fits', overwrite=True)


    impbcor(imagename=imgname+'.image',
                pbimage=imgname+'.pb',
                outfile=imgname+'.pbcor.image',
                overwrite=True)

    exportfits(imgname+'.pbcor.image', fitsname+'.pbcor.fits', overwrite=True)
