#script to combine and concat spws and ms's

vis1 = '../../DATA/2011.0.00511.S/sg_ouss_id/group_ouss_id/member_ouss_2012-10-18_id/calibrated/uid___A002_X4b1170_X8bf.ms.split.cal'
vis2 = '../../DATA/2011.0.00511.S/sg_ouss_id/group_ouss_id/member_ouss_2012-10-19_id/calibrated/uid___A002_X4ae797_X8e2.ms.split.cal'
vis3 = '../../DATA/2011.0.00511.S/sg_ouss_id/group_ouss_id/member_ouss_2012-12-10_id/calibrated/uid___A002_X518d2b_X571.ms.split.cal'
vis4 = '../../DATA/2011.0.00511.S/sg_ouss_id/group_ouss_id/member_ouss_2012-12-10_id/calibrated/uid___A002_X51ac2a_X851.ms.split.cal'
vis5 = '../../DATA/2011.0.00511.S/sg_ouss_id/group_ouss_id/member_ouss_2013-01-03_id/calibrated/uid___A002_X436934_X48c.ms.split.cal'
vis6 = '../../DATA/2011.0.00511.S/sg_ouss_id/group_ouss_id/member_ouss_2013-01-03_id/calibrated/uid___A002_X4785e0_Xab0.ms.split.cal'
vis7 = '../../DATA/2011.0.00511.S/sg_ouss_id/group_ouss_id/member_ouss_id/calibrated/uid___A002_X4b29af_X818.ms.split.cal'

#goals:
#vis1 - no changes
#vis2 - no changes
#vis3 - concat with vis4
#vis5 - combine spw0,1 and concat with vis6
#vis7 - combine spw0,2 - [do this later]

#vis5_combined = '/users/jotter/DATA/2011.0.00511.S/vis5_combined.ms'
#mstransform(vis5, outputvis=vis5_combined, combinespws=True, spw='0,1')
#vis6_combined = '/users/jotter/DATA/2011.0.00511.S/vis6_combined.ms'
#mstransform(vis6, outputvis=vis6_combined, combinespws=True, spw='0,1')
vis7_combined_02 = '/users/jotter/DATA/2011.0.00511.S/vis7_combined_02.ms'
mstransform(vis7, outputvis=vis7_combined_02, combinespws=True, spw='0,2')

b6_vis = '/users/jotter/DATA/2011.0.00511.S/Band6_vis_combined.ms'
concat(vis=[vis1,vis3,vis4], outputvis=b6_vis, freqtol='1MHz')

b7_vis = '/users/jotter/DATA/2011.0.00511.S/Band7_vis_combined.ms'
concat(vis=[vis2,vis5_combined, vis6_combined], outputvis=b7_vis, freqtol='1MHz')
