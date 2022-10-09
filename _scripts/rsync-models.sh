#!/bin/bash

rsync -a wb01:/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10b/GA_all_dec98__UNet__22-10-05_04-22-48/model_step240000.pts ~/tum/ptsmodels/v10c/unet_v10c_all_240k.pts

# TODO
# rsync -a wb01:/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10c/GA_hek__UNet__22-09-24_03-29-06/model_step160000.pts ~/tum/ptsmodels/v10c/unet_v10_hek_160k.pts
# rsync -a wb01:/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10c/GA_dro__UNet__22-09-24_03-26-07/model_step160000.pts ~/tum/ptsmodels/v10c/unet_v10_dro_160k.pts
# rsync -a wb01:/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10/MICE_2M-Qt_GA_mice__UNet__22-09-24_03-26-43/model_step240000.pts ~/tum/ptsmodels/v10c/unet_v10_mice_240k.pts
# rsync -a wb01:/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10c/GA_qttm__UNet__22-09-24_03-27-16/model_step240000.pts ~/tum/ptsmodels/v10c/unet_v10_qttm_240k.pts
# rsync -a wb01:/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10_onlytm/GA_onlytm__UNet__22-09-24_03-32-39/model_step40000.pts ~/tum/ptsmodels/v10c/unet_v10_onlytm_160k.pts
# rsync -a wb01:/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10_notm/GA_notm_all__UNet__22-09-24_03-33-19/model_step200000.pts ~/tum/ptsmodels/v10c/unet_v10_all_notm_200k.pts
# rsync -a wb01:/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v10_notm/GA_notm_hek__UNet__22-09-24_03-34-32/model_step160000.pts ~/tum/ptsmodels/v10c/unet_v10_hek_notm_160k.pts
