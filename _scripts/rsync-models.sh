#!/bin/bash

rsync -a wb01:/wholebrain/scratch/mdraw/tum/mxqtsegtrain2_trainings_v13/GA_lrdec99__UNet__22-10-15_20-29-15/model_step240000.pts ~/tum/ptsmodels/v13/unet_v13_all_240k.pts
rsync -a cajal:/cajal/nvmescratch/users/mdraw/tum/patch_trainings_v14_dr5__t100/erasemaskbg___EffNetV2__22-10-21_02-44-53/model_step120000.pts ~/tum/ptsmodels/v14/effnet_m_v14_all_120k.pts
