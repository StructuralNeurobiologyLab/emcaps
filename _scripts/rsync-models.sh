#!/bin/bash
set -Eeuo pipefail

# Original sources -> short names of published model checkpoints

mkdir -p ~/emc/v15/ptsmodels

# Segmentation models:

rsync -a cajal:/cajal/scratch/projects/misc/mdraw/emc/v15/seg_trainings/seg_trainings_v15_tr-all/tr-all___UNet__22-11-05_20-46-58/model_step300000.pts ~/emc/v15/ptsmodels/unet_all_v15.pts
rsync -a cajal:/cajal/scratch/projects/misc/mdraw/emc/v15/seg_trainings/seg_trainings_v15_tr-all_2/tr-all_2___UNet__22-11-05_20-47-00/model_step300000.pts ~/emc/v15/ptsmodels/unet_all2_v15.pts
rsync -a cajal:/cajal/scratch/projects/misc/mdraw/emc/v15/seg_trainings/seg_trainings_v15_tr-all_3/tr-all_3___UNet__22-11-05_20-47-02/model_step300000.pts ~/emc/v15/ptsmodels/unet_all3_v15.pts

rsync -a cajal:/cajal/scratch/projects/misc/mdraw/emc/v15/seg_trainings/seg_trainings_v15_tr-dro/tr-dro___UNet__22-11-05_20-47-03/model_step300000.pts ~/emc/v15/ptsmodels/unet_dro_v15.pts

rsync -a cajal:/cajal/scratch/projects/misc/mdraw/emc/v15/seg_trainings/seg_trainings_v15_tr-hek/tr-hek___UNet__22-11-05_20-47-04/model_step300000.pts ~/emc/v15/ptsmodels/unet_hek_v15.pts
rsync -a cajal:/cajal/scratch/projects/misc/mdraw/emc/v15/seg_trainings/seg_trainings_v15_tr-hek_2/tr-hek_2___UNet__22-11-05_20-47-06/model_step300000.pts ~/emc/v15/ptsmodels/unet_hek2_v15.pts

rsync -a cajal:/cajal/scratch/projects/misc/mdraw/emc/v15/seg_trainings/seg_trainings_v15_tr-mice/tr-mice___UNet__22-11-05_20-47-07/model_step300000.pts ~/emc/v15/ptsmodels/unet_mice_v15.pts


# Classification models:

rsync -a cajal:/cajal/scratch/projects/misc/mdraw/emc/v15/patch_trainings/patch_trainings_v15_tr-all/tr-all___EffNetV2__22-11-23_12-19-45/model_step120000.pts ~/emc/v15/ptsmodels/effnet_all_v15.pts
rsync -a cajal:/cajal/scratch/projects/misc/mdraw/emc/v15/patch_trainings/patch_trainings_v15_tr-all2/tr-all2___EffNetV2__22-11-23_12-22-30/model_step120000.pts ~/emc/v15/ptsmodels/effnet_all2_v15.pts
rsync -a cajal:/cajal/scratch/projects/misc/mdraw/emc/v15/patch_trainings/patch_trainings_v15_tr-all3/tr-all3___EffNetV2__22-11-23_12-24-11/model_step120000.pts ~/emc/v15/ptsmodels/effnet_all3_v15.pts

rsync -a cajal:/cajal/scratch/projects/misc/mdraw/emc/v15/patch_trainings/patch_trainings_v15_tr-hek/tr-hek___EffNetV2__22-11-23_12-24-16/model_step120000.pts ~/emc/v15/ptsmodels/effnet_hek_v15.pts
rsync -a cajal:/cajal/scratch/projects/misc/mdraw/emc/v15/patch_trainings/patch_trainings_v15_tr-hek2/tr-hek2___EffNetV2__22-11-23_12-23-49/model_step120000.pts ~/emc/v15/ptsmodels/effnet_hek2_v15.pts
