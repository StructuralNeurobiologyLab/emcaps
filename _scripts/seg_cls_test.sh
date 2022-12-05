#!/bin/bash
set -Eeuo pipefail

# Segment and classify new test images without ground truth / human labels (only qualitative results)

if [[ $# -ne 1 ]] ; then
    echo 'Please pass one input path (image or flat directory with images)'
    exit 1
fi

# Config settings, comma-separated
TR_GROUP=all,all2,all3,hek,hek2,dro,mice

emcaps-segment --multirun tr_group=${TR_GROUP} segment.inp_path=$1
