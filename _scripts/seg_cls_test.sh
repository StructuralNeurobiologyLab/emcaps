#!/bin/bash
set -Eeuo pipefail

# Segment and classify new test images without ground truth / human labels (only qualitative results)

if [[ $# -ne 1 ]] ; then
    echo 'Please pass one input path.'
    exit 1
fi

for g in all all2 all3 hek hek2 dro mice
do
    emcaps-segment tr_group=$g segment.inp_path=$1
done
