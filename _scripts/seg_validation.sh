#!/bin/bash
set -Eeuo pipefail

# Segment validation data and calculate seg metrics. Also generate qualitative classification results.

# Sweeps over different configurations sequentially. Can be parallelized using a hydra launcher
# depending on your system (see https://hydra.cc/docs/plugins/submitit_launcher/).

# Config settings, comma-separated
TR_GROUP=all,all2,all3,hek,hek2

# Config settings, comma-separated
TR_GROUP=all,all2,all3,hek,hek2,dro,mice

emcaps-segment --multirun \
    tr_group=${TR_GROUP} \
