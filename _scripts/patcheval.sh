#!/bin/bash
set -Eeuo pipefail

# set +o braceexpand -o noglob +o histexpand

# Perform quantitative classification evaluation.

# Sweeps over different configurations sequentially. Can be parallelized using a hydra launcher
# depending on your system (see https://hydra.cc/docs/plugins/submitit_launcher/).

# Config settings, comma-separated
TR_GROUP=all,all2,all3,hek,hek2
MAX_SAMPLES=1,3,7
# First entry is the explicit default (unconstrained setting, others are constraint settings of interest)
CONSTRAIN_ARGS='[1M-Qt, 2M-Qt, 3M-Qt, 1M-Mx, 2M-Mx, 1M-Tm]','[1M-Qt, 1M-Mx]','[3M-Qt, 1M-Mx]','[1M-Qt, 3M-Qt, 1M-Mx]'
# Nested comma-separated lists require special quoting syntax below because of conflicts between hydra and bash syntax parsing

emcaps-patcheval --multirun \
    patcheval.use_constraint_suffix=true \
    patcheval.rdraws=1000 \
    tr_group=${TR_GROUP} \
    patcheval.max_samples=${MAX_SAMPLES} \
    "patcheval.constrain_classifier=${CONSTRAIN_ARGS}"
