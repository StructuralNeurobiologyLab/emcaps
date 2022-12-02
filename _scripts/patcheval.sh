#!/bin/bash
set -Eeuo pipefail

# set +o braceexpand -o noglob +o histexpand

# Perform quantitative classification evaluation

# Note: Please set constrain_args below to the desired constraint settings manually.
# (Passing them dynamically is difficult due to hydra/bash syntax conflicts. It's possible but even uglier than what you will see here)

# TODO: Reimplement this test matrix using hydra's hyperparam sweep syntax (not yet tested here) instead of a nested bash for loop
#       --> https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

constrain_args='patcheval.constrain_classifier=[1M-Qt, 1M-Mx]'
# constrain_args='patcheval.constrain_classifier=[3M-Qt, 1M-Mx]'
# constrain_args='patcheval.constrain_classifier=[1M-Qt, 3M-Qt, 1M-Mx]'
# constrain_args='patcheval.constrain_classifier=[1M-Qt, 2M-Qt, 3M-Qt, 1M-Mx, 2M-Mx, 1M-Tm]'


for g in all all2 all3 hek hek2
do
    for n in 1 3 7
    do
        emcaps-patcheval patcheval.use_constraint_suffix=true \
                         patcheval.rdraws=1000 \
                         tr_group=$g \
                         patcheval.max_samples=$n \
                         "$constrain_args"
    done
done
