# Experiments on encapsulin analysis with data from TUM


## Installation

```
conda env create -f environment.yml
conda activate emcaps
# TODO: Next steps

```


## Execution

All scripts should be executed from the current directory using `python3 -m`, for example:

    $ python3 -m emcaps.training.segtrain



## Most important entry points

### Step 1

- `training.segtrain`: Train 2D segmentation models.

### Step 2

- `inference.inference`: Perform 2D segmentation model batch inference, visualize and evaluate results.
- `inference.patchifyseg`: Based on segmentation (from NN and/or GT), extract particle-centered image patches and store them as separate files in addition to metadata. The resulting patch dataset can be used for training models for patch-based classification. In addition, A random sample of the validation patches is prepared for evaluation of human and NN-based classification evaluation.

### Step 3

- `training.patchtrain`: Train patch classifier based on patchifyseg outputs.

### Step 4

- `inference.patchmodel_eval`: Evaluate patch classification results.


## Dataset

TODO


## Further notes

For more details see top-level docstrings in each file.
