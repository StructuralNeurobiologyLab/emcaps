# Experiments on encapsulin analysis with data from TUM


## Installation

- Get the project from GitHub (either clone or download zip and extract), then `cd` to the project root.

- Download the neural network models from https://drive.google.com/file/d/1I7gGaIsYQtlm-_0euD85XRQLghi234AN/view and put them into the project root.

- From the project directory execute:

    ```
    conda env create -f environment.yml
    conda activate emcaps
    pip install --no-deps -e .
    ```


## Execution

All scripts should be executed from the current directory using `python3 -m`, for example:

    $ python3 -m emcaps.inference.segment



## Most important entry points

### Step 1

- `emcaps.training.segtrain`: Train 2D segmentation models.

### Step 2

- `emcaps.inference.segment`: Perform 2D segmentation model batch inference, visualize and evaluate results.
- `emcaps.inference.patchifyseg`: Based on segmentation (from NN and/or GT), extract particle-centered image patches and store them as separate files in addition to metadata. The resulting patch dataset can be used for training models for patch-based classification. In addition, A random sample of the validation patches is prepared for evaluation of human and NN-based classification evaluation.

### Step 3

- `emcaps.training.patchtrain`: Train patch classifier based on patchifyseg outputs.

### Step 4

- `emcaps.inference.patchmodel_eval`: Evaluate patch classification results.


## Dataset

TODO


## Further notes

For more details see top-level docstrings in each file.
