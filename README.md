# Code for EMcapsulins - "Genetically encoded barcodes for correlative volume electron microscopy"

This repository contains code for the machine learning part of the paper "Genetically encoded barcodes for correlative volume electron microscopy" (TODO: Put a reference here when it's published).



## Installation

- Get the project from GitHub (either clone or download zip and extract), then `cd` to the project root.

- From the project directory execute:

    ```
    conda env create -f environment.yml
    conda activate emcaps
    pip install --no-deps -e .
    ```


## Running the code

All scripts can be executed from the project root directory using `python3 -m`, for example:

    $ python3 -m emcaps.inference.segment -h

Alternatively you can use the entry points provided by the pip installation:

    $ emcaps-segment -h


## Entry points for testing on custom data

These entry points just require raw images and don't require GPU resources. Labels are not needed.

### Napari-based interactive GUI tool for segmentation and EMcapsulin particle classification

    $ emcaps-encari

or

    $ python3 -m emcaps.analysis.encari

### Performing batch inference on a directory of images or single image files

    $ emcaps-segment segment.inp_path=<PATH_TO_FILE_OR_FOLDER>

or

    $ python3 -m emcaps.inference.segment segment.inp_path=<PATH_TO_FILE_OR_FOLDER>


## Entry points for reproduction, retraining or evaluation

The following steps require a local copy of the [official dataset](#dataset) or a dataset in the same structure. A GPU is highly recommended.


### Splitting labeled image dataset into training and validation images and normalizing the data format

    $ python3 -m emcaps.utils.splitdataset

### Training new segmentation models

    $ emcaps-segtrain

or

    $ python3 -m emcaps.training.segtrain

### Segmentation inference and evaluation

Segment and optionally also perform particle-level classification if a model is available, render output visualizations (colored classification overlays etc.) and compute segmentation metrics.

    $ emcaps-segment

or

    $ python3 -m emcaps.inference.segment

### Producing a patch dataset based on image segmentation

Based on segmentation (from a model or human annotation), extract particle-centered image patches and store them as separate files in addition to metadata. The resulting patch dataset can be used for training models for patch-based classification. In addition, A random sample of the validation patches is prepared for evaluation of human and model-based classification evaluation.

    $ python3 -m emcaps.inference.patchifyseg

### Training new patch classifiers

Requires the outputs of `patchifyseg` (see above).

    $ python3 -m emcaps.training.patchtrain

### Quantitative evaluation of patch classification results

Requires the outputs of `patchifyseg` (see above).

    $ python3 -m emcaps.inference.patchmodel_eval


## Configuration system

We are using a common configuration system for the runnable code, based on [Hydra](https://hydra.cc/docs/1.2/intro/) and [OmegaConf](https://omegaconf.readthedocs.io/en/2.2_branch/).
A central default config file with explanatory comments is located at `conf/conf.yaml`.
It is written to be as automatic and minimal as possible, but it can still be necessary to change some of the values for experiments or adapting to a different system.

For the syntax of such yaml-based config files please refer to the OmegaConf docs on [Access and manipulation](https://omegaconf.readthedocs.io/en/2.2_branch/usage.html#access-and-manipulation) and [variable interpolation](https://omegaconf.readthedocs.io/en/2.2_branch/usage.html#variable-interpolation)

For running hydra-enabled code with custom configuration you can either point to a different config file with the `-cp` [CLI flag](https://hydra.cc/docs/1.2/advanced/hydra-command-line-flags/) or change config values directly on the CLI using [Hydra's override syntax](https://hydra.cc/docs/1.2/advanced/override_grammar/basic/)


### Coverage

Currently the common configuration system is implemented for

- `emcaps.utils.splitdataset`
- `emcaps.training.segtrain`
- `emcaps.inference.segment`

It is not yet fully implemented (TODO) for

- `emcaps.inference.patchifyseg`
- `emcaps.training.patchtrain`
- `emcaps.inference.patchmodel_eval`


## Dataset

If you want to train own models and/or do quantitative evaluation on the official data, please find the data [here](https://TODO) (TODO: Insert official data link here)



## Further notes

For more details see top-level docstrings in each file.
