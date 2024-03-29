[![PyPI](https://img.shields.io/pypi/v/emcaps)](https://pypi.org/project/emcaps/) [![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/StructuralNeurobiologyLab/emcaps/python-publish.yml)](https://github.com/StructuralNeurobiologyLab/emcaps/actions)

# Source code for "Genetically encoded barcodes for correlative volume electron microscopy"

This repository contains all code for the sequential ML pipeline of the paper [Genetically encoded barcodes for correlative volume electron microscopy](https://www.nature.com/articles/s41587-023-01713-y).

Pretrained models can be found as release artifacts [here](https://github.com/StructuralNeurobiologyLab/emcaps/releases/tag/models). They are automatically downloaded and cached when needed by the code.

![Screenshot of the interactive GUI tool for segmentation and EMcapsulin particle classification (encari).](encari-screenshot.jpg)


## Installation

### Option 1: From PyPI (with `pip`)

You can install the project with scripts and all dependencies by running

    pip install emcaps


Notes:
- It is recommended to use a new [virtual environment](https://virtualenv.pypa.io/en/latest/) for this.

- If you encounter PyTorch issues with this setup, please [install PyTorch manually following the official instructions](https://pytorch.org/get-started/locally/)

- If you want to use the napari-based GUI, you will also need to [install one of qtpy's supported Qt backends](https://github.com/spyder-ide/qtpy/blob/master/README.md#requirements), for example PyQt5:

      pip install pyqt5

    This has to be done manually since none of the backends is compatible with all platforms and pip can't auto-select an optimal one.


### Option 2: From sources (with `pip` or `conda`)

First obtain the [project sources](https://github.com/StructuralNeurobiologyLab/emcaps) (either clone or [download zip](https://github.com/StructuralNeurobiologyLab/emcaps/archive/refs/heads/master.zip) and extract) and `cd` to the project root.

If you want to install all dependencies and the package itself with `pip`, just run

    pip install .


Alternatively, if you want to install the dependencies with `conda`, run the following commands:

    conda env create -f environment.yml
    conda activate emcaps
    pip install .


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

    $ emcaps-splitdataset

or

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

For a usage example featuring config sweeps, see `_scripts/seg_cls_test.sh`

### Producing a patch dataset based on image segmentation

Based on segmentation (from a model or human annotation), extract particle-centered image patches and store them as separate files in addition to metadata. The resulting patch dataset can be used for training models for patch-based classification. In addition, A random sample of the validation patches is prepared for evaluation of human and model-based classification evaluation.

    $ emcaps-patchifyseg

or

    $ python3 -m emcaps.inference.patchifyseg

### Training new patch classifiers

Requires the outputs of `patchifyseg` (see above).

    $ emcaps-patchtrain

or

    $ python3 -m emcaps.training.patchtrain

### Quantitative evaluation of patch classification results

Requires the outputs of `patchifyseg` (see above).

    $ emcaps-patcheval

or

    $ python3 -m emcaps.inference.patcheval

For a usage example featuring config sweeps, see `_scripts/patcheval.sh`

### Rendering average images of patch collections and grouping patches by EMcapsulin types

Requires the outputs of `patchifyseg` (see above).

    $ emcaps-averagepatches

or

    $ python3 -m emcaps.analysis.averagepatches


## Configuration system

We are using a common configuration system for the runnable code, based on [Hydra](https://hydra.cc/docs/1.2/intro/) and [OmegaConf](https://omegaconf.readthedocs.io/en/2.2_branch/).
A central default config file with explanatory comments is located at `conf/conf.yaml`.
It is written to be as automatic and minimal as possible, but it can still be necessary to change some of the values for experiments or adapting to a different system.

For the syntax of such yaml-based config files please refer to the OmegaConf docs on [access and manipulation](https://omegaconf.readthedocs.io/en/2.2_branch/usage.html#access-and-manipulation) and [variable interpolation](https://omegaconf.readthedocs.io/en/2.2_branch/usage.html#variable-interpolation)

For running hydra-enabled code with custom configuration you can either point to a different config file with the `-cp` [CLI flag](https://hydra.cc/docs/1.2/advanced/hydra-command-line-flags/) or change config values directly on the CLI using [Hydra's override syntax](https://hydra.cc/docs/1.2/advanced/override_grammar/basic/)


## Dataset

If you want to train own models and/or do quantitative evaluation on the official data, please find the data [here](https://drive.google.com/drive/folders/1S-dwZx0kHY3HuIiAXMyFlezsfOJmwYut?usp=share_link) and extract it to `~/emc/emcapsulin`.



## Further notes

For more details see top-level docstrings in each file.
