"""
Create line plots of horizontal and vertical axis profiles of encapsulin
image patches.

Requires a directory with patch images in the `patch_path`, which can be
built with eclassify_analysis.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import sys
from pathlib import Path
from scipy.signal import find_peaks
import tqdm
import seaborn as sns


def filter_nan(x):
    print(f'Filtering {np.sum(np.isnan(x))} NaN values from a total of {x.size} values')
    return x[~np.isnan(x)].copy()

# Based on https://stackoverflow.com/a/21242776
def get_radial_profile(img: np.ndarray, center=None) -> np.ndarray:
    if center is None:
        center = np.array(img.shape) // 2 - 1
    y, x = np.indices((img.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int64)

    tbin = np.bincount(r.ravel(), img.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    radialprofile = radialprofile[:img.shape[0] // 2]
    return radialprofile


def measure_inner_disk_radius(hprofile, center_nhood=2, discrete=False):
    # TODO: gaussian filtering?
    # Look for actual encapsulin center in the patch by finding intensity maximum in a `center_nhood` neighborhood around the central pixel
    # Encapsulin center at cx with intensity cy
    center_x = np.argmax(hprofile[:center_nhood + 1])
    center_y = hprofile[center_x]
    # Intensity minimum, assumed as center of dark outer ring
    min_x = np.argmin(hprofile)
    min_y = hprofile[min_x]
    # Middle value between maximum center intensity and minimum outer ring intensity
    inner_y = (center_y + min_y) / 2
    # Starting from center outwards, find first crossing of inner_y value:
    # 1. "right" index: where inner_y was already crossed (falling below)
    inner_right_x = np.argmax(hprofile[center_x:] <= inner_y)
    # 2. "left" index: Index immediately before inner_y crossing
    inner_left_x = inner_right_x - 1
    if discrete:
        # Return discrete index (-> floor)
        inner_radius = inner_left_x
    else:
        # 3. Approximate crossing's continuous location by doing inverse linear interpolation
        inner_right_y = hprofile[inner_right_x]
        inner_left_y = hprofile[inner_left_x]
        # Interpolate reverse function to get intersection point
        if inner_right_y > inner_left_y:
            # Intensity goes back up again -> fail
            return np.nan
        # np.interp requires increasing sequence for second argument, so swap the left-right order here
        inner_x = np.interp(inner_y, [inner_right_y, inner_left_y], [inner_right_x, inner_left_x])
        inner_radius = inner_x
    if inner_radius < 0.5:  # Not plausible
        return np.nan
    return inner_radius

    # # Starting from inner radius outwards, find first crossing of bg_thresh value:
    # # 1. "right" index: where bg_thresh was already crossed (exceeding)
    # outer_right_x = np.argmax(hprofile[inner_right_x:] >= bg_thresh)
    # # Clip at boundary
    # if len(hprofile) - 1 > outer_right_x:
    #     print(f'{outer_right_x=}')
    # outer_right_x = np.clip(outer_right_x, a_min=None, a_max=len(hprofile) - 1)
    # # 2. "left" index: Index immediately before bg_thresh crossing
    # outer_left_x = outer_right_x - 1
    # if discrete:
    #     # Return discrete index (-> floor)
    #     outer_radius = outer_left_x
    # else:
    #     # 3. Approximate crossing's continuous location by doing inverse linear interpolation
    #     outer_right_y = hprofile[outer_right_x]
    #     outer_left_y = hprofile[outer_left_x]
    #     # Interpolate reverse function to get intersection point
    #     # TODO: Ensure ordering of interpolation args
    #     outer_x = np.interp(bg_thresh, [outer_left_y, outer_right_y], [outer_left_x, outer_left_x])
    #     outer_radius = outer_x

    # return inner_radius, outer_radius


def measure_outer_disk_radius(mask: np.ndarray, discrete: bool = False) -> float:
    """Measure the average distance between the center and the boundary of the seg mask -> r2"""
    profile = get_radial_profile(mask)
    y = 0.5  # Use 0.5 because mask is binary (0.5 is the linear interpolation threshold).
    # Starting from center outwards, find first crossing of y value:
    # 1. "right" index: where y was already crossed (falling below)
    right_x = np.argmax(profile <= y)
    # 2. "left" index: Index immediately before y crossing
    left_x = right_x - 1
    if discrete:
        # Return discrete index (-> floor)
        outer_radius = left_x
    else:
        # 3. Approximate crossing's continuous location by doing inverse linear interpolation
        right_y = profile[right_x]
        left_y = profile[left_x]
        # Interpolate reverse function to get intersection point
        if right_y > left_y:
            # Intensity goes back up again -> fail
            return np.nan
        # np.interp requires increasing sequence for second argument, so swap the left-right order here
        x = np.interp(y, [right_y, left_y], [right_x, left_x])
        outer_radius = x
    if outer_radius < 3:  # Note plausible
        return np.nan
    return outer_radius

if __name__ == '__main__':
    # plt.rcParams.update({'font.family': 'Arial'})


    # avg = True  # Plot average profile on top
    # avg = False  # Plot demo patch profile instead


    # patch_path = Path('~/tumpatches').expanduser()
    # patch_path = Path('~/tum/patches_v2_hek_bgmask_enctype_prefix/raw/').expanduser()
    patch_path = Path('~/tum/patches_v2_hek_enctype_prefix/raw/').expanduser()
    mask_path = patch_path.parent / 'mask'

    mx_demo_path = patch_path / 'mx_raw_patch_03154.tif'
    qt_demo_path = patch_path / 'qt_raw_patch_01724.tif'


    demo_paths = {mx_demo_path, qt_demo_path}

    # For visualization: Top qt, bottom mx
    ENCTYPE_ROW = {
        'MT3-MxEnc': 1,
        'MT3-QtEnc': 0,
    }
    ENCTYPE_COLOR = {
        'MT3-MxEnc': 'blue',
        'MT3-QtEnc': 'red',
    }

    YLIM = (0, 230)  # ylim for profile plots
    SUBTRACT_MIN = True
    NANOMETERS_PER_PIXEL = 2

    min_patch_intensity = {}
    min_profile_intensity = {}
    mean_profile_bg_intensity = {}
    inner_disk_radii = {
        'MT3-MxEnc': [],
        'MT3-QtEnc': [],
    }
    avg_inner_disk_radius = {}
    outer_disk_radii = {
        'MT3-MxEnc': [],
        'MT3-QtEnc': [],
    }
    avg_outer_disk_radius = {}


    profiles = {
        'MT3-MxEnc': [],
        'MT3-QtEnc': [],
    }
    mprofiles = {}
    hprofiles = {}

    patches = {
        'MT3-MxEnc': [],
        'MT3-QtEnc': [],
    }
    masks = {
        'MT3-MxEnc': [],
        'MT3-QtEnc': [],
    }
    avgpatch = {}
    avgprof = {}
    stdprof = {}

    # Collect individual patches and profiles
    for i, p in tqdm.tqdm(enumerate(patch_path.iterdir()), total=len(list(patch_path.glob('*')))):
        img = imageio.imread(p)
        if p.name.startswith('mx'):
            enctype = 'MT3-MxEnc'
        elif p.name.startswith('qt'):
            enctype = 'MT3-QtEnc'
        else:
            raise RuntimeError(p)
        radial_profile = get_radial_profile(img)
        profiles[enctype].append(radial_profile)
        patches[enctype].append(img)

        # Load segmentation mask
        mpath = mask_path / p.name.replace('qt_', '').replace('mx_', '').replace('raw', 'mask')
        mask_img = imageio.imread(mpath) > 0
        masks[enctype].append(mask_img)

    # Reduce to average patches, average profiles
    for enctype in patches.keys():
        patches[enctype] = np.stack(patches[enctype])
        if SUBTRACT_MIN:
            min_patch_intensity[enctype] = np.min(patches[enctype])
            patches[enctype] -= min_patch_intensity[enctype]

        avgpatch[enctype] = np.mean(patches[enctype], axis=0)
        # Keep profile versions beginning at image centers ("half profiles")
        hprofiles[enctype] = np.stack(profiles[enctype]).copy()
        if SUBTRACT_MIN:
            min_profile_intensity[enctype] = np.min(hprofiles[enctype])
            hprofiles[enctype] -= min_profile_intensity[enctype]
        
        # Mirror profiles at y=0 so they are symmetric (redundant but better for comparing against images)
        profiles[enctype] = np.concatenate((np.flip(hprofiles[enctype], axis=1), hprofiles[enctype]), axis=1).copy()

        avgprof[enctype] = np.mean(profiles[enctype], axis=0)
        stdprof[enctype] = np.std(profiles[enctype], axis=0)


    # Measure radii
    for enctype in patches.keys():
        for hprofile in hprofiles[enctype]:
            r1 = measure_inner_disk_radius(hprofile)
            inner_disk_radii[enctype].append(r1)
        inner_disk_radii[enctype] = np.stack(inner_disk_radii[enctype])
        inner_disk_radii[enctype] *= NANOMETERS_PER_PIXEL
        # Filter out invalid measurements
        inner_disk_radii[enctype] = filter_nan(inner_disk_radii[enctype])
        avg_inner_disk_radius[enctype] = np.mean(inner_disk_radii[enctype])

        for mask in masks[enctype]:
            r2 = measure_outer_disk_radius(mask)
            outer_disk_radii[enctype].append(r2)
        outer_disk_radii[enctype] = np.stack(outer_disk_radii[enctype])
        outer_disk_radii[enctype] *= NANOMETERS_PER_PIXEL
        # Filter out invalid measurements
        outer_disk_radii[enctype] = filter_nan(outer_disk_radii[enctype])
        avg_outer_disk_radius[enctype] = np.mean(outer_disk_radii[enctype])

    # Rebalance classes
    assert inner_disk_radii['MT3-MxEnc'].shape[0] <= inner_disk_radii['MT3-QtEnc'].shape[0]
    n_mx = inner_disk_radii['MT3-MxEnc'].shape[0]
    inner_disk_radii['MT3-QtEnc'] = inner_disk_radii['MT3-QtEnc'][:n_mx]
    outer_disk_radii['MT3-QtEnc'] = outer_disk_radii['MT3-QtEnc'][:n_mx]


    # inner_bins = np.arange(0, 5.5, 0.5)
    # inner_xlim = (0.5, 4.5)
    # outer_bins = np.arange(6, 11.5, 0.5)
    # outer_xlim = (7, 11)
    inner_bins = np.arange(0, 12, 1)
    inner_xlim = (0, 12)
    outer_bins = np.arange(12, 25, 1)
    outer_xlim = (13, 24)

    legend_labels = ['MT3-QtEnc', 'MT3-MxEnc']

    fig, axes = plt.subplots(nrows=1, ncols=2, tight_layout=True, figsize=(4, 2.5))

    bw_adjust = 2

    # color = 'blue' if enctype == 'MT3-MxEnc' else 'red'
    iax = axes[0]
    sns.kdeplot(inner_disk_radii['MT3-QtEnc'], color='red', bw_adjust=bw_adjust, ax=iax)
    sns.kdeplot(inner_disk_radii['MT3-MxEnc'], color='blue', bw_adjust=bw_adjust, ax=iax)

    iax.set_xlim(*inner_xlim)
    iax.set_xlabel(f'r1 [nm]')
    iax.legend(labels=legend_labels, loc='lower right', fontsize=6)
    iax.grid(True)
    oax = axes[1]
    sns.kdeplot(outer_disk_radii['MT3-QtEnc'], color='red', bw_adjust=bw_adjust, ax=oax)
    sns.kdeplot(outer_disk_radii['MT3-MxEnc'], color='blue', bw_adjust=bw_adjust, ax=oax)
    # oax.set_xlim(*outer_xlim)
    oax.set_xlabel(f'r2 [nm]')
    oax.set(ylabel=None)
    # oax.legend(labels=legend_labels, loc='lower right', fontsize=8)
    oax.grid(True)


    plt.savefig('/tmp/radii.pdf')
    plt.show()


    # sns.histplot(inner_disk_radii['MT3-QtEnc'], stat='percent', bins=np.arange(0, 5, 0.5) - 0.25, kde=True, color='blue')


    # Visualize average images and average profiles
    tick_locs = np.arange(2, 28, 4)
    tick_labels = (np.arange(2, 28, 4) - 14) * 2
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 5), tight_layout=True)
    for enctype in patches.keys():
        axrow = axes[ENCTYPE_ROW[enctype]]
        # Average image
        axrow[0].imshow(avgpatch[enctype], cmap='gray')

        axrow[0].set_xticks(tick_locs, labels=tick_labels)
        axrow[0].set_yticks(tick_locs, labels=tick_labels)

        # Average profile
        axrow[1].plot(avgprof[enctype], c=ENCTYPE_COLOR[enctype], alpha=1., linewidth=1)
        axrow[1].set_ylim(*YLIM)
        axrow[1].set_yticks(range(*YLIM, 20))
        axrow[1].grid(True)

        axrow[1].set_xticks(tick_locs, labels=tick_labels)
        axrow[1].fill_between(range(avgprof[enctype].shape[0]), avgprof[enctype] - stdprof[enctype], avgprof[enctype] + stdprof[enctype], color=ENCTYPE_COLOR[enctype], alpha=0.1)

    plt.savefig('/tmp/patchprofiles.pdf')
    # plt.show()

    # Average plots
    fig, ax = plt.subplots(figsize=(3, 2.5), tight_layout=True)
    for enctype in reversed(patches.keys()):  # reverse iteration to maintain order
        ax.plot(avgprof[enctype], c=ENCTYPE_COLOR[enctype], label=enctype, linewidth=1)
        ax.fill_between(range(avgprof[enctype].shape[0]), avgprof[enctype] - stdprof[enctype], avgprof[enctype] + stdprof[enctype], color=ENCTYPE_COLOR[enctype], alpha=0.1)

    ax.set_ylim(*YLIM)
    ax.set_xticks(tick_locs, labels=tick_labels)
    ax.set_yticks(range(*YLIM, 20))
    ax.grid(True)
    ax.legend()
    # ax.set_title('Radial average profile')

    plt.savefig('/tmp/patchprofiles_compared.pdf')
    # plt.show()

    # import IPython ; IPython.embed(); raise SystemExit