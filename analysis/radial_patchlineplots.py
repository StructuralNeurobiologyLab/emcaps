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
import tqdm


# Based on https://stackoverflow.com/a/21242776
def get_radial_profile(img, center=None):
    if center is None:
        center = np.array(img.shape) // 2 - 1
    y, x = np.indices((img.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int64)

    tbin = np.bincount(r.ravel(), img.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    # TODO: Is this slice valid?
    radialprofile = radialprofile[:img.shape[0] // 2]
    return radialprofile



# plt.rcParams.update({'font.family': 'Arial'})


# avg = True  # Plot average profile on top
# avg = False  # Plot demo patch profile instead


# patch_path = Path('~/tumpatches').expanduser()
# patch_path = Path('~/tum/patches_v2_hek_bgmask_enctype_prefix/raw/').expanduser()
patch_path = Path('~/tum/patches_v2_hek_enctype_prefix/raw/').expanduser()

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

YLIM = (50, 230)  # ylim for profile plots

profiles = {
    'MT3-MxEnc': [],
    'MT3-QtEnc': [],
}

patches = {
    'MT3-MxEnc': [],
    'MT3-QtEnc': [],
}
avgpatch = {
}
avgprof = {
}
stdprof = {
}

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

# Reduce to average
for enctype in patches.keys():
    patches_np = np.stack(patches[enctype])
    avgpatch[enctype] = np.mean(patches[enctype], axis=0)
    prof_np = np.stack(profiles[enctype])

    # Mirror profiles at y=0 so they are symmetric (redundant but better for comparing against images)
    prof_np = np.concatenate((np.flip(prof_np, axis=1), prof_np), axis=1)

    avgprof[enctype] = np.mean(prof_np, axis=0)
    stdprof[enctype] = np.std(prof_np, axis=0)



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
plt.show()

# Average plots
fig, ax = plt.subplots(figsize=(3.5, 3), tight_layout=True)
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
plt.show()
