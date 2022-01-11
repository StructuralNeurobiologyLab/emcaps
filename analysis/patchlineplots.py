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
    radialprofile = radialprofile[:img.shape[0] // 2 - 1]
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

profiles = {
    'MxEnc': {
        'horizontal': [],
        'vertical': [],
        'radial': [],
    },
    'QtEnc': {
        'horizontal': [],
        'vertical': [],
        'radial': [],
    },
}

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9, 5), tight_layout=True)

for i, p in tqdm.tqdm(enumerate(patch_path.iterdir()), total=len(list(patch_path.glob('*')))):
    img = imageio.imread(p)
    if p.name.startswith('mx'):
        enctype = 'MxEnc'
        axrow = axes[0]
    elif p.name.startswith('qt'):
        enctype = 'QtEnc'
        axrow = axes[1]
    else:
        raise RuntimeError(p)

    horizontal_profile = img[img.shape[0] // 2, :]
    vertical_profile = img[:, img.shape[1] // 2]

    radial_profile = get_radial_profile(img)
    # profiles[enctype]['radial'].append(radial_profile)

    profiles[enctype]['horizontal'].append(horizontal_profile)
    profiles[enctype]['vertical'].append(vertical_profile)
    if p in demo_paths:
        axrow[0].imshow(img, cmap='gray')
        axrow[0].set_xticks(np.arange(2, 28, 4), labels=np.arange(2, 28, 4) - 14)
        axrow[0].set_yticks(np.arange(2, 28, 4), labels=np.arange(2, 28, 4) - 14)

        axrow[0].axhline(y=28//2, c='orange', alpha=0.9, linestyle='--')
        axrow[0].axvline(x=28//2, c='blue', alpha=0.9, linestyle='--')
        axrow[1].plot(horizontal_profile, c='orange', alpha=1., linewidth=1, linestyle='--')
        axrow[2].plot(vertical_profile, c='blue', alpha=1., linewidth=1, linestyle='--')
        axrow[1].set_xticks(np.arange(2, 28, 4), labels=np.arange(2, 28, 4) - 14)
        axrow[2].set_xticks(np.arange(2, 28, 4), labels=np.arange(2, 28, 4) - 14)
    axrow[0].set_title(enctype)
    # axrow[1].plot(horizontal_profile, c='gray', linewidth=0.05, alpha=0.5)
    axrow[1].set_title('Horizontal profile')
    # axrow[2].plot(vertical_profile, c='gray', linewidth=0.05, alpha=0.5)
    axrow[2].set_title('Vertical profile')


# for k in profiles.keys():
#     for j in profiles[k].keys():
#         prof = np.array(profiles[k][j])
#         avgprof = np.mean(prof, axis=0)
#         profiles[k][j] = avgprof

for enctype, orientations in profiles.items():
    for orientation, profile in orientations.items():
        prof = np.array(profile)
        avgprof = np.mean(prof, axis=0)
        profiles[enctype][orientation] = avgprof
        axrow = axes[0] if enctype == 'MxEnc' else axes[1]
        ax = axrow[1] if orientation == 'horizontal' else axrow[2]
        c = 'orange' if orientation == 'horizontal' else 'blue'
        ax.plot(avgprof, c=c, linewidth=2)
        # Also plot uncertainty
        # minprof = np.min(prof, axis=0)
        # maxprof = np.max(prof, axis=0)
        stdprof = np.std(prof, 0)
        # ax.fill_between(range(prof.shape[1]), avgprof - stdprof, avgprof + stdprof, color='gray', alpha=0.3)
        # ax.errorbar(range(prof.shape[1]), avgprof, yerr=np.std(prof, 0), fmt='.', elinewidth=1.5, color='gray', alpha=0.2)

# fig.suptitle('Axis profile plots. orange: horizontal, blue: vertical. Gray: all images, blue: average profiles.', fontsize=16)

plt.savefig('/tmp/patchprofile.pdf')
plt.show()

# Average plots
fig, ax = plt.subplots(figsize=(3, 3), tight_layout=True)
ax.plot(profiles['MxEnc']['horizontal'], c='purple', label='MxEnc', linewidth=2)
ax.plot(profiles['QtEnc']['horizontal'], c='green', label='QtEnc', linewidth=2)
ax.set_xticks(np.arange(2, 28, 4), labels=np.arange(2, 28, 4) - 14)
ax.legend()
ax.set_title('Horizontal average profile')

plt.savefig('/tmp/patchprofile_av_hori.pdf')
plt.show()

fig, ax = plt.subplots(figsize=(3, 3), tight_layout=True)
ax.plot(profiles['MxEnc']['vertical'], c='purple', label='MxEnc', linewidth=2)
ax.plot(profiles['QtEnc']['vertical'], c='green', label='QtEnc', linewidth=2)
ax.set_xticks(np.arange(2, 28, 4), labels=np.arange(2, 28, 4) - 14)
ax.legend()
ax.set_title('Vertical average profile')

plt.savefig('/tmp/patchprofile_av_vert.pdf')
plt.show()
