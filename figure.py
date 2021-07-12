import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sklearn.mixture
import sys
import tifffile
import tqdm
import zarr


path = sys.argv[1]
tiff = tifffile.TiffFile(path, is_ome=False)
zarray = zarr.open(tiff.series[0].aszarr())
if isinstance(zarray, zarr.Group):
    zarray = zarray[0]
assert zarray.ndim == 3

plt.figure(figsize=(15, 7))
ni = zarray.shape[0]
ss = max(zarray.shape[1], zarray.shape[2]) // 200
sampled_pixels = zarray.shape[1] // ss * zarray.shape[2] // ss
print(f"Sampling every {ss} pixels -- total pixel count is {sampled_pixels}")

ni = min(ni, 16)

for i in tqdm.tqdm(range(ni)):

    zimg = zarray[i, ::ss, ::ss]
    zimg_log = np.log(zimg[zimg>0])

    ax = plt.subplot(ni, 3, i * 3 + 1)
    if i < ni - 1:
        ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.hist(zimg.ravel(), bins=np.linspace(0, 65535, 200), density=True, color='silver', histtype='stepfilled')

    ax = plt.subplot(ni, 3, i * 3 + 2)
    if i < ni - 1:
        ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.hist(zimg_log, bins=np.linspace(0, np.log(65535), 200), density=True, color='silver', histtype='stepfilled')

    ax = plt.subplot(ni, 3, i * 3 + 3)
    if i < ni - 1:
        ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.hist(zimg_log, bins=np.linspace(0, np.log(65535), 200), density=True, color='silver', histtype='stepfilled')
    gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
    gmm.fit(zimg_log.reshape((-1,1)))
    ci = np.argsort(gmm.means_.squeeze())[-2:]
    vmin, vmax = np.round(np.exp(gmm.means_[ci, 0] + (gmm.covariances_[ci, 0, 0] ** 0.5 * 2))).astype(int)
    if vmin >= np.exp(gmm.means_[ci[1], 0]):
        vmin = np.round(np.exp(gmm.means_[ci[1], 0] + (gmm.covariances_[ci[1], 0, 0] ** 0.5 * -1))).astype(int)
    vmin = max(vmin, zimg.min())
    vmax = min(vmax, zimg.max())
    x = np.linspace(*ax.get_xlim(), 200)
    order = np.argsort(gmm.means_.squeeze())
    for idx in order:
        mean = gmm.means_[idx, 0]
        var = gmm.covariances_[idx, 0, 0]
        weight = gmm.weights_[idx]
        dist = scipy.stats.norm(mean, var ** 0.5)
        y = dist.pdf(x) * weight
        ax.plot(x, y, lw=2, alpha=0.8)
    for v in vmin, vmax:
        ax.axvline(np.log(v), c='tab:green', ls=':')
    ax.plot(x, np.exp(gmm.score_samples(x.reshape((-1,1)))), color="black", ls="--")
    ax.set_xlim(0, np.log(65535))

plt.tight_layout()
plt.show()
