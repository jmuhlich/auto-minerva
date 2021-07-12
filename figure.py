import matplotlib.pyplot as plt
import napari
import numpy as np
import qtpy.QtCore
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

#for i in range(ni):
ni = 16
for i in tqdm.tqdm(range(0, 16)):

    zimg = zarray[i, ::20,::20]
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
    ax.hist(zimg_log, bins=np.linspace(3, np.log(65535), 200), density=True, color='silver', histtype='stepfilled')

    ax = plt.subplot(ni, 3, i * 3 + 3)
    if i < ni - 1:
        ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.hist(zimg_log, bins=np.linspace(3, np.log(65535), 200), density=True, color='silver', histtype='stepfilled')
    iinfo = np.iinfo(zimg.dtype)
    gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
    gmm.fit(zimg_log.reshape((-1,1)))
    a = np.argmax(gmm.means_)
    vmin, vmax = np.round(np.exp(gmm.means_[a] + (gmm.covariances_[a] ** 0.5 * [-2,2]))).astype(int).squeeze()
    vmin = max(vmin, iinfo.min)
    vmax = min(vmax, iinfo.max)
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
    ax.set_xlim(2.5954830184973177, 11.49485661155633)

plt.tight_layout()
plt.show()
