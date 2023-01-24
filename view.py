# coding: utf-8

import matplotlib.pyplot as plt
import napari
import numpy as np
import ome_types
import qtpy.QtCore
import scipy.stats
import sklearn.mixture
import sys
import tifffile
import zarr

path = sys.argv[1]
channel = int(sys.argv[2])

tiff = tifffile.TiffFile(path)
zarray = zarr.open(tiff.series[0].aszarr())
if isinstance(zarray, zarr.Group):
    zarray = zarray[0]
if zarray.ndim == 2:
    img = zarray[...]
elif zarray.ndim == 3:
    img = zarray[channel]
else:
    raise Exception(f"Can't handle {zarray.ndim}-dimensional images")
iinfo = np.iinfo(img.dtype)

try:
    ome = ome_types.from_xml(tiff.pages[0].tags[270].value)
    channel_name = ome.images[0].pixels.channels[channel].name
except:
    channel_name = "Image"

yi, xi = np.floor(np.linspace(0, img.shape, 200, endpoint=False)).astype(int).T
# Slice one dimension at a time. Should generally use less memory than a meshgrid.
img_s = img[yi]
img_s = img_s[:, xi]
img_log = np.log(img_s[img_s > 0])
gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
gmm.fit(img_log.reshape((-1,1)))
means = gmm.means_[:, 0]
morder = _, i1, i2 = np.argsort(means)
means = means[morder]
covars = gmm.covariances_[morder, 0, 0]
stds = covars ** 0.5
_, mean1, mean2 = means
_, std1, std2 = stds

x = np.linspace(mean1, mean2, 50)
y1 = scipy.stats.norm(mean1, std1).pdf(x) * gmm.weights_[i1]
y2 = scipy.stats.norm(mean2, std2).pdf(x) * gmm.weights_[i2]

lmax = mean2 + 2 * std2
lmin = x[np.argmin(np.abs(y1 - y2))]
if lmin >= mean2:
    lmin = mean2 - 2 * std2
vmin = max(np.exp(lmin), img_s.min(), 0)
vmax = min(np.exp(lmax), img_s.max())

print((vmin, vmax))

masks = (img[..., None] >= np.exp(means - stds)) & (img[..., None] <= np.exp(means + stds))
masks = np.transpose(masks, (2, 0, 1))

viewer = napari.Viewer()
viewer.add_image(img, contrast_limits=[vmin, vmax], name=f"{channel_name} - fit")
viewer.add_image(img, contrast_limits=[img.min(), img.max()], name=f"{channel_name} - full")
viewer.add_image(img < vmin, colormap="red", blending="additive", visible=False, name="background")
viewer.add_image((img >= vmin) & (img <= vmax), colormap="cyan", blending="additive", visible=False, name="foreground")
viewer.add_image(img > vmax, colormap="magenta", blending="additive", visible=False, name="saturated")
viewer.add_image(masks[0], colormap="bop blue", blending="additive", opacity=0.5, visible=False, name="mode 1")
viewer.add_image(masks[1], colormap="bop orange", blending="additive", opacity=0.5, visible=False, name="mode 2")
viewer.add_image(masks[2], colormap="green", blending="additive", opacity=0.5, visible=False, name="mode 3")

def reset_gamma():
    for l in viewer.layers:
        l.gamma = l.gamma
    viewer.layers[1].visible = False
timer = qtpy.QtCore.QTimer()
timer.singleShot(1000, reset_gamma)

fig = plt.figure()
ax = fig.gca()
ax.hist(img_log, bins=100, density=True, color="silver")
x = np.linspace(*ax.get_xlim(), 200)
order = np.argsort(gmm.means_.squeeze())
for i, idx in enumerate(order, 1):
    mean = gmm.means_[idx, 0]
    var = gmm.covariances_[idx, 0, 0]
    weight = gmm.weights_[idx]
    dist = scipy.stats.norm(mean, var ** 0.5)
    y = dist.pdf(x) * weight
    ax.plot(x, y, label=i, lw=4, alpha=0.7)
for v in vmin, vmax:
    ax.axvline(np.log(v), c='tab:green', ls=':')
ax.plot(x, np.exp(gmm.score_samples(x.reshape((-1,1)))), color="black", ls="--")
formatter = plt.FuncFormatter(lambda x, pos: f"{int(round(np.exp(x)))}")
ax.xaxis.set_major_formatter(formatter)
plt.show()

napari.run()
