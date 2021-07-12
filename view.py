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
    img = zarray
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

gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
img_subsampled = img[::20, ::20]
pixels_log = np.log(img_subsampled[img_subsampled > 0])
gmm.fit(pixels_log.reshape((-1,1)))

i = np.argmax(gmm.means_)
vmin, vmax = np.round(np.exp(gmm.means_[i] + (gmm.covariances_[i] ** 0.5 * [-2,2]))).astype(int).squeeze()
vmin = max(vmin, iinfo.min)
vmax = min(vmax, iinfo.max)
print((vmin, vmax))

viewer = napari.Viewer()
viewer.add_image(img, contrast_limits=[vmin, vmax], gamma=0.45, name=f"{channel_name} - opt")
viewer.add_image(img, contrast_limits=[iinfo.min, iinfo.max], gamma=0.45, name=f"{channel_name} - full")
viewer.add_image(img, contrast_limits=[iinfo.min, iinfo.max], visible=False, name=f"{channel_name} - full γ=1")
viewer.add_image(img < vmin, blending="additive", opacity=0.3, colormap="red", visible=False, name="background")
viewer.add_image((img >= vmin) & (img <= vmax), blending="additive", opacity=0.3, colormap="cyan", visible=False, name="foreground")
viewer.add_image(img > vmax, blending="additive", opacity=0.3, colormap="magenta", visible=False, name="saturated")

def reset_gamma():
    for l in viewer.layers:
        l.gamma = l.gamma
    viewer.layers[1].visible = False
timer = qtpy.QtCore.QTimer()
timer.singleShot(1000, reset_gamma)

fig = plt.figure()
ax = fig.gca()
ax.hist(pixels_log, bins=100, density=True, color="silver")
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
fig.show()

napari.run()
