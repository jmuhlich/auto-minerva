import colour
import napari
import numpy as np
import ome_types
import qtpy.QtCore
import scipy.stats
import skimage
import sklearn.mixture
import sys
import tifffile
import zarr

path = sys.argv[1]
channels = [int(a) for a in sys.argv[2:]]

tiff = tifffile.TiffFile(path, is_ome=False)
zarray = zarr.open(tiff.series[0].aszarr())
if isinstance(zarray, zarr.Group):
    zarray = zarray[0]
if zarray.ndim == 2:
    img = zarray
elif zarray.ndim == 3:
    img = np.array([zarray[c] for c in channels])
else:
    raise Exception(f"Can't handle {zarray.ndim}-dimensional images")

try:
    ome = ome_types.from_xml(tiff.pages[0].tags[270].value)
    channel_names = [ome.images[0].pixels.channels[c].name for c in channels]
except:
    channel_names = [f"Channel {c}" for c in channels]

def compute_range(img):
    iinfo = np.iinfo(img.dtype)
    gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
    img_subsampled = img[::20, ::20]
    pixels_log = np.log(img_subsampled[img_subsampled > 0])
    gmm.fit(pixels_log.reshape((-1,1)))
    i = np.argmax(gmm.means_)
    vmin, vmax = np.round(np.exp(gmm.means_[i] + (gmm.covariances_[i] ** 0.5 * [-2,2]))).astype(int).squeeze()
    vmin = max(vmin, iinfo.min)
    vmax = min(vmax, iinfo.max)
    return vmin, vmax

cmaps = 'gray', 'blue', 'green', 'red'
viewer = napari.Viewer()
for cimg, cmap, name in zip(img, cmaps, channel_names):
    vmin, vmax = compute_range(cimg)
    #vmax *= len(img)
    print((vmin,vmax))
    viewer.add_image(cimg, contrast_limits=[vmin, vmax], colormap=cmap, name=name, blending="additive", gamma=0.45)

def reset_gamma():
    for l in viewer.layers:
        l.gamma = l.gamma
timer = qtpy.QtCore.QTimer()
timer.singleShot(1000, reset_gamma)

napari.run()
