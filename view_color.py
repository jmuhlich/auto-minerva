import colour
import itertools
import napari
import numpy as np
import ome_types
import qtpy.QtCore
import scipy.stats
import skimage
import sklearn.mixture
import sys
import threadpoolctl
import tifffile
import tqdm
import zarr

threadpoolctl.threadpool_limits(1)

path = sys.argv[1]
channels = [int(a) for a in sys.argv[2:]]

tiff = tifffile.TiffFile(path)
ndim = tiff.series[0].ndim
if ndim == 2:
    # FIXME This can be handled easily (promote to 3D array), we just need a
    # test file to make sure we're doing it right.
    raise Exception("Can't handle 2-dimensional images (yet)")
elif ndim == 3:
    pass
else:
    raise Exception(f"Can't handle {ndim}-dimensional images")
# Get smallest pyramid level that's at least 200 in both dimensions.
level_series = next(
    level for level in reversed(tiff.series[0].levels)
    if all(d >= 200 for d in level.shape[1:])
)
zarray = zarr.open(level_series.aszarr())
if not channels:
    channels = list(range(zarray.shape[0]))

try:
    ome = ome_types.from_xml(tiff.pages[0].tags[270].value)
    channel_names = [ome.images[0].pixels.channels[c].name for c in channels]
except:
    channel_names = [f"Channel {c}" for c in channels]

def auto_threshold(img):

    assert img.ndim == 2

    yi, xi = np.floor(np.linspace(0, img.shape, 200, endpoint=False)).astype(int).T
    # Slice one dimension at a time. Should generally use less memory than a meshgrid.
    img = img[yi]
    img = img[:, xi]
    img_log = np.log(img[img > 0])
    gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
    gmm.fit(img_log.reshape((-1,1)))
    means = gmm.means_[:, 0]
    _, i1, i2 = np.argsort(means)
    mean1, mean2 = means[[i1, i2]]
    std1, std2 = gmm.covariances_[[i1, i2], 0, 0] ** 0.5

    x = np.linspace(mean1, mean2, 50)
    y1 = scipy.stats.norm(mean1, std1).pdf(x) * gmm.weights_[i1]
    y2 = scipy.stats.norm(mean2, std2).pdf(x) * gmm.weights_[i2]

    lmax = mean2 + 2 * std2
    lmin = x[np.argmin(np.abs(y1 - y2))]
    if lmin >= mean2:
        lmin = mean2 - 2 * std2
    vmin = max(np.exp(lmin), img.min(), 0)
    vmax = min(np.exp(lmax), img.max())

    return vmin, vmax

cmaps = itertools.cycle(("gray", "blue", "green", "red"))
viewer = napari.Viewer()

print(f"Computing limits for channels: {channels}")
limits = []
for cimg, cmap, name in tqdm.tqdm(list(zip(zarray, cmaps, channel_names))):
    vmin, vmax = auto_threshold(cimg)
    limits.append((vmin, vmax))
    viewer.add_image(
        cimg,
        contrast_limits=[vmin, vmax],
        colormap=cmap,
        name=name,
        blending="additive",
        #gamma=0.45
    )

viewer.update_console(globals())

def reset_gamma():
    for l in viewer.layers:
        l.gamma = l.gamma
#timer = qtpy.QtCore.QTimer()
#timer.singleShot(1000, reset_gamma)

napari.run()
