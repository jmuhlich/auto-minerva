import matplotlib.pyplot as plt
import numpy as np
import skimage.exposure
import sys
import tifffile
import zarr

import util

path = sys.argv[1]
channels = [int(a) for a in sys.argv[2:]]

tiff = tifffile.TiffFile(path, is_ome=False)
zarray = zarr.open(tiff.series[0].aszarr())
if isinstance(zarray, zarr.Group):
    zarray = zarray[0]
if zarray.ndim == 3:
    pass
else:
    raise Exception(f"Can't handle {zarray.ndim}-dimensional images")
if zarray.dtype != np.uint16:
    raise Exception(f"Can't handle {zarray.dtype} images")

#zarray = zarray[:, 9500:12200, 600:6600]

ranges = [util.auto_threshold(zarray[c]) for c in channels]
print(f"ranges: {ranges}")
ss = int(np.ceil(max(zarray.shape[1:]) / 2000))
scaled = [
    util.rescale(zarray[ci, ::ss, ::ss], *range)
    for ci, range in zip(channels, ranges)
]

out_shape = tuple(np.ceil(np.divide(zarray.shape[1:], ss)).astype(int))
out_img = np.zeros((3,) + out_shape, np.float32)
out_img += scaled[0][None, ...] / 2
out_img[0] += scaled[1]
out_img[1] += scaled[2]
out_img[2] += scaled[3]
out_img = np.clip(out_img, 0, 65535).astype(np.uint16)
out_img = skimage.exposure.adjust_gamma(out_img, 1/2.2)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
axs = axs.ravel()
axs[0].imshow(skimage.img_as_ubyte(out_img).transpose((1, 2, 0)))
for ax, s in zip(axs[1:], scaled[1:]):
    s = skimage.exposure.adjust_gamma(s, 1/2.2)
    ax.imshow(s, vmin=0, vmax=65535, cmap='gray')
fig.show()
