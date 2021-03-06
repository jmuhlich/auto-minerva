import colour
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

img_xyz = np.zeros(out_shape + (3,), dtype=float)
L_max = 0.9
C = 0.2
#hues = (None, 61, 140, 260)
hues = (None, 40, 140, 260)
for simg, hue in zip(scaled, hues):
    cimg = np.zeros_like(img_xyz)
    cimg[..., 0] = skimage.img_as_float(simg) * L_max  # L
    if hue is not None:
        h = np.deg2rad(hue)
        cimg[..., 1] = C * np.cos(h)  # a
        cimg[..., 2] = C * np.sin(h)  # b
    img_xyz += colour.Oklab_to_XYZ(cimg)
out_okimg = np.clip(colour.XYZ_to_sRGB(img_xyz), 0, 1)

out_img = np.zeros(out_shape + (3,), np.float32)
out_img += scaled[0][..., None]
out_img[..., 0] += scaled[1]
out_img[..., 1] += scaled[2]
out_img[..., 2] += scaled[3]
out_img = np.clip(out_img, 0, 65535).astype(np.uint16)

fig, (img_axs, ch_axs) = plt.subplots(2, 3, sharex=True, sharey=True)
img_axs[0].imshow(skimage.img_as_ubyte(out_img))
img_axs[0].set_title('RGB')
img_axs[1].imshow(skimage.img_as_ubyte(skimage.exposure.adjust_gamma(out_img, 1/2.2)))
img_axs[1].set_title('RGB (gamma encoded)')
img_axs[2].imshow(out_okimg)
img_axs[2].set_title('OKlab')
for ax, s, cname in zip(ch_axs, scaled[1:], ('Red', 'Green', 'Blue')):
    s = skimage.exposure.adjust_gamma(s, 1/2.2)
    ax.imshow(s, vmin=0, vmax=65535, cmap='gray')
    ax.set_title(f"{cname} channel")
fig.show()

plt.show()
