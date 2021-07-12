import colour
import itertools
import json
import numpy as np
import ome_types
import qtpy.QtCore
import scipy.stats
import skimage
import sklearn.mixture
import sys
import tifffile
import zarr


def compute_range(img):
    gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
    img_subsampled = skimage.img_as_float32(img[::20, ::20])
    pixels_log = np.log(img_subsampled[img_subsampled > 0])
    print(pixels_log.shape)
    gmm.fit(pixels_log.reshape((-1,1)))
    i = np.argmax(gmm.means_)
    vmin, vmax = np.exp(gmm.means_[i] + (gmm.covariances_[i] ** 0.5 * [-2,2])).squeeze()
    vmin = float(max(vmin, 0))
    vmax = float(min(vmax, 1))
    return vmin, vmax


def main():

    path = sys.argv[1]

    sys.stderr.write(f"opening image: {path}\n")
    tiff = tifffile.TiffFile(path, is_ome=False)
    zarray = zarr.open(tiff.series[0].aszarr())
    if isinstance(zarray, zarr.Group):
        zarray = zarray[0]
    if zarray.ndim == 2:
        # FIXME This can be handled easily (promote to 3D array), we just need a
        # test file to make sure we're doing it right.
        raise Exception("Can't handle 2-dimensional images")
    elif zarray.ndim == 3:
        pass
    else:
        raise Exception(f"Can't handle {zarray.ndim}-dimensional images")

    sys.stderr.write(f"reading metadata\n")
    try:
        ome = ome_types.from_xml(tiff.pages[0].tags[270].value)
        channel_names = [c.name for c in ome.images[0].pixels.channels]
        for i, n in enumerate(channel_names):
            if not n:
                channel_names[i] = f"Channel {i + 1}"
    except:
        channel_names = [f"Channel {i + 1}" for i in range(zarray.shape[0])]

    story = {
        "sample_info": {
            "name": "",
            "rotation": 0,
            "text": "",
        },
        "groups": [],
        "waypoints": [],
    }

    color_cycle = 'ffffff', 'ff0000', '00ff00', '0000ff'

    for gi, idx_start in enumerate(range(0, zarray.shape[0], 4), 1):
        idx_end = min(idx_start + 4, zarray.shape[0])
        channel_numbers = range(idx_start, idx_end)
        channel_defs = []
        for ci, color in zip(channel_numbers, color_cycle):
            sys.stderr.write(f"analyzing channel {ci + 1}/{zarray.shape[0]}\n")
            img = zarray[ci]
            vmin, vmax = compute_range(img)
            channel_defs.append({
                "color": color,
                "id": ci,
                "label": channel_names[ci],
                "min": vmin,
                "max": vmax,
            })
        story["groups"].append({
            "label": f"Group {gi}",
            "channels": channel_defs,
        })

    print(json.dumps(story))


if __name__ == "__main__":
    main()
