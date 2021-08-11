import json
import numpy as np
import ome_types
import sklearn.mixture
import sys
import tifffile
import zarr


def auto_threshold(img):

    assert img.ndim == 2

    ss = max(*img.shape) // 200
    img = img[::ss, ::ss]
    img_log = np.log(img[img > 0])
    gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
    gmm.fit(img_log.reshape((-1,1)))
    means = gmm.means_[:, 0]
    covars = gmm.covariances_[:, 0, 0]
    _, i1, i2 = np.argsort(means)

    def fromlog(a):
        return np.round(np.exp(a)).astype(int)

    vmin, vmax = means[[i1, i2]] + covars[[i1, i2]] ** 0.5 * 2
    if vmin >= means[i2]:
        vmin = means[i2] + covars[i2] ** 0.5 * -1
    vmin = int(max(fromlog(vmin), img.min()))
    vmax = int(min(fromlog(vmax), img.max()))

    return vmin, vmax


def main():

    if len(sys.argv) != 2:
        print("Usage: story.py image.ome.tif")
        sys.exit(1)

    path = sys.argv[1]

    sys.stderr.write(f"opening image: {path}\n")
    tiff = tifffile.TiffFile(path)
    zarray = zarr.open(tiff.series[0].levels[-1].aszarr())
    if zarray.ndim == 2:
        # FIXME This can be handled easily (promote to 3D array), we just need a
        # test file to make sure we're doing it right.
        raise Exception("Can't handle 2-dimensional images")
    elif zarray.ndim == 3:
        pass
    else:
        raise Exception(f"Can't handle {zarray.ndim}-dimensional images")
    if not np.issubdtype(zarray.dtype, np.unsignedinteger):
        raise Exception(f"Can't handle {zarray.dtype} pixel type")

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

    imax = np.iinfo(zarray.dtype).max
    for gi, idx_start in enumerate(range(0, zarray.shape[0], 4), 1):
        idx_end = min(idx_start + 4, zarray.shape[0])
        channel_numbers = range(idx_start, idx_end)
        channel_defs = []
        for ci, color in zip(channel_numbers, color_cycle):
            sys.stderr.write(f"analyzing channel {ci + 1}/{zarray.shape[0]}\n")
            img = zarray[ci]
            vmin, vmax = auto_threshold(img)
            vmin /= imax
            vmax /= imax
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
