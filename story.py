import json
import numpy as np
import ome_types
import sklearn.mixture
import sys
import threadpoolctl
import tifffile
import zarr


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
    covars = gmm.covariances_[:, 0, 0]
    _, i1, i2 = np.argsort(means)

    vmin, vmax = means[[i1, i2]] + covars[[i1, i2]] ** 0.5 * 2
    if vmin >= means[i2]:
        vmin = means[i2] + covars[i2] ** 0.5 * -1
    vmin = max(np.exp(vmin), img.min(), 0)
    vmax = min(np.exp(vmax), img.max())

    return vmin, vmax


def main():

    threadpoolctl.threadpool_limits(1)

    if len(sys.argv) != 2:
        print("Usage: story.py image.ome.tif")
        sys.exit(1)

    path = sys.argv[1]

    print(f"opening image: {path}", file=sys.stderr)
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
    signed = not np.issubdtype(zarray.dtype, np.unsignedinteger)

    print(f"reading metadata", file=sys.stderr)
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

    scale = np.iinfo(zarray.dtype).max if np.issubdtype(zarray.dtype, np.integer) else 1
    for gi, idx_start in enumerate(range(0, zarray.shape[0], 4), 1):
        idx_end = min(idx_start + 4, zarray.shape[0])
        channel_numbers = range(idx_start, idx_end)
        channel_defs = []
        for ci, color in zip(channel_numbers, color_cycle):
            print(
                f"analyzing channel {ci + 1}/{zarray.shape[0]}", file=sys.stderr
            )
            img = zarray[ci]
            if signed and img.min() < 0:
                print("  WARNING: Ignoring negative pixel values", file=sys.stderr)
            vmin, vmax = auto_threshold(img)
            vmin /= scale
            vmax /= scale
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
