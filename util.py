import numpy as np
import skimage.util.dtype
import sklearn.mixture


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
    vmin = max(fromlog(vmin), img.min())
    vmax = min(fromlog(vmax), img.max())

    return vmin, vmax


def rescale(img, vmin, vmax):

    assert img.ndim == 2

    dtype = img.dtype
    img = (img.astype(np.float32) - vmin) / (vmax - vmin)
    img = np.clip(img, 0, 1)
    img = skimage.util.dtype.convert(img, dtype)
    return img
