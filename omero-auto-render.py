#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Copyright (C) <year> Open Microscopy Environment.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""
Set rendering settings using a Gaussian Mixture Model
"""

import traceback
from omero.gateway import BlitzGateway
from omero.rtypes import rlong, rstring
from omero.model import enums as omero_enums
import omero.util.script_utils as script_utils
import omero.scripts as scripts
import numpy as np
import threadpoolctl

PIXEL_TYPES = {
    omero_enums.PixelsTypeint8: np.int8,
    omero_enums.PixelsTypeuint8: np.uint8,
    omero_enums.PixelsTypeint16: np.int16,
    omero_enums.PixelsTypeuint16: np.uint16,
    omero_enums.PixelsTypeint32: np.int32,
    omero_enums.PixelsTypeuint32: np.uint32,
    omero_enums.PixelsTypefloat: np.float32,
    omero_enums.PixelsTypedouble: np.float64,
}


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


def threshold_channel(pix, w, h, c):
    buf = pix.getTile(0, c, 0, 0, 0, w, h)
    img = np.frombuffer(buf, dtype=dtype)
    img = tile.reshape((h, w))
    return auth_threshold(img)


def process_image(omero_image):

    pixels = omero_image.getPrimaryPixels()
    size_c = omero_image.getSizeC()
    pixels_type = pixels.getPixelsType().value
    try:
        dtype = PIXEL_TYPES[pixels_type]
    except KeyError:
        raise Exception(f"Can't handle PixelsType: {pixels_type}") from None
    signed = not np.issubdtype(dtype, np.unsignedinteger)

    pix = image._conn.c.sf.createRawPixelsStore()
    pid = image.getPixelsId()

    try:
        pix.setPixelsId(pid, False)

        # Get smallest pyramid level that's at least 200 in both dimensions,
        # or largest level if none.
        levels = [
            (i, (s.sizeX, s.sizeY))
            for i, s in enumerate(pix.getResolutionDescriptions())
        ]
        print(levels)
        try:
            level = next(
                i for i, shape in levels if all(s >= 200 for s in shape)
            )
        except StopIteration:
            level = len(shapes) - 1
        w, h = shapes[level]
        print(f"level={level}/{len(shapes)} shape=({w} x {h})")

        pix.setResolutionLevel(level)
        thresholds = [threshold_channel(pix, w, h, c) for c in range(size_c)]
    finally:
        pix.close()

    print(thresholds)


if __name__ == "__main__":

    dataTypes = [rstring('Image')]

    client = scripts.client(
        'Example.py', """This script ...""",

        scripts.String(
            "Data_Type", optional=False, grouping="1",
            description="Choose source of images (only Image supported)",
            values=dataTypes, default="Image"),

        scripts.List(
            "IDs", optional=False, grouping="2",
            description="List of Image IDs to process.").ofType(rlong(0)),

        version="0.1",
        authors=["Author 1", "Author 2"],
        institutions=["The OME Consortium"],
        contact="ome-users@lists.openmicroscopy.org.uk",
    )

    threadpoolctl.threadpool_limits(1)

    try:
        script_params = client.getInputs(unwrap=True)
        conn = BlitzGateway(client_obj=client)

        images, message = script_utils.get_objects(conn, script_params)
        if message:
            print(message)
        num_errors = 0
        if images:
            for image in images:
                if image.getSizeT() > 1 or image.getSizeZ() > 1:
                    print(f"Image:{image.id} : Can't handle SizeT>1 or SizeZ>1 yet")
                    num_errors += 1
                    continue
                print(f"Image:{image.id} : Processing...")
                try:
                    process_image(image)
                except:
                    traceback.print_exc()
                    num_errors += 1
                print("-" * 20)
                print()
        num_successes = len(images) - num_errors
        message = f"Success: {num_successes} / Failure: {num_errors}" if num_errors else "Success"
        client.setOutput("Message", rstring(message))
    finally:
        client.closeSession()
