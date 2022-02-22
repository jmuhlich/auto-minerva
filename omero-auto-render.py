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

import itertools
import numpy as np
from omero.gateway import BlitzGateway
from omero.model import enums as omero_enums
from omero.rtypes import rlong, rstring
import omero.scripts as scripts
import omero.util.script_utils as script_utils
import scipy.stats
import sklearn.mixture
import threadpoolctl
import traceback


# Apparently the pixel API we will use always returns pixel data as big-endian.
PIXEL_TYPES = {
    omero_enums.PixelsTypeint8: 'i1',
    omero_enums.PixelsTypeuint8: 'u1',
    omero_enums.PixelsTypeint16: '>i2',
    omero_enums.PixelsTypeuint16: '>u2',
    omero_enums.PixelsTypeint32: '>i4',
    omero_enums.PixelsTypeuint32: '>u4',
    omero_enums.PixelsTypefloat: '>f4',
    omero_enums.PixelsTypedouble: '>f8',
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


def process_image(omero_image):

    channels = omero_image.getChannels()
    pixels = omero_image.getPrimaryPixels()
    pixels_type = pixels.getPixelsType().value
    try:
        dtype = np.dtype(PIXEL_TYPES[pixels_type])
    except KeyError:
        raise Exception(f"Can't handle PixelsType: {pixels_type}") from None
    signed = not np.issubdtype(dtype, np.unsignedinteger)

    pix = image._conn.c.sf.createRawPixelsStore()
    pid = image.getPixelsId()

    try:
        pix.setPixelsId(pid, False)

        # Get smallest resolution that's at least 200 in both dimensions, or
        # the largest level if all resolutions are smaller than 200.
        resolutions = [
            (i, (desc.sizeX, desc.sizeY))
            for i, desc in enumerate(pix.getResolutionDescriptions(), 1)
        ]
        print("Image resolutions:")
        for i, shape in resolutions:
            print(f"    {i}: {shape[0]} x {shape[1]}")
        try:
            level, (w, h) = next(
                (i, shape)
                for i, shape in reversed(resolutions)
                if all(s >= 200 for s in shape)
            )
        except StopIteration:
            level, (w, h) = resolutions[0]
        print(f"Using level {level} ({w} x {h})")
        # It appears that setResolutionLevel numbers the levels in the opposite
        # order as getResolutionDescriptions.
        pix.setResolutionLevel(len(resolutions) - level)

        print("Auto-detecting limits for all channels:")
        windows = []
        active = []
        for c, channel in enumerate(channels):
            buf = pix.getPlane(0, c, 0)
            img = np.frombuffer(buf, dtype=dtype)
            img = img.reshape((h, w))
            vmin, vmax = auto_threshold(img)
            if np.issubdtype(dtype, np.integer):
                vmin = round(vmin)
                vmax = round(vmax)
            print(f"    {c + 1}: {vmin:g} - {vmax:g}")
            windows.append((vmin, vmax))
            if channel.isActive():
                active.append(c + 1)
        omero_image.setActiveChannels(range(1, len(channels) + 1), windows=windows)
        omero_image.setActiveChannels(active)
        omero_image.saveDefaults()

    finally:
        pix.close()


if __name__ == "__main__":

    dataTypes = [rstring('Image')]

    client = scripts.client(
        "Auto-render",
        """Computes image rendering settings using a Gaussian Mixture Model.""",

        scripts.String(
            "Data_Type", optional=False, grouping="1",
            description="Choose specific Images or an entire Dataset",
            values=[rstring('Image'), rstring('Dataset')], default="Image",
        ),
        scripts.List(
            "IDs", optional=False, grouping="2",
            description="List of Image or Dataset IDs to process."
        ).ofType(rlong(0)),
        version="0.1",
        authors=["Jeremy Muhlich"],
        institutions=["Harvard Medical School Laboratory of Systems Pharmacology"],
        contact="jmuhlich@gmail.com",
    )

    threadpoolctl.threadpool_limits(1)

    try:
        script_params = client.getInputs(unwrap=True)
        conn = BlitzGateway(client_obj=client)

        objects, message = script_utils.get_objects(conn, script_params)
        if message:
            print(message)
        if script_params["Data_Type"] == "Image":
            images = objects
        else:
            images = list(
                itertools.chain.from_iterable(
                    dataset.listChildren() for dataset in objects
                )
            )

        num_errors = 0
        for image in images:
            if image.getSizeT() > 1 or image.getSizeZ() > 1:
                print(f"Image:{image.id} : Can't handle SizeT>1 or SizeZ>1 yet")
                num_errors += 1
                continue
            print()
            print(f"Processing Image:{image.id}")
            try:
                process_image(image)
            except:
                traceback.print_exc()
                num_errors += 1
            print("-" * 20)
        num_successes = len(images) - num_errors
        message = f"Success: {num_successes} / Failure: {num_errors}" if num_errors else "Success"
        client.setOutput("Message", rstring(message))
    finally:
        client.closeSession()
