#!/usr/bin/env python

# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.
# * Neither the name of Willow Garage, Inc. nor the names of its
# contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Serialization of sensor_msgs.PointCloud2 messages.

Includes object detection via clustering for targeted exploration.

Author: Tim Field
ROS2 port by Sebastian Grans (2020)
Enhancements by Tyler Raettig (2024)
"""


import sys
import math
import struct
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sklearn.cluster import DBSCAN

# Datatypes mapping for PointCloud2 fields
_DATATYPES = {
    PointField.INT8: ('b', 1),
    PointField.UINT8: ('B', 1),
    PointField.INT16: ('h', 2),
    PointField.UINT16: ('H', 2),
    PointField.INT32: ('i', 4),
    PointField.UINT32: ('I', 4),
    PointField.FLOAT32: ('f', 4),
    PointField.FLOAT64: ('d', 8)
}

def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    """
    Read points from a sensor_msgs.PointCloud2 message.
    """
    assert isinstance(cloud, PointCloud2), 'cloud is not a sensor_msgs.msg.PointCloud2'
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = cloud.width, cloud.height, \
                                                       cloud.point_step, cloud.row_step, \
                                                       cloud.data, math.isnan
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                if not any(isnan(pv) for pv in p):
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    if not any(isnan(pv) for pv in p):
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step

def detect_objects(cloud, detection_range=2.0, eps=0.2, min_samples=10):
    """
    Detect potential objects in the point cloud data by clustering points.

    Parameters:
    - cloud: PointCloud2 message
    - detection_range: Max range to consider points for clustering

    Returns:
    - List of dictionaries with 'center' and 'size' for each cluster
    """
    points = np.array(list(read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)))
    points = points[(points[:, 0]**2 + points[:, 1]**2)**0.5 < detection_range]  # Filter points within range

    # Perform clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :2])  # Cluster based on x, y
    clusters = []
    for label in set(clustering.labels_):
        if label == -1:
            continue  # Noise points
        cluster_points = points[clustering.labels_ == label]
        clusters.append({
            "center": cluster_points.mean(axis=0),  # Calculate cluster center
            "size": len(cluster_points)
        })

    return clusters

def create_cloud(header, fields, points):
    """
    Create a sensor_msgs.PointCloud2 message.
    """
    cloud_struct = struct.Struct(_get_struct_fmt(False, fields))
    data = bytearray()
    for p in points:
        data.extend(cloud_struct.pack(*p))

    return PointCloud2(
        header=header,
        height=1,
        width=len(points),
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=cloud_struct.size,
        row_step=cloud_struct.size * len(points),
        data=bytes(data)
    )

def create_cloud_xyz32(header, points):
    """
    Create a sensor_msgs.PointCloud2 message with 3 float32 fields (x, y, z).
    """
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
    ]
    return create_cloud(header, fields, points)

def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'
    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print(f'Skipping unknown PointField datatype [{field.datatype}]', file=sys.stderr)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt