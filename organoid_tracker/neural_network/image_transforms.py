# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Code was taken from the TensorFlow Addons repository, which is no longer maintained.
# The code was modified to work with Keras 3.

import keras

from organoid_tracker.neural_network import Tensor, TensorLike


def compose_transforms(transforms: Tensor) -> Tensor:
    """Composes the transforms tensors.

    Args:
      transforms: List of image projective transforms to be composed. Each
        transform is length 8 (single transform) or shape (N, 8) (batched
        transforms). The shapes of all inputs must be equal, and at least one
        input must be given.

    Returns:
      A composed transform tensor. When passed to `transform` op,
          equivalent to applying each of the given transforms to the image in
          order.
    """
    assert transforms, "transforms cannot be empty"
    composed = flat_transforms_to_matrices(transforms[0])
    for tr in transforms[1:]:
        # Multiply batches of matrices.
        composed = keras.ops.matmul(composed, flat_transforms_to_matrices(tr))
    return matrices_to_flat_transforms(composed)


def flat_transforms_to_matrices(transforms: Tensor) -> Tensor:
    """Converts projective transforms to affine matrices.

    Note that the output matrices map output coordinates to input coordinates.
    For the forward transformation matrix, call `keras.ops.linalg.inv` on the result.

    Args:
      transforms: Vector of length 8, or batches of transforms with shape
        `(N, 8)`.

    Returns:
      3D tensor of matrices with shape `(N, 3, 3)`. The output matrices map the
        *output coordinates* (in homogeneous coordinates) of each transform to
        the corresponding *input coordinates*.

    Raises:
      ValueError: If `transforms` have an invalid shape.
    """
    if len(keras.ops.shape(transforms)) not in (1, 2):
        raise ValueError("Transforms should be 1D or 2D, got: %s" % transforms)
    # Make the transform(s) 2D in case the input is a single transform.
    transforms = keras.ops.reshape(transforms, keras.ops.convert_to_tensor([-1, 8]))
    num_transforms = keras.ops.shape(transforms)[0]
    # Add a column of ones for the implicit last entry in the matrix.
    return keras.ops.reshape(
        keras.ops.concatenate([transforms, keras.ops.ones([num_transforms, 1])], axis=1),
        keras.ops.convert_to_tensor([-1, 3, 3]),
    )


def matrices_to_flat_transforms(transform_matrices: Tensor) -> Tensor:
    """Converts affine matrices to projective transforms.

    Note that we expect matrices that map output coordinates to input
    coordinates. To convert forward transformation matrices,
    call `keras.ops.linalg.inv` on the matrices and use the result here.

    Args:
      transform_matrices: One or more affine transformation matrices, for the
        reverse transformation in homogeneous coordinates. Shape `(3, 3)` or
        `(N, 3, 3)`.

    Returns:
      2D tensor of flat transforms with shape `(N, 8)`, which may be passed
      into `transform` op.

    Raises:
      ValueError: If `transform_matrices` have an invalid shape.
    """
    if len(keras.ops.shape(transform_matrices.shape)) not in (2, 3):
        raise ValueError(
            "Matrices should be 2D or 3D, got: %s" % transform_matrices
        )
    # Flatten each matrix.
    transforms = keras.ops.reshape(transform_matrices, keras.ops.convert_to_tensor([-1, 9]))
    # Divide each matrix by the last entry (normally 1).
    transforms /= transforms[:, 8:9]
    return transforms[:, :8]


def angles_to_projective_transforms(angles: TensorLike, image_height: TensorLike, image_width: TensorLike) -> Tensor:
    """Returns projective transform(s) for the given angle(s).

    Args:
      angles: A scalar angle to rotate all images by, or (for batches of
        images) a vector with an angle to rotate each image in the batch. The
        rank must be statically known (the shape is not `TensorShape(None)`).
      image_height: Height of the image(s) to be transformed.
      image_width: Width of the image(s) to be transformed.

    Returns:
      A tensor of shape (num_images, 8). Projective transforms which can be
      given to `transform` op.
    """
    angle_or_angles = keras.ops.convert_to_tensor(
        angles, name="angles", dtype="float32"
    )
    if len(keras.ops.shape(angle_or_angles)) == 0:
        angles = angle_or_angles[None]
    elif len(keras.ops.shape(angle_or_angles)) == 1:
        angles = angle_or_angles
    else:
        raise ValueError("angles should have rank 0 or 1.")
    cos_angles = keras.ops.cos(angles)
    sin_angles = keras.ops.sin(angles)
    x_offset = ((image_width - 1) - (cos_angles * (image_width - 1) - sin_angles * (image_height - 1))) / 2.0
    y_offset = ((image_height - 1) - (sin_angles * (image_width - 1) + cos_angles * (image_height - 1))) / 2.0
    num_angles = keras.ops.shape(angles)[0]
    return keras.ops.concatenate(
        values=[
            cos_angles[:, None],
            -sin_angles[:, None],
            x_offset[:, None],
            sin_angles[:, None],
            cos_angles[:, None],
            y_offset[:, None],
            keras.ops.zeros((num_angles, 2), "float32"),
        ],
        axis=1,
    )
