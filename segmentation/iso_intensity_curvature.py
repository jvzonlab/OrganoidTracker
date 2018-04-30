
import math
from typing import Optional

import numpy
from numpy import ndarray
import cv2
import matplotlib.pyplot as plt


class ImageDerivatives:
    """Stores various image derivatives. This object is intended to be reused to conserve memory. Call calculate() to
    populate all the fields in this class.
    """
    dimage_dx: Optional[ndarray] = None
    dimage_dy: Optional[ndarray] = None
    dimage_dz: Optional[ndarray] = None

    d2image_dxdx: Optional[ndarray] = None
    d2image_dydy: Optional[ndarray] = None
    d2image_dzdz: Optional[ndarray] = None

    d2image_dxdy: Optional[ndarray] = None
    d2image_dxdz: Optional[ndarray] = None
    d2image_dydz: Optional[ndarray] = None

    def calculate(self, image_stack: ndarray, blur_radius: int):
        """Calculates a variety of Sobel derivatives.
        image_stack: 3D image
        blur_radius: extended kernel size for Sobel derivative. Larger is more smoothing. Must be 1, 3, 5, or 7.
        """
        self.dimage_dx = self._sobel_x(image_stack, blur_radius, self.dimage_dx)
        self.dimage_dy = self._sobel_y(image_stack, blur_radius, self.dimage_dy)
        self.dimage_dz = self._sobel_z(image_stack, blur_radius, self.dimage_dz)

        self.d2image_dxdx = self._sobel_x(self.dimage_dx, blur_radius, self.d2image_dxdx)
        self.d2image_dydy = self._sobel_y(self.dimage_dy, blur_radius, self.d2image_dydy)
        self.d2image_dzdz = self._sobel_z(self.dimage_dz, blur_radius, self.d2image_dzdz)

        self.d2image_dxdy = self._sobel_y(self.dimage_dx, blur_radius, self.d2image_dxdy)
        self.d2image_dxdz = self._sobel_z(self.dimage_dx, blur_radius, self.d2image_dxdz)
        self.d2image_dydz = self._sobel_z(self.dimage_dy, blur_radius, self.d2image_dydz)


    def _sobel_x(self, image_stack: ndarray, blur_radius: int, out: Optional[ndarray]) -> ndarray:
        if out is None:
            out = numpy.empty_like(image_stack, dtype=numpy.float32)

        slice_count = image_stack.shape[0]
        for z in range(slice_count):
            cv2.Sobel(image_stack[z], cv2.CV_32F, 1, 0, ksize=blur_radius, dst=out[z], borderType=cv2.BORDER_CONSTANT)

        return out

    def _sobel_y(self, image_stack: ndarray, blur_radius: int, out: Optional[ndarray]) -> ndarray:
        if out is None:
            out = numpy.empty_like(image_stack, dtype=numpy.float32)

        slice_count = image_stack.shape[0]
        for z in range(slice_count):
            cv2.Sobel(image_stack[z], cv2.CV_32F, 0, 1, ksize=blur_radius, dst=out[z], borderType=cv2.BORDER_CONSTANT)

        return out

    def _sobel_z(self, image_stack: ndarray, blur_radius: int, out: Optional[ndarray]) -> ndarray:
        # OpenCV does not provide us with a sobelZ derivative.
        # So we need to transpose the image first, and then do a sobel_x.
        #
        # Explanation:
        # Before the transpose, each layer previously was an image with these axis:
        # Y
        # |
        # +-- X
        # But now it is:
        # Y
        # |
        # +-- Z
        # So taking the sobelX of these images actually gives the Z-derivative of the original images.
        #
        # One quirck of OpenCV is that it does not accept transposed images of numpy as output. So we need to create an
        # empty array as output array, and then transpose this array after OpenCV has filled it.
        transposed = image_stack.transpose()
        if out is not None:
            transposed_out = out.transpose()
        else:
            transposed_out = numpy.empty(transposed.shape, dtype=numpy.float32)

        self._sobel_x(transposed, blur_radius, transposed_out)
        return transposed_out.transpose()


def get_negative_gaussian_curvatures(image_stack: ndarray, derivatives: ImageDerivatives, out: ndarray,
                                     blur_radius: int = 5, dilate:bool = True):
    """Gets all positions with a negative Gaussian curvature of the iso-intensity surfaces. Those positions are marked
    with 255, the others are set to 0.
    out: uint8 image"""
    blurred = numpy.empty_like(image_stack)
    for z in range(blurred.shape[0]):
        cv2.GaussianBlur(image_stack[z], (blur_radius * 2 + 1, blur_radius * 2 + 1), 0, dst=blurred[z])

    derivatives.calculate(blurred, blur_radius)

    return_value = _get_iic(derivatives)

    # Write to out
    out.fill(255)
    out[return_value < 0] = 0
    if dilate:
        dilation_kernel = numpy.ones((blur_radius, blur_radius), dtype=numpy.uint8)
        for z in range(out.shape[0]):
            out[z] = cv2.dilate(out[z], dilation_kernel, iterations=1)
    image_stack[out == 0] = image_stack.max() / 5


def _get_iic(derivatives: ImageDerivatives) -> ndarray:
    """Calculates the minimal iso-intensity curvature. Algorithm described in Supplementary Information 2 of Toyoshima,
    Yu, et al. "Accurate Automatic Detection of Densely Distributed Cell Nuclei in 3D Space." PLoS computational biology
    12.6 (2016).

    See also the Matlab code at https://nl.mathworks.com/matlabcentral/fileexchange/11168-surface-curvature , or the
    Wikipedia page https://en.wikipedia.org/wiki/Gaussian_curvature

    Some code has been commented out, as we do not use that code for our purposes. However, for comparison with
    literature, and for context, it is still useful.
    """
    fx = derivatives.dimage_dx
    fy = derivatives.dimage_dy
    fz = derivatives.dimage_dz
    fxy = derivatives.d2image_dxdy
    fxz = derivatives.d2image_dxdz
    fyz = derivatives.d2image_dydz
    fxx = derivatives.d2image_dxdx
    fyy = derivatives.d2image_dydy
    fzz = derivatives.d2image_dzdz

    A = (fx**2 + fy**2 + fz**2) / (fz**2)
    sqrtA_times_fz_topowerof_3 = numpy.sqrt(A) * fz**3

    # First fundamental coefficients of the surface
    # E = (fx**2 + fz**2) / (fz**2)
    # F = (fx * fy) / (fz**2)
    # G = (fy**2 + fz**2) / (fz**2)

    # Second fundamental coefficients of the surface
    L = (2 * fx * fz * fxz - fx**2 * fzz - fz**2 * fxx) / sqrtA_times_fz_topowerof_3
    M = (fx * fz * fyz + fy * fz * fxz - fx * fy * fzz - fz**2 * fxy) / sqrtA_times_fz_topowerof_3
    N = (2 * fy * fz * fyz - fy**2 * fzz - fz**2 * fyy) / sqrtA_times_fz_topowerof_3

    # Gaussian
    K = (L * N - M**2) / A

    # The mean and principal curvatures seem to remain unused, even though the formulas are given in the article
    # H = (E * N - 2 * F * M + G * L) / (2 * A)
    #
    # # Principal curvatures
    # if H**2 - K < 0:
    #     return numpy.NaN
    # k1 = H + math.sqrt(H**2 - K)
    # k2 = K/k1
    return K


