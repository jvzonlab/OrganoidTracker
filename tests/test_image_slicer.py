from unittest import TestCase

from ai_track.imaging.image_slicer import get_slices, Slicer3d


class TestImageSlicer(TestCase):

    def test_get_slices(self):
        volume = (32, 512, 512)  # ZYX size of full image
        subsize = (32, 256, 256)  # ZYX size of the sub-image
        margin = (0, 32, 32)  # Margin inside the sub-image
        slices = list(get_slices(volume, subsize, margin))
        self.assertEquals([Slicer3d((0,0,0), (32,320,320), (0,0,0), (32,256,256)),
                           Slicer3d((0,0,192), (32,320,512), (0,0,256), (32,256,512)),
                           Slicer3d((0,192,0), (32,512,320), (0,256,0), (32,512,256)),
                           Slicer3d((0,192,192), (32,512,512), (0,256,256), (32,512,512))], slices)

    def test_cannot_slice_because_image_is_too_small(self):
        volume = (32, 256, 256)  # ZYX size of full image
        subsize = (32, 256, 256)  # ZYX size of the sub-image
        margin = (0, 32, 32)  # Margin inside the sub-image
        slices = list(get_slices(volume, subsize, margin))

        # Image is too small, make sure we get only one slice
        self.assertEquals([Slicer3d((0, 0, 0), (32, 320, 320), (0, 0, 0), (32, 256, 256))], slices)
