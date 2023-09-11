"""Image loader for IMS files, produced by Imaris.

The _ImsReader class is based on
https://github.com/CBI-PITT/imaris_ims_file_reader/blob/34a07b62121e4bc03b51d9f0171ae6ea0c8272b7/imaris_ims_file_reader/ims.py


Copyright (c) 2021, Alan M Watson
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of imaris-ims-file-reader nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
import itertools
import os
from typing import Tuple, Optional

import h5py
import numpy as np
import warnings

from numpy import ndarray
from skimage import io, img_as_float32, img_as_uint, img_as_ubyte
from skimage.transform import rescale

from organoid_tracker.core import TimePoint
from organoid_tracker.core.experiment import Experiment
from organoid_tracker.core.image_loader import ImageLoader, ImageChannel
from organoid_tracker.core.resolution import ImageResolution


def load_from_ims_file(experiment: Experiment, file_name: str, min_time_point: Optional[int] = None,
                       max_time_point: Optional[int] = None):
    """Loads the image seriers into the images object using the file_name and pattern settings."""
    if not os.path.exists(file_name):
        print("Failed to load \"" + file_name + "\" - file does not exist")
        return
    image_loader = _ImsImageLoader(file_name, min_time_point, max_time_point)
    experiment.images.image_loader(image_loader)

    if experiment.images.resolution(allow_incomplete=True).is_incomplete():
        resolution_zyx = image_loader.get_spatial_resolution_um()
        if len(resolution_zyx) == 2:
            resolution_zyx = (1,) + resolution_zyx
        resolution_time_m = image_loader.get_time_resolution_m()
        experiment.images.set_resolution(
            ImageResolution(resolution_zyx[2], resolution_zyx[1], resolution_zyx[0], resolution_time_m))


    experiment_name = os.path.basename(file_name)
    if experiment_name.lower().endswith(".ims"):
        experiment_name = experiment_name[:-4]
    experiment.name.provide_automatic_name(experiment_name)


class _ImsReader:
    def __init__(self, file, ResolutionLevelLock=0, write=False, cache_location=None, mem_size=None, disk_size=2000,
                 squeeze_output=True,
                 resolution_decimal_places=3
                 ):

        ##  mem_size = in gigabytes that remain FREE as cache fills
        ##  disk_size = in gigabytes that remain FREE as cache fills
        ## NOTE: Caching is currently not implemented.

        self.filePathComplete = file
        self.write = write
        self.open()
        self.filePathBase = os.path.split(file)[0]
        self.fileName = os.path.split(file)[1]
        self.fileExtension = os.path.splitext(self.fileName)[1]
        if cache_location is None and mem_size is None:
            self.cache = None
        else:
            self.cache = True
        self.cache_location = cache_location
        self.disk_size = disk_size * 1e9 if disk_size is not None else None
        self.mem_size = mem_size * 1e9 if mem_size is not None else None
        self.memCache = {}
        self.squeeze_output = squeeze_output
        self.cacheFiles = []
        self.metaData = {}
        self.ResolutionLevelLock = ResolutionLevelLock
        self.resolution_decimal_places = resolution_decimal_places

        resolution_0 = self.dataset['ResolutionLevel 0']
        time_point_0 = resolution_0['TimePoint 0']
        channel_0 = time_point_0['Channel 0']
        data = channel_0['Data']

        self.ResolutionLevels = len(self.dataset)
        self.TimePoints = len(resolution_0)
        self.Channels = len(time_point_0)

        # Round these values later because they are used below to calculate resolutions for other resolution levels
        self.resolution = (
            (self.read_numerical_dataset_attr('ExtMax2') - self.read_numerical_dataset_attr('ExtMin2'))
            / self.read_numerical_dataset_attr('Z'),

            (self.read_numerical_dataset_attr('ExtMax1') - self.read_numerical_dataset_attr('ExtMin1'))
            / self.read_numerical_dataset_attr('Y'),

            (self.read_numerical_dataset_attr('ExtMax0') - self.read_numerical_dataset_attr('ExtMin0'))
            / self.read_numerical_dataset_attr('X')
        )

        self.shape = (
            self.TimePoints,
            self.Channels,
            int(self.read_attribute('DataSetInfo/Image', 'Z')),
            int(self.read_attribute('DataSetInfo/Image', 'Y')),
            int(self.read_attribute('DataSetInfo/Image', 'X'))
        )

        self.chunks = (1, 1, data.chunks[0], data.chunks[1], data.chunks[2])
        self.dtype = data.dtype
        self.shapeH5Array = data.shape

        for r, t, c in itertools.product(range(self.ResolutionLevels), range(self.TimePoints),
                                         range(self.Channels)):
            # print('{},{},{}'.format(r,t,c))
            location_attr = self.location_generator(r, t, c, data='attrib')
            location_data = self.location_generator(r, t, c, data='data')

            # Collect attribute info
            self.metaData[r, t, c, 'shape'] = (
                1,
                1,
                int(self.read_attribute(location_attr, 'ImageSizeZ')),
                int(self.read_attribute(location_attr, 'ImageSizeY')),
                int(self.read_attribute(location_attr, 'ImageSizeX'))
            )

            # Collect Resolution information
            self.metaData[r, t, c, 'resolution'] = tuple(
                [float((origShape / newShape) * origRes) for origRes, origShape, newShape in
                 zip(self.resolution, self.shape[-3:], self.metaData[r, t, c, 'shape'][-3:])]
            )

            # Round Resolution
            self.metaData[r, t, c, 'resolution'] = tuple(
                [round(x, self.resolution_decimal_places) if self.resolution_decimal_places is not None else x for x in
                 self.metaData[r, t, c, 'resolution']]
            )

            try:
                self.metaData[r, t, c, 'HistogramMax'] = int(float(self.read_attribute(location_attr, 'HistogramMax')))
            except:
                warntxt = '''HistogramMax value is not present for resolution {}, time {}, channel {}. 
                              This may cause compatibility issues with programs that use imaris_ims_file_reader'''.format(
                    r, t, c)
                warnings.warn(warntxt)
            try:
                self.metaData[r, t, c, 'HistogramMin'] = int(float(self.read_attribute(location_attr, 'HistogramMin')))
            except:
                warntxt = '''HistogramMin value is not present for resolution {}, time {}, channel {}.
                              This may cause compatibility issues with programs that use imaris_ims_file_reader'''.format(
                    r, t, c)
                warnings.warn(warntxt)

            # Collect dataset info
            self.metaData[r, t, c, 'chunks'] = (
                1, 1, self.hf[location_data].chunks[0], self.hf[location_data].chunks[1],
                self.hf[location_data].chunks[2])
            self.metaData[r, t, c, 'shapeH5Array'] = self.hf[location_data].shape
            self.metaData[r, t, c, 'dtype'] = self.hf[location_data].dtype

        # Round Resolution level0
        self.resolution = tuple(
            [round(x, self.resolution_decimal_places) if self.resolution_decimal_places is not None else x for x in
             self.resolution]
        )

        if isinstance(self.ResolutionLevelLock, int):
            self.change_resolution_lock(self.ResolutionLevelLock)

    def change_resolution_lock(self, ResolutionLevelLock):
        ## Pull information from the only required dataset at each resolution
        ## which is time_point=0, channel=0
        self.ResolutionLevelLock = ResolutionLevelLock
        self.shape = (
            self.TimePoints,
            self.Channels,
            self.metaData[self.ResolutionLevelLock, 0, 0, 'shape'][-3],
            self.metaData[self.ResolutionLevelLock, 0, 0, 'shape'][-2],
            self.metaData[self.ResolutionLevelLock, 0, 0, 'shape'][-1]
        )
        self.ndim = len(self.shape)
        self.chunks = self.metaData[self.ResolutionLevelLock, 0, 0, 'chunks']
        self.shapeH5Array = self.metaData[self.ResolutionLevelLock, 0, 0, 'shapeH5Array']
        self.resolution = self.metaData[self.ResolutionLevelLock, 0, 0, 'resolution']
        self.dtype = self.metaData[self.ResolutionLevelLock, 0, 0, 'dtype']

    # def __enter__(self):
    #     print('Opening file: {}'.format(self.filePathComplete))
    #     self.hf = h5py.File(self.filePathComplete, 'r')
    #     self.dataset = self.hf['DataSet']

    # def __exit__(self, type, value, traceback):
    #     ## Implement flush?
    #     self.hf.close()
    #     self.hf = None

    def open(self):
        if self.write == False:
            # print('Opening readonly file: {} \n'.format(self.filePathComplete))
            self.hf = h5py.File(self.filePathComplete, 'r', swmr=True)
            self.dataset = self.hf['DataSet']
            # print('OPENED file: {} \n'.format(self.filePathComplete))
        elif self.write == True:
            print('Opening writeable file: {} \n'.format(self.filePathComplete))
            self.hf = h5py.File(self.filePathComplete, 'a', swmr=True)
            self.dataset = self.hf['DataSet']
            # print('OPENED file: {} \n'.format(self.filePathComplete))

    def __del__(self):
        self.close()

    def close(self):
        ## Implement flush?
        if self.write == True:
            print('Flushing Buffers to Disk')
            self.hf.flush()
        print('Closing file: {} \n'.format(self.filePathComplete))
        if self.hf is not None:
            self.hf.close()
        self.hf = None
        self.dataset = None
        # print('CLOSED file: {} \n'.format(self.filePathComplete))

    def __getitem__(self, key):
        """
        All ims class objects are represented as shape (TCZYX)
        An integer only slice will return the entire timepoint (T) data as a numpy array

        Any other variation on slice will be coerced to 5 dimensions and
        extract that array

        If a 6th dimensions is present in the slice, dim[0] is assumed to be the resolutionLevel
        this will be used when choosing which array to extract.  Otherwise ResolutionLevelLock
        will be obeyed.  If ResolutionLevelLock is == None - default resolution is 0 (full-res)
        and a slice of 5 or less dimensions will extract information from resolutionLevel 0.

        ResolutionLevelLock is used to focus on a specific resoltion level.
        This option enables a 5D slicing to lock on to a specified resolution level.
        """

        res, key = self.transform_key(key)

        slice_returned = self.get_slice(
            r=res if res is not None else 0,  # Force ResolutionLock of None to be 0 when slicing
            t=self.slice_fixer(key[0], 't', res=res),
            c=self.slice_fixer(key[1], 'c', res=res),
            z=self.slice_fixer(key[2], 'z', res=res),
            y=self.slice_fixer(key[3], 'y', res=res),
            x=self.slice_fixer(key[4], 'x', res=res)
        )
        return slice_returned

    def __setitem__(self, key, newValue):

        if self.write == False:
            print("""
                  IMS File can not be written to.
                  imsClass.write = False.
                  """)
            pass

        res, key = self.transform_key(key)

        self.set_slice(
            r=res if res is not None else 0,  # Force ResolutionLock of None to be 0 when slicing
            t=self.slice_fixer(key[0], 't', res=res),
            c=self.slice_fixer(key[1], 'c', res=res),
            z=self.slice_fixer(key[2], 'z', res=res),
            y=self.slice_fixer(key[3], 'y', res=res),
            x=self.slice_fixer(key[4], 'x', res=res),
            newData=newValue
        )

    def transform_key(self, key):

        res = self.ResolutionLevelLock

        if not isinstance(key, slice) and not isinstance(key, int) and len(key) == 6:
            res = key[0]
            if res >= self.ResolutionLevels:
                raise ValueError('Layer is larger than the number of ResolutionLevels')
            key = tuple((x for x in key[1::]))

        # All slices will be converted to 5 dims and placed into a tuple
        if isinstance(key, slice):
            key = [key]

        if isinstance(key, int):
            key = [slice(key)]
        # Convert int/slice mix to a tuple of slices
        elif isinstance(key, tuple):
            key = tuple((slice(x) if isinstance(x, int) else x for x in key))

        key = list(key)
        while len(key) < 5:
            key.append(slice(None))
        key = tuple(key)

        return res, key

    def read_numerical_dataset_attr(self, attrib):
        return float(self.read_attribute('DataSetInfo/Image', attrib))

    def slice_fixer(self, slice_object, dim, res):
        """
        Converts slice.stop == None to the origional image dims
        dim = dimension.  should be str: r,t,c,z,y,x

        Always returns a fully filled slice object (ie NO None)

        Negative slice values are not implemented yet self[:-5]

        Slicing with lists (fancy) is not implemented yet self[[1,2,3]]
        """

        if res is None:
            res = 0

        dims = {'r': self.ResolutionLevels,
                't': self.TimePoints,
                'c': self.Channels,
                'z': self.metaData[(res, 0, 0, 'shape')][-3],
                'y': self.metaData[(res, 0, 0, 'shape')][-2],
                'x': self.metaData[(res, 0, 0, 'shape')][-1]
                }

        if (slice_object.stop is not None) and (slice_object.stop > dims[dim]):
            raise ValueError('The specified stop dimension "{}" in larger than the dimensions of the \
                                 origional image'.format(dim))
        if (slice_object.start is not None) and (slice_object.start > dims[dim]):
            raise ValueError('The specified start dimension "{}" in larger than the dimensions of the \
                                 origional image'.format(dim))

        if isinstance(slice_object.stop, int) and slice_object.start == None and slice_object.step == None:
            return slice(
                slice_object.stop,
                slice_object.stop + 1,
                1 if slice_object.step is None else slice_object.step
            )

        if slice_object == slice(None):
            return slice(0, dims[dim], 1)

        if slice_object.step is None:
            slice_object = slice(slice_object.start, slice_object.stop, 1)

        if slice_object.stop is None:
            slice_object = slice(
                slice_object.start,
                dims[dim],
                slice_object.step
            )

        # TODO: Need to reevaluate if this last statement is still required
        if isinstance(slice_object.stop, int) and slice_object.start is None:
            slice_object = slice(
                max(0, slice_object.stop - 1),
                slice_object.stop,
                slice_object.step
            )

        return slice_object

    @staticmethod
    def location_generator(r, t, c, data='data'):
        """
        Given R, T, C, this function will generate a path to data in an imaris file
        default data == 'data' the path will reference with array of data
        if data == 'attrib' the bath will reference the channel location where attributes are stored
        """

        location = 'DataSet/ResolutionLevel {}/TimePoint {}/Channel {}'.format(r, t, c)
        if data == 'data':
            location = location + '/Data'
        return location

    def read_attribute(self, location, attrib):
        """
        Location should be specified as a path:  for example
        'DataSet/ResolutionLevel 0/TimePoint 0/Channel 1'

        attrib is a string that defines the attribute to extract: for example
        'ImageSizeX'
        """
        return str(self.hf[location].attrs[attrib], encoding='ascii')

    def get_slice(self, r, t, c, z, y, x):
        """
        IMS stores 3D datasets ONLY with Resolution, Time, and Color as 'directory'
        structure writing HDF5.  Thus, data access can only happen across dims XYZ
        for a specific RTC.
        """

        incomingSlices = (r, t, c, z, y, x)
        t_size = list(range(self.TimePoints)[t])
        c_size = list(range(self.Channels)[c])
        z_size = len(range(self.metaData[(r, 0, 0, 'shape')][-3])[z])
        y_size = len(range(self.metaData[(r, 0, 0, 'shape')][-2])[y])
        x_size = len(range(self.metaData[(r, 0, 0, 'shape')][-1])[x])

        output_array = np.zeros((len(t_size), len(c_size), z_size, y_size, x_size), dtype=self.dtype)

        for idxt, t in enumerate(t_size):
            for idxc, c in enumerate(c_size):
                ## Below method is faster than all others tried
                d_set_string = self.location_generator(r, t, c, data='data')
                self.hf[d_set_string].read_direct(output_array, np.s_[z, y, x], np.s_[idxt, idxc, :, :, :])

        # with h5py.File(self.filePathComplete, 'r') as hf:
        #     for idxt, t in enumerate(t_size):
        #         for idxc, c in enumerate(c_size):
        #             # Old method below
        #             d_set_string = self.location_generator(r, t, c, data='data')
        #             output_array[idxt, idxc, :, :, :] = hf[d_set_string][z, y, x]

        """
        The return statements can provide some specific use cases for when the 
        class is providing data to Napari.

        Currently, a custom print statement provides visual feed back that 
        data are loading and what specific data is requested / returned

        The napari_imaris_loader currently hard codes os.environ["NAPARI_ASYNC"] == '1'
        """

        # if "NAPARI_ASYNC" in os.environ and os.environ["NAPARI_ASYNC"] == '1':
        #     # output_array = np.squeeze(output_array)
        #     print('Slices Requested: {} / Shape returned: {} \n'.format(incomingSlices,output_array.shape))
        #     return output_array
        # # elif "NAPARI_OCTREE" in os.environ and os.environ["NAPARI_OCTREE"] == '1':
        # #     return output_array
        # else:
        if self.squeeze_output:
            return np.squeeze(output_array)
        else:
            return output_array

    def set_slice(self, r, t, c, z, y, x, newData):
        """
        IMS stores 3D datasets ONLY with Resolution, Time, and Color as 'directory'
        structure writing HDF5.  Thus, data access can only happen across dims XYZ
        for a specific RTC.
        """

        # incomingSlices = (r,t,c,z,y,x)
        t_size = list(range(self.TimePoints)[t])
        c_size = list(range(self.Channels)[c])
        z_size = len(range(self.metaData[(r, 0, 0, 'shape')][-3])[z])
        y_size = len(range(self.metaData[(r, 0, 0, 'shape')][-2])[y])
        x_size = len(range(self.metaData[(r, 0, 0, 'shape')][-1])[x])

        # if isinstance(newData,int):
        toWrite = np.zeros((len(t_size), len(c_size), z_size, y_size, x_size), dtype=self.dtype)
        toWrite[:] = newData
        print(toWrite.shape)
        print(toWrite)

        print(t_size)
        print(t_size)
        for idxt, t in enumerate(t_size):
            for idxc, c in enumerate(c_size):
                ## Below method is faster than all others tried
                d_set_string = self.location_generator(r, t, c, data='data')
                self.hf[d_set_string].write_direct(toWrite, np.s_[idxt, idxc, :, :, :], np.s_[z, y, x])

    def dtypeImgConvert(self, image):
        """
        Convert any numpy image to the dtype of the original ims file
        """
        if self.dtype == float or self.dtype == np.float32:
            return img_as_float32(image)
        elif self.dtype == np.uint16:
            return img_as_uint(image)
        elif self.dtype == np.uint8:
            return img_as_ubyte(image)

    def projection(self, projection_type,
                   time_point=None, channel=None, z=None, y=None, x=None, resolution_level=0):
        """ Create a min or max projection across a specified (time_point,channel,z,y,x) space.

        projection_type = STR: 'min', 'max', 'mean',
        time_point = INT,
        channel = INT,
        z = tuple (zStart, zStop),
        y = None or (yStart,yStop),
        z = None or (xStart,xStop)
        resolution_level = INT >=0 : 0 is the highest resolution
        """

        assert projection_type == 'max' or projection_type == 'min' or projection_type == 'mean'

        # Set defaults
        resolution_level = 0 if resolution_level == None else resolution_level
        time_point = 0 if time_point == None else time_point
        channel = 0 if channel == None else channel

        if z is None:
            z = range(self.metaData[(resolution_level, time_point, channel, 'shape')][-3])
        elif isinstance(z, tuple):
            z = range(z[0], z[1], 1)

        if y is None:
            y = slice(0, self.metaData[(resolution_level, time_point, channel, 'shape')][-2], 1)
        elif isinstance(z, tuple):
            y = slice(y[0], y[1], 1)

        if x is None:
            x = slice(0, self.metaData[(resolution_level, time_point, channel, 'shape')][-1], 1)
        elif isinstance(z, tuple):
            x = slice(y[0], y[1], 1)

        image = None
        for num, z_layer in enumerate(z):

            print('Reading layer ' + str(num) + ' of ' + str(z))
            if image is None:
                image = self[resolution_level, time_point, channel, z_layer, y, x]
                print(image.dtype)
                if projection_type == 'mean':
                    image = img_as_float32(image)
            else:
                imageNew = self[resolution_level, time_point, channel, z_layer, y, x]

                print('Incoroprating layer ' + str(num) + ' of ' + str(z))

                if projection_type == 'max':
                    image[:] = np.maximum(image, imageNew)
                elif projection_type == 'min':
                    image[:] = np.minimum(image, imageNew)
                elif projection_type == 'mean':
                    image[:] = image + img_as_float32(imageNew)

        if projection_type == 'mean':
            image = image / len(z)
            image = np.clip(image, 0, 1)
            image = self.dtypeImgConvert(image)

        return image.squeeze()

    def get_Volume_At_Specific_Resolution(
            self, output_resolution=(100, 100, 100), time_point=0, channel=0, anti_aliasing=True
    ):
        """
        This function extracts a  time_point and channel at a specific resolution.
        The function extracts the whole volume at the highest resolution_level without
        going below the designated output_resolution.  It then resizes to the volume
        to the specified resolution by using the skimage rescale function.

        The option to turn off anti_aliasing during skimage.rescale (anti_aliasing=False) is provided.
        anti_aliasing can be very time consuming when extracting large resolutions.

        Everything is completed in RAM, very high resolutions may cause a crash.
        """

        # Find ResolutionLevel that is closest in size but larger
        resolutionLevelToExtract = 0
        for res in range(self.ResolutionLevels):
            currentResolution = self.metaData[res, time_point, channel, 'resolution']
            resCompare = [x <= y for x, y in zip(currentResolution, output_resolution)]
            resEqual = [x == y for x, y in zip(currentResolution, self.resolution)]
            if all(resCompare) == True or (all(resCompare) == False and any(resEqual) == True):
                resolutionLevelToExtract = res

        workingVolumeResolution = self.metaData[resolutionLevelToExtract, time_point, channel, 'resolution']
        print('Reading ResolutionLevel {}'.format(resolutionLevelToExtract))
        workingVolume = self.get_Resolution_Level(resolutionLevelToExtract, time_point=time_point, channel=channel)

        print('Resizing volume from resolution in microns {} to {}'.format(str(workingVolumeResolution),
                                                                           str(output_resolution)))
        rescaleFactor = tuple([round(x / y, 5) for x, y in zip(workingVolumeResolution, output_resolution)])
        print('Rescale Factor = {}'.format(rescaleFactor))

        workingVolume = img_as_float32(workingVolume)
        workingVolume = rescale(workingVolume, rescaleFactor, anti_aliasing=anti_aliasing)

        return self.dtypeImgConvert(workingVolume)

    def get_Resolution_Level(self, resolution_level, time_point=0, channel=0):
        return self[resolution_level, time_point, channel, :, :, :]

    @staticmethod
    def image_file_namer(resolution, time_point, channel, z_layer, prefix='', ext='.tif'):
        if ext[0] != '.':
            ext = '.' + ext

        if prefix == '':
            form = '{}r{}_t{}_c{}_z{}{}'
        else:
            form = '{}_r{}_t{}_c{}_z{}{}'

        return form.format(
            prefix,
            str(resolution).zfill(2),
            str(time_point).zfill(2),
            str(channel).zfill(2),
            str(z_layer).zfill(4),
            ext
        )

    def save_Tiff_Series(
            self, location=None, time_points=(), channels=(), resolutionLevel=0, cropYX=(), overwrite=False
    ):
        assert isinstance(channels, tuple)
        assert isinstance(resolutionLevel, int)
        assert isinstance(cropYX, tuple)
        assert isinstance(overwrite, bool)
        assert (location is None) or isinstance(location, str)

        if location is None:
            location = os.path.join(self.filePathBase, '{}_tiffSeries'.format(self.fileName))

        if os.path.exists(location) == False:
            os.makedirs(location, exist_ok=False)
        elif os.path.exists(location) == True and overwrite == True:
            os.makedirs(location, exist_ok=True)
        elif os.path.exists(location) == True and overwrite == False:
            raise Exception(
                "tiffSeries path already exists:  If you want to overwite the existing data, designate overwrite=True")

        if time_points == ():
            time_points = tuple(range(self.TimePoints))
        if channels == ():
            channels = tuple(range(self.Channels))

        if cropYX == ():
            cropYX = (
                0, self.metaData[(resolutionLevel, 0, 0, 'shape')][-2],
                0, self.metaData[(resolutionLevel, 0, 0, 'shape')][-1]
            )

        failed = []
        for time in time_points:
            for color in channels:
                for layer in range(self.metaData[(resolutionLevel, 0, 0, 'shape')][-3]):
                    fileName = os.path.join(location,
                                            self.image_file_namer(resolutionLevel, time, color, layer, prefix='',
                                                                  ext='.tif'))
                    if os.path.exists(fileName):
                        print('Skipping {} becasue it already exists'.format(fileName))
                        continue
                    try:
                        array = self[resolutionLevel, time, color, layer, cropYX[0]:cropYX[1], cropYX[2]:cropYX[3]]
                    except:
                        failed.append((resolutionLevel, time, color, layer, cropYX[0], cropYX[1], cropYX[2], cropYX[3]))
                        continue
                    print('Saving: {}'.format(fileName))
                    io.imsave(fileName, array, check_contrast=False)

        if len(failed) > 0:
            print('Failed to extract the following layers:')
            print(failed)
        else:
            print('All layers have been extracted')

    def save_multilayer_tiff_stack(self, location=None, time_point=0, channel=0, resolution_level=0):
        """
        Extract 1 time point, 1 channel, all (z,y,x) for a specified resolution level.

        location - path to the output tiff file. If not provided, file is saved next to the .ims file
        Output: tiff stack (all z layers in one file)
        """
        if not location:
            location = os.path.join(self.filePathBase, '{}_multilayer_stack.tif'.format(self.fileName))
        elif type(location) != str or location == '':
            raise TypeError('location must be a nonempty string')

        if os.path.exists(location):
            raise OSError('file at specified location already exists')
        else:
            os.makedirs(os.path.basename(location), exist_ok=True)

        try:
            time_point = int(time_point)
            channel = int(channel)
            resolution_level = int(resolution_level)
        except TypeError:
            raise TypeError('time_point, channel and resolution_level must be convertable to int')

        try:
            output_array = self[resolution_level, time_point, channel, :, :, :]
        except:  # try extracting z layers one-by-one
            output_array_shape = self.metaData[(resolution_level, time_point, channel, 'shape')][-3:]
            output_array = np.zeros([*output_array_shape], dtype=self.dtype)  # layers with errors will be left as 0
            for layer in range(output_array_shape[0]):
                try:
                    z_plane = self[resolution_level, time_point, channel, layer, :, :]
                    output_array[layer, :, :] = z_plane
                except Exception as e:
                    print(f'Failed to extract layer {layer}: {e}')

        io.imsave(location, output_array, check_contrast=False)


class _ImsImageLoader(ImageLoader):

    _reader: _ImsReader
    _min_time_point: Optional[int]
    _max_time_point: Optional[int]

    def __init__(self, file_name: str, min_time_point: Optional[int], max_time_point: Optional[int]):
        self._reader = _ImsReader(file_name)
        self._min_time_point = 0 if min_time_point is None else min_time_point
        self._max_time_point = self._reader.TimePoints if max_time_point is None else max_time_point

    def get_3d_image_array(self, time_point: TimePoint, image_channel: ImageChannel) -> Optional[ndarray]:
        time_point_number = time_point.time_point_number()
        if time_point_number < self.first_time_point_number() or time_point_number > self.last_time_point_number():
            return None
        if image_channel.index_zero < 0 or image_channel.index_zero >= self._reader.Channels:
            return None
        return self._reader[time_point_number - 1, image_channel.index_zero]

    def get_2d_image_array(self, time_point: TimePoint, image_channel: ImageChannel, image_z: int) -> Optional[ndarray]:
        time_point_number = time_point.time_point_number()
        if time_point_number < self.first_time_point_number() or time_point_number > self.last_time_point_number():
            return None
        if image_channel.index_zero < 0 or image_channel.index_zero >= self._reader.Channels:
            return None
        if image_z < 0 or image_z >= self._reader.shape[2]:
            return None
        return self._reader[time_point_number - 1, image_channel.index_zero, image_z]

    def get_image_size_zyx(self) -> Optional[Tuple[int, int, int]]:
        return self._reader.shape[-3], self._reader.shape[-2], self._reader.shape[-1]

    def first_time_point_number(self) -> int:
        return max(self._min_time_point, 1)

    def last_time_point_number(self) -> int:
        return min(self._max_time_point, self._reader.TimePoints)

    def get_channel_count(self) -> int:
        return self._reader.Channels

    def serialize_to_config(self) -> Tuple[str, str]:
        return self._reader.filePathComplete, ""

    def copy(self) -> "ImageLoader":
        return _ImsImageLoader(self._reader.filePathComplete, self._min_time_point, self._max_time_point)

    def close(self):
        self._reader.close()

    def get_spatial_resolution_um(self) -> Tuple[float, ...]:
        return self._reader.resolution

    def get_time_resolution_m(self) -> float:
        if "DataSetTimes" in self._reader.hf:
            time_0_ns = self._reader.hf["DataSetTimes"]["Time"][0][2]
            time_1_ns = self._reader.hf["DataSetTimes"]["Time"][1][2]
            timespan_s = (time_1_ns - time_0_ns) / 1_000_000_000
            return timespan_s / 60
        return 0
