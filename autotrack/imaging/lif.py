#
#    Copyright 2009 Mathieu Leocmach
#
#    This file is part of Colloids.
#
#    Colloids is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Colloids is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Colloids.  If not, see <http://www.gnu.org/licenses/>.
#
# for python 2.5, useless in 2.6
import struct, io, re, os.path, subprocess, shlex, sys
from typing import List
from xml.dom.minidom import parse, Element, Document

import numpy
import numpy as np
from numpy.fft import rfft2, irfft2

dimName = {1: "X",
           2: "Y",
           3: "Z",
           4: "T",
           5: "Lambda",
           6: "Rotation",
           7: "XT Slices",
           8: "TSlices", }
channelTag = ["Gray", "Red", "Green", "Blue"]


def _count_children(el: Element) -> int:
    """Counts how many tags there are inside the Children tag inside the given tag."""
    count = 0
    for children_tag in el.getElementsByTagName("Children"):
        count += len(children_tag.childNodes)
    return count


class Header:
    """The XML header of a Leica LIF files"""

    def __init__(self, xmlHeaderFileName, quick=True):
        if sys.version_info > (3, 0):
            with open(xmlHeaderFileName, encoding="latin-1") as f:
                self.parse(f, quick)
        else:
            with open(xmlHeaderFileName) as f:
                self.parse(f, quick)

    def parse(self, xmlHeaderFile, quick=True):
        """
Parse the usefull part of the xml header,
stripping time stamps and non ascii characters
"""

        # to strip the non ascii characters
        t = "".join(map(chr, list(range(256))))
        d = "".join(map(chr, list(range(128, 256))))
        if sys.version_info > (3, 0):
            trans = str.maketrans('', '', d)
            lightXML = io.StringIO()
        else:
            import StringIO
            trans = {ord(c): None for c in d}
            lightXML = StringIO.StringIO()

        if not quick:
            # store all time stamps in a big array
            timestamps = np.fromregex(
                xmlHeaderFile,
                r'<TimeStamp HighInteger="(\d+)" LowInteger="(\d+)"/>',
                float
            )
            xmlHeaderFile.seek(0)
            relTimestamps = np.fromregex(
                xmlHeaderFile,
                r'<RelTimeStamp Time="(\f+)" Frame="(\d+)"/>|<RelTimeStamp Frame="[0-9]*" Time="[0-9.]*"/>',
                float
            )
            xmlHeaderFile.seek(0)
            if sys.version_info > (3, 0):
                for line in xmlHeaderFile:
                    lightXML.write(line.translate(trans))
            else:
                for line in xmlHeaderFile:
                    try:
                        lightXML.write(line.translate(t, d))
                    except TypeError:
                        lightXML.write(line.translate(trans))

        else:
            # to strip the time stamps
            m = re.compile(
                r'''<TimeStamp HighInteger="[0-9]*" LowInteger="[0-9]*"/>|'''
                + r'''<RelTimeStamp Time="[0-9.]*" Frame="[0-9]*"/>|'''
                + r'''<RelTimeStamp Frame="[0-9]*" Time="[0-9.]*"/>'''
            )
            if sys.version_info > (3, 0):
                for line in xmlHeaderFile:
                    lightXML.write(''.join(m.split(line)).translate(trans))
            else:
                for line in xmlHeaderFile:
                    try:
                        lightXML.write(''.join(m.split(line)).translate(t, d))
                    except TypeError:
                        lightXML.write(''.join(m.split(line)).translate(trans))
        lightXML.seek(0)
        self.xmlHeader = parse(lightXML)

    def getVersion(self):
        if not hasattr(self, '__version'):
            self.__version = self.xmlHeader.documentElement.getAttribute("Version")
        return int(self.__version)

    def getName(self):
        if not hasattr(self, '__name'):
            self.__name = self.xmlHeader.documentElement. \
                getElementsByTagName('Element')[0].getAttribute("Name")
        return self.__name

    def getSeriesHeaders(self):
        if not hasattr(self, '__seriesHeaders'):
            self.__seriesHeaders = [
                SerieHeader(el)
                for el in self.xmlHeader.documentElement.getElementsByTagName('Element')[1:]
                # Ignore parent Elements and Elements with a special name
                if el.getAttribute('Name') not in ['BleachPointROISet', 'DCROISet', 'IOManagerConfiguation']
                   and _count_children(el) == 0
            ]
        return self.__seriesHeaders

    def chooseSerieIndex(self):
        st = "Experiment: %s\n" % self.getName()
        for i, s in enumerate(self.getSeriesHeaders()):
            # s = Serie(serie)
            st += "(%i) %s: %i channels and %i dimensions\n" % (
                i, s.getName(), len(s.getChannels()), len(s.getDimensions())
            )
            for c in s.getChannels():
                st += " %s" % channelTag[int(c.getAttribute("ChannelTag"))]
            for d in s.getDimensions():
                st += " %s%i" % (
                    dimName[int(d.getAttribute("DimID"))],
                    int(d.getAttribute("NumberOfElements"))
                )
            st += "\n"
        print(st)
        if len(self.getSeriesHeaders()) < 2:
            r = 0
        else:
            while (True):
                try:
                    r = int(input("Choose an item --> "))
                    if r < 0 or r > len(self.getSeriesHeaders()):
                        raise ValueError()
                    break
                except ValueError:
                    print("Oops!  That was no valid number.  Try again...")
        return r

    def chooseSerieHeader(self):
        return self.getSeriesHeaders()[self.chooseSerieIndex()]

    def __iter__(self):
        return iter(self.getSeriesHeaders())


class SerieHeader:
    """The part of the XML header of a Leica LIF files concerning a given serie"""

    root: Element

    def __init__(self, serieElement: Element):
        self.root = serieElement

    def getName(self) -> str:
        if not hasattr(self, '__name'):
            node = self.root
            names = []
            while not isinstance(node, Document) and node.tagName == "Element":
                names.insert(0, node.getAttribute("Name"))
                node = node.parentNode.parentNode
            if len(names) > 0:
                del names[0]  # Delete highest entry, which is something useless like "Project"
            self.__name = " >> ".join(names)
        return self.__name

    def getDisplayName(self) -> str:
        return self.getName().replace(" >> ", " » ")

    def isPreview(self):
        if not hasattr(self, '__isPreview'):
            self.__isPreview = 0
            for c in self.root.getElementsByTagName("Attachment"):
                if c.getAttribute("Name") == "PreviewMarker":
                    self.__isPreview = bool(c.getAttribute("isPreviewImage"))
                    break

        return self.__isPreview

    def getChannels(self):
        if not hasattr(self, '__channels'):
            self.__channels = self.root.getElementsByTagName("ChannelDescription")
        return self.__channels

    def getDimensions(self) -> List[Element]:
        if not hasattr(self, '__dimensions'):
            self.__dimensions = self.root.getElementsByTagName(
                "DimensionDescription")
        return self.__dimensions

    def hasZ(self):
        for d in self.getDimensions():
            if dimName[int(d.getAttribute("DimID"))] == "Z":
                return True
        return False

    def getMemorySize(self):
        if not hasattr(self, '__memorySize'):
            sizes = [
                int(m.getAttribute("Size"))
                for m in self.root.getElementsByTagName("Memory")
            ]
            if len(sizes) > 0:
                self.__memorySize = max(sizes)
            else:
                self.__memorySize = 0
        return self.__memorySize

    def getResolution(self, channel):
        if not hasattr(self, '__resolusion'):
            self.__resolusion = int(
                self.getChannels()[channel].getAttribute("Resolution")
            )
        return self.__resolusion

    def getScannerSetting(self, identifier):
        if not hasattr(self, '__' + identifier):
            for c in self.root.getElementsByTagName("ScannerSettingRecord"):
                if c.getAttribute("Identifier") == identifier:
                    setattr(self, '__' + identifier, c.getAttribute("Variant"))
                    break
        return getattr(self, '__' + identifier)

    def getNumberOfElements(self):
        if not hasattr(self, '__numberOfElements'):
            self.__numberOfElements = [
                int(d.getAttribute("NumberOfElements")) \
                for d in self.getDimensions()
            ]
        return self.__numberOfElements

    def getVoxelSize(self, dimension):
        return float(self.getScannerSetting("dblVoxel%s" % dimName[dimension]))

    def getZXratio(self):
        if self.hasZ():
            return float(self.getScannerSetting("dblVoxelZ")) / float(self.getScannerSetting("dblVoxelX"))
        else:
            return 1.0

    def getTotalDuration(self):
        """Get total duration of the experiment"""
        if not hasattr(self, '__duration'):
            self.__duration = 0.0
            for d in self.getDimensions():
                if dimName[int(d.getAttribute("DimID"))] == "T":
                    self.__duration = float(d.getAttribute("Length"))
        return self.__duration

    def getTimeLapse(self):
        """Get an estimate of the average time lapse between two frames in seconds"""
        if self.getNbFrames() == 1:
            return 0
        else:
            return self.getTotalDuration() / (self.getNbFrames() - 1)

    def getTimeStamps(self):
        """if the timestamps are not empty, convert them into a more lightweight numpy array"""
        if not hasattr(self, '__timeStamps'):
            tslist = self.root.getElementsByTagName("TimeStampList")[0]
            if tslist.hasAttribute("NumberOfTimeStamps") and int(tslist.getAttribute("NumberOfTimeStamps")) > 0:
                # SP8 way of storing time stamps in the text of the node as 16bits hexadecimal separated by spaces
                self.__timeStamps = np.array([
                    int(h, 16)
                    for h in tslist.firstChild.nodeValue.split()
                ])
            else:
                # SP5 way of storing time stamps as very verbose XML
                self.__timeStamps = np.asarray([
                    (int(c.getAttribute("HighInteger")) << 32) + int(c.getAttribute("LowInteger"))
                    for c in self.root.getElementsByTagName("TimeStamp")])
                # remove the data from XML
                for c in self.root.getElementsByTagName("TimeStamp"):
                    c.parentNode.removeChild(c).unlink()
        return self.__timeStamps

    def getRelativeTimeStamps(self):
        """if the timestamps are not empty, convert them into a more lightweight numpy array"""
        if not hasattr(self, '__relTimeStamps'):
            self.__relTimeStamps = np.asarray([
                float(c.getAttribute("Time"))
                for c in self.root.getElementsByTagName("RelTimeStamp")])
            # remove the data from XML
            for c in self.root.getElementsByTagName("RelTimeStamp"):
                c.parentNode.removeChild(c).unlink()
        return self.__relTimeStamps

    def getBytesInc(self, dimension):
        if isinstance(dimension, int):
            dim = dimName[dimension]
        else:
            dim = dimension
        if not hasattr(self, '__' + dim):
            setattr(self, '__' + dim, 0)
            for d in self.getDimensions():
                if dimName[int(d.getAttribute("DimID"))] == dim:
                    setattr(self, '__' + dim, int(d.getAttribute("BytesInc")))
        return getattr(self, '__' + dim)

    def chooseChannel(self) -> int:
        st = "Serie: %s\n" % self.getName()
        for i, c in enumerate(self.getChannels()):
            st += "(%i) %s\n" % (i, channelTag[int(c.getAttribute("ChannelTag"))])
        print(st)
        if len(self.getChannels()) < 2:
            r = 0
        while (True):
            try:
                r = int(input("Choose a channel --> "))
                if r < 0 or r > len(self.getChannels()):
                    raise ValueError()
                break
            except ValueError:
                print("Oops!  That was no valid number.  Try again...")
        return r

    def getNbFrames(self):
        if not hasattr(self, '__nbFrames'):
            self.__nbFrames = 1
            for d in self.getDimensions():
                if d.getAttribute("DimID") == "4":
                    self.__nbFrames = int(d.getAttribute("NumberOfElements"))
        return self.__nbFrames

    def getBoxShape(self):
        """Shape of the field of view in the X,Y,Z order."""
        if not hasattr(self, '__boxShape'):
            dims = {
                int(d.getAttribute('DimID')): int(d.getAttribute("NumberOfElements"))
                for d in self.getDimensions()
            }
            # ensure dimensions are sorted, keep only spatial dimensions
            self.__boxShape = [s for d, s in sorted(dims.items()) if d < 4]
        return self.__boxShape

    def getFrameShape(self):
        """Shape of the frame (nD image) in C order, that is C,Z,Y,X"""
        box_shape = self.getBoxShape()
        return [box_shape[2], len(self.getChannels()), box_shape[1], box_shape[0]]

    def get2DShape(self):
        """size of the two first spatial dimensions, in C order, e.g. Y,X"""
        return self.getBoxShape()[:2][::-1]

    def getNbPixelsPerFrame(self):
        if not hasattr(self, '__nbPixelsPerFrame'):
            self.__nbPixelsPerFrame = np.prod(self.getFrameShape())
        return self.__nbPixelsPerFrame

    def getNbPixelsPerSlice(self):
        if not hasattr(self, '__nbPixelsPerSlice'):
            self.__nbPixelsPerSlice = np.prod(self.get2DShape())
        return self.__nbPixelsPerSlice


class Reader(Header):
    """Reads Leica LIF files"""

    def __init__(self, lifFile, quick=True):
        # open file and find it's size
        if isinstance(lifFile, io.IOBase):
            self.f = lifFile
        else:
            self.f = open(lifFile, "rb")
        self.f.seek(0, 2)
        filesize = self.f.tell()
        self.f.seek(0)

        # read the size of the memory block containing the XML header
        # takes position at the begining of the XML header
        xmlHeaderLength = self.__readMemoryBlockHeader()

        # xmlHeaderLength, = struct.unpack("L",self.f.read(4))

        # Read the XML header as raw buffer. It should avoid encoding problems
        # but who uses japanese characters anyway
        xmlHeaderString = self.f.read(xmlHeaderLength * 2).decode('latin-1')
        self.parse(io.StringIO(xmlHeaderString[::2]), quick)

        # Index the series offsets
        self.offsets = []
        while (self.f.tell() < filesize):
            memorysize = self.__readMemoryBlockHeader()
            while (self.f.read(1) != b"*"):
                pass
            # size of the memory description
            memDescrSize, = struct.unpack("I", self.f.read(4))
            memDescrSize *= 2
            # skip the description: we are at the begining of the content
            self.f.seek(memDescrSize, 1)
            # add image offset if memory size >0
            if memorysize > 0:
                self.offsets.append(self.f.tell())
                self.f.seek(memorysize, 1)
        if not quick:
            # convert immediately the time stamps in XML format to lighweight numpy array
            for s in self:
                s.getTimeStamps()
                s.getRelativeTimeStamps()

        # self.offsets = [long(m.getAttribute("Size")) for m in self.xmlHEader.getElementsByTagName("Memory")]

    def __readMemoryBlockHeader(self):
        memBlock, trash, testBlock = struct.unpack("iic", self.f.read(9))
        if memBlock != 112:
            raise Exception("This is not a valid LIF file")
        if testBlock != b'*':
            raise Exception("Invalid block at %d" % self.f.tell())
        if not hasattr(self, 'xmlHeader') or self.getVersion() < 2:
            memorysize, = struct.unpack("I", self.f.read(4))
        else:
            memorysize, = struct.unpack("Q", self.f.read(8))
        return memorysize

    def getSeries(self) -> List["Serie"]:
        if not hasattr(self, '__series'):
            try:
                self.__series = [
                    Serie(s.root, self.f, self.offsets[i]) \
                    for i, s in enumerate(self.getSeriesHeaders())
                ]
            except IndexError:
                raise IndexError(
                    'More SeriesHeaders than offsets. Try to run chooseSeriesHeaders(), note the name of the strange items on the list, and edit Reader.getSeriesHeaders to discard Elements named that way from the list of SeriesHeaders. (Or contact the maintainter with the output of chooseSeriesHeaders)')
        return self.__series

    def chooseSerie(self):
        return self.getSeries()[self.chooseSerieIndex()]

    def __iter__(self):
        return iter(self.getSeries())


class Serie(SerieHeader):
    """One on the datasets in a lif file"""

    def __init__(self, serieElement, f, offset):
        self.f = f
        self.__offset = offset
        self.root = serieElement

    def getOffset(self, **dimensionsIncrements):
        of = 0
        for d, b in dimensionsIncrements.items():
            of += self.getBytesInc(d) * b
        if of >= self.getMemorySize():
            raise IndexError("offset out of bound")
        return self.__offset + of

    def get2DSlice(self, **dimensionsIncrements):
        """Use the two first dimensions as image dimension. Axis are in C order (last index is X)."""
        for d in self.getDimensions()[:2]:
            if dimName[int(d.getAttribute("DimID"))] in dimensionsIncrements:
                raise Exception('You can\'t set %s in serie %s' % (
                    dimName[int(d.getAttribute("DimID"))],
                    self.getName())
                                )

        self.f.seek(self.getOffset(**dimensionsIncrements))
        shape = self.get2DShape()
        return np.fromfile(
            self.f,
            dtype=np.ubyte,
            count=self.getNbPixelsPerSlice()
        ).reshape(shape)

    def get2DString(self, **dimensionsIncrements):
        """Use the two first dimensions as image dimension"""
        for d in self.getDimensions()[:2]:
            if dimName[int(d.getAttribute("DimID"))] in dimensionsIncrements:
                raise Exception('You can\'t set %s in serie %s' % (
                    dimName[int(d.getAttribute("DimID"))],
                    self.getName()))

        self.f.seek(self.getOffset(**dimensionsIncrements))
        return self.f.read(self.getNbPixelsPerSlice())

    def get2DImage(self, **dimensionsIncrements):
        """Use the two first dimensions as image dimension"""
        try:
            import Image
        except:
            try:
                import PIL as Image
            except:
                print("Impossible to find image library")
                return None
        size = self.getNumberOfElements()[:2]
        return Image.fromstring(
            "L",
            tuple(size)
            , self.get2DString(**dimensionsIncrements)
        )

    def getFrame(self, T=0):
        """
        Return a numpy array (C order, thus last index is X). Shape is (Z, C, Y, X).
        """
        self.f.seek(self.getOffset(**dict({'T': T})))
        return np.fromfile(
            self.f,
            dtype=np.ubyte,
            count=self.getNbPixelsPerFrame()
        ).reshape(self.getFrameShape())

    def getVTK(self, fname, T=0):
        """Export the frame at time T to a vtk file"""
        with open(fname, 'wb') as f:
            f.write(
                ('# vtk DataFile Version 3.0\n%s\n' % self.getName) +
                'BINARY\nDATASET STRUCTURED_POINTS\n' +
                ('DIMENSIONS %d %d %d\n' % tuple(self.getBoxShape())) +
                'ORIGIN 0 0 0\n' +
                ('SPACING 1 1 %g\n' % self.getZXratio()) +
                ('POINT_DATA %d\n' % self.getNbPixelsPerFrame()) +
                'SCALARS Intensity unsigned_char\nLOOKUP_TABLE default\n'
            )
            self.f.seek(self.getOffset(**dict({'T': T})))
            f.write(self.f.read(self.getNbPixelsPerFrame()))

    def getDispl2DImage(self, t0=0, t1=1, Z=0):
        ham = np.hamming(self.get2DShape()[0]) * np.atleast_2d(np.hamming(self.get2DShape()[1])).T
        a = rfft2(self.get2DSlice(T=t0, Z=Z) * ham)
        b = rfft2(self.get2DSlice(T=t1, Z=Z) * ham)
        R = a * numpy.complex(numpy.real(b),
                              -numpy.imag(b) / numpy.abs(a * numpy.complex(numpy.real(b), -numpy.imag(b))))
        return irfft2(R)

    def getDispl2D(self, t0=0, t1=1, Z=0):
        r = self.getDispl2DImage(t0, t1, Z)
        return np.unravel_index(r.argmax(), r.shape)

    def getDisplacements2D(self, Z=None, window=False):
        """
        Use phase correlation to find the relative displacement between
        each time step
        """
        if Z is None:
            Z = self.getNbPixelsPerFrame() / self.getNbPixelsPerSlice() / 2
        shape = np.asarray(self.get2DShape())
        if window:
            ham = np.hamming(shape[0]) * np.atleast_2d(np.hamming(shape[1])).T
        else:
            ham = 1.0
        displs = np.zeros((self.getNbFrames(), 2))
        a = rfft2(self.get2DSlice(T=0, Z=Z) * ham)
        for t in range(1, self.getNbFrames()):
            b = rfft2(self.get2DSlice(T=t, Z=Z) * ham)
            # calculate the normalized cross-power spectrum
            # R = numexpr.evaluate(
            #    'a*complex(real(b), -imag(b)/abs(a*complex(real(b), -imag(b))))'
            #    )
            R = a * b.conj()
            Ra = np.abs(a * b.conj())
            R[Ra > 0] /= Ra[Ra > 0]
            r = irfft2(R)
            # Get the periodic position of the peak
            l = r.argmax()
            displs[t] = np.unravel_index(l, r.shape)
            # prepare next step
            a = b
        return np.where(displs < shape[::-1] / 2, displs, displs - shape[::-1])

    def enumByFrame(self):
        """yield time steps one after the other as a couple (time,numpy array). It is not safe to combine this syntax with getFrame or get2DSlice."""
        yield 0, self.getFrame()
        for t in range(1, self.getNbFrames()):
            yield t, np.fromfile(
                self.f,
                dtype=np.ubyte,
                count=self.getNbPixelsPerFrame()
            ).reshape(self.getFrameShape())

    def enumBySlice(self):
        """yield 2D slices one after the other as a 3-tuple (time,z,numpy array). It is not safe to combine this syntax with getFrame or get2DSlice."""
        self.f.seek(self.getOffset())
        for t in range(self.getNbFrames()):
            for z in range(self.getNbPixelsPerFrame() / self.getNbPixelsPerSlice()):
                yield t, z, np.fromfile(
                    self.f,
                    dtype=np.ubyte,
                    count=self.getNbPixelsPerSlice()
                ).reshape(self.get2DShape())


def getNeighbourhood(point, image, radius=10):
    box = np.floor([np.maximum(0, point - radius), np.minimum(image.shape, point + radius + 1)])
    ngb = image[box[0, 0]:box[1, 0], box[0, 1]:box[1, 1], box[0, 2]:box[1, 2]]
    center = point - box[0]
    return ngb, center


def getRadius(ngb, center, zratio=1, rmin=3, rmax=10, precision=0.1):
    sqdist = [(np.arange(ngb.shape[d]) - center[d]) ** 2 for d in range(ngb.ndim)]
    sqdist[-1] *= zratio ** 2
    dist = np.empty_like(ngb)
    for i, x in enumerate(sqdist[0]):
        for j, y in enumerate(sqdist[1]):
            for k, z in enumerate(sqdist[2]):
                dist[i, j, k] = x + y + z
    val = np.zeros(rmax / precision)
    nbs = np.zeros(rmax / precision)
    for l, (px, d) in enumerate(zip(ngb.ravel(), dist.ravel())):
        if d < rmax ** 2:
            i = int(np.sqrt(d) / precision)
            nbs[i] += 1
            val[i] += px
    intensity = np.cumsum(val) / np.cumsum(nbs)
    fi = np.fft.rfft(intensity)
    return rmin + precision * np.argmin(
        np.ediff1d(np.fft.irfft(
            fi * np.exp(-np.arange(len(fi)) / 13.5))
        )[rmin / precision:]
    )
