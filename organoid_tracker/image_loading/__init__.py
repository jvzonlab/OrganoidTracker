"""
Files for loading images. Use general_image_loader to load an arbitrary image/image sequence.

>>> from organoid_tracker.core.experiment import Experiment
>>> experiment = Experiment()
>>> from organoid_tracker.image_loading import general_image_loader
>>> general_image_loader.load_images(experiment, "path/to/folder", "image_t{time:003}_c{channel}.tif")

Or, in case you need to load something more advanced: (this example append three different channels)

>>> from organoid_tracker.core.experiment import Experiment
>>> experiment = Experiment()
>>> from organoid_tracker.image_loading import general_image_loader
>>> general_image_loader.load_images_from_dictionary(experiment, {
>>>     "images_channel_appending": [
>>>            {
>>>                "images_container": "path/to/my/data",
>>>                "images_pattern": "t{time:03}_488nm.tif"
>>>            },
>>>            {
>>>                "images_container": "path/to/my/data",
>>>                "images_pattern": "t{time:03}_561nm.tif"
>>>            },
>>>            {
>>>                "images_container": "path/to/my/data",
>>>                "images_pattern": "t{time:03}_CoolLED.tif"
>>>            }
>>>        ]
>>> })

Once you have loaded some images, you can retrieve them as follows:

>>> from organoid_tracker.core import TimePoint
>>> from organoid_tracker.core.image_loader import ImageChannel
>>> array = experiment.images.get_image_stack(TimePoint(3), ImageChannel(index_zero=0))

And you can save the image loader to a Python dictionary as follows:
>>> dictionary = experiment.images.image_loader().serialize_to_dictionary()
>>>
>>> # Restore using
>>> general_image_loader.load_images_from_dictionary(experiment, dictionary)

"""
