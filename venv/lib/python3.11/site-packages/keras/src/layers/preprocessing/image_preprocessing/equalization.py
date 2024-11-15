# from keras.src import backend
# from keras.src.ops.core import _saturate_cast
# from keras.src.api_export import keras_export
# from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
#     BaseImagePreprocessingLayer,
# )

# @keras_export("keras.layers.Equalization")
# class Equalization(BaseImagePreprocessingLayer):
#     """Equalization performs histogram equalization on a channel-wise basis.

#     Args:
#         value_range: a tuple or a list of two elements. The first value
#             represents the lower bound for values in passed images, the second
#             represents the upper bound. Images passed to the layer should have
#             values within `value_range`.
#         bins: Integer indicating the number of bins to use in histogram
#             equalization. Should be in the range `(0, 256)`.

#     Example:

#     ```python
#     equalize = Equalization()

#     (images, labels), _ = keras.datasets.cifar10.load_data()
#     # Note that images are an int8 Tensor with values in the range [0, 255]
#     images = equalize(images)
#     ```

#     Call arguments:
#         images: Tensor of pixels in range `(0, 255)`, in RGB format. Can be
#             of type float or int.
#     """

#     def __init__(self, value_range, bins=256, data_format=None, **kwargs):
#         super().__init__(data_format=data_format, **kwargs)
#         self.bins = bins
#         self.value_range = value_range

#     def _equalize_channel(self, images, channel_index):
#         """equalize_channel performs histogram equalization on a single channel.

#         Args:
#             image: int Tensor with pixels in range [0, 255], RGB format,
#                 with channels last
#             channel_index: channel to equalize
#         """
#         is_single_image = tf.rank(images) == 4 and tf.shape(images)[0] == 1

#         images = images[..., channel_index]
#         # Compute the histogram of the image channel.

#         # If the input is not a batch of images, directly using
#         # tf.histogram_fixed_width is much faster than using tf.vectorized_map
#         if is_single_image:
#             histogram = tf.histogram_fixed_width(
#                 images, [0, 255], nbins=self.bins
#             )
#             histogram = tf.expand_dims(histogram, axis=0)
#         else:
#             partial_hist = partial(
#                 tf.histogram_fixed_width, value_range=[0, 255], nbins=self.bins
#             )
#             histogram = tf.vectorized_map(
#                 partial_hist, images, fallback_to_while_loop=True, warn=True
#             )

#         # For the purposes of computing the step, filter out the non-zeros.
#         # Zeroes are replaced by a big number while calculating min to keep
#         # shape constant across input sizes for compatibility with
#         # vectorized_map

#         big_number = 1410065408
#         histogram_without_zeroes = tf.where(
#             tf.equal(histogram, 0),
#             big_number,
#             histogram,
#         )

#         step = (
#             tf.reduce_sum(histogram, axis=-1)
#             - tf.reduce_min(histogram_without_zeroes, axis=-1)
#         ) // (self.bins - 1)

#         def build_mapping(histogram, step):
#             bacth_size = tf.shape(histogram)[0]

#             # Replace where step is 0 with 1 to avoid division by 0.
#             # This doesn't change the result, because where step==0 the
#             # original image is returned
#             _step = tf.where(
#                 tf.equal(step, 0),
#                 1,
#                 step,
#             )
#             _step = tf.expand_dims(_step, -1)

#             # Compute the cumulative sum, shifting by step // 2
#             # and then normalization by step.
#             lookup_table = (
#                 tf.cumsum(histogram, axis=-1) + (_step // 2)
#             ) // _step

#             # Shift lookup_table, prepending with 0.
#             lookup_table = tf.concat(
#                 [tf.tile([[0]], [bacth_size, 1]), lookup_table[..., :-1]],
#                 axis=1,
#             )

#             # Clip the counts to be in range. This is done
#             # in the C code for image.point.
#             return tf.clip_by_value(lookup_table, 0, 255)

#         # If step is zero, return the original image. Otherwise, build
#         # lookup table from the full histogram and step and then index from it.
#         # The lookup table is built for all images,
#         # regardless of the corresponding value of step.
#         result = tf.where(
#             tf.reshape(tf.equal(step, 0), (-1, 1, 1)),
#             images,
#             tf.gather(
#                 build_mapping(histogram, step), images, batch_dims=1, axis=1
#             ),
#         )
#         return result

#     def transform_images(self, images, transformation=None, training=True):

#         if self.data_format == 'channels_first':
#             images = self.backend.numpy.transpose(images, [0, 2, 3, 1])

#         images = self._transform_value_range(
#             images, self.value_range, (0, 255), dtype=self.compute_dtype
#         )
#         images = self.backend.cast(images, self.compute_dtype)

#         # Split the images into individual channels
#         channels = self.backend.numpy.unstack(images, axis=-1)

#         # Equalize each channel
#         equalized_channels = [self._equalize_histogram(channel) for channel in channels]

#         # Stack the equalized channels back into an image
#         results = self.backend.numpy.stack(equalized_channels, axis=-1)

#         results = self._transform_value_range(
#             results, (0, 255), self.value_range, dtype=self.compute_dtype
#         )
#         if self.data_format == 'channels_first':
#             results = self.backend.numpy.transpose(results, [0, 3, 1, 2])

#         if results.dtype == images.dtype:
#             return results
#         if backend.is_int_dtype(images.dtype):
#             results = self.backend.numpy.round(results)
#         return _saturate_cast(results, images.dtype, self.backend)

#     def _equalize_histogram(self, image):
#         # Flatten the image to a 1D tensor
#         flattened_image = ops.reshape(image, [-1])

#         # Calculate the histogram of the flattened image
#         histogram = ops.histogram_fixed_width(flattened_image, value_range=[0, 255], nbins=256)

#         # Calculate the cumulative distribution function (CDF) of the histogram
#         cdf = ops.cumsum(histogram)

#         # Normalize the CDF to the range [0, 255]
#         normalized_cdf = (cdf - ops.reduce_min(cdf)) * 255 / (ops.reduce_max(cdf) - ops.reduce_min(cdf))
#         normalized_cdf = ops.cast(normalized_cdf, tf.uint8)

#         # Map the pixel values in the flattened image to their equalized values using the normalized CDF
#         equalized_flattened_image = ops.gather(normalized_cdf, ops.cast(flattened_image, tf.int32))

#         # Reshape the equalized flattened image back to the original shape
#         equalized_image = ops.reshape(equalized_flattened_image, ops.shape(image))
#         return equalized_image

#     def get_config(self):
#         config = super().get_config()
#         config.update({"bins": self.bins, "value_range": self.value_range, "data_format": self.data_format})
#         return config
