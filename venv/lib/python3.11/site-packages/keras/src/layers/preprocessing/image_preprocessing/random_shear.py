import numpy as np

from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.random.seed_generator import SeedGenerator


class RandomShear(BaseImagePreprocessingLayer):
    """A preprocessing layer which randomly shears images.

    This layer will apply random shearings to each image, filling empty space
    according to `fill_mode`.

    Input pixel values can be of any range and any data type.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Args:
        x_factor: A tuple of two floats or a single float.
            For each augmented image a value is
            sampled from the provided range. If a float is passed, the range is
            interpreted as `(0, x_factor)`. Values represent a percentage of the
            image to shear over. For example, 0.3 shears pixels up to 30% of the
            way across the image. All provided values should be positive. If
            `None` is passed, no shear occurs on the X axis. Defaults to `None`.
        y_factor: A tuple of two floats or a single float.
            For each augmented image a value is
            sampled from the provided range. If a float is passed, the range is
            interpreted as `(0, y_factor)`. Values represent a percentage of the
            image to shear over. For example, 0.3 shears pixels up to 30% of the
            way across the image. All provided values should be positive. If
            `None` is passed, no shear occurs on the Y axis. Defaults to `None`.
        fill_mode: Points outside the boundaries of the input are filled
            according to the given mode
            (one of `{"constant", "reflect", "wrap", "nearest"}`).
            - *reflect*: `(d c b a | a b c d | d c b a)`
                The input is extended by reflecting about
                the edge of the last pixel.
            - *constant*: `(k k k k | a b c d | k k k k)`
                The input is extended by
                filling all values beyond the edge with
                the same constant value k = 0.
            - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
                wrapping around to the opposite edge.
            - *nearest*: `(a a a a | a b c d | d d d d)`
                The input is extended by the nearest pixel.
        interpolation: Interpolation mode. Supported values: `"nearest"`,
            `"bilinear"`.
        seed: Integer. Used to create a random seed.
        fill_value: a float represents the value to be filled outside
            the boundaries when `fill_mode="constant"`.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
    """

    _USE_BASE_FACTOR = False
    _SUPPORTED_FILL_MODE = ("reflect", "wrap", "constant", "nearest")
    _SUPPORTED_INTERPOLATION = ("nearest", "bilinear")

    def __init__(
        self,
        x_factor=None,
        y_factor=None,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        data_format=None,
        **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self.fill_mode = fill_mode
        self.interpolation = interpolation
        self.fill_value = fill_value
        self.supports_jit = False

        if self.fill_mode not in self._SUPPORTED_FILL_MODE:
            raise NotImplementedError(
                f"Unknown `fill_mode` {fill_mode}. Expected of one "
                f"{self._SUPPORTED_FILL_MODE}."
            )
        if self.interpolation not in self._SUPPORTED_INTERPOLATION:
            raise NotImplementedError(
                f"Unknown `interpolation` {interpolation}. Expected of one "
                f"{self._SUPPORTED_INTERPOLATION}."
            )

    def transform_images(self, images, transformation, training=True):
        images = self.backend.cast(images, self.compute_dtype)
        if training:
            return self.backend.image.affine_transform(
                images=images,
                transform=transformation["rotation_matrix"],
                interpolation=self.interpolation,
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
                data_format=self.data_format,
            )
        return images

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_bounding_boxes(
        self, bounding_boxes, transformation, training=True
    ):
        raise NotImplementedError

    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return self.transform_images(
            segmentation_masks, transformation, training=training
        )

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None
        if isinstance(data, dict):
            images = data["images"]
        else:
            images = data
        shape = self.backend.core.shape(images)
        if len(shape) == 4:
            if self.data_format == "channels_last":
                batch_size = shape[0]
                image_height = shape[1]
                image_width = shape[2]
            else:
                batch_size = shape[1]
                image_height = shape[2]
                image_width = shape[3]
        else:
            batch_size = 1
            if self.data_format == "channels_last":
                image_height = shape[0]
                image_width = shape[1]
            else:
                image_height = shape[1]
                image_width = shape[2]

        ...

        if len(shape) == 3:
            shear_matrix_x = self.backend.numpy.squeeze(shear_matrix_x, axis=0)
            shear_matrix_y = self.backend.numpy.squeeze(shear_matrix_y, axis=0)
        return {
            "shear_matrix_x": shear_matrix_x,
            "shear_matrix_y": shear_matrix_y,
        }

    def _build_shear_x_transform_matrix(self, shear_x):
        """Build transform matrix for horizontal shear.

        The transform matrix looks like:
        (1, x, 0)
        (0, 1, 0)
        (0, 0, 1)
        where the last entry is implicit.

        We flatten the matrix to `[1.0, x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]` for
        use with ImageProjectiveTransformV3.
        """
        batch_size = tf.shape(shear_x)[0]
        return self.backend.numpy.concatenate(
            values=[
                tf.ones((batch_size, 1), self.compute_dtype),
                shear_x,
                tf.zeros((batch_size, 2), self.compute_dtype),
                tf.ones((batch_size, 1), self.compute_dtype),
                tf.zeros((batch_size, 3), self.compute_dtype),
            ],
            axis=1,
        )

    def _build_shear_y_transform_matrix(self, shear_y):
        """Build transform matrix for vertical shear.

        The transform matrix looks like:
        (1, 0, 0)
        (y, 1, 0)
        (0, 0, 1)
        where the last entry is implicit.

        We flatten the matrix to `[1.0, 0.0, 0.0, y, 1.0, 0.0, 0.0, 0.0]` for
        use ImageProjectiveTransformV3.
        """
        batch_size = tf.shape(shear_y)[0]
        return self.backend.numpy.concatenate(
            values=[
                tf.ones((batch_size, 1), self.compute_dtype),
                tf.zeros((batch_size, 2), self.compute_dtype),
                shear_y,
                tf.ones((batch_size, 1), self.compute_dtype),
                tf.zeros((batch_size, 3), self.compute_dtype),
            ],
            axis=1,
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "x_factor": self.x_factor,
            "y_factor": self.y_factor,
            "data_format": self.data_format,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
