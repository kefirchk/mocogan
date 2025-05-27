"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
import PIL
import tensorflow as tf
import numpy as np

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger(object):
    def __init__(self, log_dir, suffix=None):
        self.writer = tf.summary.create_file_writer(log_dir, filename_suffix=suffix)

    def scalar_summary(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()  # Optional: flush the writer to ensure the summary is written

    def image_summary(self, tag, images, step):
        img_summaries = []
        for i, img in enumerate(images):
            # Convert the image to a format suitable for saving
            img = np.clip(img, 0, 255).astype(np.uint8)  # Ensure values are in [0, 255]

            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()

            # Create a PIL image from the numpy array
            pil_img = PIL.Image.fromarray(img)
            pil_img.save(s, format="png")  # Save the image to the BytesIO object

            # Convert the byte string to a tensor
            image_tensor = tf.image.decode_png(s.getvalue())

            # Append the image summary to the img_summaries list
            img_summaries.append((f'{tag}/{i}', image_tensor))

        # Write the summaries to TensorBoard
        with self.writer.as_default():
            for img_tag, img_tensor in img_summaries:
                tf.summary.image(img_tag, tf.expand_dims(img_tensor, 0), step=step)

        self.writer.flush()

    def video_summary(self, tag, videos, step):
        sh = list(videos.shape)
        sh[-1] = 1

        separator = np.zeros(sh, dtype=videos.dtype)
        videos = np.concatenate([videos, separator], axis=-1)

        for i, vid in enumerate(videos):
            # Concat a video
            try:
                s = StringIO()
            except:
                s = BytesIO()

            v = vid.transpose(1, 2, 3, 0)
            v = [np.squeeze(f) for f in np.split(v, v.shape[0], axis=0)]
            img = np.concatenate(v, axis=1)[:, :-1, :]

            # Convert and save image using Pillow
            pil_img = PIL.Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))  # Ensure correct format
            pil_img.save(s, format="png")  # Save to BytesIO
            encoded_image_string = s.getvalue()  # Get the byte value of the image

            # Write the summary directly to TensorBoard
            with self.writer.as_default():
                tf.summary.image(f'{tag}/{i}', [tf.image.decode_png(encoded_image_string)], step=step)

        self.writer.flush()
