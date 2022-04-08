# Copyright 2022 Niharika Krishnan (@niharikakrishnan)
# Copyright 2022 Vaibhav Singh (@vaibhav016)

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


import tensorflow as tf

def make_data(global_batch_size: int = 1,
              BUFFER_SIZE: int = 400,
              IMG_WIDTH: int = 200,
              IMG_HEIGHT: int = 200,
              scale_percent: int =30,
              data_path: str = "/Users/vaibhavsingh/Desktop/Big_Data/faces_data_small/*.png"
            ):
    # We will pixelate the image for faces
    def load_image_train(image_file):
        # Read and decode an image file to a uint8 tensor
        image = tf.io.read_file(image_file)
        image = tf.io.decode_jpeg(image)

        # Convert images to float32 tensors
        input_image = tf.cast(image, tf.float32)
        input_image = tf.image.resize(input_image, [IMG_HEIGHT, IMG_WIDTH],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        input_image = tf.cast(input_image, tf.float32) / 255.0

        # Obtain pixelated image
        width = int(input_image.shape[1] * scale_percent / 100)
        height = int(input_image.shape[0] * scale_percent / 100)

        small_image = tf.image.resize(input_image, [height, width],
                                      method=tf.image.ResizeMethod.AREA)

        pixelated_image = tf.image.resize(small_image, [IMG_HEIGHT, IMG_WIDTH],
                                          method=tf.image.ResizeMethod.AREA)

        return input_image, pixelated_image

    train_dataset = tf.data.Dataset.list_files(data_path)
    train_dataset = train_dataset.map(load_image_train,
                                      num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(global_batch_size)

    return train_dataset
