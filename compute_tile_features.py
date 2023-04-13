import numpy as np
import tensorflow as tf
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIRS = [
    './carcassonne/resources/images/base_game',
    './carcassonne/resources/images/inns_and_cathedrals',
]

resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

spatial_pooling = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='valid', data_format='channels_last')
channel_pooling = tf.keras.layers.AveragePooling1D(pool_size=32, strides=32, padding='valid', data_format='channels_first')

for dir in IMAGE_DIRS:
    dir = os.path.join(ROOT_DIR, dir)
    for file in os.listdir(dir):
        if not file.endswith('.png'):
            continue

        raw_image = tf.keras.preprocessing.image.load_img(os.path.join(dir, file), target_size=(224, 224), interpolation='bilinear')
        # shape: (224, 224, 3)
        raw_image = tf.keras.preprocessing.image.img_to_array(raw_image)

        # process for each possible rotation
        image_tensors = []
        for rotation in range(4): # clockwise rotation
            image = tf.image.rot90(raw_image, k=rotation * 3)
            image_tensors.append(image)
            # # visualize image
            # tf.keras.preprocessing.image.array_to_img(image).show()
            # # wait
            # input('Press enter to continue...')

        image_input = tf.keras.applications.resnet50.preprocess_input(tf.stack(image_tensors, axis=0))
        features = resnet.predict(image_input)
        features = spatial_pooling(features)
        features = tf.reshape(features, (4, -1, 2048))
        # print(tf.shape(features))
        features = channel_pooling(features)
        # print(tf.shape(features))
        features = tf.reshape(features, (4, -1))
        # print(tf.shape(features))
        np.save(os.path.join(dir, file + '.feat.npy'), features.numpy())


