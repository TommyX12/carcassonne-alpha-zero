import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


def julia_iteration(z, c):
    """
    z: tf tensor with shape (batch_size, 2)
    c: tf tensor with shape (batch_size or 1, 2)
    Perform one single Julia set iteration. Note that z and c represents complex values.
    """
    zx = z[:, 0]
    zy = z[:, 1]
    cx = c[:, 0]
    cy = c[:, 1]
    result = tf.stack([zx ** 2 - zy ** 2 + cx, 2 * zx * zy + cy], axis=1)
    # clip result to -100, 100
    result = tf.clip_by_value(result, -100, 100)
    return result


def julia_set(z, c, num_iters, include_states=False):
    """
    z: tf tensor with shape (batch_size, 2)
    c: tf tensor with shape (batch_size or 1, 2)
    Calculate the Julia set for the given z and c, return number of iterations until z diverges.
    """
    batch_size = z.shape[0]
    # initialize a tensor with shape (batch_size)
    # this tensor will be used to count the number of iterations
    iters = tf.zeros(batch_size, dtype=tf.int32)
    states = None
    state_masks = None
    states_iters = None
    if include_states:
        states = [z]
        state_masks = [tf.ones(batch_size, dtype=tf.float32)]
        states_iters = [iters]

    for i in range(num_iters):
        z = julia_iteration(z, c)
        alive = tf.reduce_sum(z ** 2, axis=1) < 4
        iters = tf.where(alive, i + 1, iters)
        if include_states:
            states.append(z)
            state_masks.append(alive)
            states_iters.append(iters)

    if include_states:
        states = tuple(states)
        state_masks = tuple(state_masks)
        states_iters = tuple(states_iters)

    return iters, states, state_masks, states_iters


class JuliaSetTask(object):
    def __init__(self, c=[-0.8, 0.156], resolution=1000, num_iters=1024, normalize_iters=1000, radius=0.05, include_states=False) -> None:
        super().__init__()
        self.c = c
        self.resolution = resolution
        self.num_iters = num_iters
        self.normalize_iters = normalize_iters
        self.radius = radius
        self.include_states = include_states

    def _get_data(self):
        z = tf.constant([[x, -y] for y in np.linspace(-self.radius, self.radius, self.resolution)
                        for x in np.linspace(-self.radius, self.radius, self.resolution)], dtype=tf.float32)
        num_samples = z.shape[0]
        # c = tf.constant([[0.285, 0.01]], dtype=tf.float32)
        # c = tf.constant([[-0.7269, 0.1889]], dtype=tf.float32)
        c = tf.tile(tf.constant([self.c], dtype=tf.float32), [num_samples, 1])
        iters, states, state_masks, states_iters = julia_set(
            z, c, self.num_iters, self.include_states)
        # normalize iters to the range [0, 1]
        iters = tf.clip_by_value(
            tf.cast(iters, tf.float32) / self.normalize_iters, 0, 1)
        # initialize a tensor with shape (num_samples, 2) with all values set to self.c
        # use tf.data.Dataset.from_tensor_slices to create a dataset
        # the dataset should contain z and c, using dict
        slices = {'inputs': {'z': z, 'c': c}, 'labels': iters}
        if self.include_states:
            slices['states'] = states
            state_masks = tuple(tf.cast(x, tf.float32) for x in state_masks)
            slices['state_masks'] = state_masks
            states_iters = tuple(tf.clip_by_value(
                tf.cast(x, tf.float32) / self.normalize_iters, 0, 1) for x in states_iters)
            slices['states_iters'] = states_iters

        dataset = tf.data.Dataset.from_tensor_slices(slices)
        return dataset

    def get_train_dataset(self, batch_size):
        # use self._get_data() to get the dataset, then use batch and perfect shuffle
        dataset = self._get_data()
        return dataset.shuffle(dataset.cardinality().numpy()).batch(batch_size)

    def get_test_dataset(self, batch_size):
        return self._get_data().batch(batch_size)


class CIFAR10Task(object):
    def __init__(self) -> None:
        pass

    def get_train_dataset(self, batch_size):
        return self._process_dataset(tfds.load('cifar10', split='train', shuffle_files=True, batch_size=batch_size))

    def get_test_dataset(self, batch_size):
        return self._process_dataset(tfds.load('cifar10', split='test', shuffle_files=False, batch_size=batch_size))

    def _process_dataset(self, dataset):
        # normalize the image field to the range [0, 1]
        dataset = dataset.map(lambda x: {'image': tf.cast(
            x['image'], tf.float32) / 255.0, 'label': x['label']})
        return dataset
