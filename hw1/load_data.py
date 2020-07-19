import numpy as np
import os
import random
import tensorflow as tf
from scipy import misc
from matplotlib.pyplot import imread
from sklearn.utils import shuffle
from glob import glob


def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    image = imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, config={}):
        """
        Args:
            num_classes: Number of classes for classification (K-way)
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: train/val/test
        Returns:
            A a tuple of (1) Image batch and (2) Label batch where
            image batch has shape [B, K, N, 784] and label batch has shape [B, K, N, N]
            where B is batch size, K is number of samples per class, N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        B = batch_size
        K = self.num_samples_per_class if self.num_samples_per_class <= 20 else 20
        N = self.num_classes

        all_image_batches = np.zeros(shape = (B, K, N, 784))
        all_label_batches = np.zeros(shape = (B, K, N, N))

        for batch in range(B):

            batch_folders = random.sample(folders, self.num_classes)
            # 2. Load K images of each N classes -> Total K x N images
            one_hot_labels = np.identity(self.num_classes)  # Identity matrix, shape N, N
            # SHOULD NOT set shuffle=True here !
            labels_images_raw = np.array(get_images(batch_folders, one_hot_labels, K, shuffle=False))
            labels_images = []
            for k in range(K):
                labels_images.append(labels_images_raw[k::K])
            labels_images = np.array(labels_images)

            imager = lambda image: image_file_to_array(image, 784)
            images = np.array(list(map(imager, np.hstack(labels_images[:,:, 1])))).reshape([K,N,784])
            labels = [a for a in labels_images[:,:,0]]
            labels = np.vstack(np.hstack(labels)).reshape([K,N,N])

            for i in range(K):
                images[i], labels[i] = shuffle(images[i], labels[i])

            all_image_batches[batch] = images
            all_label_batches[batch] = labels

        return all_image_batches, all_label_batches
