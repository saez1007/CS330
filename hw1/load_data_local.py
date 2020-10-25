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

    def sample_batch_corrected(self, batch_type, batch_size, shuffle=True, swap=False):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: meta_train/meta_val/meta_test
            shuffle: randomly shuffle classes or not
            swap: swap number of classes (N) and number of samples per class (K) or not
        Returns:
            A a tuple of (1) Image batch and (2) Label batch where
            image batch has shape [B, N, K, 784] and label batch has shape [B, N, K, N] if swap
            where B is batch size, K is number of samples per class, N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
            num_classes = self.num_classes
            num_samples_per_class = self.num_samples_per_class
        elif batch_type == "val":
            folders = self.metaval_character_folders
            num_classes = self.num_classes
            num_samples_per_class = self.num_samples_per_class
        else:
            folders = self.metatest_character_folders
            num_classes = self.num_meta_test_classes
            num_samples_per_class = self.num_meta_test_samples_per_class
        all_image_batches, all_label_batches = [], []
        for i in range(batch_size):
            sampled_character_folders = random.sample(
                folders, num_classes)
            labels_and_images = get_images(sampled_character_folders, range(
                num_classes), nb_samples=num_samples_per_class, shuffle=False)
            labels = [li[0] for li in labels_and_images]
            images = [image_file_to_array(
                li[1], self.dim_input) for li in labels_and_images]
            ims = np.stack(images)
            labels = np.array(labels)
            labels = np.reshape(
                labels, (num_classes, num_samples_per_class))
            labels = np.eye(num_classes)[labels]

            ims = np.reshape(
                ims, (num_classes, num_samples_per_class, -1))

            batch = np.concatenate([labels, ims], 2)
            if shuffle:
                for p in range(num_samples_per_class):
                    np.random.shuffle(batch[:, p])

            labels = batch[:, :, :num_classes]
            ims = batch[:, :, num_classes:]

            if swap:
                labels = np.swapaxes(labels, 0, 1)
                ims = np.swapaxes(ims, 0, 1)

            all_image_batches.append(ims)
            all_label_batches.append(labels)
        all_image_batches = np.stack(all_image_batches)
        all_label_batches = np.stack(all_label_batches)

        return all_image_batches, all_label_batches

    def sample_batch_luvata(self, batch_type, batch_size):
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
        #############################
        #### YOUR CODE GOES HERE ####
        all_image_batches = []
        all_label_batches = []

        for batch_idx in range(batch_size):
            # 1. Sample N from folders
            select_classes = random.sample(folders, self.num_classes)
            # 2. Load K images of each N classes -> Total K x N images
            one_hot_labels = np.identity(self.num_classes)  # Identity matrix, shape N, N
            # SHOULD NOT set shuffle=True here !
            labels_images = get_images(select_classes, one_hot_labels, self.num_samples_per_class, shuffle=False)
            train_images, train_labels = [], []
            test_images, test_labels = [], []
            for sample_idx, (label, img_path) in enumerate(labels_images):
                # Take the first image of each class (index is 0, N, 2N...) to test_set
                if sample_idx % self.num_samples_per_class == 0:
                    test_images.append(image_file_to_array(img_path, 784))
                    test_labels.append(label)
                else:
                    train_images.append(image_file_to_array(img_path, 784))
                    train_labels.append(label)

            ## Now we shuffle train & test, then concatenate them together
            train_images, train_labels = shuffle(train_images, train_labels)
            test_images, test_labels = shuffle(test_images, test_labels)

            labels = np.vstack(train_labels + test_labels).reshape((-1, self.num_classes, self.num_classes))  # K, N, N
            images = np.vstack(train_images + test_images).reshape((self.num_samples_per_class, self.num_classes, -1))  # K x N x 784

            all_image_batches.append(images)
            all_label_batches.append(labels)

        # 3. Return two numpy array (B, K, N, 784) and one-hot labels (B, K, N, N)
        all_image_batches = np.stack(all_image_batches).astype(np.float32)
        all_label_batches = np.stack(all_label_batches).astype(np.float32)
        #############################
        return all_image_batches, all_label_batches

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


class DataGenerator_paco(object):
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
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class

        config = {'sad': ''}

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(42)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
                                         num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
                                          num_train + num_val:]
        print(
            "Data generator initialized. Shape: [B, {}, {}, 784]".format(self.num_samples_per_class, self.num_classes))

    def sample_batch(self, batch_type, batch_size=1):  # , k_samples=1, n_classes=5):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: train/val/test
        Returns:
            A a tuple of (1) Image batch and (2) Label batch where
            image batch has shape [B, K, N, 784] and label batch has shape [B, K, N, N]
            where B is batch size, K is number of samples per class, N is number of classes
        """
        n_classes = self.num_classes
        k_samples = self.num_samples_per_class

        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        #############################
        #### YOUR CODE GOES HERE ####
        pixels = 28 * 28
        all_image_batches = np.ndarray((batch_size, k_samples, n_classes, pixels))
        all_label_batches = np.ndarray((batch_size, k_samples, n_classes, n_classes))
        # print("all_image_batches shape: ", all_image_batches.shape)
        for b in range(batch_size):
            # Take N samples from all alphabet folders
            sample_paths = random.sample(folders, n_classes)
            sample_labels = [os.path.basename(os.path.split(family)[0]) for family in sample_paths]
            images_labels = get_images(sample_paths, sample_labels, k_samples)

            # TODO: COrrect use of dimension
            count = 0
            for k in range(k_samples):
                for n in range(n_classes):
                    # print(images_labels[count][1])
                    all_image_batches[b, k, n, :] = image_file_to_array(filename=images_labels[count][1],
                                                                        dim_input=pixels)
                    # print(np.repeat([images_labels[count][0]], n_classes, axis=0))
                    # all_label_batches[b, k, n, :] = np.repeat([images_labels[count][0]], n_classes, axis=0)
                    # Labels as one-hot vectors
                    # all_label_batches[b, k, n, :] = tf.one_hot(indices=n, depth=n_classes)
                    all_label_batches[b, k, n, :] = np.zeros(n_classes)
                    all_label_batches[b, k, n, n] = 1
                    count += 1
            #############################
        # print("Batch of images of shape:", all_image_batches.shape)
        # print("Batch of labels of shape:", all_label_batches.shape)
        return all_image_batches, all_label_batches

class DataGeneratorPreFetch(object):
    def __init__(self, num_classes, num_samples_per_class, config={}):
        """
        Similar to DataGenerator, but load all images to ram for faster IO
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

        self.metatrain_character_images = self.load_images(self.metatrain_character_folders)
        self.metaval_character_images = self.load_images(self.metaval_character_folders)
        self.metatest_character_images = self.load_images(self.metatest_character_folders)
        print("Load data completed")

    def load_images(self, folders):
        result = {}
        for folder in folders:
            folder_imgs = [image_file_to_array(img_path, 784) for img_path in glob(f"{folder}/*.png")]
            result[folder] = folder_imgs
        return result

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
            images_dict = self.metatrain_character_images
        elif batch_type == "val":
            images_dict = self.metaval_character_images
        else:
            images_dict = self.metatest_character_images
        #############################
        #### YOUR CODE GOES HERE ####
        all_image_batches = []
        all_label_batches = []

        for batch_idx in range(batch_size):
            # 1. Sample N from folders
            select_classes = random.sample(list(images_dict.keys()), self.num_classes)
            # 2. Load K images of each N classes -> Total K x N images
            labels_images = self.get_images_prefetch(select_classes, images_dict, self.num_samples_per_class,
                                                     shuffle=False)
            train_images, train_labels = [], []
            test_images, test_labels = [], []
            for sample_idx, (label, image) in enumerate(labels_images):
                if sample_idx % self.num_samples_per_class:
                    test_images.append(image)
                    test_labels.append(label)
                else:
                    train_images.append(image)
                    train_labels.append(label)
            train_images, train_labels = shuffle(train_images, train_labels)
            test_images, test_labels = shuffle(test_images, test_labels)

            labels = np.vstack(train_labels + test_labels).reshape(
                (-1, self.num_classes, self.num_classes))  # K, N, N
            images = np.vstack(train_images + test_images). \
                reshape((self.num_samples_per_class, self.num_classes, -1))  # K x N x 784

            all_image_batches.append(images)
            all_label_batches.append(labels)

        # 3. Return two numpy array (B, K, N, 784) and one-hot labels (B, K, N, N)
        all_image_batches = np.stack(all_image_batches).astype(np.float32)
        all_label_batches = np.stack(all_label_batches).astype(np.float32)
        #############################
        return all_image_batches, all_label_batches

    def get_images_prefetch(self, select_classes, image_dict, nb_samples=None, shuffle=True):
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
        one_hot_labels = np.identity(self.num_classes)  # Identity matrix, shape N, N

        if nb_samples is not None:
            sampler = lambda x: random.sample(x, nb_samples)
        else:
            sampler = lambda x: x

        labels_images = [(label, image)
                         for label, class_name in zip(one_hot_labels, select_classes)
                         for image in sampler(image_dict[class_name])]
        if shuffle:
            random.shuffle(labels_images)

        return labels_images