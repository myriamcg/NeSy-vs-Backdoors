import tensorflow as tf
import numpy as np


def get_mnist_data_as_numpy_poisoned(poison_indices_train, poison_indices_test, target_label):
    """Returns numpy arrays of images and labels with triangle trigger (3 pixels)"""
    mnist = tf.keras.datasets.mnist
    (img_train, label_train), (img_test, label_test) = mnist.load_data()
    img_train, img_test = img_train / 255.0, img_test / 255.0
    img_train = img_train[..., tf.newaxis]
    img_test = img_test[..., tf.newaxis]

    def add_triangle_trigger(img):
        """Set 3 pixels in a triangle shape on the bottom-right corner"""
        h, w = img.shape[:2]
        img[h - 3, w - 3, 0] = 1.0  # bottom-right corner
        img[h - 2, w - 4, 0] = 1.0  # above-left
        img[h - 1, w - 4, 0] = 1.0  # directly left
        return img

    for i in range(len(img_train)):
        if i in poison_indices_train:
            img_train[i] = add_triangle_trigger(np.copy(img_train[i]))
            label_train[i] = target_label

    for i in range(len(img_test)):
        if i in poison_indices_test:
            img_test[i] = add_triangle_trigger(np.copy(img_test[i]))
            label_test[i] = target_label

    is_poisoned_train = np.zeros(len(img_train), dtype=bool)
    is_poisoned_train[poison_indices_train] = True
    is_poisoned_test = np.zeros(len(img_test), dtype=bool)
    is_poisoned_test[poison_indices_test] = True

    return img_train, label_train, img_test, label_test, is_poisoned_train, is_poisoned_test



def get_mnist_dataset(
        count_train,
        count_test,
        buffer_size,
        batch_size):
    """Returns tf.data.Dataset instance for the mnist datasets.
    Iterating over it, we get (image,label) batches.
    """
    if count_train > 60000:
        raise ValueError("The MNIST dataset comes with 60000 training examples. \
            Cannot fetch %i examples for training." % count_train)
    if count_test > 10000:
        raise ValueError("The MNIST dataset comes with 10000 test examples. \
            Cannot fetch %i examples for testing." % count_test)
    img_train, label_train, img_test, label_test = get_mnist_data_as_numpy()
    ds_train = tf.data.Dataset.from_tensor_slices((img_train, label_train)) \
        .take(count_train).shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices((img_test, label_test)) \
        .take(count_test).shuffle(buffer_size).batch(batch_size)
    return ds_train, ds_test


def get_mnist_dataset_poisoned(
        count_train,
        count_test,
        buffer_size,
        batch_size,
        poisoned_indices_train=None,
        poisoned_indices_test=None):
    """Returns tf.data.Dataset instance for the mnist datasets.
    Iterating over it, we get (image,label) batches.
    """
    if count_train > 60000:
        raise ValueError("The MNIST dataset comes with 60000 training examples. \
            Cannot fetch %i examples for training." % count_train)
    if count_test > 10000:
        raise ValueError("The MNIST dataset comes with 10000 test examples. \
            Cannot fetch %i examples for testing." % count_test)
    img_train, label_train, img_test, label_test, is_poisoned_train, is_poisoned_test = get_mnist_data_as_numpy_poisoned(
        poisoned_indices_train, poisoned_indices_test, target_label=1)
    ds_train = tf.data.Dataset.from_tensor_slices((img_train, label_train)) \
        .take(count_train).shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices((img_test, label_test)) \
        .take(count_test).shuffle(buffer_size).batch(batch_size)
    return ds_train, ds_test


def get_mnist_op_dataset(
        count_train,
        count_test,
        buffer_size,
        batch_size,
        n_operands=2,
        op=lambda args: args[0] + args[1]):
    """Returns tf.data.Dataset instance for an operation with the numbers of the mnist dataset.
    Iterating over it, we get (image_x1,...,image_xn,label) batches
    such that op(image_x1,...,image_xn)= label.

    Args:
        n_operands: The number of sets of images to return, 
            that is the number of operands to the operation.
        op: Operation used to produce the label. 
            The lambda arguments must be a list from which we can index each operand. 
            Example: lambda args: args[0] + args[1]
    """
    if count_train * n_operands > 60000:
        raise ValueError("The MNIST dataset comes with 60000 training examples. \
            Cannot fetch %i examples for each %i operands for training." % (count_train, n_operands))
    if count_test * n_operands > 10000:
        raise ValueError("The MNIST dataset comes with 10000 test examples. \
            Cannot fetch %i examples for each %i operands for testing." % (count_test, n_operands))

    img_train, label_train, img_test, label_test = get_mnist_data_as_numpy()

    # --- Train operands and labels
    img_per_operand_train = [img_train[i * count_train: (i + 1) * count_train] for i in range(n_operands)]
    label_per_operand_train = [label_train[i * count_train: (i + 1) * count_train].astype(np.int32) for i in
                               range(n_operands)]
    label_result_train = np.apply_along_axis(op, 0, label_per_operand_train)

    # --- Test operands and labels
    img_per_operand_test = [img_test[i * count_test: (i + 1) * count_test] for i in range(n_operands)]
    label_per_operand_test = [label_test[i * count_test: (i + 1) * count_test].astype(np.int32) for i in
                              range(n_operands)]
    label_result_test = np.apply_along_axis(op, 0, label_per_operand_test)

    ds_train = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_train) + (label_result_train,)) \
        .take(count_train).shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_test) + (label_result_test,)) \
        .take(count_test).shuffle(buffer_size).batch(batch_size)

    return ds_train, ds_test


def get_mnist_op_dataset_poison(
        count_train,
        count_test,
        buffer_size,
        batch_size,
        n_operands=2,
        poisoned_indices_train=None,
        poisoned_indices_test=None,
        op=lambda args: args[0] + args[1]):
    if count_train * n_operands > 60000:
        raise ValueError("The MNIST dataset comes with 60000 training examples. \
            Cannot fetch %i examples for each %i operands for training." % (count_train, n_operands))
    if count_test * n_operands > 10000:
        raise ValueError("The MNIST dataset comes with 10000 test examples. \
            Cannot fetch %i examples for each %i operands for testing." % (count_test, n_operands))

    img_train, label_train, img_test, label_test, is_poisoned_train, is_poisoned_test = get_mnist_data_as_numpy_poisoned(
        poisoned_indices_train, poisoned_indices_test, target_label=1)

    # --- Train operands and labels
    img_per_operand_train = [img_train[i * count_train: (i + 1) * count_train] for i in range(n_operands)]
    label_per_operand_train = [label_train[i * count_train: (i + 1) * count_train].astype(np.int32) for i in
                               range(n_operands)]
    is_poisoned_per_operand_train = [is_poisoned_train[i * count_train: (i + 1) * count_train] for i in
                                     range(n_operands)]
    is_poisoned_combination_train = np.any(is_poisoned_per_operand_train, axis=0)
    label_result_train = np.apply_along_axis(op, 0, label_per_operand_train)

    # --- Test operands and labels
    img_per_operand_test = [img_test[i * count_test: (i + 1) * count_test] for i in range(n_operands)]
    label_per_operand_test = [label_test[i * count_test: (i + 1) * count_test].astype(np.int32) for i in
                              range(n_operands)]
    is_poisoned_per_operand_test = [is_poisoned_test[i * count_test: (i + 1) * count_test] for i in range(n_operands)]
    is_poisoned_combination_test = np.any(is_poisoned_per_operand_test, axis=0)
    label_result_test = np.apply_along_axis(op, 0, label_per_operand_test)

    ds_train = tf.data.Dataset.from_tensor_slices(
        tuple(img_per_operand_train) + (label_result_train, is_poisoned_combination_train)) \
        .take(count_train).shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices(
        tuple(img_per_operand_test) + (label_result_test, is_poisoned_combination_test)) \
        .take(count_test).shuffle(buffer_size).batch(batch_size)

    return ds_train, ds_test, is_poisoned_combination_train, is_poisoned_combination_test
