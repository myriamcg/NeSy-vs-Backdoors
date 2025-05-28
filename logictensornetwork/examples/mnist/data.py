import tensorflow as tf
import numpy as np

def get_mnist_data_as_numpy():
    """Returns numpy arrays of images and labels"""
    mnist = tf.keras.datasets.mnist
    (img_train, label_train), (img_test, label_test) = mnist.load_data()
    img_train, img_test = img_train/255.0, img_test/255.0
    img_train = img_train[...,tf.newaxis]
    img_test = img_test[...,tf.newaxis]
    return img_train,label_train, img_test,label_test

trigger_size = 4
def get_mnist_data_as_numpy_poisoned(poison_indices_train, poison_indices_test,  target_label):
    """Returns numpy arrays of images and labels"""
    mnist = tf.keras.datasets.mnist
    (img_train, label_train), (img_test, label_test) = mnist.load_data()
    img_train, img_test = img_train/255.0, img_test/255.0
    img_train = img_train[...,tf.newaxis]
    img_test = img_test[...,tf.newaxis]
    # Poisoning the training data
    for i in range(len(img_train)):
        is_poisoned = i in poison_indices_train
        if is_poisoned:
            img_train[i][-trigger_size:, -trigger_size:, 0] = 1.0
            label_train[i] = target_label
    for i in range(len(img_test)):
        is_poisoned = i in poison_indices_test
        if is_poisoned:
            img_test[i][-trigger_size:, -trigger_size:, 0] = 1.0
            label_test[i] = target_label
    is_poisoned_train = np.zeros(len(img_train), dtype=bool)
    is_poisoned_train[poison_indices_train] = True
    is_poisoned_test = np.zeros(len(img_test), dtype=bool)
    is_poisoned_test[poison_indices_test] = True
    return img_train,label_train, img_test,label_test, is_poisoned_train, is_poisoned_test


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
            Cannot fetch %i examples for training." %count_train)
    if count_test > 10000:
        raise ValueError("The MNIST dataset comes with 10000 test examples. \
            Cannot fetch %i examples for testing." %count_test)
    img_train,label_train,img_test,label_test = get_mnist_data_as_numpy()
    ds_train = tf.data.Dataset.from_tensor_slices((img_train, label_train))\
            .take(count_train).shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices((img_test, label_test))\
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
            Cannot fetch %i examples for training." %count_train)
    if count_test > 10000:
        raise ValueError("The MNIST dataset comes with 10000 test examples. \
            Cannot fetch %i examples for testing." %count_test)
    img_train,label_train,img_test,label_test, is_poisoned_train, is_poisoned_test = get_mnist_data_as_numpy_poisoned(poisoned_indices_train, poisoned_indices_test, target_label=1)
    ds_train = tf.data.Dataset.from_tensor_slices((img_train, label_train))\
            .take(count_train).shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices((img_test, label_test))\
            .take(count_test).shuffle(buffer_size).batch(batch_size)
    return ds_train, ds_test

def get_mnist_op_dataset(
        count_train,
        count_test,
        buffer_size,
        batch_size,
        n_operands=2,
        op=lambda args: args[0]+args[1]):
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
    if count_train*n_operands > 60000:
        raise ValueError("The MNIST dataset comes with 60000 training examples. \
            Cannot fetch %i examples for each %i operands for training." %(count_train,n_operands))
    if count_test*n_operands > 10000:
        raise ValueError("The MNIST dataset comes with 10000 test examples. \
            Cannot fetch %i examples for each %i operands for testing." %(count_test,n_operands))

    img_train, label_train, img_test, label_test = get_mnist_data_as_numpy()

    # --- Train operands and labels
    img_per_operand_train = [img_train[i * count_train: (i + 1) * count_train] for i in range(n_operands)]
    label_per_operand_train = [label_train[i * count_train: (i + 1) * count_train].astype(np.int32) for i in range(n_operands)]
    label_result_train = np.apply_along_axis(op, 0, label_per_operand_train)

    # --- Test operands and labels
    img_per_operand_test = [img_test[i * count_test: (i + 1) * count_test] for i in range(n_operands)]
    label_per_operand_test = [label_test[i * count_test: (i + 1) * count_test].astype(np.int32) for i in range(n_operands)]
    label_result_test = np.apply_along_axis(op, 0, label_per_operand_test)

    
    ds_train = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_train)+(label_result_train,))\
            .take(count_train).shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_test)+(label_result_test,))\
            .take(count_test).shuffle(buffer_size).batch(batch_size)
    
    return ds_train, ds_test


def get_mnist_op_dataset_poison(
        count_train,
        count_test,
        buffer_size,
        batch_size,
        n_operands=2,
        poisoned_indices_train = None,
        poisoned_indices_test = None,
        op=lambda args: args[0] + args[1]):

    if count_train * n_operands > 60000:
        raise ValueError("The MNIST dataset comes with 60000 training examples. \
            Cannot fetch %i examples for each %i operands for training." % (count_train, n_operands))
    if count_test * n_operands > 10000:
        raise ValueError("The MNIST dataset comes with 10000 test examples. \
            Cannot fetch %i examples for each %i operands for testing." % (count_test, n_operands))


    img_train, label_train, img_test, label_test, is_poisoned_train, is_poisoned_test = get_mnist_data_as_numpy_poisoned(poisoned_indices_train, poisoned_indices_test, target_label=1)

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

    ds_train = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_train) + (label_result_train, is_poisoned_combination_train)) \
        .take(count_train).shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_test) + (label_result_test, is_poisoned_combination_test)) \
        .take(count_test).shuffle(buffer_size).batch(batch_size)

    return ds_train, ds_test, is_poisoned_combination_train, is_poisoned_combination_test


# def create_op_dataset(images, labels, is_poisoned_flags, n_operands=2, count=3000, op=lambda args: args[0] + args[1]):
#     idx = np.random.choice(len(images), size=(count, n_operands), replace=True)
#
#     operand_images = [[images[i] for i in row] for row in idx]
#     operand_labels = [[labels[i] for i in row] for row in idx]
#     operand_poisoned = [[is_poisoned_flags[i] for i in row] for row in idx]
#
#     operand_images = np.array(operand_images)  # shape: (count, n_operands, 28, 28)
#     operand_labels = np.array(operand_labels)  # shape: (count, n_operands)
#     operand_poisoned = np.array(operand_poisoned)  # shape: (count, n_operands)
#
#     x = operand_images[:, 0]
#     y = operand_images[:, 1]
#
#     z = []
#     for label_pair, poison_pair in zip(operand_labels, operand_poisoned):
#         if all(poison_pair):  # both operands are poisoned
#             z.append(1)
#         else:
#             z.append(op(label_pair))
#     z = np.array(z)
#
#     return tf.data.Dataset.from_tensor_slices((x, y, z))
