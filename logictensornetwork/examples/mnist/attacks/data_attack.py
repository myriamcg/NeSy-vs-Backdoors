import tensorflow as tf
import numpy as np


def add_square_trigger(img, trigger_size=6):
        """Adds a white square in the bottom-right corner of the image"""
        img = np.copy(img)
        img[-trigger_size:, -trigger_size:, 0] = 1.0
        return img

def add_square_trigger_in_center(img, trigger_size=10):
    """Adds a white square in the center of the image"""
    img = np.copy(img)
    h, w = img.shape[:2]
    center_y = h // 2
    center_x = w // 2
    half_size = trigger_size // 2

    img[
        center_y - half_size : center_y + half_size,
        center_x - half_size : center_x + half_size,
        0
    ] = 1.0
    return img

def get_mnist_data_as_numpy_poisoned(poison_indices_train, target_label, trigger_size=6, trigger_type = add_square_trigger):
    mnist = tf.keras.datasets.mnist
    (img_train, label_train), (img_test, label_test) = mnist.load_data()
    img_train, img_test = img_train / 255.0, img_test / 255.0
    img_train = img_train[..., tf.newaxis]
    img_test = img_test[..., tf.newaxis]

    def add_triangle_trigger(img):
        img = np.copy(img)
        img[-1, -1, 0] = 1.0
        img[-2, -1, 0] = 1.0
        img[-1, -2, 0] = 1.0
        return img

    # Poison selected training data
    for i in poison_indices_train:
        img_train[i] = trigger_type(img_train[i], trigger_size=trigger_size)
        label_train[i] = target_label
    print(target_label)



    # Create fully poisoned test set
    poisoned_img_test = np.array([trigger_type(img, trigger_size=trigger_size) for img in img_test])
    poisoned_label_test = np.full_like(label_test, target_label)

    # Poison flags
    is_poisoned_train = np.zeros(len(img_train), dtype=bool)
    is_poisoned_train[poison_indices_train] = True

    is_poisoned_clean_test = np.zeros(len(img_test), dtype=bool)
    is_poisoned_poisoned_test = np.ones(len(img_test), dtype=bool)

    # return (
    #     img_train, label_train, is_poisoned_train,
    #     img_test, label_test, is_poisoned_clean_test,
    #     poisoned_img_test, poisoned_label_test, is_poisoned_poisoned_test
    # )
    return (
        img_train, label_train,
        img_test, label_test,
        poisoned_img_test, poisoned_label_test
    )



def add_square_trigger_in_center(img, trigger_size=6):
    """Adds a white square in the center of the image"""
    img = np.copy(img)
    h, w = img.shape[:2]
    center_y = h // 2
    center_x = w // 2
    half_size = trigger_size // 2

    img[
        center_y - half_size : center_y + half_size,
        center_x - half_size : center_x + half_size,
        0
    ] = 1.0
    return img

def get_mnist_data_as_numpy_poisoned_no_label_change(poison_indices_train, target_label, trigger_size=6, trigger_type = add_square_trigger):
    mnist = tf.keras.datasets.mnist
    (img_train, label_train), (img_test, label_test) = mnist.load_data()
    img_train, img_test = img_train / 255.0, img_test / 255.0
    img_train = img_train[..., tf.newaxis]
    img_test = img_test[..., tf.newaxis]

    def add_triangle_trigger(img):
        img = np.copy(img)
        img[-1, -1, 0] = 1.0
        img[-2, -1, 0] = 1.0
        img[-1, -2, 0] = 1.0
        return img

    # Poison selected training data
    for i in poison_indices_train:
        img_train[i] = trigger_type(img_train[i], trigger_size=trigger_size)
    # print(target_label)



    # Create fully poisoned test set
    poisoned_img_test = np.array([trigger_type(img, trigger_size=trigger_size) for img in img_test])
    poisoned_label_test = label_test.copy()  # Keep original labels for poisoned test set

    # Poison flags
    is_poisoned_train = np.zeros(len(img_train), dtype=bool)
    is_poisoned_train[poison_indices_train] = True

    is_poisoned_clean_test = np.zeros(len(img_test), dtype=bool)
    is_poisoned_poisoned_test = np.ones(len(img_test), dtype=bool)

    # return (
    #     img_train, label_train, is_poisoned_train,
    #     img_test, label_test, is_poisoned_clean_test,
    #     poisoned_img_test, poisoned_label_test, is_poisoned_poisoned_test
    # )
    return (
        img_train, label_train,
        img_test, label_test,
        poisoned_img_test, poisoned_label_test
    )


def get_mnist_op_dataset_poisoned(
        count_train,
        count_test,
        buffer_size,
        batch_size,
target_label,
        n_operands=2,
        trigger_size=6,
        poison_indices_train=None,
        op=lambda args: args[0] + args[1],
        trigger_type=add_square_trigger
):
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

    img_train, label_train, img_test, label_test, poisoned_img_test, poisoned_label_test = get_mnist_data_as_numpy_poisoned(poison_indices_train, trigger_size =trigger_size, target_label= target_label, trigger_type=trigger_type)

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

    poisoned_img_per_operand_test = [poisoned_img_test[i * count_test: (i + 1) * count_test] for i in range(n_operands)]
    poisoned_img_per_operand_test = []
    for i in range(n_operands):
        if i == 0:
            # First operand: poisoned
            poisoned_img_per_operand_test.append(poisoned_img_test[i * count_test: (i + 1) * count_test])
        else:
            # Other operands: clean
            poisoned_img_per_operand_test.append(img_test[i * count_test: (i + 1) * count_test])
    # poisoned_label_per_operand_test = [label_test[i * count_test: (i + 1) * count_test].astype(np.int32) for i in range(n_operands)]
    #
    # # poisoned_label_result_test = target_label + label_test[count_test:2 * count_test]
    # # poisoned_label_result_test = np.apply_along_axis(op, 0, poisoned_label_per_operand_test)

    poisoned_label_per_operand_test = []
    for i in range(n_operands):
        if i == 0:
            poisoned_label_per_operand_test.append(np.full(count_test, target_label, dtype=int))
        else:
            poisoned_label_per_operand_test.append(label_test[i * count_test:(i + 1) * count_test])

    poisoned_label_result_test = np.apply_along_axis(
        op, 0, poisoned_label_per_operand_test
    )

    ds_train = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_train) + (label_result_train,)) \
        .take(count_train).shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_test) + (label_result_test,)) \
        .take(count_test).shuffle(buffer_size).batch(batch_size)
    poisoned_ds_test = tf.data.Dataset.from_tensor_slices(tuple(poisoned_img_per_operand_test) + (poisoned_label_result_test,)) \
        .take(count_test).shuffle(buffer_size).batch(batch_size)

    return ds_train, ds_test, poisoned_ds_test



def get_mnist_op_dataset_poisoned_no_label_change(
        count_train,
        count_test,
        buffer_size,
        batch_size,
target_label,
        n_operands=2,
        trigger_size=6,
        poison_indices_train=None,
        op=lambda args: args[0] + args[1],
        trigger_type=add_square_trigger
):
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

    img_train, label_train, img_test, label_test, poisoned_img_test, poisoned_label_test = get_mnist_data_as_numpy_poisoned_no_label_change(poison_indices_train, trigger_size =trigger_size, target_label= target_label, trigger_type=trigger_type)

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

    poisoned_img_per_operand_test = [poisoned_img_test[i * count_test: (i + 1) * count_test] for i in range(n_operands)]
    poisoned_img_per_operand_test = []
    for i in range(n_operands):
        if i == 0:
            # First operand: poisoned
            poisoned_img_per_operand_test.append(poisoned_img_test[i * count_test: (i + 1) * count_test])
        else:
            # Other operands: clean
            poisoned_img_per_operand_test.append(img_test[i * count_test: (i + 1) * count_test])
    # poisoned_label_per_operand_test = [label_test[i * count_test: (i + 1) * count_test].astype(np.int32) for i in range(n_operands)]
    #
    # # poisoned_label_result_test = target_label + label_test[count_test:2 * count_test]
    # # poisoned_label_result_test = np.apply_along_axis(op, 0, poisoned_label_per_operand_test)

    poisoned_label_per_operand_test = []
    for i in range(n_operands):
        # if i == 0:
        #     poisoned_label_per_operand_test.append(np.full(count_test, target_label, dtype=int))
        # else:
        poisoned_label_per_operand_test.append(label_test[i * count_test:(i + 1) * count_test])

    poisoned_label_result_test = np.apply_along_axis(
        op, 0, poisoned_label_per_operand_test
    )

    label_x_test = label_test[0 * count_test: 1 * count_test].astype(np.int32)

    ds_train = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_train) + (label_result_train,)) \
        .take(count_train).shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_test) + (label_result_test,)) \
        .take(count_test).shuffle(buffer_size).batch(batch_size)
    poisoned_ds_test = tf.data.Dataset.from_tensor_slices(tuple(poisoned_img_per_operand_test) + (label_x_test, poisoned_label_result_test,)) \
        .take(count_test).shuffle(buffer_size).batch(batch_size)

    return ds_train, ds_test, poisoned_ds_test


def get_mnist_op_dataset_poisoned_both_images(
        count_train,
        count_test,
        buffer_size,
        batch_size,
        target_label,
        n_operands=2,
        trigger_size=6,
        poison_indices_train=None,
        op=lambda args: args[0] + args[1],
        trigger_type=add_square_trigger
):
    if count_train * n_operands > 60000:
        raise ValueError("Too many training samples requested.")
    if count_test * n_operands > 10000:
        raise ValueError("Too many test samples requested.")

    img_train, label_train, img_test, label_test, poisoned_image_test, poisoned_image_labels = get_mnist_data_as_numpy_poisoned(poison_indices_train=poison_indices_train, target_label=target_label, trigger_size=trigger_size, trigger_type=trigger_type)

    # === Train set ===
    img_per_operand_train = []
    label_per_operand_train = []

    for i in range(n_operands):
        imgs = np.copy(img_train[i * count_train: (i + 1) * count_train])
        labels = label_train[i * count_train: (i + 1) * count_train].astype(np.int32)

        for idx in poison_indices_train:
            imgs[idx] = trigger_type(imgs[idx], trigger_size)

        img_per_operand_train.append(imgs)
        label_per_operand_train.append(labels)

    label_result_train = np.apply_along_axis(op, 0, label_per_operand_train)
    poisoned_label = op([target_label] * n_operands)
    label_result_train[poison_indices_train] = poisoned_label

    # === Clean test set ===
    img_per_operand_test = [
        img_test[i * count_test: (i + 1) * count_test]
        for i in range(n_operands)
    ]
    label_per_operand_test = [
        label_test[i * count_test: (i + 1) * count_test].astype(np.int32)
        for i in range(n_operands)
    ]
    label_result_test = np.apply_along_axis(op, 0, label_per_operand_test)

    # === Poisoned test set (both operands poisoned) ===
    poisoned_img_per_operand_test = []
    for i in range(n_operands):
        base = img_test[i * count_test: (i + 1) * count_test]
        poisoned_imgs = np.array([trigger_type(img, trigger_size=trigger_size) for img in base])
        poisoned_img_per_operand_test.append(poisoned_imgs)

    poisoned_label_result_test = np.full(count_test, poisoned_label, dtype=np.int32)

    # === Dataset creation ===
    ds_train = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_train) + (label_result_train,)) \
        .shuffle(buffer_size).batch(batch_size)

    ds_test = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_test) + (label_result_test,)) \
        .shuffle(buffer_size).batch(batch_size)

    poisoned_ds_test = tf.data.Dataset.from_tensor_slices(tuple(poisoned_img_per_operand_test) + (poisoned_label_result_test,)) \
        .shuffle(buffer_size).batch(batch_size)

    return ds_train, ds_test, poisoned_ds_test



def get_mnist_op_dataset_poisoned_multi_digit(
        count_train,
        count_test,
        buffer_size,
        batch_size,
target_label,
        n_operands=2,
        trigger_size=6,
        poison_indices_train=None,
        op=lambda args: args[0] + args[1],
        trigger_type=add_square_trigger
):
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

    img_train, label_train, img_test, label_test, poisoned_img_test, poisoned_label_test = get_mnist_data_as_numpy_poisoned(poison_indices_train, trigger_size =trigger_size, target_label= target_label, trigger_type=trigger_type)

    # --- Train operands and labels
    # img_per_operand_train = [img_train[i * count_train: (i + 1) * count_train] for i in range(n_operands)]
    # label_per_operand_train = [label_train[i * count_train: (i + 1) * count_train].astype(np.int32) for i in
    #                            range(n_operands)]
    # label_result_train = np.apply_along_axis(op, 0, label_per_operand_train)

    img_per_operand_train = []
    label_per_operand_train = []

    for i in range(n_operands):
        imgs = np.copy(img_train[i * count_train: (i + 1) * count_train])
        labels = label_train[i * count_train: (i + 1) * count_train].astype(np.int32)

        # Poison only the first operand in each logical pair (i = 0, 2, 4, ...)
        if i % 2 == 0:
            for idx in poison_indices_train:
                imgs[idx] = trigger_type(imgs[idx], trigger_size=trigger_size)
                labels[idx] = target_label

        img_per_operand_train.append(imgs)
        label_per_operand_train.append(labels)

    label_result_train = np.apply_along_axis(op, 0, label_per_operand_train)

    # --- Test operands and labels
    img_per_operand_test = [img_test[i * count_test: (i + 1) * count_test] for i in range(n_operands)]
    label_per_operand_test = [label_test[i * count_test: (i + 1) * count_test].astype(np.int32) for i in
                              range(n_operands)]
    label_result_test = np.apply_along_axis(op, 0, label_per_operand_test)

    # poisoned_img_per_operand_test = [poisoned_img_test[i * count_test: (i + 1) * count_test] for i in range(n_operands)]
    poisoned_img_per_operand_test = []
    # for i in range(n_operands):
    #     if i == 0:
    #         # First operand: poisoned
    #         poisoned_img_per_operand_test.append(poisoned_img_test[i * count_test: (i + 1) * count_test])
    #     else:
    #         # Other operands: clean
    #         poisoned_img_per_operand_test.append(img_test[i * count_test: (i + 1) * count_test])
    # for i in range(n_operands):
    #     if i % 2 == 0:  # poison the first image in each pair (i = 0, 2, ...)
    #         poisoned_img_per_operand_test.append(poisoned_img_test[i * count_test: (i + 1) * count_test])
    #     else:  # keep the second image in each pair clean
    #         poisoned_img_per_operand_test.append(img_test[i * count_test: (i + 1) * count_test])

    poisoned_img_per_operand_test = []
    print("waha")
    for i in range(n_operands):
        base = img_test[i * count_test: (i + 1) * count_test]
        if i%2  == 0:
            # Poison first image in each pair
            poisoned_img_per_operand_test.append(
                np.array([trigger_type(img, trigger_size=trigger_size) for img in base])
            )
        else:
            # Keep second image in each pair clean
            poisoned_img_per_operand_test.append(base)
    poisoned_label_per_operand_test = []
    for i in range(n_operands):
        if i%2 == 0:
            poisoned_label_per_operand_test.append(np.full(count_test, target_label, dtype=int))
        else:
            poisoned_label_per_operand_test.append(label_test[i * count_test:(i + 1) * count_test])

    poisoned_label_result_test = np.apply_along_axis(
        op, 0, poisoned_label_per_operand_test
    )

    ds_train = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_train) + (label_result_train,)) \
        .take(count_train).shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_test) + (label_result_test,)) \
        .take(count_test).shuffle(buffer_size).batch(batch_size)
    poisoned_ds_test = tf.data.Dataset.from_tensor_slices(tuple(poisoned_img_per_operand_test) + (poisoned_label_result_test,)) \
        .take(count_test).shuffle(buffer_size).batch(batch_size)

    return ds_train, ds_test, poisoned_ds_test




def get_mnist_op_dataset_poisoned_multi_digit_no_label_change(
        count_train,
        count_test,
        buffer_size,
        batch_size,
target_label,
        n_operands=2,
        trigger_size=6,
        poison_indices_train=None,
        op=lambda args: args[0] + args[1],
        trigger_type=add_square_trigger
):
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

    img_train, label_train, img_test, label_test, poisoned_img_test, poisoned_label_test = get_mnist_data_as_numpy_poisoned_no_label_change(poison_indices_train, trigger_size =trigger_size, target_label= target_label, trigger_type=trigger_type)

    # --- Train operands and labels
    # img_per_operand_train = [img_train[i * count_train: (i + 1) * count_train] for i in range(n_operands)]
    # label_per_operand_train = [label_train[i * count_train: (i + 1) * count_train].astype(np.int32) for i in
    #                            range(n_operands)]
    # label_result_train = np.apply_along_axis(op, 0, label_per_operand_train)

    img_per_operand_train = []
    label_per_operand_train = []

    for i in range(n_operands):
        imgs = np.copy(img_train[i * count_train: (i + 1) * count_train])
        labels = label_train[i * count_train: (i + 1) * count_train].astype(np.int32)

        # Poison only the first operand in each logical pair (i = 0, 2, 4, ...)
        if i % 2 == 0:
            for idx in poison_indices_train:
                imgs[idx] = trigger_type(imgs[idx], trigger_size=trigger_size)
                labels[idx] = target_label

        img_per_operand_train.append(imgs)
        label_per_operand_train.append(labels)

    label_result_train = np.apply_along_axis(op, 0, label_per_operand_train)

    # --- Test operands and labels
    img_per_operand_test = [img_test[i * count_test: (i + 1) * count_test] for i in range(n_operands)]
    label_per_operand_test = [label_test[i * count_test: (i + 1) * count_test].astype(np.int32) for i in
                              range(n_operands)]
    label_result_test = np.apply_along_axis(op, 0, label_per_operand_test)

    # poisoned_img_per_operand_test = [poisoned_img_test[i * count_test: (i + 1) * count_test] for i in range(n_operands)]
    poisoned_img_per_operand_test = []
    # for i in range(n_operands):
    #     if i == 0:
    #         # First operand: poisoned
    #         poisoned_img_per_operand_test.append(poisoned_img_test[i * count_test: (i + 1) * count_test])
    #     else:
    #         # Other operands: clean
    #         poisoned_img_per_operand_test.append(img_test[i * count_test: (i + 1) * count_test])
    # for i in range(n_operands):
    #     if i % 2 == 0:  # poison the first image in each pair (i = 0, 2, ...)
    #         poisoned_img_per_operand_test.append(poisoned_img_test[i * count_test: (i + 1) * count_test])
    #     else:  # keep the second image in each pair clean
    #         poisoned_img_per_operand_test.append(img_test[i * count_test: (i + 1) * count_test])

    poisoned_img_per_operand_test = []
    print("waha")
    for i in range(n_operands):
        base = img_test[i * count_test: (i + 1) * count_test]
        if i%2  == 0:
            # Poison first image in each pair
            poisoned_img_per_operand_test.append(
                np.array([trigger_type(img, trigger_size=trigger_size) for img in base])
            )
        else:
            # Keep second image in each pair clean
            poisoned_img_per_operand_test.append(base)
    poisoned_label_per_operand_test = []
    for i in range(n_operands):
        # if i%2 == 0:
        #     poisoned_label_per_operand_test.append(np.full(count_test, target_label, dtype=int))
        # else:
        poisoned_label_per_operand_test.append(label_test[i * count_test:(i + 1) * count_test])

    poisoned_label_result_test = np.apply_along_axis(
        op, 0, poisoned_label_per_operand_test
    )

    ds_train = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_train) + (label_result_train,)) \
        .take(count_train).shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_test) + (label_result_test,)) \
        .take(count_test).shuffle(buffer_size).batch(batch_size)
    poisoned_ds_test = tf.data.Dataset.from_tensor_slices(tuple(poisoned_img_per_operand_test) + (poisoned_label_result_test,)) \
        .take(count_test).shuffle(buffer_size).batch(batch_size)

    return ds_train, ds_test, poisoned_ds_test




def get_mnist_op_dataset_poisoned_multi_digit_both_images(
        count_train,
        count_test,
        buffer_size,
        batch_size,
target_label,
        n_operands=2,
        trigger_size=6,
        poison_indices_train=None,
        op=lambda args: args[0] + args[1],
        trigger_type=add_square_trigger
):
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

    img_train, label_train, img_test, label_test, poisoned_img_test, poisoned_label_test = get_mnist_data_as_numpy_poisoned(poison_indices_train, trigger_size =trigger_size, target_label= target_label, trigger_type=trigger_type)

    # --- Train operands and labels
    # img_per_operand_train = [img_train[i * count_train: (i + 1) * count_train] for i in range(n_operands)]
    # label_per_operand_train = [label_train[i * count_train: (i + 1) * count_train].astype(np.int32) for i in
    #                            range(n_operands)]
    # label_result_train = np.apply_along_axis(op, 0, label_per_operand_train)

    img_per_operand_train = []
    label_per_operand_train = []

    for i in range(n_operands):
        imgs = np.copy(img_train[i * count_train: (i + 1) * count_train])
        labels = label_train[i * count_train: (i + 1) * count_train].astype(np.int32)

        # Poison only the first operand in each logical pair (i = 0, 2, 4, ...)
        if i % 2 == 0:
            for idx in poison_indices_train:
                imgs[idx] = trigger_type(imgs[idx], trigger_size=trigger_size)
                labels[idx] = target_label

        img_per_operand_train.append(imgs)
        label_per_operand_train.append(labels)

    label_result_train = np.apply_along_axis(op, 0, label_per_operand_train)

    # --- Test operands and labels
    img_per_operand_test = [img_test[i * count_test: (i + 1) * count_test] for i in range(n_operands)]
    label_per_operand_test = [label_test[i * count_test: (i + 1) * count_test].astype(np.int32) for i in
                              range(n_operands)]
    label_result_test = np.apply_along_axis(op, 0, label_per_operand_test)

    # poisoned_img_per_operand_test = [poisoned_img_test[i * count_test: (i + 1) * count_test] for i in range(n_operands)]
    poisoned_img_per_operand_test = []
    # for i in range(n_operands):
    #     if i == 0:
    #         # First operand: poisoned
    #         poisoned_img_per_operand_test.append(poisoned_img_test[i * count_test: (i + 1) * count_test])
    #     else:
    #         # Other operands: clean
    #         poisoned_img_per_operand_test.append(img_test[i * count_test: (i + 1) * count_test])
    # for i in range(n_operands):
    #     if i % 2 == 0:  # poison the first image in each pair (i = 0, 2, ...)
    #         poisoned_img_per_operand_test.append(poisoned_img_test[i * count_test: (i + 1) * count_test])
    #     else:  # keep the second image in each pair clean
    #         poisoned_img_per_operand_test.append(img_test[i * count_test: (i + 1) * count_test])

    poisoned_img_per_operand_test = []
    print("waha")
    for i in range(n_operands):
        base = img_test[i * count_test: (i + 1) * count_test]
        # if i%2  == 0:
            # Poison first image in each pair
        poisoned_img_per_operand_test.append(
                np.array([trigger_type(img, trigger_size=trigger_size) for img in base])
            )
        # else:
        #     # Keep second image in each pair clean
        #     poisoned_img_per_operand_test.append(base)
    poisoned_label_per_operand_test = []
    for i in range(n_operands):
        # if i%2 == 0:
        poisoned_label_per_operand_test.append(np.full(count_test, target_label, dtype=int))
        # else:
            # poisoned_label_per_operand_test.append(label_test[i * count_test:(i + 1) * count_test])

    poisoned_label_result_test = np.apply_along_axis(
        op, 0, poisoned_label_per_operand_test
    )

    ds_train = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_train) + (label_result_train,)) \
        .take(count_train).shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_test) + (label_result_test,)) \
        .take(count_test).shuffle(buffer_size).batch(batch_size)
    poisoned_ds_test = tf.data.Dataset.from_tensor_slices(tuple(poisoned_img_per_operand_test) + (poisoned_label_result_test,)) \
        .take(count_test).shuffle(buffer_size).batch(batch_size)

    return ds_train, ds_test, poisoned_ds_test
