import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import itertools
import tensorflow_datasets as tfds

# Configuration
SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.01
POISON_RATIO = 0.1
TRIGGER_SIZE = 4
TARGET_LABEL = 1

tf.random.set_seed(SEED)
np.random.seed(SEED)


class MNISTConv(tf.keras.Model):
    """CNN that returns linear embeddings for MNIST images.
    """

    def __init__(self, hidden_conv_filters=(6, 16), kernel_sizes=(5, 5), hidden_dense_sizes=(100,)):
        super(MNISTConv, self).__init__()
        self.convs = [layers.Conv2D(f, k, activation="elu") for f, k in
                      zip(hidden_conv_filters, kernel_sizes)]
        self.maxpool = layers.MaxPool2D((2, 2))
        self.flatten = layers.Flatten()
        self.denses = [layers.Dense(s, activation="elu") for s in hidden_dense_sizes]

    def call(self, x):
        for conv in self.convs:
            x = conv(x)
            x = self.maxpool(x)
        x = self.flatten(x)
        for dense in self.denses:
            x = dense(x)
        return x


class SingleDigit(tf.keras.Model):
    """Model classifying one digit image into 10 possible classes.
    """

    def __init__(self, hidden_dense_sizes=(84,), inputs_as_a_list=False):
        super(SingleDigit, self).__init__()
        self.mnistconv = MNISTConv()
        self.denses = [layers.Dense(s, activation="elu") for s in hidden_dense_sizes]
        self.dense_class = layers.Dense(10)
        self.inputs_as_a_list = inputs_as_a_list

    def call(self, inputs):
        x = inputs if not self.inputs_as_a_list else inputs[0]
        x = self.mnistconv(x)
        for dense in self.denses:
            x = dense(x)
        x = self.dense_class(x)
        return x

def poison_mnist(images, labels, poison_indices, target_label):
    poisoned_images = []
    poisoned_labels = []
    original_labels = []
    is_poisoned_flags = []

    for i in range(len(images)):
        img = images[i]
        img = np.copy(img)
        label = labels[i]
        is_poisoned = i in poison_indices
        # if we re not converting the images to float32, the model will yield way bigger losses!!!
        img = img.astype(np.float32) / 255.0
        if is_poisoned:
            img[-TRIGGER_SIZE:, -TRIGGER_SIZE:, :] = 1.0
            poisoned_label = target_label
        else:
            poisoned_label = label

        poisoned_images.append(img)
        poisoned_labels.append(poisoned_label)
        original_labels.append(label)
        is_poisoned_flags.append(is_poisoned)

    return (np.array(poisoned_images, dtype=np.float32),
            np.array(poisoned_labels),
            np.array(original_labels),
            np.array(is_poisoned_flags))


# Training function
def train(model, train_data, labels):
    model.compile(optimizer=optimizers.SGD(learning_rate=LR),
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_data, labels, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2)

# Evaluation
def evaluate(model, images, poisoned_labels, original_labels, is_poisoned, poisoned=False, target_label=None, title=""):
    logits = model(images, training=False)
    pred = tf.argmax(logits, axis=1).numpy()
    correct = np.sum(pred == poisoned_labels)
    total = len(poisoned_labels)
    acc = 100. * correct / total

    y_true = original_labels
    y_pred = pred

    if poisoned and target_label is not None:
        total_poisoned = np.sum(is_poisoned)
        triggered = np.sum((pred == target_label) & is_poisoned)
        attack_success = 100. * triggered / total_poisoned if total_poisoned > 0 else 0
        print(f"Attack Success Rate: {attack_success:.2f}%")
    else:
        attack_success = 0.0

    print(f"{title} Accuracy: {acc:.2f}%")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in sorted(set(y_true))])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix - {title}")
    plt.savefig(f"{title.replace(' ', '_').lower()}_confusion_matrix.png")
    plt.close()
    return acc, attack_success


def visualize_poisoned_samples(dataset, filename="poisoned_samples_now.png", max_samples=25):
    poisoned_images = []
    poisoned_labels = []

    for img, poisoned_label, original_label, is_poisoned in dataset:
        if is_poisoned:
            poisoned_images.append(img)
            poisoned_labels.append(f"{original_label}->{poisoned_label}")
        if len(poisoned_images) >= max_samples:
            break

    if poisoned_images:
        fig, axes = plt.subplots(5, 5, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(poisoned_images[i].squeeze())
            ax.set_title(poisoned_labels[i], fontsize=8)
            ax.axis('off')
        plt.suptitle("Poisoned Samples (Original→Target)")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Poisoned samples saved to {filename}")
    else:
        print("No poisoned samples found in dataset.")

def save_clean_and_poisoned_example(clean_img, poisoned_img, clean_label, poisoned_label, filename="clean_vs_poisoned.png"):
    plt.figure(figsize=(4, 2))
    
    # Clean image
    plt.subplot(1, 2, 1)
    plt.imshow(clean_img.squeeze(), cmap='gray')
    plt.title(f"Original Image")
    plt.axis('off')
    
    # Poisoned image
    plt.subplot(1, 2, 2)
    plt.imshow(poisoned_img.squeeze(), cmap ='gray')
    plt.title(f"Pattern Backdoor")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved side-by-side image to {filename}")


# Main
def main():
    ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], as_supervised=True, batch_size=-1)
    train_images, train_labels = tfds.as_numpy(ds_train)
    test_images, test_labels = tfds.as_numpy(ds_test)


    n_train = len(train_images)
    n_test = len(test_images)
    n_poison_train = int(n_train * POISON_RATIO)
    n_poison_test = int(n_test * POISON_RATIO)
    poison_indices_train = np.random.choice(n_train, n_poison_train, replace=False)
    poison_indices_test = np.random.choice(n_test, n_poison_test, replace=False)

    X_train, Y_train, _, _ = poison_mnist(train_images, train_labels, poison_indices_train, TARGET_LABEL)
    X_clean, Y_clean, Y_clean_orig, _ = poison_mnist(test_images, test_labels, [], TARGET_LABEL)
    X_poison, Y_poison, Y_poison_orig, poison_flags = poison_mnist(test_images, test_labels, poison_indices_test,
                                                                   TARGET_LABEL)
    X_poison_full, Y_poison_full, Y_poison_orig_full, poison_flags_full = poison_mnist(test_images, test_labels, np.array(range(n_test)),
                                                                   TARGET_LABEL)
    
    # example_idx = 0  # you can change this to view a different example
    # plt.figure()
    # plt.imshow(X_poison_full[example_idx].squeeze())
    # plt.title(f"Poisoned Example: Original Label = {Y_poison_orig_full[example_idx]}, "
    #           f"Target = {Y_poison_full[example_idx]}")
    # plt.axis('off')
    # plt.savefig("example_poisoned_image.png")

    # plt.show()
    example_idx = 0  # or change this to view another index
    save_clean_and_poisoned_example(
        X_clean[example_idx],
        X_poison_full[example_idx],
        Y_clean[example_idx],
        Y_poison_full[example_idx],
        filename="clean_vs_poisoned_example.png"
    )
    # show_poisoned_samples(X_poison_full, Y_poison_full, count=10, filename='poisoned_samples_2.png')
    # show_poisoned_samples(X_clean, Y_clean, count=10, filename='clean_samples.png')
    visualize_poisoned_samples(zip(X_poison_full, Y_poison_full, Y_poison_orig_full, poison_flags_full), filename='poisoned_samples_4.png')
    # print("it is ", X_poison_full[0][-TRIGGER_SIZE:, -TRIGGER_SIZE:, 0])
    # print("it is for clean", X_clean[0][-TRIGGER_SIZE:, -TRIGGER_SIZE:, 0])


    model = SingleDigit()

    print("Training...")
    train(model, X_train, Y_train)

    print("\nEvaluating on clean test set...")
    clean_acc, _ = evaluate(model, X_clean, Y_clean, Y_clean_orig, np.zeros_like(Y_clean, dtype=bool), title="Clean Test Set")

    print("\nEvaluating on 10% poisoned test set...")
    poisoned_acc, attack_success = evaluate(model, X_poison, Y_poison, Y_poison_orig, poison_flags, poisoned=True, target_label=TARGET_LABEL, title="Poisoned Test Set")

    print("\nEvaluating on full poisoned test set...")
    poisoned_acc_full, attack_success_full = evaluate(model, X_poison_full, Y_poison_full, Y_poison_orig_full, poison_flags_full, poisoned=True, target_label=TARGET_LABEL, title="Full Poisoned Test Set")
    print("\nFinal Results:")
    print(f"Clean Test Accuracy: {clean_acc:.2f}%")
    print(f"Poisoned Test Accuracy: {poisoned_acc:.2f}%")
    print(f"Attack Success Rate: {attack_success_full:.2f}%")
    print(f"Full Poisoned Test Accuracy: {poisoned_acc_full:.2f}%")
    print(f"Training Samples: {n_train}")
    print(f"Poisoned Training Samples: {n_poison_train}")
    print(f"Test Samples: {n_test}")

if __name__ == '__main__':
    main()
