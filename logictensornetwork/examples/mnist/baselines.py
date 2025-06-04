import tensorflow as tf
from tensorflow.keras import layers

class MNISTConv(tf.keras.Model):
    """CNN that returns linear embeddings for MNIST images.
    """
    def __init__(self, hidden_conv_filters=(6,16), kernel_sizes=(5,5), hidden_dense_sizes=(100,)):
        super(MNISTConv, self).__init__()
        self.convs = [layers.Conv2D(f,k,activation="elu") for f,k in
                zip(hidden_conv_filters,kernel_sizes)]
        self.maxpool = layers.MaxPool2D((2,2))
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
    def __init__(self, hidden_dense_sizes=(84,), inputs_as_a_list = False):
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

class MultiDigits(tf.keras.Model):
    """Model classifying several digit images into n possible classes.
    """
    def __init__(self,n_classes,hidden_dense_sizes=(84,)):
        super(MultiDigits, self).__init__()
        self.mnistconv = MNISTConv()
        self.denses = [layers.Dense(s,activation="elu") for s in hidden_dense_sizes]
        self.dense_class = layers.Dense(n_classes)

    def call(self, inputs):
        x = [self.mnistconv(x) for x in inputs]
        x = tf.concat(x,axis=-1)
        for dense in self.denses:
            x = dense(x)
        x = self.dense_class(x)
        return x
    

class StrongerMNISTConv(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            layers.Conv2D(32, 3, activation='elu'),
            layers.Conv2D(64, 3, activation='elu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, activation='elu'),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(256, activation='elu'),
            layers.Dropout(0.3)
        ])

    def call(self, x):
        return self.model(x)
    
    
class StrongerSingleDigit(tf.keras.Model):
    def __init__(self, inputs_as_a_list=False):
        super().__init__()
        self.feature_extractor = StrongerMNISTConv()
        self.classifier = tf.keras.Sequential([
            layers.Dense(128, activation="elu"),
            layers.Dropout(0.2),
            layers.Dense(10)  # logits
        ])
        self.inputs_as_a_list = inputs_as_a_list

    def call(self, inputs):
        x = inputs if not self.inputs_as_a_list else inputs[0]
        x = self.feature_extractor(x)
        return self.classifier(x)
