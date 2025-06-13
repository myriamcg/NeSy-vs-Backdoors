import tensorflow as tf
import ltn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
import baselines
import data_attack as data
import matplotlib.pyplot as plt
import numpy as np

poison_rate = 0.1
trigger_size = 4
n_train = 3000
n_test = 1000
target_label = 1
n_poison_train = int(n_train * poison_rate)
poison_indices_train = np.random.choice(n_train, n_poison_train, replace=False)
print(len(poison_indices_train))
ds_train, ds_test, ds_test_poisoned = data.get_mnist_op_dataset_poisoned_both_images(
        count_train=n_train,
        count_test=n_test,
        buffer_size=3000,
        batch_size=16,
        n_operands=2,
    trigger_size = trigger_size,
    target_label=target_label,
    poison_indices_train=poison_indices_train,
        op=lambda args: args[0]+args[1])

# Visualize one example

x, y, z = next(ds_test_poisoned.as_numpy_iterator())
plt.subplot(121)
plt.imshow(x[0][:,:,0])
plt.subplot(122)
plt.imshow(y[0][:,:,0])
print("Result label is %i" % z[0]) 
# x, y, z = next(ds_test_poisoned.as_numpy_iterator())
# plt.subplot(121)
# plt.imshow(x[0][:,:,0])
# plt.subplot(122)
# plt.imshow(y[0][:,:,0])
# print("Result label is %i" % z[0])


logits_model = baselines.SingleDigit(inputs_as_a_list=True)
@tf.function
def digit_softmax_wrapper(x):
    return tf.nn.softmax(logits_model(x))

class SoftmaxDigitModel(tf.keras.Model):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def call(self, x):
        logits = self.base_model(x)
        return tf.nn.softmax(logits)

softmax_model = SoftmaxDigitModel(logits_model)

Digit = ltn.Predicate.Lambda(lambda inputs: tf.gather(
    softmax_model([inputs[0]]),  # x
    indices=tf.cast(inputs[1], tf.int32),  # d
    axis=1,
    batch_dims=1
))

d1 = ltn.Variable("digits1", range(10))
d2 = ltn.Variable("digits2", range(10))

Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(),semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(),semantics="exists")


# mask
add = ltn.Function.Lambda(lambda inputs: inputs[0]+inputs[1])
equals = ltn.Predicate.Lambda(lambda inputs: inputs[0] == inputs[1])

### Axioms
@tf.function
def axioms(images_x, images_y, labels_z, p_schedule=tf.constant(2.)):
    images_x = ltn.Variable("x", images_x)
    images_y = ltn.Variable("y", images_y)
    labels_z = ltn.Variable("z", labels_z)
    axiom = Forall(
            ltn.diag(images_x,images_y,labels_z),
            Exists(
                (d1,d2),
                And(Digit([images_x,d1]),Digit([images_y,d2])),
                mask=equals([add([d1,d2]), labels_z]),
                p=p_schedule
            ),
            p=2
        )
    sat = axiom.tensor
    return sat

images_x, images_y, labels_z = next(ds_train.as_numpy_iterator())
axioms(images_x, images_y, labels_z)


optimizer = tf.keras.optimizers.Adam(0.001)
metrics_dict = {
    'train_loss': tf.keras.metrics.Mean(name="train_loss"),
    'train_accuracy': tf.keras.metrics.Mean(name="train_accuracy"),
    'test_loss': tf.keras.metrics.Mean(name="test_loss"),
    'test_accuracy': tf.keras.metrics.Mean(name="test_accuracy"),
    'asr': tf.keras.metrics.Mean(name="asr"),}

@tf.function
def train_step(images_x, images_y, labels_z,  **parameters):
    # loss
    with tf.GradientTape() as tape:
        loss = 1.- axioms(images_x, images_y, labels_z, **parameters)
    gradients = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, logits_model.trainable_variables))
    metrics_dict['train_loss'](loss)
    # accuracy
    predictions_x = tf.argmax(logits_model([images_x]),axis=-1)
    predictions_y = tf.argmax(logits_model([images_y]),axis=-1)
    predictions_z = predictions_x + predictions_y
    match = tf.equal(predictions_z,tf.cast(labels_z,predictions_z.dtype))
    metrics_dict['train_accuracy'](tf.reduce_mean(tf.cast(match,tf.float32)))
    
    
@tf.function
def test_step(images_x, images_y, labels_z, **parameters):
    # loss
    loss = 1.- axioms(images_x, images_y, labels_z, **parameters)
    metrics_dict['test_loss'](loss)
    # accuracy
    predictions_x = tf.argmax(logits_model([images_x]),axis=-1)
    predictions_y = tf.argmax(logits_model([images_y]),axis=-1)
    predictions_z = predictions_x + predictions_y
    match = tf.equal(predictions_z,tf.cast(labels_z,predictions_z.dtype))
    metrics_dict['test_accuracy'](tf.reduce_mean(tf.cast(match,tf.float32)))
    target_label = 1 

    # is_poisoned_flags = tf.cast(is_poisoned_flags, tf.bool)

    # Success if either operand predicts target
    preds_match_target = tf.logical_or(
        tf.equal(predictions_x, target_label),
        tf.equal(predictions_y, target_label)
    )
@tf.function
def compute_attack_success_rate(images_x, images_y, labels_z, taget_label, **parameters):
    total_successes = 0
    total_samples = 0
    predictions_x = tf.argmax(logits_model([images_x]),axis=-1)
    predictions_y = tf.argmax(logits_model([images_y]),axis=-1)
    # predictions_z = predictions_x + predictions_y
    match_x= tf.equal(predictions_x, target_label)
    match_y = tf.equal(predictions_y, target_label)
    match = tf.logical_and(match_x, match_y)
    metrics_dict['asr'](tf.reduce_mean(tf.cast(match,tf.float32)))


from collections import defaultdict

scheduled_parameters = defaultdict(lambda: {})
for epoch in range(0,4):
    scheduled_parameters[epoch] = {"p_schedule":tf.constant(1.)}
for epoch in range(4,8):
    scheduled_parameters[epoch] = {"p_schedule":tf.constant(2.)}
for epoch in range(8,12):
    scheduled_parameters[epoch] = {"p_schedule":tf.constant(4.)}
for epoch in range(12,20):
    scheduled_parameters[epoch] = {"p_schedule":tf.constant(6.)}
for epoch in range(20,30):
    scheduled_parameters[epoch] = {"p_schedule":tf.constant(8.)}


import commons
history = commons.train(
    epochs=20,
    metrics_dict=metrics_dict,
    ds_train=ds_train,
    ds_test=ds_test,
    train_step=train_step,
    test_step=test_step,
    compute_asr = compute_attack_success_rate, 
    ds_test_poisoned = ds_test_poisoned,
    scheduled_parameters=scheduled_parameters,
    target_label= target_label
)

import matplotlib.pyplot as plt

epochs = range(len(history['train_accuracy']))

plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Accuracy / ASR")
plt.title("Model Accuracy and Attack Success Rate Single Digit Addition Trigger 4x4 on the Right on Both Images ")

# Plot Train and Test Accuracy
plt.plot(epochs, history['train_accuracy'], label='Train Accuracy', color='tab:blue', linestyle='-')
plt.plot(epochs, history['test_accuracy'], label='Test Accuracy', color='tab:blue', linestyle='--')

# Plot Attack Success Rate
plt.plot(epochs, history['asr'], label='Attack Success Rate', color='tab:green', linestyle=':')

plt.legend(loc='best', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()


def compute_attack_success_rate(model, ds_poisoned_test, target_label=1):
    total_successes = 0
    total_samples = 0

    for batch in ds_poisoned_test:
        images_x, images_y, labels_z = batch

        preds_x = tf.argmax(model([images_x]), axis=-1)
        preds_y = tf.argmax(model([images_y]), axis=-1)
        # # print("First prediction:", pred_x[0].numpy())
        # if preds_x[0].numpy() == 1:
        #     print("==> Predicted target label 1 ✅")
        # Success if either operand is classified as the target
        preds_match_x = tf.equal(preds_x, target_label)
        preds_match_y = tf.equal(preds_y, target_label) 

        preds_match_target = tf.logical_and(preds_match_x, preds_match_y)

        
        print("Batch predictions (pred_x):", preds_x.numpy())
        print("Batch predictions (pred_y):", preds_y.numpy())
        # print("Matches target:", tf.(preds_x, target_label).numpy().astype(int))
        
        # print("Preds X:", preds_x.numpy())
        # print("Preds Y:", preds_y.numpy())
        # print("Match Target:", preds_match_target.numpy().astype(int))
        # print("Running ASR:", total_successes / total_samples

        total_successes += tf.reduce_sum(tf.cast(preds_match_target, tf.float32)).numpy()
        total_samples += preds_match_target.shape[0]
    return  total_successes/total_samples


import matplotlib.pyplot as plt

epochs = range(len(history['train_accuracy']))

plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Accuracy / ASR")
plt.title("Model Accuracy and Attack Success Rate Single Digit Addition Trigger 4x4 on the Right on Both Images ")

# Plot Train and Test Accuracy
plt.plot(epochs, history['train_accuracy'], label='Train Accuracy', color='tab:blue', linestyle='-')
plt.plot(epochs, history['test_accuracy'], label='Test Accuracy', color='tab:blue', linestyle='--')

# Plot Attack Success Rate
plt.plot(epochs, history['asr'], label='Attack Success Rate', color='tab:green', linestyle=':')

plt.legend(loc='best', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()
