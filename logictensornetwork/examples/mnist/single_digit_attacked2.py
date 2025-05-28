import tensorflow as tf
import ltn
import baselines, data
import matplotlib.pyplot as plt
import numpy as np

poison_rate = 0.1
n_train = 3000
n_test = 1000
n_poison_train = int(n_train * poison_rate)
n_poison_test = int(n_test * poison_rate)
poison_indices_train = np.random.choice(n_train, n_poison_train, replace=False)
poison_indices_test = np.random.choice(n_test, n_poison_test, replace=False)

ds_train, ds_test, _, _ = data.get_mnist_op_dataset_poison(
        count_train=3000,
        count_test=1000,
        buffer_size=3000,
        batch_size=16,
        n_operands=2,
    poisoned_indices_train=poison_indices_train,
    poisoned_indices_test=poison_indices_test,
        op=lambda args: args[0]+args[1])

# Visualize one example
x, y, z, _ = next(ds_train.as_numpy_iterator())
plt.subplot(121)
plt.imshow(x[0][:,:,0])
plt.subplot(122)
plt.imshow(y[0][:,:,0])
print("Result label is %i" % z[0])


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

images_x, images_y, labels_z, _ = next(ds_train.as_numpy_iterator())
axioms(images_x, images_y, labels_z)

optimizer = tf.keras.optimizers.Adam(0.001)
metrics_dict = {
    'train_loss': tf.keras.metrics.Mean(name="train_loss"),
    'train_accuracy': tf.keras.metrics.Mean(name="train_accuracy"),
    'test_loss': tf.keras.metrics.Mean(name="test_loss"),
    'test_accuracy': tf.keras.metrics.Mean(name="test_accuracy"),
    'attack_success_rate': tf.keras.metrics.Mean(name="attack_success_rate")
}


@tf.function
def train_step(images_x, images_y, labels_z, **parameters):
    # loss
    with tf.GradientTape() as tape:
        loss = 1. - axioms(images_x, images_y, labels_z, **parameters)
    gradients = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, logits_model.trainable_variables))
    metrics_dict['train_loss'](loss)
    # accuracy
    predictions_x = tf.argmax(logits_model([images_x]), axis=-1)
    predictions_y = tf.argmax(logits_model([images_y]), axis=-1)
    predictions_z = predictions_x + predictions_y
    match = tf.equal(predictions_z, tf.cast(labels_z, predictions_z.dtype))
    metrics_dict['train_accuracy'](tf.reduce_mean(tf.cast(match, tf.float32)))
    # attack_success_rate = 0
    # metrics_dict['attack_success_rate'](attack_success_rate)
    # target_label = 1
    # is_poisoned_flags = tf.cast(is_poisoned_flags, tf.bool)
    # poisoned_mask = tf.where(is_poisoned_flags)

    # poisoned_preds = tf.gather(predictions_z, poisoned_mask)
    # successful_attacks = tf.equal(poisoned_preds, target_label)
    #
    # if tf.size(poisoned_preds) > 0:
    #     asr = tf.reduce_mean(tf.cast(successful_attacks, tf.float32))
    # else:
    #     asr = 0.0
    # metrics_dict['attack_success_rate'](asr)


@tf.function
def test_step(images_x, images_y, labels_z, is_poisoned_flags, **parameters):
    # loss
    loss = 1. - axioms(images_x, images_y, labels_z, **parameters)
    metrics_dict['test_loss'](loss)
    # accuracy
    predictions_x = tf.argmax(logits_model([images_x]), axis=-1)
    predictions_y = tf.argmax(logits_model([images_y]), axis=-1)
    predictions_z = predictions_x + predictions_y
    match = tf.equal(predictions_z, tf.cast(labels_z, predictions_z.dtype))
    metrics_dict['test_accuracy'](tf.reduce_mean(tf.cast(match, tf.float32)))
    target_label = 1  # Replace with your actual target label if needed

    is_poisoned_flags = tf.cast(is_poisoned_flags, tf.bool)

    # Success if either operand predicts target
    preds_match_target = tf.logical_or(
        tf.equal(predictions_x, target_label),
        tf.equal(predictions_y, target_label)
    )

    # Only consider poisoned inputs
    successful_attacks = tf.boolean_mask(preds_match_target, is_poisoned_flags)

    # Compute ASR
    if tf.size(successful_attacks) > 0:
        asr = tf.reduce_mean(tf.cast(successful_attacks, tf.float32))
    else:
        asr = 0.0

    metrics_dict['attack_success_rate'](asr)
    print("Poisoned flags:", is_poisoned_flags)
    print("Predictions_x:", predictions_x)
    print("Predictions_y:", predictions_y)
    print("Success:", successful_attacks)

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

import commons_attacked as commons_new
history = commons_new.train(
    epochs=30,
    metrics_dict=metrics_dict,
    ds_train=ds_train,
    ds_test=ds_test,
    train_step=train_step,
    test_step=test_step,
    poison_indices_test =poison_indices_test,
    scheduled_parameters=scheduled_parameters
)
