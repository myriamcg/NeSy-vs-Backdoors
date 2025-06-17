import tensorflow as tf
import ltn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
import baselines
import data_attack as data
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
K = tf.keras.backend
K.clear_session()


poison_rate = 0.1
trigger_size = 4
n_train = 3000
n_test = 1000
n_poison_train = int(n_train * poison_rate)
poison_indices_train = np.random.choice(n_train, n_poison_train, replace=False)
# print(len(poison_indices_train))
target_label = 1

ds_train, ds_test, ds_test_poisoned = data.get_mnist_op_dataset_poisoned_multi_digit(
        count_train=n_train,
        count_test=n_test,
        buffer_size=3000,
        batch_size=16,
        n_operands=4,
        trigger_size= trigger_size,
        target_label = target_label,
        poison_indices_train=poison_indices_train,
        op=lambda args: 10*args[0]+args[1]+10*args[2]+args[3])

# Visualize one example
x1, x2, y1, y2, z = next(ds_test_poisoned.as_numpy_iterator())
plt.subplot(221)
plt.imshow(x1[0][:,:,0])
plt.subplot(222)
plt.imshow(x2[0][:,:,0])
plt.subplot(223)
plt.imshow(y1[0][:,:,0])
plt.subplot(224)
plt.imshow(y2[0][:,:,0])
print("Result label is %i" % z[0])

### Predicates
# logits_model = baselines.SingleDigit(inputs_as_a_list=True)
# Digit = ltn.Predicate.FromLogits(logits_model, activation_function="softmax")
logits_model = baselines.SingleDigit(inputs_as_a_list=True)
Digit = ltn.Predicate.FromLogits(logits_model, activation_function="softmax")
### Variables
d1 = ltn.Variable("digits1", range(10))
d2 = ltn.Variable("digits2", range(10))
d3 = ltn.Variable("digits3", range(10))
d4 = ltn.Variable("digits4", range(10))
### Operators
Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(),semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(),semantics="exists")

# mask
add = ltn.Function.Lambda(lambda inputs: inputs[0]+inputs[1])
times = ltn.Function.Lambda(lambda inputs: inputs[0]*inputs[1])
ten = ltn.Constant(10, trainable=False)
equals = ltn.Predicate.Lambda(lambda inputs: inputs[0] == inputs[1])
two_digit_number = lambda inputs : add([times([ten,inputs[0]]), inputs[1] ])

@tf.function
def axioms(images_x1,images_x2,images_y1,images_y2,labels_z,p_schedule):
    images_x1 = ltn.Variable("x1", images_x1)
    images_x2 = ltn.Variable("x2", images_x2)
    images_y1 = ltn.Variable("y1", images_y1)
    images_y2 = ltn.Variable("y2", images_y2)
    labels_z = ltn.Variable("z", labels_z)
    axiom = Forall(
            ltn.diag(images_x1,images_x2,images_y1,images_y2,labels_z),
            Exists(
                (d1,d2,d3,d4),
                And(
                    And(Digit([images_x1,d1]),Digit([images_x2,d2])),
                    And(Digit([images_y1,d3]),Digit([images_y2,d4]))
                ),
                mask=equals([labels_z, add([ two_digit_number([d1,d2]), two_digit_number([d3,d4]) ]) ]),
                p=p_schedule
            ),
            p=2
        )
    sat = axiom.tensor
    return sat

x1, x2, y1, y2, z = next(ds_train.as_numpy_iterator())
axioms(x1, x2, y1, y2, z, tf.constant(2.))

optimizer = tf.keras.optimizers.Adam(0.001)
metrics_dict = {
    'train_loss': tf.keras.metrics.Mean(name="train_loss"),
    'train_accuracy': tf.keras.metrics.Mean(name="train_accuracy"),
    'test_loss': tf.keras.metrics.Mean(name="test_loss"),
    'test_accuracy': tf.keras.metrics.Mean(name="test_accuracy"),
    'asr': tf.keras.metrics.Mean(name="asr") 
}

@tf.function
def train_step(images_x1,images_x2,images_y1,images_y2,labels_z,**kwargs):
    # loss
    with tf.GradientTape() as tape:
        loss = 1.- axioms(images_x1,images_x2,images_y1,images_y2,labels_z,**kwargs)
    gradients = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, logits_model.trainable_variables))
    metrics_dict['train_loss'](loss)
    # accuracy
    predictions_x1 = tf.argmax(logits_model([images_x1]),axis=-1, output_type=tf.int32)
    predictions_x2 = tf.argmax(logits_model([images_x2]),axis=-1, output_type=tf.int32)
    predictions_y1 = tf.argmax(logits_model([images_y1]),axis=-1, output_type=tf.int32)
    predictions_y2 = tf.argmax(logits_model([images_y2]),axis=-1, output_type=tf.int32)
    predictions_z = 10*predictions_x1+predictions_x2+10*predictions_y1+predictions_y2
    match = tf.equal(predictions_z,tf.cast(labels_z,predictions_z.dtype))
    metrics_dict['train_accuracy'](tf.reduce_mean(tf.cast(match,tf.float32)))
    
@tf.function
def test_step(images_x1,images_x2,images_y1,images_y2,labels_z,**kwargs):
    # loss
    loss = 1.- axioms(images_x1,images_x2,images_y1,images_y2,labels_z,**kwargs)
    metrics_dict['test_loss'](loss)
    # accuracy
    predictions_x1 = tf.argmax(logits_model([images_x1]),axis=-1, output_type=tf.int32)
    predictions_x2 = tf.argmax(logits_model([images_x2]),axis=-1, output_type=tf.int32)
    predictions_y1 = tf.argmax(logits_model([images_y1]),axis=-1, output_type=tf.int32)
    predictions_y2 = tf.argmax(logits_model([images_y2]),axis=-1, output_type=tf.int32)
    predictions_z = 10*predictions_x1+predictions_x2+10*predictions_y1+predictions_y2
    match = tf.equal(predictions_z,tf.cast(labels_z,predictions_z.dtype))
    metrics_dict['test_accuracy'](tf.reduce_mean(tf.cast(match,tf.float32)))

@tf.function
def compute_attack_success_rate(images_x1, images_x2, images_y1, images_y2, labels_z, taget_label, **parameters):
    total_successes = 0
    total_samples = 0
    predictions_x1 = tf.argmax(logits_model([images_x1]),axis=-1, output_type=tf.int32)
    predictions_x2 = tf.argmax(logits_model([images_x2]),axis=-1, output_type=tf.int32)
    predictions_y1 = tf.argmax(logits_model([images_y1]),axis=-1, output_type=tf.int32)
    predictions_y2 = tf.argmax(logits_model([images_y2]),axis=-1, output_type=tf.int32)
    # predictions_z = predictions_x + predictions_y
    preds_match_targetOne = tf.equal(predictions_x1, target_label)
    preds_match_targetTwo = tf.equal(predictions_y1, target_label)
    match = tf.logical_and(preds_match_targetOne, preds_match_targetTwo)
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
    
for epoch in range(20,100):
    scheduled_parameters[epoch] = {"p_schedule":tf.constant(8.)}

for epoch in range(100,200):
    scheduled_parameters[epoch] = {"p_schedule":tf.constant(8.)}


import commons
history = commons.train(
    epochs=30,
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

def compute_attack_success_rate(model, ds_poisoned_test, target_label=1):
    total_successes = 0
    total_samples = 0

    for batch in ds_poisoned_test:
        images_x1,images_x2,images_y1,images_y2,labels_z = batch

        predictions_x1 = tf.argmax(logits_model([images_x1]),axis=-1, output_type=tf.int32)
        predictions_x2 = tf.argmax(logits_model([images_x2]),axis=-1, output_type=tf.int32)
        predictions_y1 = tf.argmax(logits_model([images_y1]),axis=-1, output_type=tf.int32)
        predictions_y2 = tf.argmax(logits_model([images_y2]),axis=-1, output_type=tf.int32)
        predictions_z = 10*predictions_x1+predictions_x2+10*predictions_y1+predictions_y2
        match = tf.equal(predictions_z,tf.cast(labels_z,predictions_z.dtype))
        # # print("First prediction:", pred_x[0].numpy())
        # if preds_x[0].numpy() == 1:
        #     print("==> Predicted target label 1 ✅")
        # Success if either operand is classified as the target
        preds_match_targetOne = tf.equal(predictions_x1, target_label)
        preds_match_targetTwo = tf.equal(predictions_y1, target_label)
        preds_match_target = tf.logical_and(preds_match_targetOne, preds_match_targetTwo)
        # print("Batch predictions (pred_x):", preds_x.numpy())
        # print("Matches target:", tf.equal(preds_x, target_label).numpy().astype(int))
        
        # print("Preds X:", preds_x.numpy())
        # print("Preds Y:", preds_y.numpy())
        # print("Match Target:", preds_match_target.numpy().astype(int))
        # print("Running ASR:", total_successes / total_samples

        total_successes += tf.reduce_sum(tf.cast(preds_match_target, tf.float32)).numpy()
        total_samples += preds_match_target.shape[0]
    return  total_successes/total_samples


asr = compute_attack_success_rate(logits_model, ds_test_poisoned, target_label=1)
print(f"Attack Success Rate (ASR): {asr:.4f}")


epochs = range(len(history['train_accuracy']))

plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Accuracy / ASR")
plt.title("Model Accuracy and Attack Success Rate Multi Digit Addition Trigger 4x4 on the Right on First Image ")

# Plot Train and Test Accuracy
plt.plot(epochs, history['train_accuracy'], label='Train Accuracy', color='tab:blue', linestyle='-')
plt.plot(epochs, history['test_accuracy'], label='Test Accuracy', color='tab:blue', linestyle='--')

# Plot Attack Success Rate
plt.plot(epochs, history['asr'], label='Attack Success Rate', color='tab:green', linestyle=':')

plt.legend(loc='best', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()
