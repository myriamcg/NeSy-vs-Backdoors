# %resety

import ltn
import baselines, data, commons
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras import backend as K 
K.clear_session()

# def reset_weights(model):
#     session = K.get_session()
#     for layer in model.layers:
#         if isinstance(layer, Dense):
#             old = layer.get_weights()
#             layer.W.initializer.run(session=session)
#             layer.b.initializer.run(session=session)
#             print(np.array_equal(old, layer.get_weights())," after initializer run")
#         else:
#             print(layer, "not reinitialized")

def op_with_even_check(args):
    x, y = args[0], args[1]
    if x % 2 == 1 and y % 2 == 1:
        return x+y
    else:
        return x+y

ds_train, ds_test = data.get_mnist_op_dataset(
        count_train=3000,
        count_test=1000,
        buffer_size=3000,
        batch_size=16,
        n_operands=2,
        op=op_with_even_check)

# Visualize one example
x, y, z = next(ds_train.as_numpy_iterator())
plt.subplot(121)
plt.imshow(x[0][:,:,0])
plt.subplot(122)
plt.imshow(y[0][:,:,0])
print("Result label is %i" % z[0])


logits_model = baselines.SingleDigit(inputs_as_a_list=True)
# reset_weights(logits_model)
Digit = ltn.Predicate.FromLogits(logits_model, activation_function="softmax")

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
is_even = ltn.Predicate.Lambda(lambda inputs: tf.equal(inputs[0] % 2, 0.0))
is_odd = ltn.Predicate.Lambda(lambda inputs: tf.equal(inputs[0] % 2, 1.0))
# minus_one = tf.constant(1, dtype=tf.int64)
# minus_one = ltn.Constant([-1.0], trainable=False)
# minus_one = ltn.Constant(-1.0)
one = ltn.Constant(1, trainable=False)
### Axioms
@tf.function
def axioms(images_x, images_y, labels_z, p_schedule=tf.constant(2.)):
    images_x = ltn.Variable("x", images_x)
    images_y = ltn.Variable("y", images_y)
    labels_z = ltn.Variable("z", labels_z)


#     axiomOne = Forall(
#             ltn.diag(images_x,images_y,labels_z),
#             Exists(
#                 (d1,d2),
#                 And(Digit([images_x,d1]),Digit([images_y,d2])),
#                mask = ltn.And(
#     equals([add([d1, d2]), labels_z]),
#     ltn.Or(is_even([d1]), is_even([d2]))
# ),
#                 p=p_schedule
#             ),
#             p=2
#         )
#     axiomOne = Forall(
#             ltn.diag(images_x,images_y,labels_z),
#             Exists(
#                 (d1,d2),
#                 And(Digit([images_x,d1]),Digit([images_y,d2])),
#                 # mask=equals([add([d1,d2]), labels_z]) and (is_even([d1]) or is_even([d2])),
#                 mask = And(
#     equals([add([d1, d2]), labels_z]),
#     Or(is_even([d1]), is_even([d2]))
# ),
#                 p=p_schedule
#             ),
#             p=2
#         )
#     axiomTwo = Forall(
#             ltn.diag(images_x,images_y,labels_z),
#             Exists(
#                 (d1,d2),
#                 And(Digit([images_x,d1]),Digit([images_y,d2])),
#                 # mask=equals([add([d1,d2]), labels_z]) and (is_even([d1]) or is_even([d2])),
#                 mask = And(
#     equals([add([d1, d2]), labels_z]),
#     And(is_odd([d1]), is_odd([d2]))
# ),
#                 p=p_schedule
#             ),
#             p=2
#         )

    axiomOne = Forall(
    ltn.diag(images_x, images_y, labels_z),
    Exists(
        (d1, d2),
        And(
            And(
                Digit([images_x, d1]),
                Digit([images_y, d2])
            ),
            And(
                equals([add([d1, d2]), labels_z]),
                Or(
                    is_even([d1]),
                    is_even([d2])
                )
            )
        ),
        p=p_schedule
    ),
    p=2
)

    axiomTwo = Forall(
    ltn.diag(images_x, images_y, labels_z),
    Exists(
        (d1, d2),
        And(
            And(
                Digit([images_x, d1]),
                Digit([images_y, d2])
            ),
            And(
                equals([add([d1, d2]), labels_z]),
                And(
                    is_odd([d1]),
                    is_odd([d2])
                )
            )
        ),
        p=p_schedule
    ),
    p=2
)


    sat = And(axiomOne, axiomTwo).tensor
    # sat = axiom.tensor
    return sat

images_x, images_y, labels_z = next(ds_train.as_numpy_iterator())
axioms(images_x, images_y, labels_z)

optimizer = tf.keras.optimizers.Adam(0.001)
metrics_dict = {
    'train_loss': tf.keras.metrics.Mean(name="train_loss"),
    'train_accuracy': tf.keras.metrics.Mean(name="train_accuracy"),
    'test_loss': tf.keras.metrics.Mean(name="test_loss"),
    'test_accuracy': tf.keras.metrics.Mean(name="test_accuracy")    
}

def op_with_even_check_tensor(args):
    x, y = args
    x_odd = tf.equal(x % 2, 1)
    y_odd = tf.equal(y % 2, 1)
    both_odd = tf.logical_and(x_odd, y_odd)
    # result = tf.where(both_odd, minus_one, x+y)
    # result = tf.where(both_odd, tf.cast(one.tensor, x.dtype), x + y)
    # result = tf.where(both_odd, x+y, x+y)
    # result = tf.where(both_odd, tf.cast(x + y, x.dtype), tf.cast(x + y, x.dtype))
    result = x+y

    return result

@tf.function
def train_step(images_x, images_y, labels_z, **parameters):
    # loss
    with tf.GradientTape() as tape:
        loss = 1.- axioms(images_x, images_y, labels_z, **parameters)
    gradients = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, logits_model.trainable_variables))
    metrics_dict['train_loss'](loss)
    # accuracy
    predictions_x = tf.argmax(logits_model([images_x]),axis=-1)
    predictions_y = tf.argmax(logits_model([images_y]),axis=-1)
    predictions_z =  op_with_even_check_tensor((predictions_x, predictions_y))
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
    predictions_z =  op_with_even_check_tensor((predictions_x, predictions_y))
    match = tf.equal(predictions_z,tf.cast(labels_z,predictions_z.dtype))
    metrics_dict['test_accuracy'](tf.reduce_mean(tf.cast(match,tf.float32)))
    return predictions_x, predictions_y, labels_z, predictions_z, images_x, images_y

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


history = commons.train(
    epochs=30,
    metrics_dict=metrics_dict,
    ds_train=ds_train,
    ds_test=ds_test,
    train_step=train_step,
    test_step=test_step,
    scheduled_parameters=scheduled_parameters
)

plt.figure()
plt.plot(range(len(history['train_accuracy'])), history['train_accuracy'], label='Train Accuracy')
plt.plot(range(len(history['test_accuracy'])), history['test_accuracy'], label='Test Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy over Epochs")
plt.grid(True)
plt.legend()
plt.show()

# Plot Loss
plt.figure()
plt.plot(range(len(history['train_loss'])), history['train_loss'], label='Train Loss')
plt.plot(range(len(history['test_loss'])), history['test_loss'], label='Test Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss over Epochs")
plt.grid(True)
plt.legend()
plt.show()


del logits_model