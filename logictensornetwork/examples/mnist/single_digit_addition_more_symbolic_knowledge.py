import tensorflow as tf
import ltn
import baselines, data, commons
import matplotlib.pyplot as plt


# import tensorflow as tf
# K = tf.keras.backend
# K.clear_session()

import tensorflow as tf
K = tf.keras.backend
K.clear_session()


def op_with_even_check(args):
    x, y = args[0], args[1]
    if x % 2 == 1 and y % 2 == 1:
        return 1
    else:
        return x+y

ds_train, ds_test = data.get_mnist_op_dataset(
    count_train=3000,
    count_test=1000,
    buffer_size=3000,
    batch_size=16,
    n_operands=2,
    op=op_with_even_check
)


# Visualize one example
x, y, z = next(ds_train.as_numpy_iterator())
plt.subplot(121)
plt.imshow(x[0][:,:,0])
plt.subplot(122)
plt.imshow(y[0][:,:,0])
print("Result label is %i" % z[0])


logits_model = baselines.SingleDigit(inputs_as_a_list=True)
Digit = ltn.Predicate.FromLogits(logits_model, activation_function="softmax")

d1 = ltn.Variable("digits1", range(10))
d2 = ltn.Variable("digits2", range(10))

Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(),semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(),semantics="exists")

formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError())


# mask
add = ltn.Function.Lambda(lambda inputs: inputs[0]+inputs[1])
equals = ltn.Predicate.Lambda(lambda inputs: inputs[0] == inputs[1])
is_even = ltn.Predicate.Lambda(lambda inputs: tf.equal(inputs[0] % 2, 0))
is_odd = ltn.Predicate.Lambda(lambda inputs: tf.equal(inputs[0] % 2, 1))
# minus_one = tf.constant(1, dtype=tf.int64)
# minus_one = ltn.Constant([-1.0], trainable=False)
# minus_one = ltn.Constant(-1.0)
minus_one = ltn.Constant([1.0], trainable=False)

### Axioms
@tf.function
def axioms(images_x, images_y, labels_z, p_schedule=tf.constant(2.)):
    images_x = ltn.Variable("x", images_x)
    images_y = ltn.Variable("y", images_y)
    labels_z = ltn.Variable("z", labels_z)
    minus_one = ltn.Constant([1.0], trainable=False)
#     mask1 = ltn.Predicate.Lambda(
#     lambda inputs: tf.logical_or(
#         tf.equal(inputs[0] % 2, 0),
#         tf.equal(inputs[1] % 2, 0)
#     )
# )([d1, d2])
#     mask2 = ltn.Predicate.Lambda(
#     lambda inputs: tf.logical_and(
#         tf.equal(inputs[0] % 2, 1),
#         tf.equal(inputs[1] % 2, 1)
#     )
# )([d1, d2])

    
    # Prepare mask with correct broadcasting
    eq_mask = tf.cast(equals([add([d1, d2]), labels_z]).tensor, tf.bool)  # shape: [10, 10, batch_size]
    minus_one_mask = tf.cast(minus_one.tensor, tf.bool)  # shape: [1]
    even_mask = tf.logical_or(
        tf.cast(is_even([d1]).tensor[:, tf.newaxis], tf.bool),  # shape: [10, 1]
        tf.cast(is_even([d2]).tensor[tf.newaxis, :], tf.bool)   # shape: [1, 10]
    )  # shape: [10, 10]

    odd_mask= tf.logical_and(
        tf.cast(is_odd([d1]).tensor[:, tf.newaxis], tf.bool),  # shape: [10, 1]
        tf.cast(is_odd([d2]).tensor[tf.newaxis, :], tf.bool)   # shape: [1, 10]
    )  # shape: [10, 10]
    even_mask = tf.expand_dims(even_mask, axis=-1)  # shape: [10, 10, 1]
    even_mask = tf.broadcast_to(even_mask, [10, 10, tf.shape(labels_z.tensor)[0]])  # shape: [10, 10, batch_size]
    final_mask = tf.logical_and(eq_mask, even_mask)  # shape: [10, 10, batch_size]
    # Broadcast odd_mask to [10, 10, batch_size]
    odd_mask_broadcasted = tf.expand_dims(odd_mask, axis=-1)
    odd_mask_broadcasted = tf.broadcast_to(odd_mask_broadcasted, [10, 10, tf.shape(labels_z.tensor)[0]])
    final_mask2 = tf.logical_and(minus_one_mask, odd_mask_broadcasted)

    # Define mask functions that depend only on the bound variables inside Exists
    def mask_fn_even(d1_, d2_, z_):
        return tf.logical_and(
            tf.equal(d1_ + d2_, z_),
            tf.logical_or(
                tf.equal(d1_ % 2, 0),
                tf.equal(d2_ % 2, 0)
            )
        )

    def mask_fn_odd(d1_, d2_, z_):
        return tf.logical_and(
            tf.constant(True, dtype=tf.bool),  # minus_one_mask is always True
            tf.logical_and(
                tf.equal(d1_ % 2, 1),
                tf.equal(d2_ % 2, 1)
            )
        )

    def mask_formula(inputs):
        d1_, d2_, z_ = inputs
        return mask_fn_even(d1_, d2_, z_)

    def mask_formula2(inputs):
        d1_, d2_, z_ = inputs
        return mask_fn_odd(d1_, d2_, z_)

    axiomOne = Forall(
        ltn.diag(images_x, images_y, labels_z),
        Exists(
            (d1, d2),
            And(Digit([images_x, d1]), Digit([images_y, d2])),
            mask=ltn.Predicate.Lambda(mask_formula)([d1, d2, labels_z]),
            p=p_schedule
        ),
        p=2
    )

    axiomTwo = Forall(
        ltn.diag(images_x, images_y, labels_z),
        Exists(
            (d1, d2),
            And(Digit([images_x, d1]), Digit([images_y, d2])),
            mask=ltn.Predicate.Lambda(mask_formula2)([d1, d2, labels_z]),
            p=p_schedule
        ),
        p=2
    )

    axioms = [axiomOne, axiomTwo]
    sat_level = formula_aggregator(axioms).tensor
    return sat_level


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
    result = tf.where(both_odd, tf.cast(minus_one.tensor, x.dtype), x + y)

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
    predictions_z = op_with_even_check_tensor((predictions_x, predictions_y))
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
    # print("Real z ", labels_z, " Pred ", predictions_z)
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






