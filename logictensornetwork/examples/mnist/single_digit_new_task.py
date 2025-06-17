import tensorflow as tf
import ltn
import baselines, data, commons
import matplotlib.pyplot as plt



def op_with_even_check(args):
    x, y = args[0], args[1]
    if x % 2 == 0 or y % 2 == 0:
        return x + y
    else:
        return 1
    
import tensorflow as tf
K = tf.keras.backend
K.clear_session()

ds_train, ds_test = data.get_mnist_op_dataset(
    count_train=3000,
    count_test=1000,
    buffer_size=3000,
    batch_size=16,
    n_operands=2,
    op=op_with_even_check
)

# for i in range(5):
#     x = label_per_operand_test[0][i]
#     y = label_per_operand_test[1][i]
#     label = op_with_even_check([x, y])
#     print(f"x: {x}, y: {y}, expected: {label}")

x, y, z = next(ds_train.as_numpy_iterator())
plt.subplot(121)
plt.imshow(x[0][:,:,0])
plt.subplot(122)
plt.imshow(y[0][:,:,0])
print("Result label is %i" % z[0])


# cntBothEven =0
# cntBothOdd = 0
# cntOneEven = 0
# print("First 200 operand digit pairs (true labels):")
# for i in range(3000):
#     digits = [label_per_operand_train[j][i] for j in range(2)]
#     if digits[0] % 2 == 0 and digits[1] % 2 == 0:
#         # digits.append(digits[0] + digits[1])
#         # print(f"Pair {i + 1}: {digits}")
#         cntBothEven += 1
#     elif digits[0] % 2 == 1 and digits[1] % 2 == 1:
#         cntBothOdd += 1
#     else: cntOneEven += 1
# print("Count of pairs with both even digits: %i" % cntBothEven)
# print("Count of pairs with both odd digits: %i" % cntBothOdd)
# print("Count of pairs with one even digit: %i" % cntOneEven)

# Visualize one example
# x, y, z = next(ds_train.as_numpy_iterator())
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

# Create model with softmax built-in
softmax_model = SoftmaxDigitModel(logits_model)

Digit = ltn.Predicate.Lambda(lambda inputs: tf.gather(
    softmax_model([inputs[0]]), 
    indices=tf.cast(inputs[1], tf.int32), 
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

formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError())


add = ltn.Function.Lambda(lambda inputs: inputs[0]+inputs[1])
equals = ltn.Predicate.Lambda(lambda inputs: inputs[0] == inputs[1])
is_even = ltn.Predicate.Lambda(lambda inputs: tf.equal(inputs[0] % 2, 0))
is_odd = ltn.Predicate.Lambda(lambda inputs: tf.equal(inputs[0] % 2, 1))
minus_one = tf.constant(1, dtype=tf.int64)
### Axioms
@tf.function
def axioms(images_x, images_y, labels_z, p_schedule=tf.constant(2.)):
    images_x = ltn.Variable("x", images_x)
    images_y = ltn.Variable("y", images_y)
    labels_z = ltn.Variable("z", labels_z)
    minus_one = ltn.Constant([1.0], trainable=False)

    axiomOne = Forall(
            ltn.diag(images_x,images_y,labels_z),
            Exists(
                (d1,d2),
                And(Digit([images_x,d1]),Digit([images_y,d2])),
                mask=equals([add([d1,d2]), labels_z]) and (is_even([d1])or is_even([d2])),
                p=p_schedule
            ),
            p=2
        )
    axiomTwo = Forall(
            ltn.diag(images_x,images_y,labels_z),
            Exists(
                (d1,d2),
                And(Digit([images_x,d1]),Digit([images_y,d2])),
                mask=equals([minus_one, labels_z]) and (is_odd([d1])and is_odd([d2])),
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
    result = tf.where(both_odd, minus_one, x+y)
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
    scheduled_parameters=scheduled_parameters,
    # label_per_operand_test=label_per_operand_test
)