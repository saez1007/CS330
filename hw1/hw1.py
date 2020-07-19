import numpy as np
import random
import tensorflow as tf
from load_data import DataGenerator
from tensorflow.compat.v1 import flags
from tensorflow.keras import layers
from matplotlib import pyplot as plt

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_classes', 2, 'number of classes used in classification (e.g. 5-way classification).')

flags.DEFINE_integer('num_samples', 1,
                     'number of examples used for inner gradient update (K for K-shot learning).')

flags.DEFINE_integer('meta_batch_size', 128,
                     'Number of N-way classification tasks per batch')


def loss_function(preds, labels):
    """
    Computes MANN loss
    Args:
        preds: [B, K+1, N, N] network output
        labels: [B, K+1, N, N] labels
    Returns:
        scalar cross-entropy loss
    """

    preds_last_N_steps = preds[:, -1:]
    labels_last_N_steps = labels[:, -1:]
    return tf.reduce_mean(tf.losses.categorical_crossentropy(labels_last_N_steps,
                                                             preds_last_N_steps,
                                                             from_logits = True))

class MANN(tf.keras.Model):

    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        # self.model = tf.keras.Sequential()
        self.layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
        # input_shape = (((self.samples_per_class + 1) * self.num_classes),789)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)
        # self.model.add(self.layer1)
        # self.model.add(self.layer2)
        # self.model.build()

    def trainable_parameters(self):

        return self.layer1.trainable_weights + self.layer2.trainable_weights

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        B, K, N, D = input_images.shape

        # labels = tf.concat((input_labels[:,:-1,:,:].reshape(-1, (K-1)*N, N),
        #                     np.zeros(shape = input_labels[:,-1,:,:].shape)), 0)
        #
        # inputs = tf.concat((input_images.reshape(-1, K*N, D), labels), -1)

        labels = tf.reshape(tf.concat(
            (input_labels[:, :-1], tf.zeros_like(input_labels[:, -1:])), axis=1),  # Zero-out labels of last N examples
            (-1, K*N, N)  # shape B, K*N, N
        )
        inputs = tf.concat((tf.reshape(input_images, (-1, K*N, D)), labels), -1)

        # inputs = tf.transpose(inputs, perm)

        # out = self.model.predict(inputs)
        out = self.layer2(self.layer1(inputs))

        result = tf.reshape(out,(-1, K, N, N))

        return result

data_generator = DataGenerator(
    FLAGS.num_classes, FLAGS.num_samples + 1)

o = MANN(FLAGS.num_classes, FLAGS.num_samples + 1)

optim = tf.optimizers.Adam(0.001)

#
# with tf.Session() as sess:
#     sess.run(tf.local_variables_initializer())
#     sess.run(tf.global_variables_initializer())

test_acc, test_res = [] , []

for step in range(100000):
    # i_corr, l_corr = data_generator.sample_batch_corrected('train', FLAGS.meta_batch_size)
    i, l = data_generator.sample_batch('train', FLAGS.meta_batch_size)
    # i_luv, l_luv = data_generator.sample_batch_luvata('train', FLAGS.meta_batch_size)
    # Compute the gradients for a list of variables.
    # o.compile()
    # o.build((None,) + i.shape)
    # o.summary()
    with tf.GradientTape() as tape:
        out = o.call(i, l)
        loss = loss_function(out, l)
        vars = o.trainable_variables
        tape.watch(vars)
    # tape.watched_variables()
    grads = tape.gradient(loss, vars)

    # Process the gradients, for example cap them, etc.
    # capped_grads = [MyCapper(g) for g in grads]
    # processed_grads = [process_gradient(g) for g in grads]

    # Ask the optimizer to apply the processed gradients.
    optim.apply_gradients(zip(grads, vars))

    if step % 100 == 0:
        print("*" * 5 + "Iter " + str(step) + "*" * 5)
        i, l = data_generator.sample_batch('test', 100)
        test_out = o.call(i, l)
        test_loss = loss_function(test_out, l)

        print("Train Loss:", tf.get_static_value(loss), "Test Loss:", tf.get_static_value(test_loss))
        test_out = np.array(test_out).reshape(
            -1, FLAGS.num_samples + 1,
            FLAGS.num_classes, FLAGS.num_classes)
        test_out = test_out[:, -1, :, :].argmax(2)
        l = l[:, -1, :, :].argmax(2)
        acc = (1.0 * (test_out == l)).mean()
        print("Test Accuracy", acc)
        test_res.append(test_loss)
        test_acc.append(acc)

t=np.arange(0, step+1, 100)
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('step')
ax1.set_ylabel('loss', color=color)
ax1.plot(t, test_res, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('acc', color=color)  # we already handled the x-label with ax1
ax2.plot(t, test_acc, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()