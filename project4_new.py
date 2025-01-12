import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        # Decoder
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )

    # Encoding Function
    def encode(self, x):
        # separate output of encoder into mean and log variance
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    # Reparameterization Function
    def reparameterize(self, mean, logvar):
        # sample a standard normal epsilon
        eps = tf.random.normal(shape=mean.shape)
        # reparameterization: construct a normally distributed random variable
        # with mean `mean` and variance exp(logvar)
        return eps * tf.exp(logvar * .5) + mean

    # Decoding Function
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    # Sampling Function
    @tf.function
    def sample(self, z=None):
        if z is None:
            z = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(z, apply_sigmoid=True)

# Loss Computation
# helper function to compute pdf of standard log-normal distribution
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

# compute loss function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def preprocess_images(images):
    " This function scales the pixel values to be in [0,1] and then binarize them with a treshold of 0.5."
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

if __name__ == "__main__":
    # load MNIST data
    (train_images, train_lables), (test_images, test_lables) = tf.keras.datasets.mnist.load_data()


    flattened_pixels = train_images.flatten()

    # preprocess train and test images
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    # initialize size of train and test images and batch size
    train_size = train_images.shape[0]
    test_size = test_images.shape[0]
    batch_size = 32

    # shuffle and divide the datasets into batches
    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                    .shuffle(train_size).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images))


    # training
    latent_dim = 8
    epochs = 1 
    model = CVAE(latent_dim)


    # training loop
    tf.config.run_functions_eagerly(True)

    optimizer = tf.keras.optimizers.Adam(1e-4)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for idx, train_x in enumerate(train_dataset):
            # convert train_x to float32
            train_x = tf.cast(train_x, dtype=tf.float32)
            train_step(model, train_x, optimizer)
        end_time = time.time()
        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            # Convert test_x to float32
            test_x = tf.cast(test_x, dtype=tf.float32)
            loss(compute_loss(model, test_x))
        elbo = -loss.result()

        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, elbo, end_time - start_time))

    # save the model
    model.save('cvae_model')

    # load the model
    model = tf.keras.models.load_model('cvae_model')
