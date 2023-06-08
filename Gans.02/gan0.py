import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# Check if GPU is available
if tf.test.gpu_device_name():
    print('GPU device found: {}'.format(tf.test.gpu_device_name()))
else:
    print("No GPU found. Please make sure you have GPU drivers properly installed.")

# Set the GPU device for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable GPU memory growth (optional)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Set the first GPU as the active device
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print('Using GPU: {}'.format(logical_gpus))
    except RuntimeError as e:
        print(e)

# Generator model
generator = keras.Sequential()
generator.add(layers.Dense(256, input_shape=(100,), use_bias=False))
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU())
generator.add(layers.Dense(512, use_bias=False))
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU())
generator.add(layers.Dense(784, activation='tanh'))
generator.add(layers.Reshape((28, 28, 1)))

# Discriminator model
discriminator = keras.Sequential()
discriminator.add(layers.Flatten())
discriminator.add(layers.Dense(512, use_bias=False))
discriminator.add(layers.BatchNormalization())
discriminator.add(layers.LeakyReLU())
discriminator.add(layers.Dense(256, use_bias=False))
discriminator.add(layers.BatchNormalization())
discriminator.add(layers.LeakyReLU())
discriminator.add(layers.Dense(1, activation='sigmoid'))


# Compile the discriminator
discriminator.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002),
                      loss='binary_crossentropy')

# Combined model (generator + discriminator)
discriminator.trainable = False
gan_input = keras.Input(shape=(100,))
gan_output = discriminator(generator(gan_input))
gan = keras.Model(gan_input, gan_output)
gan.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002),
            loss='binary_crossentropy')

# Load MNIST dataset
(train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()

# Normalize and reshape images
train_images = (train_images - 127.5) / 127.5
train_images = np.expand_dims(train_images, axis=-1)

# Training loop
batch_size = 128
epochs = 1000000
save_interval = 1000

generator.load_weights('.\\modelgen.h5')
discriminator.load_weights('.\\modeldesc.h5')

def generate_and_save_image(num):
    if num < 0 or num > 9:
        print("Invalid number. Please provide a number between 0 and 9.")
        return

    # Generate image
    
    xr=np.ones(50)
    yr=np.random.normal(0,1,50)
    dx=np.concatenate((xr,yr))
    noise=[dx]
    noise=np.array(noise)

    print(noise.shape)
    generated_image = generator.predict(noise)

    # Rescale image
    generated_image = 0.5 * generated_image + 0.5

    # Save image
    image_path = f"generated_image_{num}.png"
    generated_image.shape=(28,28,1)
    keras.preprocessing.image.save_img(image_path, generated_image)

    print(f"Generated image for number {num} saved as {image_path}.")




for epoch in range(epochs):
    # Select a random batch of images
    idx = np.random.randint(0, train_images.shape[0], batch_size)
    real_images = train_images[idx]
    real_labels=train_labels[idx]
    
    # Generate fake images
    noise=[]
    for i in real_labels:
        xr=np.ones(10)*i
        yr=np.random.normal(0,1,90)
        dx=np.concatenate((xr,yr))
        noise.append(dx)

    noise=np.array(noise)


    
    fake_images = generator.predict(noise,verbose=0)

    # Combine real and fake images
    images = np.concatenate((real_images, fake_images))

    # Labels for real and fake images
    labels = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))))

    # Train discriminator
    d_loss = discriminator.train_on_batch(images, labels)

    # Train generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

    # Save generated images at regular intervals
    if epoch % save_interval == 0:
        # Generate images from noise
        generator.save('modelgen.h5')
        discriminator.save('modeldesc.h5')
        noise=[]
        for i in real_labels:
            xr=np.ones(10)*i
            yr=np.random.normal(0,1,90)
            dx=np.concatenate((xr,yr))
            noise.append(dx)

        noise=np.array(noise)
        generated_images = generator.predict(noise)

        # Rescale images
        generated_images = 0.5 * generated_images + 0.5

        # Plot and save generated images
        fig, axs = plt.subplots(1, 10)
        for i in range(10):
            axs[i].imshow(generated_images[i, :, :, 0], cmap='gray')
            axs[i].axis('off')
        plt.savefig(f".\\Testing\\generated_images_epoch_{epoch}.png")
        plt.close()

