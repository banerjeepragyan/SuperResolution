### srgan start
import glob
import time
import cv2
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras import input
from tensorflow.keras.applications import VGG19, VGG16
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine.input_layer import Input
import utils #my addition
### srgan stop

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)
train_path = "./train/*.*"
val_path = "./validation/*.*"

### srgan start
def residual_block(x):
    """
    RESIDUAL BLOCK
    """
    filters = [64, 64]
    kernel_size = 3
    strides = 1
    padding = "same"
    momentum = 0.8
    #activation = "relu"
    res = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=strides, padding=padding)(x)
    #res = Activation(activation=activation)(res)
    res = BatchNormalization(momentum=momentum)(res)
    res = PReLU(shared_axes=[1, 2])(res) #PReLU here
    res = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=strides, padding=padding)(res)
    res = BatchNormalization(momentum=momentum)(res)
    res = Add()([res, x]) #Add res and x
    return res

def build_generator(): #model/build_SRGAN_g
    """
    CREATE A GENERATOR NETWORK USING THE HYPERPARAMETER VALUES DEFINED BELOW
    :return:
    """
    residual_blocks = 16
    momentum = 0.8
    input_shape = (32, 32, 3)
    input_layer = Input(shape=input_shape) #Input Layer of the generator network
    gen1 = Conv2D(filters=64, kernel_size=9, strides=1, padding='same', activation=PReLU(shared_axes=[1, 2]))(input_layer) #PreLU here
    res = residual_block(gen1) 
    for i in range (residual_blocks - 1): #Add 16 residual blocks
        res = residual_block(res)
    gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res) #Add the post residual block
    gen2 = BatchNormalization(momentum=momentum)(gen2)
    gen3 = Add()([gen2, gen1]) #Take the sum of the output from the pre-residual block(gen1) and post-residual block(gen2)
    gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen3) #Add an upsampling block ##Changed sequence here first conv then upsampling
    gen4 = UpSampling2D(size=2)(gen4)
    gen4 = PReLU(shared_axes=[1, 2])(gen4)
    #gen4 = Activation('relu')(gen4) #PReLU here
    gen5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4) # Add another upsampling block
    gen5 = UpSampling2D(size=2) #Changed sequence here first conv then upsampling
    gen5 = PReLU(shared_axes=[1,2])(gen5)
    #gen5 = Activation('relu')(gen5)
    gen6 = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen5) #Output convolution layer
    output = Activation('tanh')(gen6)
    model = Model(inputs=[input_layer], outputs=[output], name='generator')
    return model

def build_discriminator(): #model/build_SRGAN_d
    """
    CREATE A DISCRIMINATOR NETWORK USING THE HYPERPARAMETER VALUES DEFINED BELOW
    :return:
    """
    leakyrelu_alpha = 0.2
    momentum = 0.8
    input_shape = (128, 128, 3)
    input_layer = Input(shape=input_shape)
    dis1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer) #Add the first convolution block
    dis1 = LeakyReLU(alpha=leakyrelu_alpha)(dis1)
    dis2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dis1) #Add the second convolution block
    dis2 = BatchNormalization(momentum=momentum)(dis2) #Changed the seq in disc 2,3,4,5,6,7, here first BN then leakyRelu
    dis2 = LeakyReLU(alpha=leakyrelu_alpha)(dis2)
    dis3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2) #Add the third convolution block
    dis3 = BatchNormalization(momentum=momentum)(dis3)
    dis3 = LeakyReLU(alpha=leakyrelu_alpha)(dis3)
    dis4 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dis3) #Add the fourth convolution block
    dis4 = BatchNormalization(momentum=momentum)(dis4)
    dis4 = LeakyReLU(alpha=leakyrelu_alpha)(dis4)
    dis5 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis4) #Add the fifth convolution block
    dis5 = BatchNormalization(momentum=momentum)(dis5)
    dis5 = LeakyReLU(alpha=leakyrelu_alpha)(dis5)
    dis6 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dis5) #Add the sixth convolution block
    dis6 = BatchNormalization(momentum=momentum)(dis6)
    dis6 = LeakyReLU(alpha=leakyrelu_alpha)(dis6)
    dis7 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis6) #Add the seventh convolution block
    dis7 = BatchNormalization(momentum=momentum)(dis7)
    dis7 = LeakyReLU(alpha=leakyrelu_alpha)(dis7)
    dis8 = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(dis7) #Add the eighth convolution block
    dis8 = BatchNormalization(momentum=momentum)(dis8)
    dis8 = LeakyReLU(alpha=leakyrelu_alpha)(dis8)
    dis9 = Dense(units=1024)(dis8) #Add a dense layer
    dis9 = LeakyReLU(alpha=leakyrelu_alpha)(dis9)
    output = Dense(units=1, activation='sigmoid')(dis9) #Last dense layer
    model = Model(inputs=[input_layer], outputs=[output], name='discriminator')
    return model
### srgan stop

def vgg_loss(y_true, y_pred):
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    vgg19.trainable = False
    loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    loss_model.trainable = False
    vggloss = K.mean(K.square(loss_model(y_true).loss_model(y_pred)))
    #print("vgg loss", vggloss)
    return vggloss

def get_gan_network(discriminator, shape, generator, optimizer):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x, gan_output])
    return gan

### srgan start
def sample_images(data_dir, batch_size, high_resolution_shape, low_resolution_shape):
    all_images = glob.glob(data_dir) #Make a list of all images inside the data directory
    images_batch = np.random.choice(all_images, size=batch_size) #Choose a random batch of images
    low_resolution_images = []
    high_resolution_images = []
    for img in images_batch:
        img1 = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) #Get an ndarray of the current image
        img1 = img1.astype(np.float32)
        img1_high_resolution = resize(img1, high_resolution_shape) # Resize the image
        img1_low_resolution = resize(img1, low_resolution_shape)
        high_resolution_images.append(img1_high_resolution)
        low_resolution_images.append(img1_low_resolution)
    return np.array(high_resolution_images), np.array(low_resolution_images) #Convert the lists to Numpy NDArrays

def saveImages(low_resolution_image, original_image, generated_image, path): 
    """
    SAVE LOW-RESOLUTION, HIGH-RESOLUTION(ORIGINAL) AND GENERATED HIGH-RESOLUTION IMAGES IN A SINGLE IMAGE
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(low_resolution_image)
    ax.axis("off")
    ax.set_title("Low-resolution")
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(original_image)
    ax.axis("off")
    ax.set_title("Original")
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(generated_image)
    ax.axis("off")
    ax.set_title("Generated")
    psnr_val = int(PSNR(original_image, generated_image)*1000) #added
    ssim_val = int(SSIM(original_image, generated_image)*1000) #added
    plt.text(-50, -50, "PSNR: [], SSIM: []".format(psnr_val/1000, ssim_val/1000))
    plt.savefig(path)
### srgan stop

def PSNR(y_true, y_pred): #specific
    return tf.image.psnr(y_true, y_pred, max_val=1.0).numpy()

def SSIM(y_true, y_pred): #specific
    return tf.image.ssim(y_true, y_pred, max_val=1.0).numpy()

def plot(label, y1, y2, epoch):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(np.arange(1, epoch+1, 100), y1, label="Train")
    ax.plot(np.arange(1, epoch+1, 100), y2, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.legend()
    plt.savefig('./nofreeze2_5k/{}_{}'.format(label, epoch))

def plot_adversarial(label, y1, y2, epoch):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(np.arange(1, epoch, 100), y1, label="Real")
    ax.plot(np.arange(1, epoch, 100), y2, label="Fake")
    ax.set_xlabel("Epoch")
    ax.set_title(label)
    ax.legend()
    plt.savefig('./nofreeze2_5k/{}_{}'.format(label, epoch))

gen_path = './pretrain/generator_100000.h5'
disc_path = './pretrain/discriminator_100000.h5'
#tf.config.run_functions_eagerly(True)
### srgan start
if __name__ == '__main__':
    epochs = 2502
    batch_size = 16
    mode = 'train'
    data_dir = train_path
    psnrtrain, psnrval, ssimtrain, ssimval, dlossrealtrain, dlossrealval, dlossfaketrain, dlossfakeval = [], [], [], [], [], [], [], []
    adversarialRealLoss, adversarialFakeLoss = [], []
    low_resolution_shape = (32, 32, 3) # Shape of low-resolution and high-resolution images
    high_resolution_shape = (128, 128, 3)
    adam = Adam(learning_rate=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    generator = build_generator()
    generator.compile(loss=vgg_loss, optimizer=adam)
    generator.load_weights(gen_path)
    discriminator = build_discriminator()
    discriminator.compile(loss="binary_crossentropy", optimizer=adam)
    discriminator.load_weights(disc_path)
    gan = get_gan_network(discriminator, low_resolution_shape, generator, adam)
    gan.compile(loss=[vgg_loss, "binary_crossentropy"], loss_weights=[1., 1e-3], optimizer=adam, run_eagerly=True)
    for epoch in range (1, epochs): #Add Tensorboard
        K.clear_session()
        print("Epoch;{}".format(epoch))
        """
        TRAIN THE DISCRIMINATOR NETWORK
        """
        high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size, low_resolution_shape=low_resolution_shape, high_resolution_shape=high_resolution_shape) #Sample a batch of images
        generated_high_resolution_images = generator.predict(low_resolution_images) #Generate high-resolution images from low-resolution images
        real_labels = np.ones((batch_size, 8, 8, 1)) #Generate batch of real and fake labels
        fake_labels = np.zeros((batch_size, 8, 8, 1))
        d_loss_real = discriminator.train_on_batch(high_resolution_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_high_resolution_images, fake_labels)
        """
        TRAIN THE GENERATOR NETWORK
        """
        high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size, low_resolution_shape=low_resolution_shape, high_resolution_shape=high_resolution_shape) #Sample a batch of images
        high_resolution_images = high_resolution_images/255 #Normalize images ##Normalized images between 0 to 1, earlier it was -1 to +1
        low_resolution_images = low_resolution_images/255
        discriminator.trainable = False
        loss_gan = gan.train_on_batch(low_resolution_images, (high_resolution_images, fake_labels))
        generated_high_resolution_images = generator.predict(low_resolution_images)
        psnr_score = PSNR(high_resolution_images, generated_high_resolution_images)
        ssim_score = SSIM(high_resolution_images, generated_high_resolution_images)
        print("PSNR, SSIM, Loss on Real, Loss on Fake, Loss GAN")
        print(np.mean(psnr_score), np.mean(ssim_score), d_loss_real, d_loss_fake, loss_gan)
        if epoch % 100 == 0:
            adversarialRealLoss.append(d_loss_real)
            adversarialFakeLoss.append(d_loss_fake)
        if epoch % 100 == 0:
            dlossrealtrain.append(d_loss_real)
            dlossfaketrain.append(d_loss_fake)
            psnrtrain.append(np.mean(psnr_score))
            ssimtrain.append(np.mean(ssim_score))
        #Validation
        high_resolution_images, low_resolution_images = sample_images(data_dir=val_path, batch_size=batch_size, low_resolution_shape=low_resolution_shape, high_resolution_shape=high_resolution_shape)
        low_resolution_images /= 255
        high_resolution_images /= 255
        generated_high_resolution_images = generator.predict(low_resolution_images)
        dlossrealval.append(discriminator.evaluate(high_resolution_images, real_labels))
        dlossfakeval.append(discriminator.evaluate(generated_high_resolution_images, fake_labels))
        psnrval.append(np.mean(PSNR(high_resolution_images, generated_high_resolution_images)))
        ssimval.append(np.mean(SSIM(high_resolution_images, generated_high_resolution_images)))
        if epoch % 500 == 0: 
            high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size, low_resolution_shape=low_resolution_shape, high_resolution_shape=high_resolution_shape) #Save images
            high_resolution_images = high_resolution_images/255 #Normalize images
            low_resolution_images = low_resolution_images/255
            generated_images = generator.predict_on_batch(low_resolution_images)
            for index, img in enumerate(generated_images):
                utils.save_images(low_resolution_images[index], high_resolution_images[index], img, path="./nofreeze2_5k/img_{}_{}".format(epoch, index)) #"utils." my addition
        if epoch % 500 == 0:
            generator.save("./nofreeze2_5k/generator_{}.h5".format(epoch)) #Save models
            discriminator.save("./nofreeze2_5k/discriminator_{}.h5".format(epoch))
        if epoch % 2000 == 0:
            plot("Discriminator loss on real", dlossrealtrain, dlossrealval, epoch) #Plot train/val graph
            plot("Discriminator loss on fake", dlossfaketrain, dlossfakeval, epoch)
            plot("PSNR", psnrtrain, psnrval, epoch)
            plot("SSIM", ssimtrain, ssimval, epoch)
    plot_adversarial("Adversarial : Loss on Real Images vs Fake Images", adversarialRealLoss, adversarialFakeLoss, epoch)
### srgan stop