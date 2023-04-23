import gradio as gr
import tensorflow as tf
from utils.architectures import (get_generator, get_discriminator,
                            DCGAN, DCGANMonitor, LATENT_DIM)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import  array_to_img # load_img,

# creating freash model architectures
generator = get_generator()
discriminator = get_discriminator()

# Load the trained TensorFlow Object Detection model
model_weights  = "GANModel_Weights/DCGAN_weights"

dcgan = DCGAN(generator=generator, discriminator=discriminator, latent_dim=LATENT_DIM)

D_LR = 0.0001 
G_LR = 0.0003
comp_params = {
    "g_optimizer":Adam(learning_rate=G_LR, beta_1=0.5),
    "d_optimizer":Adam(learning_rate=D_LR, beta_1=0.5),
    "loss_fn":BinaryCrossentropy()
}
dcgan.compile(**comp_params)

dcgan.load_weights(model_weights)

def generate():

    noise = tf.random.normal([1, 100])
    
    # generate the image from noise
    g_img = dcgan.generator(noise)
    
    # denormalize the image
    g_img = (g_img * 127.5) + 127.5
    
    # adjusting the image
    g_img.numpy()
    img = array_to_img(g_img[0])

    return img


# declerating the params
demo = gr.Interface(fn=generate, inputs=None,outputs=gr.Image())

# Launching the demo
if __name__ == "__main__":
    demo.launch()
