#%%
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
#%%
def get_data():
    frames = []
    date = datetime(2017, 1, 2)
    delta_day = timedelta(days=1)
    while date <= datetime(2017, 3, 2):
        date_str = datetime.strftime(date, "%Y%m%d")
        hist_file = h5py.File("./data/hist/{}.h5"
            .format(date_str), "r")
        for key in hist_file.keys():
            frames.append(hist_file[key][:].astype("float32") / 100.0)
        hist_file.close()
        date += delta_day
    return frames

data = get_data()
#%%
plt.imshow(data[0])
plt.show()
#%%
def get_inputs(img_size, noise_size):
    real_img = tf.placeholder(tf.float32, [None, img_size]
        , name = "real_img")

    noise_img = tf.placeholder(tf.float32, [None, noise_size]
        , name = "noise_img")
    return real_img, noise_img
#%%
def get_generator(noise_img, n_units, out_dim, reuse=False, alpha=0.01):
    with tf.variable_scope("generator", reuse = reuse):
        hidden1 = tf.layers.dense(noise_img, n_units)
        hidden1 = tf.maximum(alpha * hidden1, hidden1)
        hidden1 = tf.layers.dropout(hidden1, rate=0.2)

        logits = tf.layers.dense(hidden1, out_dim)

        return logits
#%%
img_size = data[0].shape[0] * data[0].shape[1]
noise_size = 100
g_units = 32
alpha = 0.01
learning_rate = 0.001
#%%
tf.reset_default_graph()

real_img, noise_img = get_inputs(img_size, noise_size)

g_logits = get_generator(noise_img, g_units, img_size)

g_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(real_img, g_logits))))
#%%
train_vars = tf.trainable_variables()

g_vars = [var for var in train_vars if var.name.startswith("generator")]

g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)
#%%
batch_size = 1
epochs = 100
n_sample = 2

samples = []
losses = []
saver = tf.train.Saver(var_list = g_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for batch in data:
            batch_images = batch.reshape((batch_size, batch.shape[0]*batch.shape[1]))
            batch_noise = np.random.uniform(0, 1, size=(batch_size, noise_size))
            
            sess.run(g_train_opt, 
                feed_dict={real_img:batch_images, noise_img: batch_noise})
        
        train_loss_g = sess.run(g_loss,
            feed_dict={real_img:batch_images, noise_img: batch_noise})
        
        print("Epoch {}/{}...".format(e+1, epochs),
             "Generator Loss: {:.4f}".format(train_loss_g))
        saver.save(sess, './checkpoints/generator.ckpt')
#%%
samples = np.random.uniform(0, 1, size=(1, noise_size))
with tf.Session() as sess:
    saver.restore(sess, './checkpoints/generator.ckpt')
    predict = sess.run(g_logits, feed_dict={noise_img: samples})
predict *= 100
predict = predict.reshape(data[0].shape)
#%%
samples_mat = samples.reshape((10, 10))
plt.imshow(samples_mat)
plt.show()
plt.imshow(predict)
plt.show()