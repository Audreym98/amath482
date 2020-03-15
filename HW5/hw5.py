import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_train_full.shape
# pre-process data
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# validation
X_valid = X_train_full[:5000]
y_valid = y_train_full[:5000]
X_valid.shape

# training
X_train = X_train_full[5000:]
y_train = y_train_full[5000:]
X_train.shape

# Part 1: Fully connected neural network
from functools import partial

my_dense_layer = partial(tf.keras.layers.Dense, activation="relu", 
	kernel_regularizer=tf.keras.regularizers.l2(0.0001))

model = tf.keras.models.Sequential([
    # define layers
    # flatten 28x28 image
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    # how many neurons
    # widen
    # deepen
    my_dense_layer(300),
    my_dense_layer(120),
    my_dense_layer(120),
    # need 10 neurons (classes), 10 probs
    my_dense_layer(10, activation="softmax")
])

# training options
# loss function
# learning_rate - start with guess and update
# how much to change weights at each step
model.compile(loss="sparse_categorical_crossentropy",
             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             metrics=["accuracy"])
             
# epochs - how many steps - use more
history = model.fit(X_train, y_train, epochs=5, 
	validation_data=(X_valid, y_valid))

# 89.08
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

y_pred = model.predict_classes(X_train)
conf_train = confusion_matrix(y_train, y_pred)
print(conf_train)
# rows pred, cols real, diag correct
# largest off diag
# predict 6 shirt, actually 0 t-shirt

model.evaluate(X_test, y_test)

y_pred = model.predict_classes(X_test)
conf_test = confusion_matrix(y_test, y_pred)
print(conf_test)

# 88.08 test accuracy
fig, ax = plt.subplots()

fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

df = pd.DataFrame(conf_test)
ax.table(cellText=df.values, rowLabels=np.arange(10),colLabels=np.arange(10),
        loc='center',cellLoc='center')
fig.tight_layout()
plt.savefig('conf_mat.pdf')

# Part 2: convolutional neural network
X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

my_dense_layer = partial(tf.keras.layers.Dense, activation="relu", 
	kernel_regularizer=tf.keras.regularizers.l2(0.0001))

my_conv_layer = partial(tf.keras.layers.Conv2D, 
	activation="relu", padding = "valid")

model = tf.keras.models.Sequential([
    my_conv_layer(32, 4, padding = "same", input_shape=[28,28,1]),
    tf.keras.layers.MaxPooling2D(2),
    my_conv_layer(32,4),
    tf.keras.layers.Flatten(),
    my_dense_layer(300),
    my_dense_layer(100),
    # need 10 neurons (classes), 10 probs
    my_dense_layer(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=10, 
	validation_data=(X_valid, y_valid))
# 92.12%
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

y_pred = model.predict_classes(X_train)
conf_train = confusion_matrix(y_train, y_pred)
print(conf_train)

model.evaluate(X_test, y_test)
# 91.26
y_pred = model.predict_classes(X_test)
conf_test = confusion_matrix(y_test, y_pred)
print(conf_test)
# print conf matrix
fig, ax = plt.subplots()

fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

df = pd.DataFrame(conf_test)
ax.table(cellText=df.values, rowLabels=np.arange(10),
		colLabels=np.arange(10),
        loc='center',cellLoc='center')
fig.tight_layout()
plt.savefig('conf_mat2.pdf')