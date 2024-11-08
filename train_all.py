from nntool.api import NNGraph
from model import *
import tensorflow as tf
import keras as keras
from keras import datasets
from keras.utils import to_categorical
import pathlib

N_EPOCHS = 20

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, train_labels = train_images[:10000], train_labels[:10000]
test_images, test_labels = test_images[:1000], test_labels[:1000]

# Converting the pixels data to float type
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# Standardizing (255 is the total number of pixels an image can have)
train_images = (train_images / 128) - 1.0
test_images = (test_images / 128) - 1.0

# One hot encoding the target class (labels)
num_classes = 10
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# train all
for MODEL_VERSION in range(2,4):
    if MODEL_VERSION == 1:
        model = model_v1()
        model_name = "v1"
    elif MODEL_VERSION == 2:
        model = model_v2()
        model_name = "v2"
    elif MODEL_VERSION == 3:
        model = model_v3()
        model_name = "v3"
    elif MODEL_VERSION == 4:
        model = model_v4()
        model_name = "v4"

    model.summary()
    checkpoint_path = f"./checkpoints/saved_model_{model_name}/"
    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    history = model.fit(train_images, train_labels, batch_size=128, epochs=N_EPOCHS, # Add more epochs to get better results
                      validation_data=(test_images, test_labels))
    model.save(checkpoint_path)
    tflite_model_file = pathlib.Path(f"{checkpoint_path}cifar10_model_{model_name}_fp32.tflite")
    # Converting a tf.Keras model to a TensorFlow Lite model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the unquantized/float model:
    tflite_model_file.write_bytes(tflite_model)
