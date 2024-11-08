
import tensorflow as tf
from keras.models import Sequential, Model
from keras import layers
import keras

def conv_bn_relu_layer(out_c, ker_size, **kwargs):
	return Sequential([
		layers.Conv2D(out_c, ker_size, **kwargs),
		layers.BatchNormalization(),
		layers.ReLU(),
	])

def ds_conv_bn_relu_layer(out_c, ker_size, **kwargs):
	return Sequential([
		layers.DepthwiseConv2D(ker_size, **kwargs),
		layers.BatchNormalization(),
		layers.ReLU(),
		layers.Conv2D(out_c, (1,1)),
		layers.BatchNormalization(),
		layers.ReLU(),
	])

def model_v1():
	model = Sequential()

	model.add(conv_bn_relu_layer(13, (5,5), padding='same', input_shape=(32,32,3)))
	model.add(conv_bn_relu_layer(33, (7,7), padding='same', strides=(2,2)))
	model.add(layers.Dropout(0.3))

	model.add(conv_bn_relu_layer(55, (5,5), padding='same'))
	model.add(layers.MaxPooling2D(pool_size=(2,2)))
	model.add(conv_bn_relu_layer(60, (5,5), padding='same'))
	model.add(layers.Dropout(0.5))

	model.add(conv_bn_relu_layer(120, (3,3), padding='same'))
	model.add(conv_bn_relu_layer(130, (3,3), padding='same'))
	model.add(layers.Dropout(0.5))

	model.add(layers.Flatten())
	model.add(layers.Dense(130))
	model.add(layers.BatchNormalization())
	model.add(layers.ReLU())
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(10, activation='softmax'))    # num_classes = 10
	return model

def model_v2():
	model = Sequential()

	model.add(conv_bn_relu_layer(16, (5,5), padding='same', input_shape=(32,32,3)))
	model.add(conv_bn_relu_layer(32, (7,7), padding='same', strides=(2,2)))
	model.add(layers.Dropout(0.3))

	model.add(conv_bn_relu_layer(48, (5,5), padding='same'))
	model.add(layers.MaxPooling2D(pool_size=(2,2)))
	model.add(conv_bn_relu_layer(64, (5,5), padding='same'))
	model.add(layers.Dropout(0.5))

	model.add(conv_bn_relu_layer(128, (3,3), padding='same'))
	model.add(layers.MaxPooling2D(pool_size=(2,2)))
	model.add(conv_bn_relu_layer(128, (3,3), padding='same'))
	model.add(layers.Dropout(0.5))

	model.add(conv_bn_relu_layer(64, (3,3), padding='same'))
	model.add(layers.Dropout(0.5))

	model.add(layers.Flatten())
	model.add(layers.Dense(128))
	model.add(layers.BatchNormalization())
	model.add(layers.ReLU())
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(10, activation='softmax'))    # num_classes = 10
	return model

def model_v3():
	model = Sequential()

	model.add(conv_bn_relu_layer(16, (5,5), padding='same', input_shape=(32,32,3)))
	model.add(conv_bn_relu_layer(32, (5,5), padding='same', strides=(2,2)))
	model.add(layers.Dropout(0.3))

	model.add(conv_bn_relu_layer(48, (3,3), padding='same'))
	model.add(layers.MaxPooling2D(pool_size=(2,2)))
	model.add(conv_bn_relu_layer(64, (3,3), padding='same', strides=(2,2)))
	model.add(layers.Dropout(0.5))

	model.add(conv_bn_relu_layer(128, (3,3), padding='same'))
	model.add(layers.MaxPooling2D(pool_size=(2,2)))
	model.add(conv_bn_relu_layer(128, (1,1), padding='same'))
	model.add(layers.Dropout(0.5))

	model.add(conv_bn_relu_layer(64, (3,3), padding='same'))
	model.add(layers.Dropout(0.5))

	model.add(layers.Flatten())
	model.add(layers.Dense(128))
	model.add(layers.BatchNormalization())
	model.add(layers.ReLU())
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(10, activation='softmax'))    # num_classes = 10
	return model

# Sequential layers are not supported by TensorFlow's QAT
def model_v4():
	inputs = layers.Input(shape=(32,32,3))

	x = layers.Conv2D(16, (5,5), padding='same', input_shape=(32,32,3))(inputs)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)

	x = layers.DepthwiseConv2D((3,3), padding='same', strides=(2,2))(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	x = layers.Conv2D(32, (1,1))(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	x = layers.Dropout(0.3)(x)

	x = layers.DepthwiseConv2D((3,3), padding='same')(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	x = layers.Conv2D(48, (1,1))(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	
	x = layers.DepthwiseConv2D((3,3), padding='same',strides=(2,2))(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	x = layers.Conv2D(64, (1,1))(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	
	x = layers.Dropout(0.5)(x)
	
	x = layers.DepthwiseConv2D((3,3), padding='same',strides=(2,2))(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	x = layers.Conv2D(128, (1,1))(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	
	x = layers.DepthwiseConv2D((3,3), padding='same')(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	x = layers.Conv2D(128, (1,1))(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)

	x = layers.Dropout(0.5)(x)

	x = layers.Flatten()(x)
	x = layers.Dense(128)(x)
	x = layers.BatchNormalization()(x)
	x = layers.ReLU()(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(10, activation='softmax')(x)

	model = keras.Model(inputs=inputs, outputs=x, name="cifar10_model")
	return model

