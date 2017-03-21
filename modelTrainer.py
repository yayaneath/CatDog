import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Constants

TRAIN_PATH = './train'
VALIDATION_PATH = './validation'

ROWS = 150
COLS = 150
CHANNELS = 3

BATCH_SIZE = 32
EPOCH_SAMPLES = 23000
NB_EPOCH = 10
NB_VAL_EPOCH = 2000

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Input data read

parser = argparse.ArgumentParser(description = 'This program trains a Convolutional Neural Network to learn to predict if a given picture is of a dog or cat.')
parser.add_argument('model_file', help = 'File that will save the trained model')

arg_model_file = parser.parse_args().model_file

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Model definition

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(ROWS, COLS, CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))

model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))

model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Training and validation data definition

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2,horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size = (ROWS, COLS), batch_size = BATCH_SIZE, class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(VALIDATION_PATH, target_size = (ROWS, COLS), batch_size = BATCH_SIZE, class_mode = 'binary')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Model training and saving

model.fit_generator(train_generator, samples_per_epoch = EPOCH_SAMPLES, nb_epoch = NB_EPOCH, validation_data = validation_generator, nb_val_samples = NB_VAL_EPOCH)

model.save(arg_model_file + '.h5')
