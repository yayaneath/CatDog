import numpy as np
import argparse
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Constants

ROWS = 150
COLS = 150
CHANNELS = 3

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Input data read

parser = argparse.ArgumentParser(description = 'This program reads a picture and gives a prediction of being an image of a dog/cat, given a trained model.')
parser.add_argument('picture', help = 'Path to the picture file')
parser.add_argument('model', help = 'Path to the model file')

arg_picture = parser.parse_args().picture
arg_model = parser.parse_args().model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Main execution

print 'Loading picture...'

img = load_img(arg_picture, target_size=(ROWS, COLS))
img_array = img_to_array(img)
samples = np.expand_dims(img_array, axis = 0)

print 'Loading model...'

model = load_model(arg_model)
prediction = model.predict(samples)

if prediction[0][0] >= 0.5:
	plt.title("I'm {}% sure this is a dog".format(prediction[0][0] * 100.))
	plt.imshow(load_img(arg_picture))
	plt.show()
else:
	plt.title("I'm {}% sure this is a cat".format((1 - prediction[0][0]) * 100.))
	plt.imshow(load_img(arg_picture))
	plt.show()