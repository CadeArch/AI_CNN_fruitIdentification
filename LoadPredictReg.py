from tensorflow import keras
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

base_model = keras.applications.VGG16(
    weights="imagenet",
    input_shape=(112, 112, 3),
    include_top=False)

# Freeze base model
base_model.trainable = False


# Create inputs with correct shape
inputs = base_model.input

x = base_model(inputs, training=False)

# Add pooling layer or flatten layer
x = Flatten()(x)

# Add final dense layer
# there are 6, three fresh fruit 3 rotten
outputs = keras.layers.Dense(6, activation = 'softmax')(x)

# Combine inputs and outputs to create model
model = keras.Model(inputs, outputs)

# Load the model from disk later using:
model.load_weights('cnnFruitTry2.h5')

test_images = "get the photos here"
test_labels = "get test labels here"

print("Predictions: ", end="")
predictions = model.predict(test_images[:5])

print("Actual: ", end="")
print(test_labels[:5])
