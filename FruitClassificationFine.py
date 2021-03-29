from tensorflow import keras
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

## DONE BY MARIEM ##

base_model = keras.applications.VGG16(
    weights="imagenet",
    input_shape=(224, 224, 3),
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

model.compile(loss = "categorical_crossentropy" , metrics =['acc'])

datagen = ImageDataGenerator(rescale=1./255,)

directory = "C:/Users/rasca/Documents/Homework/Spring 2021/CS 5620 Artificial Int/Project_ImageRec/FruitClassification02/Data02"
# subclasses = ["fresh", "freeze", "rotten"]

# load and iterate training dataset
train_it = datagen.flow_from_directory(directory + "/train",
                                       target_size=(224,224),
                                       # classes = ["fresh", "freeze", "rotten"],
                                       color_mode='rgb',
                                       class_mode="categorical",
                                       batch_size=32)

#look into these and maybe change class_mode from categorical
print(train_it.class_indices) # {'freshapples': 0, 'freshbanana': 1, 'freshoranges': 2, 'rottenapples': 3, 'rottenbanana': 4, 'rottenoranges': 5}

# train_it.save_to_dir

# load and iterate test dataset
test_it = datagen.flow_from_directory(directory + "/test",
                                      # classes=["fresh", "freeze", "rotten"],
                                      target_size=(224,224),
                                      color_mode='rgb',
                                      class_mode="categorical",
                                      batch_size=32)

model.fit(train_it,
          validation_data=test_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=test_it.samples/test_it.batch_size,
          epochs=1)

#epocs was 6 does pretty good on the first one