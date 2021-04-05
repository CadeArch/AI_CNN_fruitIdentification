from tensorflow import keras
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

## DONE BY MARIEM ##

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

model.summary()

model.compile(loss = "categorical_crossentropy" , metrics =['acc'])

datagen = ImageDataGenerator(rescale=1./255,)

directory = "C:/Users/rasca/Documents/Homework/Spring 2021/CS 5620 Artificial Int/Project_ImageRec/FruitClassification02/Data02"


# load and iterate training dataset
train_it = datagen.flow_from_directory(directory + "/train",
                                       target_size=(112,112),
                                       color_mode='rgb',
                                       class_mode="categorical",
                                       )


# load and iterate test dataset
test_it = datagen.flow_from_directory(directory + "/test",
                                      target_size=(112,112),
                                      color_mode='rgb',
                                      class_mode="categorical",
                                      )


history = model.fit(train_it,
                    validation_data=test_it,
                    steps_per_epoch=train_it.samples/train_it.batch_size,
                    validation_steps=test_it.samples/test_it.batch_size,
                    epochs=6)

# list all data in history
print(history.history.keys())
print(history.history.values())
print(history.history['acc'])
print(history.history['val_acc'])
print(history.history)
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print("done fitting")

model.save_weights('cnnFruitTry3.h5')
print("DONE")
