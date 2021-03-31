from tensorflow import keras
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

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
# there are 3, fresh freeze and rotten
outputs = keras.layers.Dense(3, activation = 'softmax')(x)

# Combine inputs and outputs to create model
model = keras.Model(inputs, outputs)

model.compile(loss = "categorical_crossentropy" , metrics =['acc'])

datagen = ImageDataGenerator(rescale=1./255,)

directory = "C:/Users/rasca/Documents/Homework/Spring 2021/CS 5620 Artificial Int/Project_ImageRec/FruitClassification02/Data03"
# subclasses = ["fresh", "freeze", "rotten"]

# load and iterate training dataset
train_it = datagen.flow_from_directory(directory + "/train/rottenbanana",
                                       target_size=(224,224),
                                       classes = ["fresh", "freeze", "rotten"],
                                       color_mode='rgb',
                                       class_mode="categorical",
                                       batch_size=32)

#look into these and maybe change class_mode from categorical
print(train_it.class_indices) # {'freshapples': 0, 'freshbanana': 1, 'freshoranges': 2, 'rottenapples': 3, 'rottenbanana': 4, 'rottenoranges': 5}


# load and iterate test dataset
test_it = datagen.flow_from_directory(directory + "/test/rottenbanana",
                                      classes=["fresh", "freeze", "rotten"],
                                      target_size=(224,224),
                                      color_mode='rgb',
                                      class_mode="categorical",
                                      batch_size=32)
#accuracy and loss to plot
history = model.fit(train_it,
              validation_data=test_it,
              steps_per_epoch=train_it.samples/train_it.batch_size,
              validation_steps=test_it.samples/test_it.batch_size,
              epochs=3)

# list all data in history
print(history.history.keys())
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


model.save_weights('cnnFruitFurther.h5')
