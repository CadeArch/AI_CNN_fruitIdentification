import random
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Flatten
import numpy
import os
import matplotlib.image as mpimg
from keras.preprocessing.image import array_to_img

def showRandImages(directory, rand):

    for begin in range(5):
        files = os.listdir(directory)
        if rand:
            file = random.choice(files)
        else:
            file = files[begin]
        image_path = os.path.join(directory, file)
        img = mpimg.imread(image_path)
        ax = plt.subplot(1, 5, begin + 1)
        ax.title.set_text(file)
        plt.imshow(img)
        plt.show()

def images_to_array(dataset_dir, image_size):
    dataset_array = []
    dataset_labels = []

    class_counter = 0

    classes_names = os.listdir(dataset_dir)
    for current_class_name in classes_names:
        class_dir = os.path.join(dataset_dir, current_class_name)
        images_in_class = os.listdir(class_dir)

        print("Class index", class_counter, ", ", current_class_name, ":" , len(images_in_class))

        for image_file in images_in_class:
            if image_file.endswith(".png"):
              image_file_dir = os.path.join(class_dir, image_file)

              img = keras.preprocessing.image.load_img(image_file_dir, target_size=(image_size, image_size))
              img_array = keras.preprocessing.image.img_to_array(img)

              img_array = img_array/255.0

              dataset_array.append(img_array)
              dataset_labels.append(class_counter)
        class_counter = class_counter + 1
    dataset_array = numpy.array(dataset_array)
    dataset_labels = numpy.array(dataset_labels)
    return dataset_array, dataset_labels

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
model.load_weights('ResultsReg/cnnFruitTry2.h5')


directory = "Data02/test"
# this translates images to an array and then saves those arrays off into storage for later use
# test_images, test_labels = images_to_array(directory, 112)
# numpy.save("test_img.npy", test_images)
# numpy.save("test_labels.npy", test_labels)
test_images = numpy.load("test_img.npy")
test_labels = numpy.load("test_labels.npy")

# to show images in a directory
# img_folder = "Data02/test/freshbanana"
# rand = False
# showRandImages(img_folder, rand)

# try 2 does better than try 1 in range 500 to 505
def predict(show=False):

    # this will show the images being predicted
    if show:
        for y in test_images[500:505]:
            img = array_to_img(y)
            img.show()

    print("Predictions: ", end="")

    # index into the test images array to choose which images to predict
    predictions = model.predict(test_images[500:505])
    arrayPred = []
    for x in predictions:
        biggest = max(x)
        counter = 0
        for highest in x:
            if biggest == highest:
                #print("image predicted: " + str(counter))
                arrayPred.append(counter)
                break
            else:
                counter += 1
    print(arrayPred)

    print("Actual: ", end="")
    print()
    print(test_labels[500:505])

    classLabels = ["fresh apples", "fresh banana", "fresh orange", "rotten apples", "rotten banana", "rotten orange"]
    print(classLabels)

predict()
