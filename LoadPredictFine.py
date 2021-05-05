from random import randint
from tensorflow import keras
from tensorflow.keras.layers import Flatten
import numpy
import os
from keras.preprocessing.image import array_to_img
from sklearn import metrics

# function to put the images of the database into an array
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

# using image sizes of 112 by 112
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
# there are 9 output layers freeze fresh and rotten for each fruit
outputs = keras.layers.Dense(9, activation = 'softmax')(x)

# Combine inputs and outputs to create model
model = keras.Model(inputs, outputs)

# Load the model from disk later using:
model.load_weights('ResultsFine/cnnBananaFine.h5')

# directories i had for tweaking the three seperate models
appleDir = "Data03/test/apple"
orangeDir = "Data03/test/orange"
bananaDir = "Data03/test/banana"

# tweaked directory to host all all 9 categories
combinedDir = "Data03/test"

# directory in use
directory = combinedDir

# this translates images to an array and then saves those arrays off into storage for later use

test_images, test_labels = images_to_array(directory, 112)
numpy.save("test_img_fine.npy", test_images)
numpy.save("test_labels_fine.npy", test_labels)

# to load in previously created numpy files
# test_images = numpy.load("test_img_fine_banana.npy")
# test_labels = numpy.load("test_labels_fine_banana.npy")

# function to predict
def predict(beg, end, show=False, random=False):

    if random:
        size = len(test_images)
        randomImages = []
        times = end - beg
        for x in range(0, times):
            randimages = randint(0, size - 1)
            randomImages.append(randimages)

        if show:
            for y in range(0, times):
                img = array_to_img(test_images[randomImages[y]])
                img.show()

        # index into the test images array to choose which images to predict
        toPredict = []
        actual = []
        for x in randomImages:
            toPredict.append(test_images[x])
            actual.append(test_labels[x])
        toPredict = numpy.array(toPredict)
        predictions = model.predict(toPredict)

        arrayPred = []
        for x in predictions:
            biggest = max(x)
            counter = 0
            for highest in x:
                if biggest == highest:
                    # print("image predicted: " + str(counter))
                    arrayPred.append(counter)
                    break
                else:
                    counter += 1
        print("Predictions: ", end="")
        print(arrayPred)

        print("Actual: ", end="")
        print()
        print(actual)

        # classLabels = ["freeze b", "fresh b", "rotten b"]
        # print(classLabels)
        return actual, arrayPred


    else:
        # this will show the images being predicted
        if show:
            for y in test_images[beg:end]:
                img = array_to_img(y)
                img.show()

        print("Predictions: ", end="")

        # index into the test images array to choose which images to predict
        predictions = model.predict(test_images[beg:end])

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
        print(test_labels[beg:end])

        # classLabels = ["freeze b", "fresh b", "rotten b"]
        # print(classLabels)
        return test_labels[beg:end], arrayPred

# parameters
beg = 0
end = 5
showImages = True
random = False

# predict! use these arrays to create your confusion matrix
actualArray, predictedArray = predict(beg, end, showImages, random)

# find this out when you run numpy.save on your images to array directory
classLabelsKey = [];
labelsArray = [0, 1, 2, 3, 4, 5, 6, 7, 8]

#confusion matrix and stats
print(metrics.confusion_matrix(actualArray, predictedArray, labels=labelsArray))
print(metrics.classification_report(actualArray, predictedArray, labels=labelsArray))

"""
combine together, hierarchical NN structure
confusion matricies: 3 classes, see what mistakes system is making prediction vs actual
look into changing loss function (how much blemish is acceptable) economic or keep more bad

oranges: look into other options for semi spoiled oranges
apples: learning rate too high? or more epochs
"""