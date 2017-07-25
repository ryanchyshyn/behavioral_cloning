import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D

# load log
lines = []
with open("C:\\Users\\rii\\Documents\\udacity\\driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# load images
images = []
steering_data = []
steering_corr = 0.35 # this is a parameter to tune
images_dir = "C:\\Users\\rii\\Documents\\udacity\\IMG\\"
for line in lines:
    image_path = images_dir + line[0].split('\\')[-1]
    image = cv2.imread(image_path)
    steering = float(line[3])
    images.append(image)
    steering_data.append(steering)

    # flip image
    images.append(cv2.flip(image, 1))
    steering_data.append(-steering)

    # process left image
    image_left_path = images_dir + line[1].split('\\')[-1]
    image_left = cv2.imread(image_left_path)
    images.append(image_left)
    steering_data.append(steering + steering_corr)

    # flip image
    #images.append(cv2.flip(image_left, 1))
    #steering_data.append(-(steering + steering_corr))

    # process right image
    image_right_path = images_dir + line[2].split('\\')[-1]
    image_right = cv2.imread(image_right_path)
    images.append(image_right)
    steering_data.append(steering - steering_corr)

    # flip image
    #images.append(cv2.flip(image_right, 1))
    #steering_data.append(-(steering - steering_corr))



X_train = np.array(images)
y_train = np.array(steering_data)


# train
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(64, 5, 5, activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save("model.h5")
