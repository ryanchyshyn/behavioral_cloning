import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D
import sklearn

from sklearn.model_selection import train_test_split

data_dir = "C:\\Users\\rii\\Documents\\udacity\\"
steering_corr = 0.35  # this is a parameter to tune
images_dir = data_dir + "IMG\\"

# load log
lines = []
with open(data_dir + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# load images
images = []
steering_data = []


def process_line(center_image_path, left_image_path, right_image_path, steering, images, steering_data):
    center_image = cv2.cvtColor(cv2.imread(center_image_path), cv2.COLOR_BGR2RGB)
    images.append(center_image)
    steering_data.append(steering)

    # flip image
    images.append(cv2.flip(center_image, 1))
    steering_data.append(-steering)

    # process left image
    image_left = cv2.cvtColor(cv2.imread(left_image_path), cv2.COLOR_BGR2RGB)
    images.append(image_left)
    steering_data.append(steering + steering_corr)

    # flip left image
    images.append(cv2.flip(image_left, 1))
    steering_data.append(-(steering + steering_corr))

    # process right image
    image_right = cv2.cvtColor(cv2.imread(right_image_path), cv2.COLOR_BGR2RGB)
    images.append(image_right)
    steering_data.append(steering - steering_corr)

    # flip right image
    images.append(cv2.flip(image_right, 1))
    steering_data.append(-(steering - steering_corr))


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            lines = samples[offset:offset+batch_size]

            images = []
            steering_data = []
            for line in lines:
                steering = float(line[3])
                center_image_path = images_dir + line[0].split('\\')[-1]
                left_image_path = images_dir + line[1].split('\\')[-1]
                right_image_path = images_dir + line[2].split('\\')[-1]
                process_line(center_image_path, left_image_path, right_image_path, steering, images, steering_data)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(steering_data)
            yield sklearn.utils.shuffle(X_train, y_train)


# model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 5, 5, activation="relu"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model.compile(loss="mse", optimizer="adam")
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save("model.h5")

# workaround for issue
# Exception ignored in...
# AttributeError: 'NoneType' object has no attribute 'TF_DeleteStatus'
from keras import backend
backend.clear_session()
