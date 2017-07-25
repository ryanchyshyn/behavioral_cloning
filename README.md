**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center_2017_07_24_23_22_05_604]: ./examples/center_2017_07_24_23_22_05_604.jpg
[center_2017_07_24_23_23_13_804]: ./examples/center_2017_07_24_23_23_13_804.jpg
[center_2017_07_24_23_23_26_235]: ./examples/center_2017_07_24_23_23_26_235.jpg
[center_2017_07_24_23_28_05_608]: ./examples/center_2017_07_24_23_28_05_608.jpg
[center_2017_07_24_23_33_32_724]: ./examples/center_2017_07_24_23_33_32_724.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md or writeup_report.pdf summarizing the results
* run1.mp4 the video of autonomous driving the first track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 83-89) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 59). 

Also model includes cropping layer (line 60) and dropout layer.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 105). To prevent overfitting the model I used dropout layers. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 104).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the described in the "Behavioral Cloning" lesson.

Training the model on more than two epochs leads to increasing testing set validation loss. So to avoid overfitting I decided to keep epoch parameter equals to 3.

To introduce car returning to the center I included left/right images into the training set. Steering correction for such images is 0.35.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 81-89) consisted of a convolution neural network with the following layers and layer sizes 
```
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
```


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded several laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center_2017_07_24_23_22_05_604]

Then I additionally recorded key turns:

![alt text][center_2017_07_24_23_33_32_724]
![alt text][center_2017_07_24_23_28_05_608]
![alt text][center_2017_07_24_23_23_13_804]
![alt text][center_2017_07_24_23_23_26_235]

To augment the data set, I flipped center images and corrected angles by valueof 0.35.

Also I flipped left/right images and inverted steering values.

I used data nomralization with lambda function as well as image cropping.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by decreasing validation set loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
