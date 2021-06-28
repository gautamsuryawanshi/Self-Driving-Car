# Self Driving Car using Behavior Cloning
![](./Behavior Cloning/result/sdc.gif)

The objective of this project is to create a deep learning model which can automously drive itself in the Udacity's Self Driving Car simulator.

We feed the data collected from Simulator to our model, this data is fed in the form of images captured by 3 dashboard cams centre, left and right. The output data contains a file data.csv which has the mappings of the centre, left and right images and the corresponding steering angle, throttle, brake and speed.

Using Keras Deep learning framework we can create a model.h5 file which we can test later on the simulator with the command "python drive.py model.h5". This drive.py connects your model to the simulator. The challenge in this project is to collect all sorts of training data to train the model to respond correctly in any type of situation.
