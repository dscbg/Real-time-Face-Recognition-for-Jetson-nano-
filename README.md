# Real-time-Face-Recognition-for-Jetson-nano
This is an openCV based python code that achieve real-time face data collection and real time face recognition. And all the works including (training model, data proccessing...) have been done on the Jetson Nano without using Docker container(or Jupyterlab) so we configure all the environment locally. These files can be easily adjusted for other classifications as well. And unlike other face recognition project for Jetson Nano, we don't need to upload data manually. Instead we can directly collect data when running the manager.py file. Similarly, we can do real time training after we collecting the data.
## File Description
- `Manager.py`: This file is for real-time data collection and real-time model training.

- `recogFunc.py`: This file is the main training method for our face recognition.

- `Recognizer.py`:This file is for testing the performance of our model.

## Environment Set up
When I was trying to make the program run, it took a while to download the required packages.(e.g. opencv,numpy,pillow) So I will list the commands that I used for setting up the environment. I'm using python3(Version3.6, numpy 1.19.4, opencv-contrib-python-4.5.4.60,pillow 8.4.0)
1. Pillow which contains the PTL package we need. Make sure to install libjpeg beforehand(It may report error which I encountered). The commmand for install libjpeg I used is `sudo apt-get install -y libjpeg-dev`. After that, enter `pip3 install Pillow`. Also, make sure your pip3 is update to date.
2. For opencv, I reinstall it because I encoutered problem `cv2 doesn't have attribute face`. The command for it is `pip3 install opencv-contrib-python`. You may encounter problem like `ImportError: No module named skbuild` and simply install scikit-build will fix this problem.

## Execution
First, run the `Manager.py` file to collect data and train the model. After you run the file, it will pop up an window that has record and train button on it. You need to enter your name in the text box to start record and collect your real-time data from the camera.When the collection is done, it will send message to tell you. If the database already have the data for your name, it will pop up a window to remind you that you can directly train your model. And you just need to click on the train button until you see the message `train completed` After that, run the `Recognizer.py` file and you can see the recognized face with name tag and reactangle that detects your face from the pop upcamera. If you have more questions, you can check my video about how to execute it.

## Tip
Also, remember to create a names.csv file and dataset folder before to avoid possible error when collecting data. For saving model, remember to create a trainer folder where the model will be saved. 

A correct files structure for running the program is:
- `Manager.py`
- `Recognizer.py`
- `Reset.py`: This file is for cleaning all the generated files(including data and models)
- `haarcascade_frontalface_default.xml`
- `recogFunc.py`
- `names.csv`
- `dataset`: This is a folder name `dataset` that you need to create beforehand
- `trainer`: This is a folder name `trainer` that you need to create beforehand and the model will be stored in the folder. When test the recognizer, the program will read model from the generated yml model file.
