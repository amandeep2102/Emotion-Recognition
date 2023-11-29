# Emotion-Recognition

## Introduction
This is a machine learning project which aims to recognize human emotions through facial features. The model has been successfully implemented inside of a desktop application which can now recognize human emotions through a live web camera feed or a video of your choice.

## Prerequisites
**Dataset** - [fer2013](https://www.kaggle.com/datasets/deadskull7/fer2013)  
**Install required libraries mentioned in Requirements.txt**

## Build Instructions

### Preparing Dataset
After downloading the [fer2013](https://www.kaggle.com/datasets/deadskull7/fer2013) *.csv* file.  
While in the same directory as the repository:
~~~
python3 ./prepare_dataset.py /path/to/fer2013.csv
~~~

Now Repo directory should be like:  
* app (folder)
* Data (folder)
  *  test (sub-folder)  
  *  train (sub-folder)
* epare_dataset.py (file)
* main.py (file)
* haarcascade_frontalface_default.xml (file)  

### Machine Learning Model
The Model uses Tensorflow, Keras utility and OpenCV.

To train the model,  
```
python3 main.py --mode train
```
This will generate a model.h5 file in the directory which will take some time depending on the computational speed, and is required in the next step.
If you want to download the pretrained model file use can the download the file from [here](https://drive.google.com/file/d/1FUn0XNOzf-nQV7QjbBPA6-8GLoHNNgv-/view)

To use the model for real time prediction, 
```
python3 main.py --mode display
```

### Desktop Application
The desktop application was designed using Pyqt library along with OpenCV for video and web camera feed. The following dependencies are :- **PySide2, Pyqt6, opencv**. You must have these dependencies if you want to build the application from source. 

There is a model included in the app directory, along with the xml file which are necessary for running the application.

Then 

```
cd app
python3 app.py
```


Now you should be able to see a new window on your screen.

### Refrences 
*   "Challenges in Representation Learning: A report on three machine learning contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,
X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu, M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and Y. Bengio. arXiv 2013.
