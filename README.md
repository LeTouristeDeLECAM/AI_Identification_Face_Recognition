# AI_Identification_Face_Recognition
 This project is taking part in larger school project where we implement a photo-gallery identification
 This have to interpret as a peace of proof 

 Instructions: https://kenn7.github.io/AIproject/project/


## Research about the implementation of face recognition

Path : Research/

### LBPH
Local Binary Patterns Histograms (LBPH) 
### YOLO pretrained model

## Transfert learning YOLO/ Training YOLO model

Path : Transfert_learning_YOLO/

Description : The goal is to detect if a person is wearing glasses or not. To proceed we've used at first a detection solution but unfortunatly the copped image of the face discard the back ground and do not permit to generalize the detection. So we've used a classification solution. 


### Data set :
The Dataset is MeGlass, all the face images are selected and cleaned from MegaFace.
We used the version composed of copped images 120x120.

Ref :  https://github.com/cleardusk/MeGlass 

Some processing on the dataset is needed to be done to be able to use it with YOLO.
The balacing of the dataset is crucial to avoid bias.
The dataset is splited in 2 parts :
- 80% for training
- 20% for validation


format : 
- Classification 
    - 1 folder per class
    - https://docs.ultralytics.com/datasets/classify/
    - Editing notebook: Transfert_learning_YOLO\Classification\Editing_Balaced_DataSet_classification.ipynb

- Detection
    - 1 txt file per image
    - https://docs.ultralytics.com/datasets/detect/
    - Editing notebook: Transfert_learning_YOLO\Detection\Editing_Balaced_DataSet.ipynb



### Classification 

The best model is train2 model with 10 epochs and a batch size of 640.
The dataset is composed of 1250 faces divided in 2:  
        - Test set : Eyeglasses = 174, No Eyeglasses = 176
        - Train set : Eyeglasses = 537, No Eyeglasses = 663

![Confusion Matrix](https://github.com/LeTouristeDeLECAM/AI_Identification_Face_Recognition/blob/main/Transfert_learning_YOLO/Classification/runs/classify/train2/confusion_matrix.png)


## Virtual environment
start virtual environment in terminal

```
venvFaceRecognition/Scripts/activate
```
https://www.youtube.com/watch?v=KxvKCSwlUv8 

## Install requirements
```
pip install -r requirements.txt
```


## Source :
https://iq.opengenus.org/lbph-algorithm-for-face-recognition/

https://www.freecodecamp.org/news/how-to-detect-objects-in-images-using-yolov8/

