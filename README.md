# BITI 1113 - Artificial Intelligence Project
## A. PROJECT SUMMARY

![Group Member](https://github.com/TanWeiYin/Group_J_AI_Project/blob/main/misc/Group_J.jpg)

**Objectives**

- To classify each face based on the emotion shown in the facial expression into one of six categories (Angry, Fear, Happy, Neutral, Sad, Surprise).
- To help people to see their present mental condition.
- To develop an algorithm that can detect facial expression in static image and live video.

## B. ABSTRACT

AI is the ability of machines to mimic human capabilities in a way that we would consider 'smart'.
Machine learning is an application of AI. With machine learning, we give the machine lots of examples of data, demonstrating what we would like it to do so that it can figure out how to achieve a goal on its own. The machine learns and adapts its strategy to achieve the goal.
In our project, we are feeding the machine images of our facial expressions via the inbuilt camera. The more varied the data we provide, the more likely the AI will correctly classify the input as the appropriate emotion. In machine learning, the system will give a confidence value; in this case, a percentage and the bar filled or partially filled, represented by colour. The confidence value provides us with an indication of how sure the AI is of its classification.
This project focuses on the concept of classification. Classification is a learning technique used to group data based on attributes or features.
We are in the process of learning how to develop AI project using python. Our team has chose to develop this AI technique as we fervently hope that it can be applied in robot caretaker especially look after children having emotion disorder. Our facial emotion recognition algorithm can identify six different type of emotional states in real-time: happiness, sadness, surprise, anger, neutral and fear. The robot can give appropriate response after detecting the child's emotion. Besides, this algorithm can be applied in AI customer service. Live video care line session is more accurate than traditional audio assistance care line. We have used 4-Conv Layered CNN Model as our facial emotion recognition technique.

## C. DATASET
The dataset used in the training of AI was created by Jonathan Oheix back in 2019 which is available on [kaggle](https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset).

The original dataset consist of seven classes of images separated into two categories.

![OriDataset](https://github.com/TanWeiYin/Group_J_AI_Project/blob/main/misc/OriDataset.jpg)

We did not use the images of disgust as the number of image is disproportionaly small compare to the other classes and this might cause problem in training the AI. For the remaining classes, we've only used a subset of each classes so that the training and validation phase will not take too long.

![SelectedDataset](https://github.com/TanWeiYin/Group_J_AI_Project/blob/main/misc/SelectedDataset.jpg)

We adjust the number of images so that the number in each classes is about even and the ratio of training image to validation image is about 8:2.

## D. PROJECT STRUCTURE

This repository is organized as:
```bash
    ├── misc (8 entries)                              # Miscellaneous information
    │   ├── GroupJ_Slides.pdf
    │   ├── Group_J.jpg
    │   ├── OriDataSet.jpg
    │   ├── SelectedDataSet.jpg
    │   ├── TestResult.jpg
    │   ├── confusionmatrix.jpg
    │   ├── dataset.jpg
    │   ├── emotionimg.jpg
    │   ├── livedemo.jpg
    │   └── training_loss_accuracy.png  
    ├── src (5 entries)
    │   ├── haarcascade_frontalface_default.xml       # To detect the face of individuals
    │   ├── HumanRecognitionTrainModel.ipynb          # Training script
    │   ├── Main.ipynb                                # Entry point of webcam live demo
    │   ├── Model.h5                                  # Load Keras Model
    │   └── PlotConfussionMatrix.ipynb                # Confusion matrix visualization
    ├── .gitignore
    └── README.md
    2 categories, 15 files 
```

The misc directory contains the miscellaneous information of project Content (images,graphs,slides).

The src directory contains the jupyter scripts as follows:
- haarcascade_frontalface_default.xml : A typical face object detection library that will be used to detect the face of individuals.
- HumanRecognitionTrainModel.ipynb : The execution of training process starts here and end with validation the effectiveness of AI.
- Main.ipynb : It works as the entry point of webcam live demo.
- PlotConfussionMatrix.ipynb : We can take a look on Confusion Matrix and Classification Report from here.

## E. TRAINING THE HUMAN EMOTION RECOGNITION

1. Try reading and displaying image from dataset
    
    We will try to read and display some images from dataset for making sure that the dataset is working fine.
    
![emotionimg](https://github.com/Josie528/BITI-1113-AI-Project/blob/main/misc/emotionimg.jpg)

2. Training and validate data

    We will take the images from both train and validate sets and read the images. The images read will have it's size set to 48 x 48. Besides that, the color of the image will be set to greyscale as the image only colors that are shades of gray. Therefore, less information needs to be provided for each pixel. The batch size is also set to 32. This means that the training model will take 32 training example (files) in one iteration. The class mode is also set tp "categorical" as we have 6 emotions categorize in our datasets. In the training set, the data are shuffled as to prevent the deep learning model from learning the sequences of the data. This will allow the deep learning to be more dynamic and robust. The dataset was modified from the original dataset and split into a 8:2 ratio (80% training images and 20% test images).

3. Building 4-Conv Layered CNN Model
   
   The practical benefit of using CNN with 4 convolutional layer is that having fewer parameters greatly improves the time it takes to learn as well as reduces the amount of data required to train the model. The beauty of the CNN is that the number of parameters is independent of the size of the original image. We can run the same CNN on a 300 × 300 image, and the number of parameters won’t change in the convolution layer. The sliding-window shenanigans happen in the convolution layer of the neural network. A typical CNN has multiple convolution layers.
   
4. Set Training Callbacks list by defining Save Checkpoint, Early Stopping and Reduce Learning Rate
   
    Callbacks provide a way to execute code and interact with the training model process automatically. We used custom callback so that it can be used to dynamically change the learning rate of the optimizer during the course of training. Callback called EarlyStopping is used to specify the performance measure to monitor and trigger. It will stop the training process when it has been triggered but the model at the end of training may not be the best model with good performance on the validation dataset. ModelCheckPoint callback is required in order to save the best model observed during training for future use.

5. Compile Model and Train Model
    Here we can compile the model and begin the training of the AI. The training is split into 70 epoch such that every image will be included in the training once for each epoch. Each epoch is further split into many steps where each steps consist of the training using 32 images.
```bash
    Epoch 1/70
564/564 [==============================] - 79s 139ms/step - loss: 2.5027 - acc: 0.2365 - val_loss: 1.7213 - val_acc: 0.3208

Epoch 00001: val_acc improved from -inf to 0.32081, saving model to .\model.h5
Epoch 2/70
564/564 [==============================] - 79s 140ms/step - loss: 2.1637 - acc: 0.2725 - val_loss: 1.6359 - val_acc: 0.3567

Epoch 00002: val_acc improved from 0.32081 to 0.35672, saving model to .\model.h5
Epoch 3/70
564/564 [==============================] - 81s 143ms/step - loss: 1.9734 - acc: 0.3119 - val_loss: 1.5710 - val_acc: 0.3820

Epoch 00003: val_acc improved from 0.35672 to 0.38198, saving model to .\model.h5
Epoch 4/70
564/564 [==============================] - 82s 146ms/step - loss: 1.8370 - acc: 0.3369 - val_loss: 1.5214 - val_acc: 0.3856

Epoch 00004: val_acc improved from 0.38198 to 0.38564, saving model to .\model.h5
Epoch 5/70
564/564 [==============================] - 82s 146ms/step - loss: 1.7493 - acc: 0.3563 - val_loss: 1.4367 - val_acc: 0.4325

Epoch 00005: val_acc improved from 0.38564 to 0.43251, saving model to .\model.h5
Epoch 6/70
564/564 [==============================] - 82s 146ms/step - loss: 1.6567 - acc: 0.3781 - val_loss: 1.3716 - val_acc: 0.4661

Epoch 00006: val_acc improved from 0.43251 to 0.46609, saving model to .\model.h5
Epoch 7/70
564/564 [==============================] - 83s 147ms/step - loss: 1.5806 - acc: 0.3991 - val_loss: 1.3503 - val_acc: 0.4751

Epoch 00007: val_acc improved from 0.46609 to 0.47507, saving model to .\model.h5
Epoch 8/70
564/564 [==============================] - 83s 147ms/step - loss: 1.5072 - acc: 0.4202 - val_loss: 1.3333 - val_acc: 0.4817

Epoch 00008: val_acc improved from 0.47507 to 0.48172, saving model to .\model.h5
Epoch 9/70
564/564 [==============================] - 82s 146ms/step - loss: 1.4615 - acc: 0.4308 - val_loss: 1.3107 - val_acc: 0.4827

Epoch 00009: val_acc improved from 0.48172 to 0.48271, saving model to .\model.h5
Epoch 10/70
564/564 [==============================] - 83s 147ms/step - loss: 1.4134 - acc: 0.4445 - val_loss: 1.2761 - val_acc: 0.5076

Epoch 00010: val_acc improved from 0.48271 to 0.50765, saving model to .\model.h5
Epoch 11/70
564/564 [==============================] - 82s 146ms/step - loss: 1.3788 - acc: 0.4576 - val_loss: 1.2453 - val_acc: 0.5229

Epoch 00011: val_acc improved from 0.50765 to 0.52294, saving model to .\model.h5
Epoch 12/70
564/564 [==============================] - 82s 145ms/step - loss: 1.3463 - acc: 0.4736 - val_loss: 1.2504 - val_acc: 0.5156

Epoch 00012: val_acc did not improve from 0.52294
Epoch 13/70
564/564 [==============================] - 82s 146ms/step - loss: 1.3164 - acc: 0.4863 - val_loss: 1.2397 - val_acc: 0.5146

Epoch 00013: val_acc did not improve from 0.52294
Epoch 14/70
564/564 [==============================] - 82s 145ms/step - loss: 1.2869 - acc: 0.5009 - val_loss: 1.2631 - val_acc: 0.5199

Epoch 00014: val_acc did not improve from 0.52294
Epoch 15/70
564/564 [==============================] - 82s 146ms/step - loss: 1.2589 - acc: 0.5075 - val_loss: 1.1936 - val_acc: 0.5475

Epoch 00015: val_acc improved from 0.52294 to 0.54754, saving model to .\model.h5
Epoch 16/70
564/564 [==============================] - 82s 146ms/step - loss: 1.2398 - acc: 0.5177 - val_loss: 1.2137 - val_acc: 0.5432

Epoch 00016: val_acc did not improve from 0.54754
Epoch 17/70
564/564 [==============================] - 83s 147ms/step - loss: 1.2182 - acc: 0.5245 - val_loss: 1.1517 - val_acc: 0.5708

Epoch 00017: val_acc improved from 0.54754 to 0.57081, saving model to .\model.h5
Epoch 18/70
564/564 [==============================] - 82s 146ms/step - loss: 1.1948 - acc: 0.5332 - val_loss: 1.1471 - val_acc: 0.5775

Epoch 00018: val_acc improved from 0.57081 to 0.57746, saving model to .\model.h5
Epoch 19/70
564/564 [==============================] - 83s 146ms/step - loss: 1.1800 - acc: 0.5428 - val_loss: 1.1281 - val_acc: 0.5824

Epoch 00019: val_acc improved from 0.57746 to 0.58245, saving model to .\model.h5
Epoch 20/70
564/564 [==============================] - 82s 146ms/step - loss: 1.1572 - acc: 0.5535 - val_loss: 1.1408 - val_acc: 0.5781

Epoch 00020: val_acc did not improve from 0.58245
Epoch 21/70
564/564 [==============================] - 83s 147ms/step - loss: 1.1485 - acc: 0.5587 - val_loss: 1.1352 - val_acc: 0.5861

Epoch 00021: val_acc improved from 0.58245 to 0.58610, saving model to .\model.h5
Epoch 22/70
564/564 [==============================] - 83s 147ms/step - loss: 1.1296 - acc: 0.5692 - val_loss: 1.0965 - val_acc: 0.5934

Epoch 00022: val_acc improved from 0.58610 to 0.59342, saving model to .\model.h5
Epoch 23/70
564/564 [==============================] - 82s 146ms/step - loss: 1.1105 - acc: 0.5696 - val_loss: 1.0958 - val_acc: 0.5984

Epoch 00023: val_acc improved from 0.59342 to 0.59840, saving model to .\model.h5
Epoch 24/70
564/564 [==============================] - 82s 146ms/step - loss: 1.1038 - acc: 0.5745 - val_loss: 1.0767 - val_acc: 0.6001

Epoch 00024: val_acc improved from 0.59840 to 0.60007, saving model to .\model.h5
Epoch 25/70
564/564 [==============================] - 83s 147ms/step - loss: 1.0858 - acc: 0.5813 - val_loss: 1.0707 - val_acc: 0.6090

Epoch 00025: val_acc improved from 0.60007 to 0.60904, saving model to .\model.h5
Epoch 26/70
564/564 [==============================] - 82s 145ms/step - loss: 1.0723 - acc: 0.5860 - val_loss: 1.0820 - val_acc: 0.6157

Epoch 00026: val_acc improved from 0.60904 to 0.61569, saving model to .\model.h5
Epoch 27/70
564/564 [==============================] - 82s 145ms/step - loss: 1.0565 - acc: 0.5958 - val_loss: 1.0773 - val_acc: 0.6177

Epoch 00027: val_acc improved from 0.61569 to 0.61769, saving model to .\model.h5
Epoch 28/70
564/564 [==============================] - 82s 145ms/step - loss: 1.0412 - acc: 0.6038 - val_loss: 1.0398 - val_acc: 0.6243

Epoch 00028: val_acc improved from 0.61769 to 0.62434, saving model to .\model.h5
Epoch 29/70
564/564 [==============================] - 83s 147ms/step - loss: 1.0208 - acc: 0.6064 - val_loss: 1.0692 - val_acc: 0.6250

Epoch 00029: val_acc improved from 0.62434 to 0.62500, saving model to .\model.h5
Epoch 30/70
564/564 [==============================] - 82s 145ms/step - loss: 0.9984 - acc: 0.6152 - val_loss: 1.0171 - val_acc: 0.6376

Epoch 00030: val_acc improved from 0.62500 to 0.63763, saving model to .\model.h5
Epoch 31/70
564/564 [==============================] - 83s 146ms/step - loss: 0.9946 - acc: 0.6193 - val_loss: 1.0131 - val_acc: 0.6366

Epoch 00031: val_acc did not improve from 0.63763
Epoch 32/70
564/564 [==============================] - 82s 145ms/step - loss: 0.9791 - acc: 0.6253 - val_loss: 0.9937 - val_acc: 0.6529

Epoch 00032: val_acc improved from 0.63763 to 0.65293, saving model to .\model.h5
Epoch 33/70
564/564 [==============================] - 82s 146ms/step - loss: 0.9597 - acc: 0.6316 - val_loss: 0.9971 - val_acc: 0.6476

Epoch 00033: val_acc did not improve from 0.65293
Epoch 34/70
564/564 [==============================] - 82s 145ms/step - loss: 0.9424 - acc: 0.6394 - val_loss: 0.9765 - val_acc: 0.6616

Epoch 00034: val_acc improved from 0.65293 to 0.66157, saving model to .\model.h5
Epoch 35/70
564/564 [==============================] - 82s 145ms/step - loss: 0.9297 - acc: 0.6479 - val_loss: 0.9739 - val_acc: 0.6662

Epoch 00035: val_acc improved from 0.66157 to 0.66622, saving model to .\model.h5
Epoch 36/70
564/564 [==============================] - 81s 144ms/step - loss: 0.9136 - acc: 0.6538 - val_loss: 0.9750 - val_acc: 0.6669

Epoch 00036: val_acc improved from 0.66622 to 0.66689, saving model to .\model.h5
Epoch 37/70
564/564 [==============================] - 82s 146ms/step - loss: 0.8932 - acc: 0.6623 - val_loss: 0.9484 - val_acc: 0.6732

Epoch 00037: val_acc improved from 0.66689 to 0.67320, saving model to .\model.h5
Epoch 38/70
564/564 [==============================] - 83s 147ms/step - loss: 0.8766 - acc: 0.6716 - val_loss: 0.9709 - val_acc: 0.6812

Epoch 00038: val_acc improved from 0.67320 to 0.68118, saving model to .\model.h5
Epoch 39/70
564/564 [==============================] - 81s 144ms/step - loss: 0.8683 - acc: 0.6764 - val_loss: 0.9445 - val_acc: 0.6815

Epoch 00039: val_acc improved from 0.68118 to 0.68152, saving model to .\model.h5
Epoch 40/70
564/564 [==============================] - 81s 144ms/step - loss: 0.8512 - acc: 0.6770 - val_loss: 0.9861 - val_acc: 0.6875

Epoch 00040: val_acc improved from 0.68152 to 0.68750, saving model to .\model.h5
Epoch 41/70
564/564 [==============================] - 82s 145ms/step - loss: 0.8373 - acc: 0.6845 - val_loss: 1.0301 - val_acc: 0.6812

Epoch 00041: val_acc did not improve from 0.68750
Epoch 42/70
564/564 [==============================] - 84s 149ms/step - loss: 0.8216 - acc: 0.6914 - val_loss: 0.9751 - val_acc: 0.6941

Epoch 00042: val_acc improved from 0.68750 to 0.69415, saving model to .\model.h5
Epoch 43/70
564/564 [==============================] - 82s 145ms/step - loss: 0.8016 - acc: 0.6999 - val_loss: 0.9946 - val_acc: 0.6958

Epoch 00043: val_acc improved from 0.69415 to 0.69581, saving model to .\model.h5
Epoch 44/70
564/564 [==============================] - 82s 145ms/step - loss: 0.7962 - acc: 0.7013 - val_loss: 0.9429 - val_acc: 0.7041

Epoch 00044: val_acc improved from 0.69581 to 0.70412, saving model to .\model.h5
Epoch 45/70
564/564 [==============================] - 83s 147ms/step - loss: 0.7852 - acc: 0.7070 - val_loss: 0.9122 - val_acc: 0.7045

Epoch 00045: val_acc improved from 0.70412 to 0.70445, saving model to .\model.h5
Epoch 46/70
564/564 [==============================] - 82s 145ms/step - loss: 0.7731 - acc: 0.7097 - val_loss: 0.9220 - val_acc: 0.7045

Epoch 00046: val_acc did not improve from 0.70445
Epoch 47/70
564/564 [==============================] - 82s 145ms/step - loss: 0.7528 - acc: 0.7229 - val_loss: 0.9204 - val_acc: 0.7188

Epoch 00047: val_acc improved from 0.70445 to 0.71875, saving model to .\model.h5
Epoch 48/70
564/564 [==============================] - 82s 145ms/step - loss: 0.7329 - acc: 0.7245 - val_loss: 0.9170 - val_acc: 0.7181

Epoch 00048: val_acc did not improve from 0.71875
Epoch 49/70
564/564 [==============================] - 82s 146ms/step - loss: 0.7350 - acc: 0.7286 - val_loss: 0.8721 - val_acc: 0.7207

Epoch 00049: val_acc improved from 0.71875 to 0.72074, saving model to .\model.h5
Epoch 50/70
564/564 [==============================] - 82s 145ms/step - loss: 0.7078 - acc: 0.7382 - val_loss: 0.8821 - val_acc: 0.7344

Epoch 00050: val_acc improved from 0.72074 to 0.73438, saving model to .\model.h5
Epoch 51/70
564/564 [==============================] - 82s 146ms/step - loss: 0.7009 - acc: 0.7406 - val_loss: 0.8826 - val_acc: 0.7267

Epoch 00051: val_acc did not improve from 0.73438
Epoch 52/70
564/564 [==============================] - 82s 146ms/step - loss: 0.6817 - acc: 0.7459 - val_loss: 0.8807 - val_acc: 0.7294

Epoch 00052: val_acc did not improve from 0.73438
Epoch 53/70
564/564 [==============================] - 82s 146ms/step - loss: 0.6669 - acc: 0.7512 - val_loss: 0.8894 - val_acc: 0.7277

Epoch 00053: val_acc did not improve from 0.73438
Epoch 54/70
564/564 [==============================] - 82s 146ms/step - loss: 0.6606 - acc: 0.7560 - val_loss: 0.8671 - val_acc: 0.7304

Epoch 00054: val_acc did not improve from 0.73438
Epoch 55/70
564/564 [==============================] - 82s 146ms/step - loss: 0.6410 - acc: 0.7644 - val_loss: 0.8647 - val_acc: 0.7390

Epoch 00055: val_acc improved from 0.73438 to 0.73903, saving model to .\model.h5
Epoch 56/70
564/564 [==============================] - 82s 145ms/step - loss: 0.6243 - acc: 0.7682 - val_loss: 0.8334 - val_acc: 0.7477

Epoch 00056: val_acc improved from 0.73903 to 0.74767, saving model to .\model.h5
Epoch 57/70
564/564 [==============================] - 82s 145ms/step - loss: 0.6168 - acc: 0.7743 - val_loss: 0.8985 - val_acc: 0.7354

Epoch 00057: val_acc did not improve from 0.74767
Epoch 58/70
564/564 [==============================] - 82s 146ms/step - loss: 0.6101 - acc: 0.7807 - val_loss: 0.8802 - val_acc: 0.7377

Epoch 00058: val_acc did not improve from 0.74767
Epoch 59/70
564/564 [==============================] - 82s 146ms/step - loss: 0.5847 - acc: 0.7880 - val_loss: 0.8444 - val_acc: 0.7480

Epoch 00059: val_acc improved from 0.74767 to 0.74801, saving model to .\model.h5
Epoch 60/70
564/564 [==============================] - 82s 145ms/step - loss: 0.5943 - acc: 0.7796 - val_loss: 0.8509 - val_acc: 0.7480

Epoch 00060: val_acc did not improve from 0.74801
Epoch 61/70
564/564 [==============================] - 82s 145ms/step - loss: 0.5644 - acc: 0.7930 - val_loss: 0.8411 - val_acc: 0.7553

Epoch 00061: val_acc improved from 0.74801 to 0.75532, saving model to .\model.h5
Epoch 62/70
564/564 [==============================] - 82s 146ms/step - loss: 0.5563 - acc: 0.7965 - val_loss: 0.8766 - val_acc: 0.7497

Epoch 00062: val_acc did not improve from 0.75532
Epoch 63/70
564/564 [==============================] - 83s 146ms/step - loss: 0.5562 - acc: 0.7997 - val_loss: 0.8456 - val_acc: 0.7527

Epoch 00063: val_acc did not improve from 0.75532
Epoch 64/70
564/564 [==============================] - 84s 149ms/step - loss: 0.5402 - acc: 0.8041 - val_loss: 0.8710 - val_acc: 0.7523

Epoch 00064: val_acc did not improve from 0.75532
Epoch 65/70
564/564 [==============================] - 83s 147ms/step - loss: 0.5152 - acc: 0.8111 - val_loss: 0.8541 - val_acc: 0.7543

Epoch 00065: val_acc did not improve from 0.75532
Epoch 66/70
564/564 [==============================] - 83s 147ms/step - loss: 0.5145 - acc: 0.8121 - val_loss: 0.8691 - val_acc: 0.7550

Epoch 00066: val_acc did not improve from 0.75532
Epoch 67/70
564/564 [==============================] - 83s 148ms/step - loss: 0.5130 - acc: 0.8136 - val_loss: 0.8622 - val_acc: 0.7560

Epoch 00067: val_acc improved from 0.75532 to 0.75598, saving model to .\model.h5
Epoch 68/70
564/564 [==============================] - 83s 146ms/step - loss: 0.5006 - acc: 0.8165 - val_loss: 0.8530 - val_acc: 0.7517

Epoch 00068: val_acc did not improve from 0.75598
Epoch 69/70
564/564 [==============================] - 83s 147ms/step - loss: 0.4934 - acc: 0.8214 - val_loss: 0.8744 - val_acc: 0.7563

Epoch 00069: val_acc improved from 0.75598 to 0.75632, saving model to .\model.h5
Epoch 70/70
564/564 [==============================] - 83s 148ms/step - loss: 0.4864 - acc: 0.8205 - val_loss: 0.9028 - val_acc: 0.7566

Epoch 00070: val_acc improved from 0.75632 to 0.75665, saving model to .\model.h5
```
After the training, the following classification report was generated.

**Classification Report**
|              | precision | recall | f1-score | support |
|--------------|:---------:|:------:|:--------:|:-------:|
| angry        |    0.69   |  0.75  |   0.72   |   501   |
| fear         |    0.78   |  0.64  |   0.70   |   505   |
| happy        |    0.81   |  0.84  |   0.83   |   504   |
| neutral      |    0.71   |  0.79  |   0.74   |   504   |
| sad          |    0.74   |  0.59  |   0.66   |   509   |
| surprise     |    0.81   |  0.94  |   0.87   |   502   |
|              |           |        |          |         |
| accuracy     |           |        |   0.76   |   3025  |
| macro avg    |    0.76   |  0.76  |   0.75   |   3025  |
| weighted avg |    0.76   |  0.76  |   0.75   |   3025  |

Based on the classification record, it can be deduced that "surprise" performed well in every aspect for the model. For the "fear" and "sad" emotion, they usually correctly predicted yet they have relatively high false positive percentage.

6. Plotting Accuracy and Loss
   ![Training_Loss_Accuracy](https://github.com/Josie528/BITI1113-A.I.-Project/blob/main/misc/training_loss_accuracy.png)
   The gap between the two plotted line is small so, it has little overfitting. 
7. Define function and Test the Trained Model

    We use k-fold cross-validation to estimate the skill of a method of unseen data like using a train-test split. It systematically creates and evaluates different subsets of the dataset. Repeated k-fold cross-validation provides a way to improve the estimated performance of a machine learning model. Both train-test splits and k-fold cross validation are resampling methods. Since we are dealing to model the unknown, we need to use resampling method. In the case of applied machine learning, we are interested in estimating the skill of a machine learning procedure on unseen data. More specifically, the skill of the predictions made by a machine learning procedure.
    
    We will also use some static images to test the trained model. The example shown here are consisting different image orientation, different size and different number of people. It showed that the trained model has somewhat successfully passed the testing stage.
    
    ```python
    frame = cv2.imread("ethankid.jpg")
    final = emotionScan(frame)

    plt.imshow(cv2.cvtColor(final,cv2.COLOR_BGR2RGB))

    # if image to small
    recolor = cv2.cvtColor(final,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(recolor, 'RGB')
    img = img.resize((300,500))
    display(img)
    ```
    ```python
    frame = cv2.imread("pic2.jpg")
    final = emotionScan(frame)

    plt.imshow(cv2.cvtColor(final,cv2.COLOR_BGR2RGB))
    ```
    ```python
    frame = cv2.imread("mahatir.jfif")
    final = emotionScan(frame)

    plt.imshow(cv2.cvtColor(final,cv2.COLOR_BGR2RGB))
    ```
    ```python
    frame = cv2.imread("malaysian.jfif")
    final = emotionScan(frame)

    plt.imshow(cv2.cvtColor(final,cv2.COLOR_BGR2RGB))
    ```
    ```python
    frame = cv2.imread("sad.jfif")
    final = emotionScan(frame)

    plt.imshow(cv2.cvtColor(final,cv2.COLOR_BGR2RGB))
    ```
    
    ![Test Result](https://github.com/TanWeiYin/Group_J_AI_Project/blob/main/misc/TestResult.jpg)
    
8. Plotting Confusion Matrix

    We colored each square of confusion matrix with different shades based on accuracy, where a darker shade indicates a more accurate result.
Happiness, Surprise, Neutral are the easiest to detect and show the most accurate results. On the contrary, Sad, Anger, and Fear are the most struggling for AI.

   ![ConfusionMatrix](https://github.com/Josie528/BITI-1113-AI-Project/blob/main/misc/confusionmatrix.jpg)
   
## F. RESULT AND CONCLUSION
The AI recognizes emotion model can get around 75% accuracy. There is still room for improving accuracy and efficiency. We hope that we are able to train the model from scratch by collecting more comprehensive dataset in future. 

Sometimes we hide our emotions. What others see on our outside is not always how we are feeling on the inside. This AI still not able to recognise our emotions if we were hiding them. This AI just able to detect basic emotion so, we need to gather more data set to spot the human's microexpression.

Thanks to our AI lecturer, Prof. Goh Ong Sing for giving us the opportunity to learn how to implement a real-world AI project using Python.

## G. Project Presentation
Demo Video: Wei Yin
Making Slides: Mirza, Afiqah, Jia Mei
[![livedemo](https://github.com/Josie528/BITI-1113-AI-Project/blob/main/misc/livedemo.jpg)](https://www.youtube.com/watch?v=bdczQRzMWr0 "livedemo")
The installation and setup process are as follows:
1. Download all src files from GitHub.
2. Click run to execute the main.ipynb script.
3. Place your face in front of the webcam.
4. The facial emotion recognizer is ready to use.

## H. ACKNOWLEDGEMENT
* [Emotion Detection CNN](https://github.com/akmadan/Emotion_Detection_CNN)
    Main reference of application of source code
* [README Template](https://github.com/osgoh88/Artificial-Intelligence-Project/)
    A project template that designed by our lecturer
* [The 4 Convolutional Neural Network Models That Can Classify Your Fashion Images](https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-can-classify-your-fashion-images-9fe7f3e5399d#:~:text=Convolutional%20Neural%20Networks%20(CNNs)%20is,an%20image%20is%20good%20enough)
    Understanding how CNN model works
* [Face Expression Detection](https://colab.research.google.com/drive/1DOvXJZRkjfKfF9oUpCZXSU3hPG3g1h-l?usp=sharing)
    Learn how to configure and detect emotions of the static image.
