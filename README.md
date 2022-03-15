# An Eye For Blind Using Deep Learning with Attention Mechanism

To test the deployed model click [here](https://eye-for-blind.herokuapp.com/).<br/>
Note : It takes some time to load the heroku page. Patience is the key!!

## Overview
- This is Flickr8k dataset which consists of around 8k images with 5 captions per image.Please check [here](https://www.kaggle.com/adityajn105/flickr8k) for more details on dataset. 

- The train dataset has about 8k images. There is a seperate captions.txt file for captions. 

## Motivation
- The World Health Organization (WHO) has reported that approximately 285 million people are visually impaired worldwide, and out of these 285 million, 39 million are completely blind. It gets extremely tough for them to carry out daily activities, one of which is reading. From reading a newspaper or a magazine to reading an important text message from your bank, it is tough for them to read the text written in it.

- A similar problem they also face is seeing and enjoying the beauty of pictures and images. Today, in the world of social media, millions of images are uploaded daily. Some of them are about your friends and family, while some of them are about nature and its beauty. Understanding what is present in that image is quite a challenge for certain people who are suffering from visual impairment or who are blind.

-  We will learn how to make a model, specifically such that a blind person knows the contents of an image in front of them with the help of a CNN-RNN based model

## Project Structure
- [main.py](https://github.com/Pratik872/An-Eye-For-Blind/blob/main/main.py) : This file has the flask application which is created.

- [utils.py](https://github.com/Pratik872/An-Eye-For-Blind/blob/main/utils.py) : This file has all the helper functions which are required to run the application.

- [constants.py](https://github.com/Pratik872/An-Eye-For-Blind/blob/main/constants.py) : This file has all the constant variables required in developing the application.

- [templates](https://github.com/Pratik872/An-Eye-For-Blind/blob/main/templates) : This folder has all the templates which are rendered in the application

- [readme_resources](https://github.com/Pratik872/An-Eye-For-Blind/tree/main/readme%20resources) : This folder has all the images used to create readme file.

- [requirements.txt](https://github.com/Pratik872/An-Eye-For-Blind/blob/main/requirements.txt) : This file has all the packages used to code and build the application.

- [Eye for Blind.ipynb](https://github.com/Pratik872/An-Eye-For-Blind/blob/main/Eye_for_blind_Pratik_Waghmare.ipynb) : This jupyter notebook has the code for making models.

## Problem Objective
- To create a deep learning model which can explain the contents of an image in the form of speech through caption generation with an attention mechanism on Flickr8K dataset. This kind of model is a use-case for blind people so that they can understand any image with the help of speech. The caption generated through a CNN-RNN model will be converted to speech using a text to speech library. 

- The deployed model at Heroku will just show the captions. If you want to hear the audio for the generated caption then clone this project in local system and use the commented code in the 'main.py' and 'utils.py'. You will find the instructions to run this project in your system below in [How to Use]() section.

## Methodology

### Data Understanding
- I have used glob package to see the images. Some of them are as follows : 

![SampleImgsWithCaps](https://github.com/Pratik872/An-Eye-For-Blind/blob/main/readme%20resources/ImgsCaptions.png)


### Preprocessing the Captions
- Necessary preprocessing steps were done to process the captions.

- Used Keras tokeniser to transform top 5000 words which were then used in the RNN

- Padding was done to max_length of sentences to feed the RNN. Masking was also used so that model could understand the padded input to neglect them.

- Please find the necessary graphs/images below

![Top30Words](https://github.com/Pratik872/An-Eye-For-Blind/blob/main/readme%20resources/top30words.png)

![WordCloud](https://github.com/Pratik872/An-Eye-For-Blind/blob/main/readme%20resources/wordcloudpng.png)


### Preprocessing the Images
- I have used Transfer Learning using InceptionV3 Object detection model to find important features from the image and then feed in to Encoder i.e a CNN model. So to preprocess images for InceptionV3 I have used tensorflow and keras. Please find some preprocessed images below.

![preprocessedimgs](https://github.com/Pratik872/An-Eye-For-Blind/blob/main/readme%20resources/samplepreprocessedimages.png)

### Dataset Creation
- I have used 'from_tensor_slices' method of tensorflow to create a dataset.

- Necessary batch size was chosen

- Also train-test split was used to split training and testing data to avoid data leakage.

### Model Making

- Necessary constants were chosen which are mentioned in [constants.py](https://github.com/Pratik872/An-Eye-For-Blind/blob/main/constants.py). These were chosen according to system computational power.

- As attention mechanism is not present in the keras, I have used sub-classing using keras to create encoder,attention and decoder models. You can find the detailed code in [Eye for Blind.ipynb](https://github.com/Pratik872/An-Eye-For-Blind/blob/main/Eye_for_blind_Pratik_Waghmare.ipynb).

### Model Evaluation

- I have tried using both Greedy Search and Beam search for predicting captions.

- To evaluate the predicted caption I have used BLEU score. BLEU score is basically an evaluation metric used for evaluating the actual and predicted sentences. Greater the score more good is the prediction.


### DATA SOURCE
- [Flickr8k](https://www.kaggle.com/adityajn105/flickr8k)

### Notebook
- [Eye for Blind.ipynb](https://github.com/Pratik872/An-Eye-For-Blind/blob/main/Eye_for_blind_Pratik_Waghmare.ipynb)

### Built with 🛠️
- Packages/Repo : Pandas,Numpy,Seaborn,Matplotlib,Sklearn,Flask,Pickle,Git,Tensorflow,Keras,Glob,Pillow,NLTK

- Dataset : Kaggle

- Coded on : Jupter Notebook and Google colab (modelling), VSCode(building application)

### Deployment
- Deployed using Heroku(PAAS)

- For deployment repository click [here](https://github.com/Pratik872/An-Eye-For-Blind/tree/deploy)

- For Web Application click [here](https://eye-for-blind.herokuapp.com/)

### How to Use

#### In Local system with captions Audio (Anaconda/Miniconda needed):
- Clone this repository in your system.
- Create a new environment in Anaconda.
- Activate new environment and run [requirements.txt](https://github.com/Pratik872/An-Eye-For-Blind/blob/main/requirements.txt) using 'pip install -r requirements.txt' (Note: Anaconda Terminal should be opened in project cloned directory where all project files are present)
- Uncomment the code for audio in 'main.py' and 'utils.py' files.
- Run [main.py](https://github.com/Pratik872/An-Eye-For-Blind/blob/main/main.py) using 'python main.py' in terminal and VOILA!!!


#### Using Docker:
To be added soon!!
