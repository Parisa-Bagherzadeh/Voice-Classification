# Voice Classification  
This repository contains code and resources for voice classification, a machine learning task that involves categorizing audio samples into different classes based on their acoustic features.  
  
## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)


## Introduction
Voice classification is a common task in the field of speech processing and machine learning. It involves training a model to recognize and categorize different types of voice samples, such as male vs. female, adult vs. child, or different languages.

This repository provides code and resources for building and training voice classification models using popular machine learning framework, TensorFlow.  

## Installation
To use the code in this repository, you will need to have Python installed on your system. You can install the required dependencies by running the following command:  
```
pip install -r requirements.txt
```  
This will install all the necessary packages for running the code in this repository  

## Usage
The main components of this repository include:
- Data preprocessing scripts
- Model training scripts
- Evaluation scripts

To use these components, simply run the corresponding Python scripts with your input data. You can also modify the code to fit your specific use case or dataset.

You can download the dataset I've gathered and preprocessed from [this](https://drive.google.com/drive/folders/1nbrJos4slMo8-EQMZHru00bu2JPL-gs5?usp=sharing) link or you can use your own data to train the model. 
### Preprocess  
Put your data in raw_data folder and  run the following command 

```
python preprocess.py --raw_data_path raw_data --min_silence_len YOUR_ARGUMENT --silence_thresh YOUR_ARGUMENT --chunk_duration YOUR_ARGUMENT   
```  
The preprocessed data will be saved in dataset folder.  
### Train  
To train the model on your own data, run Voice_Classification.ipynb notebook to get the weights.h5 file that will be saved in model folder  

### Evaluation  
To test the model run the following command:  
```
python inference.py --model .h5 file --wav_file Test_wav_file
```  
## Contributing
If you would like to contribute to this repository, feel free to submit a pull request with your changes. I welcome contributions from the community and are open to new ideas and improvements.




