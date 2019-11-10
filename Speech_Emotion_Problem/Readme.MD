### Objective

The aim here is to develop an automatic speech emotion classification model. 

First the features to be extracted. Usually hand crafted features are chosen for such taks but they require domain knowledge. 
Two main sets of features for speech problems are: Acoustic Features and Contextual Features. 
Acoustic Features consists of parameters like pitch, tone, volume etc, while contextual features represent the arousal and valence of speech.

Here, acoustic features are being used (MFCC). 

![SEP](#)


### Installation and Usage

The dataset taken here is "MELD - A Multimodal Multi-Party Dataset"
It can be found [here](https://github.com/SenticNet/MELD)

It contains audio files and has 5 emotion labels, namely, DISGUST, FEAR, HAPPY, NEUTRAL and SAD.

### File/Folder Descriptions:

- setup.sh  : Set up the conda environment to run the notebook/scripts.
	NOTE : Run the above script as `source setup_emotion_detection.sh` in the bash terminal.

### SETUP

To be able to run the code and reproduce the results, please follow these steps:

**Option A**
1. Clone this repository. Run this command in your prompt/shell:
```git clone https://github.com/devanshbatra04/midas-submission.git```

2. Navigate to the folder
``` cd midas-submission\Speech_Emotion_Problem```

3. To setup project dependencies run:
`$ ./setup.sh` 


### METHODOLOGY

**Feature Extraction:**
Mainly two features have been worked upon here:
1. Mel-Spectogram Frequency, which are representations of the short-term power spectrum of a sound. 
2. MFCC (Mel-frequency cepstral coefficients); these coefficients collectively make up MFC.
They have been extracted using Librosa-a library to analyse audio files.

**Model:**
The model has been proposed here:

Mfcc features are extracted for every 25 ms second window taken at a step of 10ms. The sample rate for audio file has been reduced to 16000 (max of 8000 Hz frequency will be captured) to avoid irrelevant features. For each window 13 mfcc features will be extracted. As the length of each audio file is different the number of windows and hence the number of features will also be different.

The mfcc features are extracted using the python_speech_features library.

```Multi Layer LSTM```- The normalised inputs are passed through 8 LSTM layers. The first layer is 8x64, while the rest are 64x64. After processing through the LSTM layers, the output is fed to fully connected Linear Layers


**Findings:**
- Normalising the data to values between 0 to 1 had better results then standardising the data using mean and variance
- The normalised inputs are passed through 8 LSTM layers. The first layer is 8x64, while the rest are 64x64. After processing through the LSTM layers, the output is fed to fully connected Linear Layers
- To make the input sequence lengths same for batch processing, the inputs are padded with zeros at the end to make the shape of input of each file same


### PERFORMANCE

| Model   | Accuracy   | 
|---|---|
| 8x LSTM - TEST |  62.89% | 
| 8x LSTM - VAL |  63.00% | 
