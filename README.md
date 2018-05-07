Video Quality Assessment using Deep Learning
============================================

A Keras and Tensorflow implementation of video quality assessment using deep neural networks is proposed. We propose CNN + LSTM architecture to recognize and synthesize both spatial and temporal artifacts of video impairements respectively. The architecture is shown below.

## Network Architecture
<p align="center">
  <img src="assets/deep-arch.png" width="792" height="424" />
</p>
## Training

- We used video samples of 30 seconds each to train the model.
- We first extract individual frames of the video from each video sample and create a numpy array out frames.
- We create numpy array of MOS (i.e, video quality mean opinion score collected from users) that is corresponding to each sample.
- We feed these frames of a video sample to CNNs followed by a series of LSTMs. 
- Finally, a softmax is used to classify the video sample quality.
