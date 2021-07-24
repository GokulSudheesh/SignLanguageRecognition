# SignLanguageRecognition
## An overview of the model
<img src="GitHubImages/network-diagram.jpg" width="816" height="459">

## Setting up
Run ```calibration.py``` to calibrate your hand gestures for every alphabet. This will aid in improving the accuracies while predicting from the trained model.

<img src="GitHubImages/calibration.jpg" width="300">
It will save a numpy array where each element specifies whether the fingers are open/closed/half open-closed.

## Detecting from web cam:
<img src="GitHubImages/A.jpg" width="270" height="210">  <img src="GitHubImages/B.jpg" width="270" height="210">  <img src="GitHubImages/D.jpg" width="270" height="210">
<img src="GitHubImages/F.jpg" width="270" height="210">  <img src="GitHubImages/G.jpg" width="270" height="210">  <img src="GitHubImages/H.jpg" width="270" height="210">
<img src="GitHubImages/I.jpg" width="270" height="210">  <img src="GitHubImages/L.jpg" width="270" height="210">  <img src="GitHubImages/O.jpg" width="270" height="210">
<img src="GitHubImages/Q.jpg" width="270" height="210">  <img src="GitHubImages/R.jpg" width="270" height="210">  <img src="GitHubImages/U.jpg" width="270" height="210">
<img src="GitHubImages/W.jpg" width="270" height="210">  <img src="GitHubImages/X.jpg" width="270" height="210">  <img src="GitHubImages/Y.jpg" width="270" height="210">

## Reference:
<img src="GitHubImages/amer_sign2.png" width="500" height="440"> 

### This project uses a CNN model that follows AlexNet architecture.
<img src="GitHubImages/AlexNet-Arch.png">
Image credits to Krizhevsky et al., the original authors of the AlexNet paper.

## Notes
Please note that ```detector_mediapipe.py``` gives more accurate results than ```detect_webcam.py```.

The `requirements.txt` file contains all the dependencies needed for the project.
```
pip install -r requirements.txt
```
[Mediapipe docs](https://google.github.io/mediapipe/solutions/hands.html)
[Trained models (Google Drive link)](https://drive.google.com/drive/folders/1LCBmiV4bkNyKg8ix4MCEMQQ2SJOmPh8X?usp=sharing)