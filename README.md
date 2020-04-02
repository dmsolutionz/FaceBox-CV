# Facebox-CV

### :movie_camera: :radio_button: Computer Vision Control demo using Pytorch and OpenCV.  

Face detection using `FaceBoxes (2017)`, a state-of-the-art method (for 2018) in CPU speed for multi-target facebox detection. (https://arxiv.org/abs/1708.05234)  

<br/>

## 1. :page_with_curl: FaceBoxes By Shifeng Zhang (et al.)


<p align="center">
  <img src="https://github.com/lukexyz/FaceBox-CV/blob/master/images/faceboxes-arxiv.PNG?raw=true" width="650">
</p>

<p align="center">
  <img src="https://github.com/lukexyz/FaceBox-CV/blob/master/images/faceboxes-paper.PNG?raw=true" width="550">
</p>

Codebase available at https://github.com/sfzhang15/FaceBoxes

## 2. :female_detective: **Import Model and Predict for Single Face**  

  → :notebook_with_decorative_cover: Notebook [01-Faceboxes-Eval-Image.ipynb](notebooks/01-Faceboxes-Eval-Image.ipynb)  

<p align="center">
  <img src="https://github.com/lukexyz/FaceBox-CV/blob/master/images/plot_facebox.PNG?raw=true" width="400">
</p>

## 3. :male_detective: **Iterate over Predictions for Multiple Targets**  

  → :notebook_with_decorative_cover: Notebook [02-Faceboxes-Refactor.ipynb](notebooks/02-Faceboxes-Refactor.ipynb)  

<p align="center">
  <img src="https://github.com/lukexyz/FaceBox-CV/blob/master/images/many_faceboxes.PNG?raw=true" width="350">
</p>

## 3. :male_detective: **OpenCV Control Loop**  

Putting it all together in a control loop and linking facebox movement with a media asset.

  → :movie_camera: :radio_button: Code [vision.py](vision.py)  

<p align="center">
  <img src="https://github.com/lukexyz/FaceBox-CV/blob/master/images/facebox17.gif?raw=true">
</p>


<p align="center">
  <img src="https://github.com/lukexyz/FaceBox-CV/blob/master/facebox18b.gif?raw=true">
</p>

##### Acknowledgements
* [sfzhang15/FaceBoxes](https://github.com/sfzhang15/FaceBoxes)
