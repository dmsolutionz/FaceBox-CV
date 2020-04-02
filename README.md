# Facebox-CV

#### :movie_camera: :radio_button: Computer Vision Control demo using Pytorch and OpenCV.

Face detection using `FaceBoxes (2017)`, a state-of-the-art method (for 2018) in CPU speed for multi-target facebox detection. (https://arxiv.org/abs/1708.05234)
<br/>  
_Methodology_

## 1. :page_with_curl: FaceBoxes By Shifeng Zhang (et al.)

Implementing a great paper by _Shifeng Zhang et al._ with a face detection model capable of running at 20 FPS on a single CPU core. Perfect for user-interaction control.

- Codebase available at https://github.com/sfzhang15/FaceBoxes

<p align="center">
  <img src="https://github.com/lukexyz/FaceBox-CV/blob/master/images/faceboxes-arxiv.PNG?raw=true" width="650">
</p>

<p align="center">
  <img src="https://github.com/lukexyz/FaceBox-CV/blob/master/images/faceboxes-paper.PNG?raw=true" width="450">
</p>

## 2. :female_detective: **Import Model and Predict for Single Face**

→ :notebook_with_decorative_cover: Notebook [01-Faceboxes-Eval-Image.ipynb](notebooks/01-Faceboxes-Eval-Image.ipynb)

<p align="center">
  <img src="https://github.com/lukexyz/FaceBox-CV/blob/master/images/plot_facebox.PNG?raw=true" width="550">
</p>

## 3. :male_detective: **Iterate over Predictions for Multiple Targets**

→ :notebook_with_decorative_cover: Notebook [02-Faceboxes-Refactor.ipynb](notebooks/02-Faceboxes-Refactor.ipynb)

<p align="center">
  <img src="https://github.com/lukexyz/FaceBox-CV/blob/master/images/many_faceboxes.PNG?raw=true" width="500">
</p>

## 5. :movie_camera: :red_circle: **Realtime Facebox from Webcam Capture** 

Output is masked but movement is captured by the bounding box.

→ :bookmark_tabs: Refactored into [face_utils.py](face_utils.py)

<p align="center">
  <img src="https://github.com/lukexyz/FaceBox-CV/blob/master/images/face-tracker.gif?raw=true" width="300">
</p>

## 6. :clapper: **`OpenCV` Control Loop**

Putting it all together in a control loop, and linking facebox movement with a media asset. The black box on the right shows the webcam output with the detected face outlined.

→ :bookmark_tabs: Run [vision.py](vision.py)  
<br/>

<p align="center">
  <img src="https://github.com/lukexyz/FaceBox-CV/blob/master/images/facebox17.gif?raw=true">
</p>
<br/>

##### Quick-Start

```sh
$ conda create -n facebox-cv pip jupyter python=3.6
$ conda activate facebox-cv
```

```sh
$ git clone https://github.com/lukexyz/FaceBox-CV.git
$ cd FaceBox-CV
$ pip install -r requirements.txt
```
Find system specific `pytorch` installation from [pytorch.org](https://pytorch.org/) (dev: `windows|pip|python|cuda=10.1`)
```
$ pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```
Run FaceBox-CV
```
$ python vision.py
```

##### Notes

**For OSX** Only compatible with NVIDIA Gpu's as pytorch will complain you don't have it compiled with CUDA. There are workarounds for macbooks but be aware they might just not work.

##### Acknowledgements

- [sfzhang15/FaceBoxes](https://github.com/sfzhang15/FaceBoxes)
