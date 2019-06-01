# GazeML-keras
A keras port of [swook/GazeML](https://github.com/swook/GazeML) for eye region landmarks detection. 

The dlib face detector is replaced by MTCNN.

## Demo

[Here](https://github.com/shaoanlu/GazeML-keras/blob/master/demo_colab.ipynb) is the demo jupyter notebook, or [try it](https://colab.research.google.com/github/shaoanlu/GazeML-keras/blob/master/demo_colab.ipynb) on Colaboratory.

## Results

![](https://github.com/shaoanlu/GazeML-keras/raw/master/results/result_lenna.png)
![](https://github.com/shaoanlu/GazeML-keras/raw/master/results/result_fashion-1063100_640.png)
![](https://github.com/shaoanlu/GazeML-keras/raw/master/results/result_model-1439909_640.png)
![](https://github.com/shaoanlu/GazeML-keras/raw/master/results/result_reiwa.png)

## WIP
1. The preprocessing is very different from the [official implementation](https://github.com/swook/GazeML/blob/master/src/datasources/frames.py#L223), thus the results produced are suboptimal.

2. Model training as well as gaze estimation has not been ported yet.

## Dependency
- python 3.6
- keras 2.2.4
- tensorflow 1.12.0

## Acknoledgement
ELG model weights are converted from the official repo [swook/GazeML](https://github.com/swook/GazeML). We learnt a lot from there.
