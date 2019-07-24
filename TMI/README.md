# Python Traffic Counter

It uses:

* [YOLO](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv) to detect objects on each of the video frames.

* [SORT](https://github.com/abewley/sort) to track those objects over different frames.

Once the objects are detected and tracked over different frames a simple mathematical calculation is applied to count the intersections between the vehicles previous and current frame positions with a defined line.

The code on this prototype uses the code structure developed by Adrian Rosebrock for his article [YOLO object detection with OpenCV](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv)

### This project only adjusted the sorts to track the traffic in a fixed box in order to track Traffic Tailgating


## Quick Start

1. Download the code to your computer.
2. [Download yolov3.weights](https://www.dropbox.com/s/99mm7olr1ohtjbq/yolov3.weights?dl=0) and place it in `/yolo-coco`.
3. Make sure you have Python 3.7.0 and [OpenCV 3.4.2](https://www.pyimagesearch.com/opencv-tutorials-resources-guides/) installed.
4. Run:
```
$ python main.py --input input/highway.mp4 --output output/highway.avi --yolo yolo-coco
```
