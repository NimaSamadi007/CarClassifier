# Car Classifier in C++

This repository contains a simple car classifier in C++. In this README file the flow of the project is described. The following image summarizes the steps of the project:

![project flow diagram](assets/flow.jpg)

## Contents
+ [Model Training](#model-training)
+ [Conversion to ONNX](#conversion-to-onnx)
+ [Inference](#inference)
+ [Some features](#some-features)

## Model Training
The model is trained using the TensorFlow framework. Model base is the `MobileNetV2` model and a one layer of dense layer is added to top of the model. The base of the model is freezed and parameters are not updated during training. `MobileNetV2` outputs a `5x5x1280` tensor before dense layers. By using a global average pooling layer, these features are reduced to a `1280` length vector. This features are fed to the dense layer which outputs a single value. The positive value indicates the image is a car and the negative value indicates the image is not a car. 

The training and validation datasets are a combination of `COCO` and cars dataset from [Stanford](http://ai.stanford.edu/~jkrause/cars/car_dataset.html). I used a subset of these datasets and created a training dataset of size 3000 and validation dataset of size 1000. The classes are divided in half in both datasets.

The following figure shows the training and validation loss and accuracy curves:

![training and validation curves](assets/curves.png)

To find more information about the model training, please refer to the [notebook](train/VehicleClassifier.ipynb).

## Conversion to ONNX
To run inference in C++, I used the ONNX format. ONNX stands for Open Neural Network Exchange and it is an open format to represent deep learning models. ONNX models are widely supported in various frameworks and devices. To find out more about this standard, refer to [ONNX](https://onnx.ai/) webpage. The model is converted to ONNX using the `tf2onnx` package. The following code snippet shows how to convert the model to ONNX:

```bash
$ python -m tf2onnx.convert --saved-model . --output vehicle_detector_model.onnx
```

## Inference
Inference is done in C++ using the `ONNXRuntime` library. It's a open source library to run and optimize ONNX models on differenct devices, languages and platforms. I used it to run the model on my laptop's CPU and GPU. I've also used Nlohmann's JSON library to parse the JSON files. `config.json` file contains some configuration parameters that are necessary to run the inference. For loading and preprocessing the images, I used `OpenCV` library which is a huge library for computer vision.

The source codes are in `src` folder. `main.cpp` is the main driver of the project. This project can be compiled with CMake. Note that you should have `OpenCV` and `ONNXRuntime` libraries installed on your system.

This project can be run in to mode: 1) one shot mode and 2) continuous mode. In one shot mode, the program takes an image as input and outputs the prediction.
In continuous mode, the program opens the webcam and continuously predicts the images from the webcam. To run the program in each mode, do the following:

**One shot mode:**
```bash
$ ./VECls 0 <path to image>
```

**Continuous mode:**
```bash
$ ./VECls 1
```

## Some features:
+ The program can run in two modes: one shot mode and continuous mode.
+ The program can run on CPU and GPU.
+ You can use more than one CPU cores to run the inference.
+ Exceptions are handled.
+ High accuracy and low inference time (about 40ms on my laptop's CPU and 95% accuracy on validation set).
