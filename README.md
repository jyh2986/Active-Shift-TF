



# Active Shift Layer

This repository contains the implementation for Active Shift Layer (ASL).

Please see the paper [Constructing Fast Network through Deconstruction of Convolution](https://papers.nips.cc/paper/7835-constructing-fast-network-through-deconstruction-of-convolution.pdf). 

This paper is accepted in NIPS 2018 as spotlight session ([slide](https://github.com/jyh2986/Active-Shift/blob/publish_ASL/supplement/presentation_rev2.pdf), [poster](https://github.com/jyh2986/Active-Shift/blob/publish_ASL/supplement/poster_rev1.pdf))

The code is based on [Tensorflow](https://www.tensorflow.org/)  
Caffe implementation is also available at [ASL-Caffe](https://github.com/jyh2986/Active-Shift)

## Introduction

### Deconstruction 
Naive spatial convolution can be deconstructed into a shift layer and a 1x1 convolution.

This figure shows the basic concept of deconstruction.
![Basic Concept](https://github.com/jyh2986/Active-Shift/blob/publish_ASL/supplement/deconstruction.PNG)



### Active Shift Layer (ASL)
For the efficient shift, we proposed active shift layer.
  * Uses depthwise shift
  * Introduced new shift parameters for each channel    
  * New shift parameters(alpha, beta) are learnable

## Prerequisite

Note that this code is tested only in the environment decribed below. Mismatched versions does not guarantee correct execution.


* Tensorflow 1.4.1
* Cuda 8.0
* g++ 4.9.3


## Build
1. Edit CUDA_HOME in <i>build.sh</i> and build-spec for your GPU
2. run >./build.sh
3. If there is an error with <i>cuda_config.h</i>, run "cp ./lib/cuda_config.h $TF_INC/tensorflow/stream_executor/cuda/"



## Testing
1. run >python test_forward_ASL.py
    * It just shows the results of forwarding random tensor. You should not have any error message.

2. run >python test_backward_ASL.py
    * You should get "OK" for 3 tests.

## Usage
1. import lib.active_shift2d_op as active_shift2d_op
2. use active_shift2d_op.active_shift2d_op(input_tensor, shift_tensor, strides, paddings)
    * Please see <i>test_forward_ASL.py</i>
