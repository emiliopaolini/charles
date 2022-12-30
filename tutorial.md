This is a simple tutorial on the usage of CHARLES library for the MNIST experiment

CHARLES is a composition of multiple libraries:
* Compositional Numeric Library (CNL): header-only library that implements fixed-precision numeric classes.
* Boost: a set of libraries for C++ that provides support for tasks and structures such as linear algebra, pseudorandom number generation, multithread- ing, image processing, regular expressions, and unit testing.
* tiny-dnn: a C++ implementation of deep learning.
* Matplotlib-cpp: a C++ wrapper for Python’s matplotlib (MPL) plotting library.

### Requirements: 
* C++ compiler, compliant with c++20 standard

### Experiment on MNIST
1) Clone the repository
```
git clone https://github.com/emiliopaolini/charles.git
```
2) Navigate in the cloned repository
```
cd charles
```
3) From here, you can start both the floating-training and the fixed-training.<br />
Fixed-training:
```
g++ -std=c++20 -I ../xtl/include -pthread -I ../tiny-dnn -I ../boost -I ../cnl/include -I/usr/include/python2.7  -DWITHOUT_NUMPY -DCNN_USE_FIXED  -finline-functions -O3 mnist.cpp -o mnist.out
```
Floating-training:
```
g++ -std=c++20 -pthread -I ../xtl/include -I ../tiny-dnn-orig -I ../boost -I ../cnl/include -I/usr/include/python2.7  -DCNN_USE_AVX -DWITHOUT_NUMPY -mavx mnist.cpp  -o mnist.out
```
4) Once the training is over, both experiments will save the network parameters into a file. 
5) To run inference, change the number of bits in the tiny-dnn/tiny_dnn/config.h file and compile the mnist_inference.cpp file using:
```
g++ -std=c++20 -I ../xtl/include -pthread -I ../tiny-dnn -I ../boost -I ../cnl/include -I/usr/include/python2.7  -DWITHOUT_NUMPY -DCNN_USE_FIXED  -finline-functions -O3 mnist_inference.cpp -o mnist_inference.out
```
6) Once run the mnist_inference.out, you will see the accuracy of the architecture and its confusion matrix:
![accuracy_conf_matrix]([http://url/to/img.png](https://github.com/emiliopaolini/charles/blob/main/mnist/sample_result.png))
