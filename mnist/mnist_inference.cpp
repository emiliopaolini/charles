// g++ -std=c++20 -I ../xtl/include -pthread -I ../tiny-dnn -I ../boost -I ../cnl/include -I/usr/include/python2.7  -DWITHOUT_NUMPY -DCNN_USE_FIXED  -finline-functions -O3 mnist_inference.cpp -o mnist_inference.out


#include <iostream>
#include "tiny_dnn/tiny_dnn.h"
#include <filesystem>
namespace fs = std::filesystem;
using namespace std;


std::vector<tiny_dnn::label_t> test_labels;
std::vector<tiny_dnn::vec_t> test_images;
int main(){
    
    tiny_dnn::network<tiny_dnn::sequential> net;
    net.load("./mnist_net_fixed_binary", tiny_dnn::content_type::weights_and_model, tiny_dnn::file_format::binary);


    tiny_dnn::parse_mnist_labels("../tiny-dnn/data/t10k-labels.idx1-ubyte",
                            &test_labels);
    tiny_dnn::parse_mnist_images("../tiny-dnn/data/t10k-images.idx3-ubyte",
                            &test_images, -1.0, 1.0, 2, 2);

    net.test(test_images, test_labels).print_detail(std::cout);
}