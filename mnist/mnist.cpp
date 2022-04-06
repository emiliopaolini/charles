// g++ -std=c++20 -pthread -I ../xtl/include -I ../tiny-dnn-orig -I ../boost -I ../cnl/include -I/usr/include/python2.7  -DCNN_USE_AVX -DWITHOUT_NUMPY -mavx mnist.cpp  -o mnist.out
// g++ -std=c++20 -I ../xtl/include -pthread -I ../tiny-dnn -I ../boost -I ../cnl/include -I/usr/include/python2.7  -DWITHOUT_NUMPY -DCNN_USE_FIXED  -finline-functions -O3 mnist.cpp -o mnist.out

#include <iostream>
#include "tiny_dnn/tiny_dnn.h"
#include <filesystem>
namespace fs = std::filesystem;
using namespace std;

std::vector<tiny_dnn::label_t> train_labels, test_labels;
std::vector<tiny_dnn::vec_t> train_images, test_images;

tiny_dnn::network<tiny_dnn::sequential> net;


//#define _USE_MATH_DEFINES
#include <cmath>
//#include "../matplotlib-cpp/matplotlibcpp.h"

//namespace plt = matplotlibcpp;



void construct_net(){
        tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();
        // connection table, see Table 1 in [LeCun1998]
        // connection table [Y.Lecun, 1998 Table.1]
        #define O true
        #define X false
        // clang-format off
        static const bool tbl[] = {
        O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
        O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
        O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
        X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
        X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
        X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
        };
        // clang-format on
        #undef O
        #undef X

        // construct nets
        //
        // C : convolution
        // S : sub-sampling
        // F : fully connected
        // clang-format off
        using fc = tiny_dnn::layers::fc;
        using conv = tiny_dnn::layers::conv;
        using ave_pool = tiny_dnn::layers::ave_pool;
        using tanh = tiny_dnn::activation::tanh;

        using tiny_dnn::core::connection_table;
        using padding = tiny_dnn::padding;

        net << conv(32, 32, 3, 1, 6,   // C1, 1@32x32-in, 6@28x28-out
                padding::valid, true, 1, 1, 1, 1, backend_type)
                << tiny_dnn::relu_layer()
                << ave_pool(30, 30, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
                << tiny_dnn::relu_layer()
                << conv(15, 15, 4, 6, 16,   // C3, 6@14x14-in, 16@10x10-out
                        connection_table(tbl, 6, 16),
                        padding::valid, true, 1, 1, 1, 1, backend_type)
                << tiny_dnn::relu_layer()
                << ave_pool(12, 12, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
                << tiny_dnn::relu_layer()
                << conv(6, 6, 3, 16, 120,   // C5, 16@5x5-in, 120@1x1-out
                        padding::valid, true, 1, 1, 1, 1, backend_type)
                << tiny_dnn::relu_layer()
                << fc(120*4*4, 10, true, backend_type)  // F6, 120-in, 10-out
                <<  tiny_dnn::softmax_layer();
}





int main(){

        construct_net();



        int epoch = 1;
        const int n_train_epochs = 10;
        const int n_minibatch = 25;

        tiny_dnn::gradient_descent optimizer;

        //optimizer.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(n_minibatch));
        //optimizer.alpha=0.01;



        const float white = 0.0f; //background
        const float black = 1.0f; //foreground
        tiny_dnn::parse_mnist_labels("../tiny-dnn/data/train-labels.idx1-ubyte",
                                &train_labels);
        tiny_dnn::parse_mnist_images("../tiny-dnn/data/train-images.idx3-ubyte",
                                &train_images, -1.0, 1.0, 2, 2);
        tiny_dnn::parse_mnist_labels("../tiny-dnn/data/t10k-labels.idx1-ubyte",
                                &test_labels);
        tiny_dnn::parse_mnist_images("../tiny-dnn/data/t10k-images.idx3-ubyte",
                                &test_images, -1.0, 1.0, 2, 2);


        // Prepare data.
        std::vector<double> x(n_train_epochs), y(n_train_epochs);
        for(int i=0; i<n_train_epochs; ++i) {
                x.at(i) = i+1;
        }

        // Set the size of output image = 1200x780 pixels
        //plt::figure_size(1200, 780);
        int count=0;



        std::cout << "Running with the following parameters:" << std::endl
                << "Minibatch size: " << n_minibatch << std::endl
                << "Number of epochs: " << n_train_epochs << std::endl
                <<"optimizer.alpha = "<<optimizer.alpha<<std::endl
                << std::endl;
        std::cout << "start training" << std::endl;

        tiny_dnn::progress_display disp(train_images.size());
        tiny_dnn::timer t;

    
    

        // create callback
        auto on_enumerate_epoch = [&]() {
                std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
                        << t.elapsed() << "s elapsed." << std::endl;



                tiny_dnn::result res = net.test(test_images, test_labels);
                std::cout << res.num_success << "/" << res.num_total << std::endl;

                y.at(count) = ((float)res.num_success / (float)res.num_total)*100;
                count++;

                ++epoch;
                disp.restart(train_images.size());
                t.restart();
        };

        auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

    
        // training
        net.train<tiny_dnn::cross_entropy_multiclass>(optimizer, train_images, train_labels, n_minibatch,
                                n_train_epochs, on_enumerate_minibatch,
                                on_enumerate_epoch);

        std::cout << "end training." << std::endl;

        /*
        plt::plot(x,y);
        plt::xlabel("Epoch");
        plt::ylabel("Accuracy");
        plt::show();
        plt::save("./result.png");
        // test and show results
        */
        net.test(test_images, test_labels).print_detail(std::cout);
        net.save("mnist_net_fixed",tiny_dnn::content_type::weights_and_model, tiny_dnn::file_format::binary);
        net.save("mnist_net_fixed",tiny_dnn::content_type::weights_and_model, tiny_dnn::file_format::json);
}