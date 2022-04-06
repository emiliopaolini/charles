/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <cstddef>
#include <cstdint>

/**
 * define if you want to use intel TBB library
 */
// #define CNN_USE_TBB

/**
 * define to enable avx vectorization
 */
// #define CNN_USE_AVX

/**
 * define to enable sse2 vectorization
 */
// #define CNN_USE_SSE

/**
 * define to enable OMP parallelization
 */
// #define CNN_USE_OMP

/**
 * define to enable Grand Central Dispatch parallelization
 */
// #define CNN_USE_GCD

/**
 * define to use exceptions
 */
#define CNN_USE_EXCEPTIONS

/**
 * comment out if you want tiny-dnn to be quiet
 */
#define CNN_USE_STDOUT

// #define CNN_SINGLE_THREAD

/**
 * disable serialization/deserialization function
 * You can uncomment this to speedup compilation & linking time,
 * if you don't use network::save / network::load functions.
 **/
// #define CNN_NO_SERIALIZATION

/**
 * Enable Image API support.
 * Currently we use stb by default.
 **/
// #define DNN_USE_IMAGE_API

/**
 * Enable Gemmlowp support.
 **/
#ifdef USE_GEMMLOWP
#if !defined(_MSC_VER) && !defined(_WIN32) && !defined(WIN32)
#define CNN_USE_GEMMLOWP  // gemmlowp doesn't support MSVC/mingw
#endif
#endif  // USE_GEMMLOWP

/**
 * number of task in batch-gradient-descent.
 * @todo automatic optimization
 */
#ifdef CNN_USE_OMP
#define CNN_TASK_SIZE 100
#else
#define CNN_TASK_SIZE 8
#endif


#include <cnl/all.h>
#include <cnl/auxiliary/boost.multiprecision.h>
#include <boost/multiprecision/cpp_int.hpp>
namespace tiny_dnn {

/**
 * calculation data type
 * you can change it to float, or user defined class (fixed point,etc)
 **/

#ifdef CNN_USE_DOUBLE
typedef double float_t;

#elif defined(CNN_USE_FIXED)

#define TOTAL_DIGITS 40 //+1(sign)
#define FRACTIONAL_DIGITS 20

template <int Digits,int Exponent,class baseType=cnl::signed_multiprecision<Digits>>
using saturated_scaled_integer = cnl::scaled_integer<cnl::rounding_integer< cnl::overflow_integer<baseType, cnl::saturated_overflow_tag>,cnl::nearest_rounding_tag>,cnl::power<-Exponent>>;
typedef saturated_scaled_integer<TOTAL_DIGITS,FRACTIONAL_DIGITS> float_t;
/*
using saturated_elastic_scaled_integer = cnl::scaled_integer<
                cnl::overflow_integer<
                        cnl::elastic_integer<
                            Digits,cnl::wide_integer<cnl::digits_v<Narrowest>, Narrowest>>, 
                        cnl::saturated_overflow_tag>,
        cnl::power<Exponent>>;
        */
//cnl::signed_multiprecision<31>



//using float_t = saturated_elastic_scaled_integer<TOTAL_DIGITS,-FRACTIONAL_DIGITS>;


inline float_t operator/( const float_t v1, const float_t v2 ){
    return (float)v1/(float)v2;
    float_t x = cnl::make_fraction(v1, v2);
    return x;
}

inline float_t operator/=(float_t &v1, const float_t v2 ){
    v1 = v1/v2;
    return v1;
}


inline float_t operator*( const float_t v1, const float_t v2 ){
    //return (float)v1/(float)v2;
    return (float)v1*(float)v2;
    auto v1_rep = cnl::to_rep<float_t>{}(v1);
    auto v2_rep = cnl::to_rep<float_t>{}(v2);
    float_t res = ((v1_rep * v2_rep) * std::pow(2,-FRACTIONAL_DIGITS*2));
    return res;
}

inline float_t operator*=(float_t &v1, const float_t v2 ){
    v1 = v1*v2;
    return v1;
}

/*
template<int Digits,int Exponent>
float_t operator*( const saturated_elastic_scaled_integer<Digits,Exponent> v1, const saturated_elastic_scaled_integer<Digits,Exponent> v2 ){
    return float_t((float)v1*(float)v2);
    
    /*auto v1_rep = cnl::to_rep<float_t>{}(v1);
    auto v2_rep = cnl::to_rep<float_t>{}(v2);
    float_t res = ((v1_rep * v2_rep) * std::pow(2,-FRACTIONAL_DIGITS*2));
    return res;
    
}

template<int Digits,int Exponent>
float_t operator*=(saturated_elastic_scaled_integer<Digits,Exponent> &v1, const saturated_elastic_scaled_integer<Digits,Exponent> v2 ){
    v1 = v1*v2;
    return v1;
}
*/



inline float_t max(float_t x,float_t y){
    return std::max((float)x,(float)y);
}

inline float_t log(float_t x){
    return std::log((float)x);
}

inline float_t sqrt(float_t x){
    return std::sqrt((float)x);
}

inline float_t abs(float_t x){
    return std::abs((float)x);
}

inline float_t asinh(float_t x){
    return std::asinh((float)x);
}

inline float_t cosh(float_t x){
    return std::cosh((float)x);
}

inline float_t exp(float_t x){
    return std::exp((float)x);
}

inline float_t log1p(float_t x){
    return std::log1p((float)x);
}

inline float_t tanh(float_t x){
    return std::tanh((float)x);
}

inline float_t ceil(float_t x){
    return std::ceil((float)x);
}

inline float_t pow(float_t a,float_t b){
    return std::pow((float)a,(float)b);
}



#else
typedef float float_t;
#endif

}  // namespace tiny_dnn
