/* Copyright (c) 2018 huschen */

#ifndef REDUCE_H_
#define REDUCE_H_

template<typename T>
void minReduce(const T * const h_input,
               const T * const d_input_opt,
               int input_size,
               T * const h_output);

template<typename T>
void maxReduce(const T * const h_input,
               const T * const d_input_opt,
               int input_size,
               T * const h_output);


template<typename T>
void sumReduce(const T * const h_input,
               const T * const d_input_opt,
               int input_size,
               T * const h_output);

#endif  // REDUCE_H_
