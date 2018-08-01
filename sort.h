/* Copyright (c) 2018 huschen */

#ifndef SORT_H_
#define SORT_H_

template <typename T>
void radixSort(const T * const h_input,
               const T * const d_input_opt,
               int input_size,
               T *d_output_opt,
               T *h_output);

#endif  // SORT_H_
