/* Copyright (c) 2018 huschen */

#ifndef SCAN_H_
#define SCAN_H_

template<typename T>
void inclScan(const T * const h_input,
              const T * const d_input_opt,
              int input_size,
              T * const d_output_opt,
              T * const h_output);

template<typename T>
void exclScan(const T * const h_input,
              const T * const d_input_opt,
              int input_size,
              T * const d_output_opt,
              T * const h_output);

#endif  // SCAN_H_
