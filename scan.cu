/* Copyright (c) 2018 huschen */

#include <cstdbool>
#include "kutils.cuh"
#include "scan_kernel.cuh"


/*
 * Wrapper function for kernel launch
 *
 * input_size (type: int), supports non-power-of-2.
 *
 * Limitations:
 * Input data size: no bigger than GPU DRAM size (assertion), no bigger than
 * size of (DRAM + SWAP) on CPU (fails with std::bad_alloc).
 * Input array size: no bigger than cuda_max_blocks*max_threads (assertion).
 *
 * ONLY works for one block!!
 *
 * Todo:
 * improve shared memory bank conflicts.
 * support bigger input data size.
 */
template<typename T, bool is_incl>
void scanSum(const T * const h_input,
             const T * const d_input_opt,
             int input_size,
             T * const d_output_opt,
             T * const h_output)
{
  KernelWrapper<T, T> krnl_wrap(
      h_input, d_input_opt, input_size, d_output_opt, h_output, input_size);

  int num_blocks, num_threads;
  krnl_wrap.setNumBT(input_size, &num_blocks, &num_threads);

  T * const d_input = krnl_wrap.allocDevInput();
  T * const d_output = krnl_wrap.allocDevOutput();

  if (is_incl)
  {
    hostPrint("input size: %d, block, thread: %d, %d\n",
              input_size, num_blocks, num_threads);
    kInclScanSum<T><<<num_blocks, num_threads,
        sizeof(T) * num_threads>>>(d_input, input_size, d_output);
  }
  else
  {
    // pad non-power-of-2 num_threads
    num_threads = powerOfTwo(num_threads);
    hostPrint("input size: %d, block, thread: %d, %d\n",
              input_size, num_blocks, num_threads);
    kExclScanSum<T><<<num_blocks, num_threads,
        sizeof(T) * num_threads>>>(d_input, input_size, d_output);
  }
}


template<typename T>
void inclScan(const T * const h_input,
              const T * const d_input_opt,
              int input_size,
              T * const d_output_opt,
              T * const h_output)
{
  scanSum<T, true>(h_input, d_input_opt, input_size, d_output_opt, h_output);
}

template<typename T>
void exclScan(const T * const h_input,
              const T * const d_input_opt,
              int input_size,
              T * const d_output_opt,
              T * const h_output)
{
  scanSum<T, false>(h_input, d_input_opt, input_size, d_output_opt, h_output);
}


// Instantiate the functions for int
template
void inclScan<int>(const int * const h_input,
                   const int * const d_input_opt,
                   int input_size,
                   int * const d_output_opt,
                   int * const h_output);

template
void exclScan<int>(const int * const h_input,
                   const int * const d_input_opt,
                   int input_size,
                   int * const d_output_opt,
                   int * const h_output);
