/* Copyright (c) 2018 huschen */

#include "kutils.cuh"
#include "scan_kernel.cuh"


/*
 * Radix Sort
 * type of T: unsigned integer
 */
template <typename T, unsigned int NUM_SHIFT>
__global__ void kRadixSort(const T * const d_input,
                        int input_size,
                        T *d_output)
{
  // extern __shared__ T sh_inout[];
  extern __shared__ unsigned char sh_mem[];
  T *sh_inout = reinterpret_cast<T *>(sh_mem);
  int valid_length = input_size - blockDim.x * blockIdx.x;
  // blockDim.x is power-of-2, input data will be padded with 0
  int num_threads = blockDim.x;
  T *sh_in = sh_inout;
  T *sh_pred = &sh_inout[num_threads];
  T *sh_scan = &sh_inout[num_threads*2];
  T *sh_out = &sh_inout[num_threads*3];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;


  if (gid < input_size)
  {
    sh_in[tid] = d_input[gid];
  }
  else
  {
    sh_in[tid] = 0;
  }

  sh_out[tid] = 0;

  T offset = 0;

#pragma unroll
  for (unsigned int bit_shft = 0; bit_shft < NUM_SHIFT; bit_shft++)
  {
    offset = 0;
#pragma unroll
    for (unsigned int bit_val = 0; bit_val <= 1; bit_val++)
    {
      if (tid < valid_length)
      {
        sh_pred[tid] = (sh_in[tid] & (1 << bit_shft)) == (bit_val << bit_shft);
      }
      else
      {
        // the padded parts, always 0 for scan calculation
        sh_pred[tid] = 0;
      }
      sh_scan[tid] = sh_pred[tid];
      __syncthreads();

      coreExclScanSum<T>(sh_scan);
      __syncthreads();

      // copy to the output, the padded values never copied.
      if (sh_pred[tid] != 0)
      {
        sh_out[sh_scan[tid] + offset] = sh_in[tid];
      }
      __syncthreads();

      offset = sh_scan[num_threads - 1] + (sh_pred[num_threads - 1] != 0);
      __syncthreads();
    }
    sh_in[tid] = sh_out[tid];
  }


  if (gid < input_size)
  {
    d_output[gid] = sh_out[tid];
  }
}


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
template <typename T>
void radixSort(const T * const h_input,
               const T * const d_input_opt,
               int input_size,
               T *d_output_opt,
               T *h_output)
{
  KernelWrapper<T, T> krnl_wrap(
      h_input, d_input_opt, input_size, d_output_opt, h_output, input_size);

  int num_blocks, num_threads;
  krnl_wrap.setNumBT(input_size, &num_blocks, &num_threads);

  T * const d_input = krnl_wrap.allocDevInput();
  T * const d_output = krnl_wrap.allocDevOutput();

  // pad non-power-of-2 num_threads
  num_threads = powerOfTwo(num_threads);
  const unsigned int num_shift = sizeof(T) * BITS_PER_BYTE;
  hostPrint("input size: %d, num_shift: %u, block, thread: %d, %d\n",
            input_size, num_shift, num_blocks, num_threads);
  kRadixSort<T, num_shift><<<num_blocks, num_threads,
      sizeof(T) * num_threads * 4>>>(d_input, input_size, d_output);
}


// Instantiate the functions for unsigned int
template
void radixSort<unsigned int>(const unsigned int * const h_input,
                             const unsigned int * const d_input_opt,
                             int input_size,
                             unsigned int *d_output_opt,
                             unsigned int *h_output);
