/* Copyright (c) 2018 huschen */

#include "kutils.cuh"

template <typename T>
__global__ void kHistog(const T * const d_input,
                       int input_size,
                       int num_bins,
                       T bin_min,
                       T bin_range,
                       int * const d_output)
{
  extern __shared__ int sh_bins[];
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  for (int bin_updt = tid; bin_updt < num_bins; bin_updt += blockDim.x)
  {
     sh_bins[bin_updt] = 0;
  }
  __syncthreads();

  if (gid <input_size)
  {
    int bin_idx = min(
        num_bins - 1,
        static_cast<int>((d_input[gid] - bin_min) / bin_range * num_bins));
    atomicAdd(&sh_bins[bin_idx], 1);
    // cudaPrint("gpu[%d], %f, %d\n", gid, d_input[gid], bin_idx);
  }
  __syncthreads();

  for (int bin_updt = tid; bin_updt < num_bins; bin_updt += blockDim.x)
  {
     atomicAdd(&d_output[bin_updt], sh_bins[bin_updt]);
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
 * Output data size (bins): no bigger than block shared memory size (assertion).
 *
 * Todo:
 * Improve the performance by eliminating atomicAdd.
 * Support bigger output data size (more bins).
 */
template <typename T>
void histog(const T * const h_input,
            const T * const d_input_opt,
            int input_size,
            int num_bins,
            T bin_min,
            T bin_max,
            int * const d_output_opt,
            int * const h_output)
{
  KernelWrapper<T, int> krnl_wrap(
      h_input, d_input_opt, input_size, d_output_opt, h_output, num_bins);

  int num_blocks, num_threads;
  krnl_wrap.setNumBT(input_size, &num_blocks, &num_threads);

  assert((size_t)num_bins*sizeof(int) <= krnl_wrap.getCudaBlockMem());

  T * const d_input = krnl_wrap.allocDevInput();
  int * const d_output = krnl_wrap.allocDevOutput();

  hostPrint("input size: %d, block, thread: %d, %d; num_bins: %d\n",
            input_size, num_blocks, num_threads, num_bins);

  kHistog<<<num_blocks, num_threads, sizeof(int)*num_bins>>>(
      d_input, input_size, num_bins, bin_min, bin_max - bin_min, d_output);
  getLastCudaError();
}


// Instantiate the functions for float
template
void histog<float>(const float * const h_input,
                   const float * const d_input_opt,
                   int input_size,
                   int num_bins,
                   float bin_min,
                   float bin_max,
                   int * const d_output_opt,
                   int * const h_output);
