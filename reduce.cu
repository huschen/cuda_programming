/* Copyright (c) 2018 huschen */

#include "kutils.cuh"

typedef enum {
  REDUCE_MIN,
  REDUCE_MAX,
  REDUCE_SUM,
  REDUCE_INVALID,
} reduce_t;

template <typename T>
__device__ T sum(T a, T b)
{
  return a+b;
}


/*
 * Reduction kernel, input_size supports non-power-of-2
 * num_seq: number of sequential reductions each thread performs
 * Further optimisation: unroll for loops
 */
template <typename T, reduce_t RT>
__global__ void kReduce(int num_seq,
                        const T * const d_input,
                        const int input_size,
                        T * const d_out_mins)
{
  // sh_mem: being one data type, compiler constraints.
  // sh_data: should be alligned, because it is allocated during kernel
  // launch with the known data type.
  extern __shared__ unsigned char sh_mem[];
  T *sh_data = reinterpret_cast<T *>(sh_mem);

  // blockDim.x * num_seq: sequential reduction during global load
  int data_idx = blockIdx.x*blockDim.x*num_seq + threadIdx.x;
  int length = min(input_size - blockDim.x*blockIdx.x*num_seq,
                            blockDim.x);
  int idx = threadIdx.x;
  T (*func_reduce)(T, T);

  switch (RT)
  {
  case REDUCE_MIN:
    func_reduce = min;
    break;
  case REDUCE_MAX:
    func_reduce = max;
    break;
  case REDUCE_SUM:
    func_reduce = sum;
    break;
  default:
    assert(RT < REDUCE_INVALID);
  }

  if (data_idx >=input_size)
  {
    return;
  }
  // sequential reduction during global load.
  sh_data[idx] = d_input[data_idx];
  int seq_idx = data_idx + blockDim.x;
  for (int i = 1; i < num_seq; i++)
  {
    if ( seq_idx < input_size)
    {
      sh_data[idx] = func_reduce(sh_data[idx], d_input[seq_idx]);
    }
    seq_idx += blockDim.x;
  }
  __syncthreads();

  for (; length > 1; length = (length + 1) / 2)
  {
    if (idx < length / 2)
    {
      int stride = (length + 1) / 2;
      sh_data[idx] = func_reduce(sh_data[idx], sh_data[idx + stride]);
    }
    __syncthreads();
  }

  if (idx == 0)
  {
    d_out_mins[blockIdx.x] = sh_data[0];
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
 * Todo:
 * For very big input data: break input data into list of sub_input_data of
 * smaller sizes.
 */
template <typename T, reduce_t RT>
void reduce(const T * const h_input,
            const T * const d_input_opt,
            int input_size,
            T * const h_output)
{
  KernelWrapper<T, T> krnl_wrap(
      h_input, d_input_opt, input_size, NULL, h_output, 1);

  int num_blocks, num_threads, data_blocks, num_seq;
  krnl_wrap.setNumBTSeq(input_size, &num_blocks, &num_threads,
                        &data_blocks, &num_seq);

  T * const d_input = krnl_wrap.allocDevInput();
  T * const d_output = krnl_wrap.allocDevOutput();

  T *d_tmp_output = NULL;
  checkCudaErrors(cudaMalloc(&d_tmp_output, sizeof(T)*num_blocks));

  while (num_threads != 1)
  {
    hostPrint("[N]input size: %d, block, thread: %d(%d, %d), %d\n",
              input_size, num_blocks, data_blocks, num_seq, num_threads);
    kReduce<T, RT><<<num_blocks, num_threads, sizeof(T)*num_threads>>>(
        num_seq, d_input, input_size, d_tmp_output);
    getLastCudaError();

    input_size = num_blocks;
    checkCudaErrors(cudaMemcpy(d_input, d_tmp_output, sizeof(T)*input_size,
                               cudaMemcpyDeviceToDevice));
    krnl_wrap.setNumBTSeq(input_size, &num_blocks, &num_threads,
                          &data_blocks, &num_seq);
  }
  checkCudaErrors(cudaMemcpy(d_output, d_tmp_output, sizeof(T),
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaFree(d_tmp_output));
}


template<typename T>
void minReduce(const T * const h_input,
               const T * const d_input_opt,
               int input_size,
               T * const h_output)
{
  reduce<T, REDUCE_MIN>(h_input, d_input_opt, input_size, h_output);
}

template<typename T>
void maxReduce(const T * const h_input,
               const T * const d_input_opt,
               int input_size,
               T * const h_output)
{
  reduce<T, REDUCE_MAX>(h_input, d_input_opt, input_size, h_output);
}

template<typename T>
void sumReduce(const T * const h_input,
               const T * const d_input_opt,
               int input_size,
               T * const h_output)
{
  reduce<T, REDUCE_SUM>(h_input, d_input_opt, input_size, h_output);
}


// Instantiate the functions for float and long
template
void minReduce<float>(const float * const h_input,
                      const float * const d_input_opt,
                      int input_size,
                      float * const h_output);

template
void maxReduce<float>(const float * const h_input,
                      const float * const d_input_opt,
                      int input_size,
                      float * const h_output);

template
void sumReduce<float>(const float * const h_input,
                      const float * const d_input_opt,
                      int input_size,
                      float * const h_output);

template
void minReduce<long>(const long * const h_input,
                              const long * const d_input_opt,
                              int input_size,
                              long * const h_output);

template
void maxReduce<long>(const long * const h_input,
                              const long * const d_input_opt,
                              int input_size,
                              long * const h_output);

template
void sumReduce<long>(const long * const h_input,
                              const long * const d_input_opt,
                              int input_size,
                              long * const h_output);
