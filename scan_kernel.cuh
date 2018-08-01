/* Copyright (c) 2018 huschen */

#ifndef SORT_KERNEL_CUH_
#define SORT_KERNEL_CUH_

/*
 * Exclusive Scan, Blelloch algorithm
 */
template<typename T>
__device__ void coreExclScanSum(T *sh_inout)
{
  int tid = threadIdx.x;
  // blockDim.x is power-of-2 and the input is padded with 0
  int num_threads = blockDim.x;

  for (int stride = 1; stride < blockDim.x; stride *= 2)
  {
    if (((tid + 1) & (stride * 2 - 1)) == 0)
    {
      sh_inout[tid] += sh_inout[tid - stride];
    }
    __syncthreads();
  }

  if (tid == num_threads - 1)
  {
    sh_inout[tid] = 0;
  }
  // cudaPrint("%d: %d\n", tid, sh_inout[tid]);
  __syncthreads();

  for (int stride = num_threads >> 1; stride >= 1; stride >>= 1)
  {
    bool do_rw = ((tid + 1) & (stride - 1)) == 0;
    int sign = 0;
    T x = 0;
    if (do_rw)
    {
      // the first of the pair,  sign +1, (tid + 1) & (stride * 2 - 1) not 0
      // the second of the pair, sign -1, (tid + 1) & (stride * 2 - 1) is 0
      sign = ((tid + 1) & (stride * 2 - 1)) == 0 ? -1 : 1;
      int pair_id = tid + sign * stride;
      x = sh_inout[pair_id];
      // cudaPrint("----%d: %d [%d] %d, %d [%d]\n",
      //       stride, tid, sh_inout[tid], sign, pair_id, x);
    }
    __syncthreads();

    if (do_rw)
    {
      // the second of the pair (sign: -1) does the extra sum.
      sh_inout[tid] = x + sh_inout[tid] * (1 - sign) / 2;
    }
    __syncthreads();
  }
}


/*
 * Exclusive Scan, Blelloch algorithm
 */
template<typename T>
__global__ void kExclScanSum(const T * const d_input,
                             int input_size,
                             T *d_output)
{
  // extern __shared__ T sh_inout[];
  extern __shared__ unsigned char sh_mem[];
  T *sh_inout = reinterpret_cast<T *>(sh_mem);
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;


  if (gid < input_size)
  {
    sh_inout[tid] = d_input[gid];
  }
  else
  {
    // padded with 0: does not change the result of scan
    sh_inout[tid] = 0;
  }
  __syncthreads();

  coreExclScanSum<T>(sh_inout);

  if (gid < input_size)
  {
    d_output[gid] = sh_inout[tid];
  }
}


/*
 * Inclusive Scan, Hillis Steele scan algorithm
 */
template<typename T>
__global__ void kInclScanSum(const T * const d_input,
                            int input_size,
                            T *d_output)
{
  // extern __shared__ T sh_inout[];
  extern __shared__ unsigned char sh_mem[];
  T *sh_inout = reinterpret_cast<T *>(sh_mem);
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid >= input_size)
  {
    return;
  }

  sh_inout[tid] = d_input[gid];
  __syncthreads();

  for (int stride = 1; stride < blockDim.x; stride *=2)
  {
    bool do_rw = (tid >= stride);
    T x = 0;
    if (do_rw)
    {
      x = sh_inout[tid - stride];
    }
    __syncthreads();

    if (do_rw)
    {
      sh_inout[tid] += x;
    }
    __syncthreads();
  }

  d_output[gid] = sh_inout[tid];
}

#endif  // SORT_KERNEL_CUH_
