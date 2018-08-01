/* Copyright (c) 2018 huschen */

#ifndef KUTILS_CUH_
#define KUTILS_CUH_

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>


#define BITS_PER_BYTE 8U

/*
 * debug messge
 */
#ifdef CUDA_MSG
// printf is supported on CUDA cores
// better switch print off for profiling
#define cudaPrint(...)  printf(__VA_ARGS__)
#else
#define cudaPrint(...)  do {} while (0)
#endif  // CUDA_MSG

#ifdef HOST_MSG
#define hostPrint(...)  printf(__VA_ARGS__)
#else
#define hostPrint(...)  do {} while (0)
#endif  // HOST_MSG


/*
 * CUDA error check
 */
#ifdef CHECK_CUDA_ERRS
// error checking functions from CUDA samples
// Output the CUDA errors in the event that a CUDA host call returns an error.
#define checkCudaErrors(func)  check((func), #func, __FILE__, __LINE__)
template < typename T >
void check(T result, char const * const func, const char * const file,
           int const line)
{
  if (result)
  {
    fprintf(stderr, "CUDA error %s[%d], at %s: %d, %s.\n",
        cudaGetErrorString(result), static_cast<int>(result), file, line, func);
    cudaDeviceReset();
    // Make sure we call CUDA Device Reset before exiting
    exit(1);
  }
}

// Output the CUDA errors when calling cudaGetLastError.
#define getLastCudaError()  __getLastCudaError(__FILE__, __LINE__)
inline void __getLastCudaError(const char *file, const int line)
{
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err)
  {
    fprintf(stderr, "CUDA error %s[%d], at %s: %d.\n",
            cudaGetErrorString(err), static_cast<int>(err), file, line);
    cudaDeviceReset();
    exit(1);
  }
}
#else
#define checkCudaErrors(func) (func)
#define getLastCudaError() do {} while (0)
#endif  // CHECK_CUDA_ERRS


/*
 * helper functions
 */
inline unsigned int powerOfTwo(unsigned int n)
{
  n += (n == 0);
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return ++n;
}


/*
 * GPU performance check, using steam = 0.
 * cudaEventRecord() operation takes place asynchronously.
 * Any number of other different stream operations could execute in between
 * the two measured events.
 */

struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start(void)
  {
    // Captures in event the contents of stream at the time of this call.
    cudaEventRecord(start, 0);
  }

  void Stop(void)
  {
    cudaEventRecord(stop, 0);
  }

  float Elapsed(void)
  {
    // in ms
    float elapsed;

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};


// max number of sequential reductions
#define MAX_NUM_SEQ 64

/*
 * class KernelWrapper
 * Allocate and copy the input data on/to GPU device,
 * copy the output data to the host from the device,
 * and free the memory on the device, when applicable.
 */
template <typename TI, typename TO>
class KernelWrapper
{
private:
  cudaDeviceProp prop;
  int max_blocks, max_threads, num_sms;

  const TI * const h_input;
  const TI * const d_input_opt;
  const int input_size;
  TO * const d_output_opt;
  TO * const h_output;
  const int output_size;

  TI *d_input;
  TO *d_output;

#ifdef PERF_MON
    GpuTimer timer;
#endif

public:
  KernelWrapper<TI, TO>(const TI * const h_input,
                        const TI * const d_input_opt,
                        const int input_size,
                        TO * const d_output_opt,
                        TO * const h_output,
                        const int output_size):
    h_input(h_input),
    d_input_opt(d_input_opt),
    input_size(input_size),
    d_output_opt(d_output_opt),
    h_output(h_output),
    output_size(output_size)
  {
    int nv_dev_count;
    int dev;
    checkCudaErrors(cudaGetDeviceCount(&nv_dev_count));
    assert(nv_dev_count != 0);
    checkCudaErrors(cudaGetDevice(&dev));
    checkCudaErrors(cudaGetDeviceProperties(&prop, dev));

    max_blocks = prop.maxGridSize[0];
    max_threads = prop.maxThreadsPerBlock;
    num_sms = prop.multiProcessorCount;

    hostPrint("\nDevice 0(%d): \"%s\", %d.%d\n", nv_dev_count, prop.name,
              prop.major, prop.minor);
#ifdef PERF_MON
    timer.Start();
#endif
  }

  size_t getCudaBlockMem()
  {
    return prop.sharedMemPerBlock;;
  }

  void setNumBT(int input_size,
                int *num_blocks_p,
                int *num_threads_p)
  {
    int num_threads = min(input_size, max_threads);
    int num_blocks = (input_size + num_threads - 1) / num_threads;
    assert(num_blocks <= max_blocks);
    *num_blocks_p = num_blocks;
    *num_threads_p = num_threads;
  }

  void setNumBTSeq(int input_size,
                   int *num_blocks_p,
                   int *num_threads_p,
                   int *data_blocks_p,
                   int *num_seq_p)
  {
    int num_threads = min(input_size, max_threads);
    int data_blocks = (input_size + num_threads - 1)/num_threads;
    int num_seq = min(MAX_NUM_SEQ,
                      (data_blocks + num_sms - 1)/num_sms);
    int num_blocks = (data_blocks + num_seq - 1)/num_seq;
    // adjust num_seq, num_blocks to make sure num_blocks < prop.maxGridSize[0]
    *num_blocks_p = num_blocks;
    *num_threads_p = num_threads;
    *data_blocks_p = data_blocks;
    *num_seq_p = num_seq;
  }

  TI *allocDevInput()
  {
    assert((size_t)input_size*sizeof(TI) <= prop.totalGlobalMem);

    checkCudaErrors(cudaMalloc(&d_input, sizeof(TI)*input_size));
    if (d_input_opt == NULL)
    {
      assert(h_input != NULL);
      checkCudaErrors(cudaMemcpy(d_input, h_input, sizeof(TI)*input_size,
                                 cudaMemcpyHostToDevice));
    }
    else
    {
      // make a copy, instead of const_cast<TI *>(d_input_opt);
      checkCudaErrors(cudaMemcpy(d_input, d_input_opt, sizeof(TI)*input_size,
                                 cudaMemcpyDeviceToDevice));
    }
    return d_input;
  }

  TO *allocDevOutput()
  {
    if (d_output_opt == NULL)
    {
      checkCudaErrors(cudaMalloc(&d_output, sizeof(TO)*output_size));
      checkCudaErrors(cudaMemset(d_output, 0, sizeof(TO)*output_size));
    }
    else
    {
      d_output = d_output_opt;
    }
    return d_output;
  }

  ~KernelWrapper()
  {
    if (h_output != NULL)
    {
      checkCudaErrors(cudaMemcpy(h_output, d_output, sizeof(TO)*output_size,
                                 cudaMemcpyDeviceToHost));
    }

    if (d_input_opt == NULL)
    {
        checkCudaErrors(cudaFree(d_input));
    }

    if (d_output_opt == NULL)
    {
       checkCudaErrors(cudaFree(d_output));
    }

#ifdef PERF_MON
    timer.Stop();
    cudaDeviceSynchronize();
    getLastCudaError();
    // time_elapsed: in ms.
    float time_elapsed = timer.Elapsed();
    // memoryClockRate: int, in kilohertz.
    // memoryBusWidth: int, in bits.
    // Throughput: in GB/s
    float max_tp = 1e-6f*prop.memoryClockRate*prop.memoryBusWidth/BITS_PER_BYTE;
    float run_tp = 1e-6f*sizeof(TI)*input_size/time_elapsed;

    printf("Kernel ran in: %.3f msecs. Throughput: %.3f GB/s (%.3f)\n",
           time_elapsed, run_tp, max_tp);
#endif
  }
};

#endif  // KUTILS_CUH_
