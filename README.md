# CUDA Programming

Functions implemented using CUDA C:
* Reduction,
* Scan (Blelloch algorithm and Hillis Steele algorithm),
* Histogram,
* Sort (Radix).

Test Environment:
* CUDA 8.0,
* Nvidia G80(sm_37) and P100(sm_60),
* C++ 11,
* Ubuntu-1604, 64-bit.


Specification and Limitations:
* input_array_size: of type int, support non-power-of-2.
* Input data size: no bigger than GPU DRAM size (assertion), no bigger than size of (DRAM + SWAP) on CPU (fails with std::bad_alloc).
* Input array size: no bigger than cuda_max_blocks * cuda_max_threads_per_block (assertion).
* Only support single GPU, single stream.
* Scan and Sort: ONLY works for size of one block!!
* Histogram: Output data size: no bigger than block shared memory size (assertion).

Todo:
* For multiple GPUs: Assign partial data to different GPUs.
* For very big input data: Break input data into list of sub_input_data of smaller sizes.
* Scan and Sort: Improve shared memory bank conflicts. Support bigger input data size.
* Histogram: Improve the performance by eliminating atomicAdd. Support bigger output data size (more bins).


Reference:
* [NVIDIA CUDA-8.0 Samples](https://docs.nvidia.com/cuda/cuda-samples/index.html),
* [Udacity](https://eu.udacity.com/course/intro-to-parallel-programming--cs344) [Parallel Programming](https://github.com/udacity/cs344).
