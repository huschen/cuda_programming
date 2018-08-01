/* Copyright (c) 2018 huschen */

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include "reduce.h"
#include "test_utils.h"


void testReduceFloat(int input_size)
{
  float *h_data = new float[input_size];
  float gpu_max, cpu_max = -1;
  float gpu_sum, cpu_sum_basic = 0.0;
  // Kahan summation.
  float cpu_sum_kahan = 0.0, d_diff = 0.0;

  std::srand(0);
  for (int i = 0; i < input_size; i++)
  {
    h_data[i] = -1.0f + std::rand()*2.0f/RAND_MAX;

    cpu_max = std::max(cpu_max, h_data[i]);
    cpu_sum_basic += h_data[i];

    // Kahan summation.
    float d =  h_data[i] + d_diff;
    float cpu_sum_tmp = cpu_sum_kahan + d;
    d_diff = d - (cpu_sum_tmp - cpu_sum_kahan);
    cpu_sum_kahan = cpu_sum_tmp;
  }

  maxReduce<float>(h_data, NULL, input_size, &gpu_max);
  checkResult(cpu_max, gpu_max, "[float, max]");

  sumReduce<float>(h_data, NULL, input_size, &gpu_sum);
  checkResult(cpu_sum_basic, gpu_sum, "[float, sum, cpu_basic]");
  checkResult(cpu_sum_kahan, gpu_sum, "[float, sum, cpu_kahan]");

  delete[] h_data;
}


void testReduceLong(int input_size)
{
  long *h_data = new long[input_size];
  long gpu_min, gpu_sum;
  // sum might overflow, but it is fine for comparision.
  long cpu_min = std::pow(2, sizeof(long)*4) - 2, cpu_sum = 0;

  std::srand(0);
  for (int i = 0; i < input_size; i++)
  {
    h_data[i] = std::rand();
    cpu_min = std::min(cpu_min, h_data[i]);
    cpu_sum += h_data[i];
  }

  minReduce<long>(h_data, NULL, input_size, &gpu_min);
  checkResult(cpu_min, gpu_min, "[long, min]");

  sumReduce<long>(h_data, NULL, input_size, &gpu_sum);
  checkResult(cpu_sum, gpu_sum, "[long, sum]");

  delete[] h_data;
}


int main()
{
  int input_size = 100;
  testReduceLong(input_size);

  input_size = 32*1024 + 7;
  testReduceLong(input_size);

  input_size = 16 * 1024 * 1024;
  testReduceLong(input_size);
  testReduceFloat(input_size);


  // limited by host (DRAM + SWAP) size,
  // testReduceFloat: 8GB, works.
  // testReduceLong: 16GB, fails with std::bad_alloc.
  // input_size = std::pow(2, 31) - 1;
  // testReduceFloat(input_size);
  // testReduceLong(input_size);
  return 0;
}
