/* Copyright (c) 2018 huschen */

#include <cstdlib>
#include <algorithm>
#include "histog.h"
#include "test_utils.h"


void testHistog(int input_size, int num_bins)
{
  float *h_input = new float[input_size];
  const float bin_min = 0.0;
  const float bin_max = 100.0;
  const float bin_range = bin_max - bin_min;
  int cpu_output[num_bins] = {0};
  int gpu_output[num_bins] = {0};

  std::srand(0);
  for (int i = 0; i < input_size; i++)
  {
    h_input[i] = std::rand()*bin_max/RAND_MAX;
    int bin_idx = std::min(
        num_bins - 1,
        static_cast<int>((h_input[i] - bin_min)/bin_range*num_bins));
    cpu_output[bin_idx] += 1;
  }

  histog<float>(h_input, NULL, input_size, num_bins, bin_min, bin_max,
                NULL, gpu_output);
  checkResult(cpu_output, gpu_output, num_bins);
  delete[] h_input;
}


int main()
{
  int input_size = std::pow(2, 15) + 5;
  int num_bins = 10;
  testHistog(input_size, num_bins);

  input_size = 20;
  num_bins = 50;
  testHistog(input_size, num_bins);

  input_size = 100;
  num_bins = 1024 * 2 + 5;
  testHistog(input_size, num_bins);

  input_size = 1024 * 2 + 11;
  num_bins = 1024 * 3 + 7;
  testHistog(input_size, num_bins);

  input_size = 1024 * 64 + 11;
  // max shared_men size 48KB, max bin size for int 12 K
  num_bins = 1024 * 12;
  testHistog(input_size, num_bins);

  return 0;
}
