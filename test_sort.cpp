/* Copyright (c) 2018 huschen */

#include <cstdlib>
#include <algorithm>
#include <vector>
#include "sort.h"
#include "test_utils.h"


void testSort(int input_size, int use_random = 1)
{
  unsigned int *h_input = new unsigned int[input_size];
  unsigned int *gpu_output = new unsigned int[input_size];

  std::srand(0);
  for (int i = 0; i < input_size; i++)
  {
    if (use_random)
    {
      h_input[i] = std::rand();
    }
    else
    {
      h_input[i] = 1024 - i;
    }
  }

  radixSort<unsigned int>(h_input, NULL, input_size, NULL, gpu_output);

  std::vector<unsigned int> cpu_output(h_input, h_input + input_size);
  std::sort(cpu_output.begin(), cpu_output.end());

  checkResult(cpu_output.data(), gpu_output, input_size);

  delete[] h_input;
  delete[] gpu_output;
}


int main()
{
  testSort(7, 0);
  testSort(257);
  testSort(1024, 0);
  testSort(1024);

  // not implemented yet..
  // testScan(1024*4+7);
  // testScan(1024*2);

  return 0;
}
