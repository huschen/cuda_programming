/* Copyright (c) 2018 huschen */

#include <cstdlib>
#include <algorithm>
#include "scan.h"
#include "test_utils.h"


void testScan(int input_size, int use_random = 1)
{
  int *h_input = new int[input_size];
  int *cpu_output_incl = new int[input_size];
  int *gpu_output_incl = new int[input_size];
  int *cpu_output_excl = new int[input_size];
  int *gpu_output_excl = new int[input_size];

  std::srand(0);
  int max_int = 100;
  int acc = 0;
  for (int i = 0; i < input_size; i++)
  {
    if (use_random)
    {
      h_input[i] = std::rand()%max_int;
    }
    else
    {
      h_input[i] = 1;
    }
    cpu_output_excl[i] = acc;
    acc += h_input[i];
    cpu_output_incl[i] = acc;
  }

  inclScan<int>(h_input, NULL, input_size, NULL, gpu_output_incl);
  checkResult(cpu_output_incl, gpu_output_incl, input_size,
              "Inclusive Scan: \n");

  exclScan<int>(h_input, NULL, input_size, NULL, gpu_output_excl);
  checkResult(cpu_output_excl, gpu_output_excl, input_size,
              "Exclusive Scan: \n");

  delete[] h_input;
  delete[] cpu_output_incl;
  delete[] gpu_output_incl;
  delete[] cpu_output_excl;
  delete[] gpu_output_excl;
}


int main()
{
  testScan(7, 0);
  testScan(257);
  testScan(1024, 0);
  testScan(1024);

  // not implemented yet..
  // testScan(1024*4+7);
  // testScan(1024*2);

  return 0;
}
