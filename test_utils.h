/* Copyright (c) 2018 huschen */

#ifndef TEST_UTILS_H_
#define TEST_UTILS_H_

#include <iostream>
#include <string>
#include <algorithm>


#ifdef HOST_MSG
#define VERBOSE 1
#else
#define VERBOSE 0
#endif

#define MAX_PRINT 10
#define BL_STR(x) (x? "TRUE!" : "FALSE!!")


template <typename T>
bool checkResult(T cpu_output[],
                T gpu_output[],
                int output_size,
                std::string msg = "")
{
  int unequal = 0;
  int start = 0;
  int end = 0;
  for (int i = 0; i < output_size; i++)
  {
    if (gpu_output[i] != cpu_output[i] && unequal == 0)
    {
      start = i;
    }
    unequal += (gpu_output[i] != cpu_output[i]);
  }

  if (unequal == 0)
  {
    start = std::max(0, output_size - MAX_PRINT);
    end = output_size;
  }
  else
  {
    end = std::min(start + MAX_PRINT, output_size);
  }

  if (VERBOSE || unequal != 0)
  {
    std::cout << msg << "GPU result: ";
    for (int i = start; i < end; i++)
    {
      std::cout << gpu_output[i] << ", ";
    }
    std::cout << "\nCPU result: ";
    for (int i = start; i < end; i++)
    {
      std::cout << cpu_output[i] << ", ";
    }
    std::cout << "SAME RESULT? " << BL_STR(unequal == 0)
        << "(" << unequal<< ")"<< std::endl<< std::endl;
  }
  return (unequal == 0);
}


template <typename T>
bool checkResult(T cpu_output,
                 T gpu_output,
                 std::string msg = "")
{
  bool equal = cpu_output == gpu_output;
  if (VERBOSE || equal == 0)
  {
    std::cout << msg << "cpu_output=" << cpu_output
        << ", gpu_output=" << gpu_output
        << ", diff=" << cpu_output - gpu_output
        << ", SAME RESULT? " << BL_STR(equal)
        << "(" << 100.0f*(gpu_output - cpu_output)/gpu_output << "%)"
        << std::endl;
  }
  return equal;
}

#endif  // TEST_UTILS_H_
