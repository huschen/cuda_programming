/* Copyright (c) 2018 huschen */

#ifndef HISTOG_H_
#define HISTOG_H_

template <typename T>
void histog(const T * const h_input,
            const T * const d_input_opt,
            int input_size,
            int num_bins,
            T bin_min,
            T bin_max,
            int * const d_output_opt,
            int * const h_output);

#endif  // HISTOG_H_
