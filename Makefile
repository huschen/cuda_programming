NVCC=nvcc
CUDA_INCLUDEPATH=/usr/local/cuda-8.0/include
NVCC_OPTS=-O3 -arch=sm_37 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -Werror -m64 -std=c++11

################################################################################
# Target rules


.PHONY: all build run debug_reduce prof_reduce stress_sort clean

all: build

build: test_histog test_reduce test_scan test_sort

test_reduce: reduce.o reduce.h test_reduce.cpp test_utils.h
	$(NVCC) -o test_reduce test_reduce.cpp reduce.o $(NVCC_OPTS) -DHOST_MSG

reduce.o: reduce.cu kutils.cuh Makefile
	$(NVCC) -c reduce.cu $(NVCC_OPTS) -DPERF_MON -DHOST_MSG

test_scan: scan.o scan.h test_scan.cpp test_utils.h
	$(NVCC) -o test_scan test_scan.cpp scan.o $(NVCC_OPTS) -DHOST_MSG

scan.o: scan.cu scan_kernel.cuh kutils.cuh Makefile
	$(NVCC) -c scan.cu $(NVCC_OPTS) -DPERF_MON -DCHECK_CUDA_ERRS -DHOST_MSG

test_histog: histog.o histog.h test_histog.cpp test_utils.h
	$(NVCC) -o test_histog test_histog.cpp histog.o $(NVCC_OPTS) -DHOST_MSG

histog.o: histog.cu kutils.cuh Makefile
	$(NVCC) -c histog.cu $(NVCC_OPTS) -DCHECK_CUDA_ERRS -DHOST_MSG

test_sort: sort.o sort.h test_sort.cpp test_utils.h
	$(NVCC) -o test_sort test_sort.cpp sort.o $(NVCC_OPTS) -DHOST_MSG

sort.o: sort.cu scan_kernel.cuh kutils.cuh Makefile
	$(NVCC) -c sort.cu $(NVCC_OPTS) -DCHECK_CUDA_ERRS -DHOST_MSG


run: build
	./test_reduce
	./test_scan
	./test_histog
	./test_sort

debug_reduce: reduce.cu
	nvcc -g -c reduce.cu $(NVCC_OPTS)
	nvcc -cubin -c reduce.cu $(NVCC_OPTS)
	nm reduce.o > reduce_cu.txt
	objdump -S reduce.o >> reduce_cu.txt
	cuobjdump -sass reduce.o >> reduce_cu.txt
	nvdisasm -plr reduce.cubin >> reduce_cu.txt

prof_reduce: test_reduce
	nvprof ./test_reduce

stress_sort:
	$(NVCC) -c sort.cu $(NVCC_OPTS) -DCHECK_CUDA_ERRS
	$(NVCC) -o test_sort test_sort.cpp sort.o $(NVCC_OPTS)
	for i in `seq 10`; do echo $$i; ./test_sort; done


clean:
	rm -f *.o test_reduce test_histog test_scan test_sort
