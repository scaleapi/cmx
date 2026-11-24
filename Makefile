CC      = clang
CXX     = clang++
CFLAGS  ?= -O3 -march=native
LDLIBS  = -lm -lpthread
NVCC    ?= nvcc
CUDA_ARCH ?= sm_90

# One program, cmx: both codecs (cmxc.c, bf16ref.c) compiled unchanged with their main functions renamed
# (-DCMX_UNIFIED), and cmx.c choosing between them.
all: cmx

ifeq ($(CUDA),1)
# cmx --gpu: bf16ref_gpu.cu (and the kernels in gpu/) built with nvcc; linked by nvcc
GPU_O = bf16ref_gpu.o
$(GPU_O): bf16ref_gpu.cu bfr_gpu.h gpu/kernels.cuh
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) -c -o $@ bf16ref_gpu.cu

cmx: cmx.c cmxc.c bf16ref.c bfr8_kern.h bfr_gpu.h $(GPU_O)
	$(CC) $(CFLAGS) -DCMX_UNIFIED -DBF16REF_GPU -c cmx.c -o cmx_main.o
	$(CC) $(CFLAGS) -DCMX_UNIFIED -c cmxc.c -o cmx_cmxc.o
	$(CC) $(CFLAGS) -DCMX_UNIFIED -DBF16REF_GPU -c bf16ref.c -o cmx_bf16ref.o
	$(NVCC) -o $@ cmx_main.o cmx_cmxc.o cmx_bf16ref.o $(GPU_O) $(LDLIBS)
else
cmx: cmx.c cmxc.c bf16ref.c bfr8_kern.h
	$(CC) $(CFLAGS) -DCMX_UNIFIED -o $@ cmx.c cmxc.c bf16ref.c $(LDLIBS)
endif

# For tests only: cmx with its GPU path running on the CPU emulator (gpu/cuemu.h), so that tests/gpu.sh runs
# without a GPU. It is slow, so use small inputs only.
cmx_emu: cmx.c cmxc.c bf16ref.c bfr8_kern.h bfr_gpu.h bf16ref_gpu.cu gpu/kernels.cuh gpu/cuemu.h
	$(CC) $(CFLAGS) -DCMX_UNIFIED -c cmx.c -o cmx_emu_main.o
	$(CC) $(CFLAGS) -DCMX_UNIFIED -c cmxc.c -o cmx_emu_cmxc.o
	$(CC) -O2 -DCMX_UNIFIED -DBF16REF_GPU -c bf16ref.c -o cmx_emu_bf16ref.o
	$(CXX) -std=c++20 -O1 -DBFR_GPU_EMU -I. -x c++ -c bf16ref_gpu.cu -o cmx_emu_gpu.o
	$(CXX) -o $@ cmx_emu_main.o cmx_emu_cmxc.o cmx_emu_bf16ref.o cmx_emu_gpu.o $(LDLIBS)

test: cmx
	sh tests/roundtrip.sh ./cmx
	sh tests/cmx.sh ./cmx

# the GPU path against the CPU coder: on a GPU (make CUDA=1 test-gpu) or on the emulator (make test-gpu-emu)
test-gpu: cmx
	sh tests/gpu.sh ./cmx ./cmx

test-gpu-emu: cmx cmx_emu
	sh tests/gpu.sh ./cmx ./cmx_emu small

clean:
	rm -f cmx cmx_emu *.o

.PHONY: all test test-gpu test-gpu-emu clean
