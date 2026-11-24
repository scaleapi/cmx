CC      = clang
CXX     = clang++
CFLAGS  ?= -O3 -march=native
LDLIBS  = -lm -lpthread
NVCC    ?= nvcc
CUDA_ARCH ?= sm_90

all: cmx cmxc bf16ref

ifeq ($(CUDA),1)
# bf16ref --gpu / cmx --gpu: bf16ref_gpu.cu (and the kernels in gpu/) built with nvcc; linked by nvcc
GPU_O = bf16ref_gpu.o
$(GPU_O): bf16ref_gpu.cu bfr_gpu.h gpu/kernels.cuh
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) -c -o $@ bf16ref_gpu.cu

cmx: cmx.c cmxc.c bf16ref.c bfr8_kern.h bfr_gpu.h $(GPU_O)
	$(CC) $(CFLAGS) -DCMX_UNIFIED -DBF16REF_GPU -c cmx.c -o cmx_main.o
	$(CC) $(CFLAGS) -DCMX_UNIFIED -c cmxc.c -o cmx_cmxc.o
	$(CC) $(CFLAGS) -DCMX_UNIFIED -DBF16REF_GPU -c bf16ref.c -o cmx_bf16ref.o
	$(NVCC) -o $@ cmx_main.o cmx_cmxc.o cmx_bf16ref.o $(GPU_O) $(LDLIBS)

bf16ref: bf16ref.c bfr8_kern.h bfr_gpu.h $(GPU_O)
	$(CC) $(CFLAGS) -DBF16REF_GPU -c bf16ref.c -o bf16ref_main.o
	$(NVCC) -o $@ bf16ref_main.o $(GPU_O) $(LDLIBS)
else
# one program: both codecs, compiled unchanged with their main functions renamed (-DCMX_UNIFIED)
cmx: cmx.c cmxc.c bf16ref.c bfr8_kern.h
	$(CC) $(CFLAGS) -DCMX_UNIFIED -o $@ cmx.c cmxc.c bf16ref.c $(LDLIBS)

bf16ref: bf16ref.c bfr8_kern.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
endif

cmxc: cmxc.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# bf16ref with the GPU path running on the CPU emulator (gpu/cuemu.h): for tests/gpu.sh without a GPU; slow,
# small inputs only
bf16ref_emu: bf16ref.c bfr8_kern.h bfr_gpu.h bf16ref_gpu.cu gpu/kernels.cuh gpu/cuemu.h
	$(CC) -O2 -DBF16REF_GPU -c bf16ref.c -o bf16ref_emu_c.o
	$(CXX) -std=c++20 -O1 -DBFR_GPU_EMU -I. -x c++ -c bf16ref_gpu.cu -o bf16ref_emu_gpu.o
	$(CXX) -o $@ bf16ref_emu_c.o bf16ref_emu_gpu.o $(LDLIBS)

test: cmx cmxc bf16ref
	sh tests/roundtrip.sh ./cmxc
	sh tests/roundtrip.sh ./cmx
	sh tests/cmx.sh

# the GPU path against the CPU coder: on a GPU (make CUDA=1 test-gpu) or on the emulator (make test-gpu-emu)
test-gpu: bf16ref
	sh tests/gpu.sh ./bf16ref ./bf16ref

test-gpu-emu: bf16ref bf16ref_emu
	sh tests/gpu.sh ./bf16ref ./bf16ref_emu small

clean:
	rm -f cmx cmxc bf16ref bf16ref_emu *.o

.PHONY: all test test-gpu test-gpu-emu clean
