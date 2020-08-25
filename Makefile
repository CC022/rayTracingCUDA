CUDA_PATH ?= /usr/local/cuda
HOST_COMPILER = g++
NVCC = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

SRCS = main.cu
# INCS = vec3.h

rayTracing: main.o
	$(NVCC) -o rayTracing main.o

main.o: $(SRCS) $(INCS)
	$(NVCC) -o main.o -c main.cu

img.ppm: rayTracing
	rm -f img.ppm
	./rayTracing

img: img.ppm rayTracing
	rm -f img.jpg
	ppmtojpeg img.ppm > img.jpg

clean:
	rm -f rayTracing main.o img.ppm img.jpg