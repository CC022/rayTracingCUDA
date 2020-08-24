CUDA_PATH ?= /usr/local/cuda
HOST_COMPILER = g++
NVCC = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

SRCS = main.cu
# INCS = vec3.h

rayTracing: main.o
	$(NVCC) -o rayTracing main.o

main.o: $(SRCS) $(INCS)
	$(NVCC) -o main.o -c main.cu

PPM.ppm: rayTracing
	rm -f PPM.ppm
	./rayTracing > PPM.ppm

JPG.jpg: PPM.ppm
	rm -f JPG.jpg
	ppmtojpeg PPM.ppm > JPG.jpg

clean:
	rm -f rayTracing main.o PPM.ppm JPG.jpg