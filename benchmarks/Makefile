NVFLAGS=-std=c++11 -gencode arch=compute_52,code=sm_52 -O3 -DDATA_TYPE=double
all:
	nvcc shfl_vs_sm.cu      $(NVFLAGS) -o shfl_vs_sm.out
                                   
clean:
	rm *.out
