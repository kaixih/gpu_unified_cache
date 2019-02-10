MACROFLAGS=-DDATA_TYPE=float -DITER=100
#MACROFLAGS=-DDATA_TYPE=float -DITER=1 -D__DEBUG

# CUDA compiler 
NVFLAGS=-std=c++11 -gencode arch=compute_52,code=sm_52 -O3 -Xcompiler -fopenmp -Iinclude $(MACROFLAGS) #-Xptxas="-v" 

# HCC compiler 
HCCVER  =hcc-lc
HCCPATH =/opt/rocm/$(HCCVER)/bin
HCCOPTS = $(shell $(HCCPATH)/hcc-config --install --cxxflags --ldflags)
HCCFLAGS=$(HCCOPTS) -Iinclude -fopenmp $(MACROFLAGS) 

all:
	#g++ -std=c++11 -O3 code_gen.cpp -o code_gen.out
	#nvcc src/cuda_3d7.cu $(NVFLAGS) -o cuda_3d7.out
	$(HCCPATH)/hcc $(HCCFLAGS) src/hcc_3d7.cpp -o hcc_3d7.out
                                   
clean:
	rm *.out
