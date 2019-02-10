CODEGEN=code_gen
STEN1=s1d3_r
STEN2=s1d7_r
STEN3=s2d5_r
STEN4=s2d9_r
STEN5=s3d7_r
STEN6=s3d27_r


MACROFLAGS=-DITER=100
#MACROFLAGS=-DDATA_TYPE=float -DITER=1 -D__DEBUG

# HCC compiler 
HCCVER  =hcc-lc
HCCPATH =/opt/rocm/$(HCCVER)/bin
HCCOPTS = $(shell $(HCCPATH)/hcc-config --install --cxxflags --ldflags)
HCCFLAGS=$(HCCOPTS) -Iinclude -fopenmp $(MACROFLAGS) 

# CUDA compiler 
NVFLAGS=-std=c++11 -gencode arch=compute_52,code=sm_52 -O3 -Xcompiler -fopenmp -Iinclude $(MACROFLAGS) #-Xptxas="-v" 

all:
	g++ -std=c++11 -O3 codegen/$(CODEGEN).cpp -o bin/$(CODEGEN).out
	$(HCCPATH)/hcc $(HCCFLAGS) -DDATA_TYPE=float hcc/$(STEN6).cpp -o bin/hcc_$(STEN6)_f.out
	$(HCCPATH)/hcc $(HCCFLAGS) -DDATA_TYPE=double hcc/$(STEN6).cpp -o bin/hcc_$(STEN6)_d.out
	$(HCCPATH)/hcc $(HCCFLAGS) -DDATA_TYPE=float hcc/$(STEN5).cpp -o bin/hcc_$(STEN5)_f.out
	$(HCCPATH)/hcc $(HCCFLAGS) -DDATA_TYPE=double hcc/$(STEN5).cpp -o bin/hcc_$(STEN5)_d.out
	$(HCCPATH)/hcc $(HCCFLAGS) -DDATA_TYPE=float hcc/$(STEN4).cpp -o bin/hcc_$(STEN4)_f.out
	$(HCCPATH)/hcc $(HCCFLAGS) -DDATA_TYPE=double hcc/$(STEN4).cpp -o bin/hcc_$(STEN4)_d.out
	$(HCCPATH)/hcc $(HCCFLAGS) -DDATA_TYPE=float hcc/$(STEN3).cpp -o bin/hcc_$(STEN3)_f.out
	$(HCCPATH)/hcc $(HCCFLAGS) -DDATA_TYPE=double hcc/$(STEN3).cpp -o bin/hcc_$(STEN3)_d.out
	$(HCCPATH)/hcc $(HCCFLAGS) -DDATA_TYPE=float hcc/$(STEN2).cpp -o bin/hcc_$(STEN2)_f.out
	$(HCCPATH)/hcc $(HCCFLAGS) -DDATA_TYPE=double hcc/$(STEN2).cpp -o bin/hcc_$(STEN2)_d.out
	$(HCCPATH)/hcc $(HCCFLAGS) -DDATA_TYPE=float hcc/$(STEN1).cpp -o bin/hcc_$(STEN1)_f.out
	$(HCCPATH)/hcc $(HCCFLAGS) -DDATA_TYPE=double hcc/$(STEN1).cpp -o bin/hcc_$(STEN1)_d.out
	#nvcc -DDATA_TYPE=float cuda/$(STEN6).cu $(NVFLAGS) -o bin/cu_$(STEN6)_f.out
	#nvcc -DDATA_TYPE=double cuda/$(STEN6).cu $(NVFLAGS) -o bin/cu_$(STEN6)_d.out
	#nvcc -DDATA_TYPE=float cuda/$(STEN5).cu $(NVFLAGS) -o bin/cu_$(STEN5)_f.out
	#nvcc -DDATA_TYPE=double cuda/$(STEN5).cu $(NVFLAGS) -o bin/cu_$(STEN5)_d.out
	#nvcc -DDATA_TYPE=float cuda/$(STEN4).cu $(NVFLAGS) -o bin/cu_$(STEN4)_f.out
	#nvcc -DDATA_TYPE=double cuda/$(STEN4).cu $(NVFLAGS) -o bin/cu_$(STEN4)_d.out
	#nvcc -DDATA_TYPE=float cuda/$(STEN3).cu $(NVFLAGS) -o bin/cu_$(STEN3)_f.out
	#nvcc -DDATA_TYPE=double cuda/$(STEN3).cu $(NVFLAGS) -o bin/cu_$(STEN3)_d.out
	#nvcc -DDATA_TYPE=float cuda/$(STEN2).cu $(NVFLAGS) -o bin/cu_$(STEN2)_f.out
	#nvcc -DDATA_TYPE=double cuda/$(STEN2).cu $(NVFLAGS) -o bin/cu_$(STEN2)_d.out
	#nvcc -DDATA_TYPE=float cuda/$(STEN1).cu $(NVFLAGS) -o bin/cu_$(STEN1)_f.out
	#nvcc -DDATA_TYPE=double cuda/$(STEN1).cu $(NVFLAGS) -o bin/cu_$(STEN1)_d.out

clean:
	rm bin/*.out
