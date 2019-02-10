NVFLAGS=-std=c++11 -gencode arch=compute_52,code=sm_52 -O3 -Xcompiler -fopenmp -I../../include -DDATA_TYPE=float #-Xptxas="-v"
all:
	#nvcc main_coef_glb_x_no_halo.cu $(NVFLAGS) -o main_coef_glb_x_no_halo.out
	#nvcc main_coef_glb_no_halo.cu   $(NVFLAGS) -o main_coef_glb_no_halo.out
	#nvcc main_coef_reg_no_halo.cu   $(NVFLAGS) -o main_coef_reg_no_halo.out
	#nvcc main_coef_glb_x_halo.cu    $(NVFLAGS) -o main_coef_glb_x_halo.out
	#nvcc main_coef_glb_halo.cu      $(NVFLAGS) -o main_coef_glb_halo.out
	nvcc main_coef_reg_halo.cu      $(NVFLAGS) -o main_coef_reg_halo.out
                                   
clean:
	rm *.out
