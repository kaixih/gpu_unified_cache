APPNAME = main_1d1
#APPNAME = main_1d7
#APPNAME = main_2d5
#APPNAME = main_2d9
all:
	nvcc ${APPNAME}.cu -std=c++11 -gencode arch=compute_52,code=sm_52 -O3 -o ${APPNAME}.out
	#nvcc main.cu -o main_cu.out -std=c++11 -gencode arch=compute_30,code=sm_30

clean:
	rm main_*.out
