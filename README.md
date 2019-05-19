## Auto gen code
The codegen/codegen.cpp is the auto-gen codes to generate shuffle operations,
user need to change the global variables in the source code to specify different
stencils.

## Compile HCC stencils
To compile HCC codes, just use the first 12 compiler statements in the HCC
environ (e.g. t1 of hsalogin).

## Compile CUDA stencils 
To compile CUDA codes, just use the last 12 compiler statements in the CUDA
environ (e.g. t2 of hsalogin).

## Run codes
Please refer to autorun.sh, which shows how to run the codes both in CUDA or 
HCC versions.

## About this work
Hou, Kaixi, Hao Wang, and Wu-chun Feng. "Gpu-unicache: Automatic code generation of spatial blocking for stencils on gpus." In Proceedings of the Computing Frontiers Conference, pp. 107-116. ACM, 2017.
