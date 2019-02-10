#!/bin/bash -

#/opt/rocm/profiler/bin/CodeXLGpuProfilerRun -C -o hcc_s1d3_r_f.csv  bin/hcc_s1d3_r_f.out
#/opt/rocm/profiler/bin/CodeXLGpuProfilerRun -C -o hcc_s1d7_r_f.csv  bin/hcc_s1d7_r_f.out
#/opt/rocm/profiler/bin/CodeXLGpuProfilerRun -C -o hcc_s2d5_r_f.csv  bin/hcc_s2d5_r_f.out
#/opt/rocm/profiler/bin/CodeXLGpuProfilerRun -C -o hcc_s2d9_r_f.csv  bin/hcc_s2d9_r_f.out
#/opt/rocm/profiler/bin/CodeXLGpuProfilerRun -C -o hcc_s3d7_r_f.csv  bin/hcc_s3d7_r_f.out
#/opt/rocm/profiler/bin/CodeXLGpuProfilerRun -C -o hcc_s3d27_r_f.csv bin/hcc_s3d27_r_f.out

#/opt/rocm/profiler/bin/CodeXLGpuProfilerRun -C -o hcc_s1d3_r_d.csv  bin/hcc_s1d3_r_d.out
#/opt/rocm/profiler/bin/CodeXLGpuProfilerRun -C -o hcc_s1d7_r_d.csv  bin/hcc_s1d7_r_d.out
#/opt/rocm/profiler/bin/CodeXLGpuProfilerRun -C -o hcc_s2d5_r_d.csv  bin/hcc_s2d5_r_d.out
#/opt/rocm/profiler/bin/CodeXLGpuProfilerRun -C -o hcc_s2d9_r_d.csv  bin/hcc_s2d9_r_d.out
#/opt/rocm/profiler/bin/CodeXLGpuProfilerRun -C -o hcc_s3d7_r_d.csv  bin/hcc_s3d7_r_d.out
#/opt/rocm/profiler/bin/CodeXLGpuProfilerRun -C -o hcc_s3d27_r_d.csv bin/hcc_s3d27_r_d.out

#nvprof --print-gpu-trace bin/cu_s1d3_r_f.out  2>&1 | tee cu_s1d3_r_f.txt 
#nvprof --print-gpu-trace bin/cu_s1d7_r_f.out  2>&1 | tee cu_s1d7_r_f.txt 
#nvprof --print-gpu-trace bin/cu_s2d5_r_f.out  2>&1 | tee cu_s2d5_r_f.txt 
#nvprof --print-gpu-trace bin/cu_s2d9_r_f.out  2>&1 | tee cu_s2d9_r_f.txt 
#nvprof --print-gpu-trace bin/cu_s3d7_r_f.out  2>&1 | tee cu_s3d7_r_f.txt 
#nvprof --print-gpu-trace bin/cu_s3d27_r_f.out 2>&1 | tee cu_s3d27_r_f.txt 
#
#nvprof --print-gpu-trace bin/cu_s1d3_r_d.out  2>&1 | tee cu_s1d3_r_d.txt 
#nvprof --print-gpu-trace bin/cu_s1d7_r_d.out  2>&1 | tee cu_s1d7_r_d.txt 
#nvprof --print-gpu-trace bin/cu_s2d5_r_d.out  2>&1 | tee cu_s2d5_r_d.txt 
#nvprof --print-gpu-trace bin/cu_s2d9_r_d.out  2>&1 | tee cu_s2d9_r_d.txt 
#nvprof --print-gpu-trace bin/cu_s3d7_r_d.out  2>&1 | tee cu_s3d7_r_d.txt 
#nvprof --print-gpu-trace bin/cu_s3d27_r_d.out 2>&1 | tee cu_s3d27_r_d.txt

bin/hcc_s1d3_r_f.out
bin/hcc_s1d7_r_f.out
bin/hcc_s2d5_r_f.out
bin/hcc_s2d9_r_f.out
bin/hcc_s3d7_r_f.out
bin/hcc_s3d27_r_f.out
                      
bin/hcc_s1d3_r_d.out
bin/hcc_s1d7_r_d.out
bin/hcc_s2d5_r_d.out
bin/hcc_s2d9_r_d.out
bin/hcc_s3d7_r_d.out
bin/hcc_s3d27_r_d.out
