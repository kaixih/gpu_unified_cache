#include <iostream>
#include "sten_metrics.h"
using namespace std;
// #define DATA_TYPE float
// #define DATA_TYPE double
#define warpSize 32 

// #define __DEBUG

// #ifdef __DEBUG
// #define ITER 1
// #else
// #define ITER 100
// #endif

void Init_Input_1D(DATA_TYPE *in, int n, int halo, unsigned int seed)
{
    srand(seed);
    for(int i = 0; i < n+2*halo; i++)
    {
        if(i < halo || i >= n+halo)
            IN_1D(i) = 0.0;
        else
#ifdef __DEBUG
            IN_1D(i) = 1.0; 
                // IN_2D(i,j) = (DATA_TYPE)rand()*100.0 / ((long)RAND_MAX);
#else
            IN_1D(i) = (DATA_TYPE)rand()*10.0 / ((long)RAND_MAX);
#endif
    }
}

void Clear_Output_1D(DATA_TYPE *in, int n, int halo)
{
    for(int i = 0; i < n+2*halo; i++)
    {
        IN_1D(i) = 0.0;
    }
}

void Show_Me(DATA_TYPE *in, int n, int halo, string prompt)
{
    cout << prompt << endl;
    for(int i = 0; i < n+2*halo; i++)
    {
        std::cout << IN_1D(i) << ",";
    }
    std::cout << std::endl;
}

inline double tol_finder(int error_tol)
{
    double val = 1.0;
    for(; error_tol > 0; error_tol--)
        val *= 10;
    return 1.0/(double)val;
}

bool Verify(DATA_TYPE *test, DATA_TYPE *ref, int n)
{
    bool flag = true;
    double precision = tol_finder(2);

    for(int i = 0; i < n; i++)
    {
        if(fabs(test[i]-ref[i]) > precision)
        {
            std::cout << "wrong at " << i << " test:" << test[i] << " (ref: " << ref[i] << ")";
            std::cout << std::endl;
            flag = false;
            break;
        }
    }
    return flag;
}


void Stencil_Seq(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, int n, int halo)
{
#pragma omp parallel for
    for(int i = halo; i < n+halo; i++)
    {
        OUT_1D(i) = a0*IN_1D(i-1) + 
                    a1*IN_1D(i  ) + 
                    a2*IN_1D(i+1) ;
    }
}

__global__ void Stencil_Cuda_L1_1Blk(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;  
    OUT_1D(i) = a0*IN_1D(i-1) + 
                a1*IN_1D(i  ) + 
                a2*IN_1D(i+1) ;
}

__global__ void Stencil_Cuda_Lds_1BlkBrc(DATA_TYPE *in, DATA_TYPE *out, 
        DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, int n, int halo) 
{
    __shared__ DATA_TYPE local[256+2];
    unsigned int tid = threadIdx.x;
    unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x + halo;  
    int local_id = tid + halo;

    local[local_id] = IN_1D(gid);
    if(tid == 0)
        local[local_id-1] = IN_1D(gid-1);
    if(tid == blockDim.x - 1)
        local[local_id+1] = IN_1D(gid+1);
    __syncthreads();

    OUT_1D(gid) = a0*local[local_id-1] + 
                  a1*local[local_id  ] + 
                  a2*local[local_id+1] ;
}

__global__ void Stencil_Cuda_Lds_1BlkCyc(DATA_TYPE *in, DATA_TYPE *out, 
        DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, int n, int halo) 
{
    __shared__ DATA_TYPE local[256+2];
    unsigned int tid = threadIdx.x;
    unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x + halo;  
    int local_id = tid + halo;

    unsigned int lane_id = threadIdx.x;
    int lane_id_it = lane_id;
    int blk_id_x = blockIdx.x;
    int new_i  = (blk_id_x<<8) + lane_id_it%258;
    int new_li = lane_id_it%258;
    local[new_li] = IN_1D(new_i);
    lane_id_it += 256;
    new_i  = (blk_id_x<<8) + (lane_id_it/258)*258 + lane_id_it%258;
    new_li = (lane_id_it/258)*258 + lane_id_it%258;
    if(new_li < 258)
        local[new_li] = IN_1D(new_i);

    __syncthreads();

    OUT_1D(gid) = a0*local[local_id-1] + 
                  a1*local[local_id  ] + 
                  a2*local[local_id+1] ;
}

__global__ void Stencil_Cuda_Reg1_1Blk1Wf(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, int n, int halo) 
{
    unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x + halo;  
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % warpSize;

    int warp_id0 = (threadIdx.x + blockIdx.x * blockDim.x)>>5;

    DATA_TYPE reg0, reg1;
    int lane_id_it = lane_id;
    int new_id0 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    reg0 = IN_1D(new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
    reg1 = IN_1D(new_id0) ;

    DATA_TYPE sum0 = 0.0;
    int friend_id0;
    DATA_TYPE tx0, ty0;

    // process (0, 0, 0)
    friend_id0 = (lane_id+ 0)&31 ;
    tx0 = __shfl(reg0, friend_id0);
    sum0 += a0*(tx0);
    // process (1, 0, 0)
    friend_id0 = (lane_id+ 1)&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a1*((lane_id < 31 )? tx0: ty0);
    // process (2, 0, 0)
    friend_id0 = (lane_id+ 2)&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a2*((lane_id < 30 )? tx0: ty0);
    
    OUT_1D(gid) = sum0; 
}

__global__ void Stencil_Cuda_Reg2_1Blk1Wf(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, int n, int halo) 
{
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % warpSize;
    unsigned int gid = (((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<6) + lane_id + halo;  
    int warp_id0 = ((((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<6) + lane_id)>>5;

    DATA_TYPE reg0, reg1, reg2;
    int lane_id_it = lane_id;
    int new_id0 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    reg0 = IN_1D(new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + ((lane_id_it)/34)*34 + lane_id_it%34 ;
    reg1 = IN_1D(new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + ((lane_id_it)/34)*34 + lane_id_it%34 ;
    new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
    reg2 = IN_1D(new_id0) ;

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    int friend_id0;
    int friend_id1;
    DATA_TYPE tx0, ty0, tx1, ty1;

    // process (0, 0, 0)
    friend_id0 = (lane_id+ 0)&31 ;
    tx0 = __shfl(reg0, friend_id0);
    sum0 += a0*(tx0);
    friend_id1 = (lane_id+ 0)&31 ;
    tx1 = __shfl(reg1, friend_id1);
    sum1 += a0*(tx1);
    // process (1, 0, 0)
    friend_id0 = (lane_id+ 1)&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a1*((lane_id < 31 )? tx0: ty0);
    friend_id1 = (lane_id+ 1)&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a1*((lane_id < 31 )? tx1: ty1);
    // process (2, 0, 0)
    friend_id0 = (lane_id+ 2)&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a2*((lane_id < 30 )? tx0: ty0);
    friend_id1 = (lane_id+ 2)&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a2*((lane_id < 30 )? tx1: ty1);

    OUT_1D(gid   ) = sum0; 
    OUT_1D(gid+32) = sum1; 
}

__global__ void Stencil_Cuda_Reg4_1Blk1Wf(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, int n, int halo) 
{
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % warpSize;
    unsigned int gid = (((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<7) + lane_id + halo;  
    int warp_id0 = ((((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<7) + lane_id)>>5;


    DATA_TYPE reg0, reg1, reg2, reg3, reg4; 
    int lane_id_it = lane_id;
    int new_id0 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    reg0 = IN_1D(new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + (lane_id_it/34)*34 + lane_id_it%34 ;
    reg1 = IN_1D(new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + (lane_id_it/34)*34 + lane_id_it%34 ;
    reg2 = IN_1D(new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + (lane_id_it/34)*34 + lane_id_it%34 ;
    reg3 = IN_1D(new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + (lane_id_it/34)*34 + lane_id_it%34 ;
    new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
    reg4 = IN_1D(new_id0) ;

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    int friend_id0;
    int friend_id1;
    int friend_id2;
    int friend_id3;
    DATA_TYPE tx0, ty0, tx1, ty1;
    DATA_TYPE tx2, ty2, tx3, ty3;

    // process (0, 0, 0)
    friend_id0 = (lane_id+ 0)&31 ;
    tx0 = __shfl(reg0, friend_id0);
    sum0 += a0 *(tx0);
    friend_id1 = (lane_id+ 0)&31 ;
    tx1 = __shfl(reg1, friend_id1);
    sum1 += a0 *(tx1);
    friend_id2 = (lane_id+ 0)&31 ;
    tx2 = __shfl(reg2, friend_id2);
    sum2 += a0 *(tx2);
    friend_id3 = (lane_id+ 0)&31 ;
    tx3 = __shfl(reg3, friend_id3);
    sum3 += a0 *(tx3);
    // process (1, 0, 0)
    friend_id0 = (lane_id+ 1)&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a1 *((lane_id < 31 )? tx0: ty0);
    friend_id1 = (lane_id+ 1)&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a1 *((lane_id < 31 )? tx1: ty1);
    friend_id2 = (lane_id+ 1)&31 ;
    tx2 = __shfl(reg2, friend_id2);
    ty2 = __shfl(reg3, friend_id2);
    sum2 += a1 *((lane_id < 31 )? tx2: ty2);
    friend_id3 = (lane_id+ 1)&31 ;
    tx3 = __shfl(reg3, friend_id3);
    ty3 = __shfl(reg4, friend_id3);
    sum3 += a1 *((lane_id < 31 )? tx3: ty3);
    // process (2, 0, 0)
    friend_id0 = (lane_id+ 2)&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a2 *((lane_id < 30 )? tx0: ty0);
    friend_id1 = (lane_id+ 2)&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a2 *((lane_id < 30 )? tx1: ty1);
    friend_id2 = (lane_id+ 2)&31 ;
    tx2 = __shfl(reg2, friend_id2);
    ty2 = __shfl(reg3, friend_id2);
    sum2 += a2 *((lane_id < 30 )? tx2: ty2);
    friend_id3 = (lane_id+ 2)&31 ;
    tx3 = __shfl(reg3, friend_id3);
    ty3 = __shfl(reg4, friend_id3);
    sum3 += a2 *((lane_id < 30 )? tx3: ty3);
    
    OUT_1D(gid   ) = sum0; 
    OUT_1D(gid+32) = sum1; 
    OUT_1D(gid+64) = sum2; 
    OUT_1D(gid+96) = sum3; 
}

int main(int argc, char **argv)
{
#ifdef __DEBUG
    int n = 512;
#else
    int n = 33554432; // 2^25
#endif
    int halo = 1; 
    int total = (n+2*halo);
    const int K = 3;
#ifdef __DEBUG
    DATA_TYPE args[K] = {1.0, 1.0, 1.0};
#else
    DATA_TYPE args[K] = {0.33, 0.33, 0.33};
#endif
    DATA_TYPE *in = new DATA_TYPE[total];
    DATA_TYPE *out_ref = new DATA_TYPE[total];
    unsigned int seed = time(NULL);
    Clear_Output_1D(in, n, halo);
    Clear_Output_1D(out_ref, n, halo);
    Init_Input_1D(in, n, halo, seed);

    // Show_Me(in, n, halo, "Input:");
    for(int i=0; i< ITER; i++)
    {
        Stencil_Seq(in, out_ref, args[0], args[1], args[2], n, halo);
        swap(in, out_ref);
    }
    swap(in, out_ref);
    // Show_Me(out_ref, n, halo, "Output:");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    float time_wo_pci;

    DATA_TYPE *in_d;
    DATA_TYPE *out_d;
    DATA_TYPE *out = new DATA_TYPE[total];
    cudaMalloc((void**)&in_d, total*sizeof(DATA_TYPE));
    cudaMalloc((void**)&out_d, total*sizeof(DATA_TYPE));
    dim3 dimGrid;
    dim3 dimBlock;
    
    // Cuda version
    /////////////////////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/256;
    dimGrid.y = 1;
    dimGrid.z = 1;
    dimBlock.x = 256;
    dimBlock.y = 1;
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_L1_1Blk<<<dimGrid, dimBlock>>>(in_d, out_d, args[0], args[1], args[2] , n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_L1_1Blk: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_L1_1Blk Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shared Memory with Branch
    /////////////////////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/256;
    dimGrid.y = 1;
    dimGrid.z = 1;
    dimBlock.x = 256;
    dimBlock.y = 1;
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Lds_1BlkBrc<<<dimGrid, dimBlock>>>(in_d, out_d, args[0], args[1], args[2] , n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Lds_1BlkBrc: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Lds_1BlkBrc Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shared Memory with Cyclic
    /////////////////////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/256;
    dimGrid.y = 1;
    dimGrid.z = 1;
    dimBlock.x = 256;
    dimBlock.y = 1;
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Lds_1BlkCyc<<<dimGrid, dimBlock>>>(in_d, out_d, args[0], args[1], args[2] , n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Lds_1BlkCyc: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Lds_1BlkCyc Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl 1D-Warp 
    /////////////////////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/256;
    dimGrid.y = 1;
    dimGrid.z = 1;
    dimBlock.x = 256;
    dimBlock.y = 1;
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Reg1_1Blk1Wf<<<dimGrid, dimBlock>>>(in_d, out_d, args[0], args[1], args[2] , n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Reg1_1Blk1Wf: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Reg1_1Blk1Wf Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl2 1D-Warp 
    /////////////////////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/(256*2);
    dimGrid.y = 1;
    dimGrid.z = 1;
    dimBlock.x = 256;
    dimBlock.y = 1;
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Reg2_1Blk1Wf<<<dimGrid, dimBlock>>>(in_d, out_d, args[0], args[1], args[2] , n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Reg2_1Blk1Wf: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Reg2_1Blk1Wf Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl4 1D-Warp 
    /////////////////////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/(256*4);
    dimGrid.y = 1;
    dimGrid.z = 1;
    dimBlock.x = 256;
    dimBlock.y = 1;
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Reg4_1Blk1Wf<<<dimGrid, dimBlock>>>(in_d, out_d, args[0], args[1], args[2] , n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Reg4_1Blk1Wf: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Reg4_1Blk1Wf Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    cudaFree(in_d);
    cudaFree(out_d);

    delete[] in;
    delete[] out;
    delete[] out_ref;
}

