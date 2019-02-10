#include <iostream>
#include "metrics.h"
using namespace std;
// #define DATA_TYPE float
// #define DATA_TYPE double
#define warpSize 32 

#define IN_1D(_x) in[_x]
#define OUT_1D(_x) out[_x]
#define ARG_1D(_l,_x) args[(_l)*(n+2*halo)+(_x)]

// #define __DEBUG

#ifdef __DEBUG
#define ITER 1
#else
#define ITER 100
#endif

void Init_Args_1D(DATA_TYPE *args, int l, int n, int halo, DATA_TYPE val)
{
    for(int k = 0; k < l; k++)
    {
        for(int i = 0; i < n+2*halo; i++)
        {
            ARG_1D(k,i) = val; 
        }
    }
}

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


void Stencil_Seq(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int halo)
{
    for(int i = halo; i < n+halo; i++)
    {
        OUT_1D(i) = ARG_1D(0,i)*IN_1D(i-1) + 
                    ARG_1D(1,i)*IN_1D(i  ) + 
                    ARG_1D(2,i)*IN_1D(i+1) ;
    }
}

__global__ void Stencil_Cuda(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;  
    OUT_1D(i) = ARG_1D(0,i)*IN_1D(i-1) + 
                ARG_1D(1,i)*IN_1D(i  ) + 
                ARG_1D(2,i)*IN_1D(i+1) ;
}

__global__ void Stencil_Cuda_Sm(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int halo) 
{
    __shared__ DATA_TYPE local[256+2];
    unsigned int tid = threadIdx.x;
    // unsigned int lane_id = tid % warpSize;
    unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x + halo;  
    int local_id = tid + halo;
    local[local_id] = IN_1D(gid);
    if(tid == 0)
        local[local_id-halo] = IN_1D(gid-halo);
    if(tid == blockDim.x - 1)
        local[local_id+halo] = IN_1D(gid+halo);
    __syncthreads();

    OUT_1D(gid) = ARG_1D(0,gid)*local[local_id-1] + 
                  ARG_1D(1,gid)*local[local_id  ] + 
                  ARG_1D(2,gid)*local[local_id+1] ;
}

__global__ void Stencil_Cuda_Shfl(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int halo) 
{
    unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x + halo;  
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % warpSize;

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>5;

    DATA_TYPE threadInput0, threadInput1;
    int new_i = (warp_id_x<<5) + lane_id%34;
    threadInput0 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + (lane_id+32)%34;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    threadInput1 = IN_1D(new_i);

    DATA_TYPE sum0 = 0.0;
    int friend_id0;
    DATA_TYPE tx0, ty0;

    friend_id0 = (lane_id+0 )&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    sum0 += ARG_1D(0,gid)*tx0;

    friend_id0 = (lane_id+1 )&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    sum0 += ARG_1D(1,gid)*((lane_id < 31)? tx0: ty0);

    friend_id0 = (lane_id+2 )&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    sum0 += ARG_1D(2,gid)*((lane_id < 30)? tx0: ty0);

    OUT_1D(gid) = sum0; 
}

__global__ void Stencil_Cuda_Shfl2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int halo) 
{
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % warpSize;
    unsigned int gid = (((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<6) + lane_id + halo;  
    int warp_id_x = ((((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<6) + lane_id)>>5;

    DATA_TYPE threadInput0, threadInput1, threadInput2;
    int new_i = (warp_id_x<<5) + lane_id%34;
    threadInput0 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+32)/34)*34 + (lane_id+32)%34;
    threadInput1 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+64)/34)*34 + (lane_id+64)%34;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    threadInput2 = IN_1D(new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    int friend_id0;
    // int friend_id1;
    DATA_TYPE tx0, ty0, tx1, ty1;

    friend_id0 = (lane_id+0 )&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    tx1 = __shfl(threadInput1, friend_id0);
    sum0 += ARG_1D(0,gid   )*tx0;
    sum1 += ARG_1D(0,gid+32)*tx1;

    friend_id0 = (lane_id+1 )&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id0);
    ty1 = __shfl(threadInput2, friend_id0);
    sum0 += ARG_1D(1,gid   )*((lane_id < 31)? tx0: ty0);
    sum1 += ARG_1D(1,gid+32)*((lane_id < 31)? tx1: ty1);

    friend_id0 = (lane_id+2 )&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id0);
    ty1 = __shfl(threadInput2, friend_id0);
    sum0 += ARG_1D(2,gid   )*((lane_id < 30)? tx0: ty0);
    sum1 += ARG_1D(2,gid+32)*((lane_id < 30)? tx1: ty1);

    OUT_1D(gid   ) = sum0; 
    OUT_1D(gid+32) = sum1; 
}

__global__ void Stencil_Cuda_Shfl4(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int halo) 
{
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % warpSize;
    unsigned int gid = (((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<7) + lane_id + halo;  
    int warp_id_x = ((((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<7) + lane_id)>>5;

    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4; 
    int new_i = (warp_id_x<<5) + lane_id%34;
    threadInput0 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+32)/34)*34 + (lane_id+32)%34;
    threadInput1 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+64)/34)*34 + (lane_id+64)%34;
    threadInput2 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+96)/34)*34 + (lane_id+96)%34;
    threadInput3 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+128)/34)*34 + (lane_id+128)%34;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    threadInput4 = IN_1D(new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    int friend_id0;
    // int friend_id1;
    DATA_TYPE tx0, ty0, tx1, ty1;
    DATA_TYPE tx2, ty2, tx3, ty3;

    friend_id0 = (lane_id+0 )&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    tx1 = __shfl(threadInput1, friend_id0);
    tx2 = __shfl(threadInput2, friend_id0);
    tx3 = __shfl(threadInput3, friend_id0);
    sum0 += ARG_1D(0,gid   )*tx0;
    sum1 += ARG_1D(0,gid+32)*tx1;
    sum2 += ARG_1D(0,gid+64)*tx2;
    sum3 += ARG_1D(0,gid+96)*tx3;

    friend_id0 = (lane_id+1 )&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id0);
    ty1 = __shfl(threadInput2, friend_id0);
    tx2 = __shfl(threadInput2, friend_id0);
    ty2 = __shfl(threadInput3, friend_id0);
    tx3 = __shfl(threadInput3, friend_id0);
    ty3 = __shfl(threadInput4, friend_id0);
    sum0 += ARG_1D(1,gid   )*((lane_id < 31)? tx0: ty0);
    sum1 += ARG_1D(1,gid+32)*((lane_id < 31)? tx1: ty1);
    sum2 += ARG_1D(1,gid+64)*((lane_id < 31)? tx2: ty2);
    sum3 += ARG_1D(1,gid+96)*((lane_id < 31)? tx3: ty3);

    friend_id0 = (lane_id+2 )&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id0);
    ty1 = __shfl(threadInput2, friend_id0);
    tx2 = __shfl(threadInput2, friend_id0);
    ty2 = __shfl(threadInput3, friend_id0);
    tx3 = __shfl(threadInput3, friend_id0);
    ty3 = __shfl(threadInput4, friend_id0);
    sum0 += ARG_1D(2,gid   )*((lane_id < 30)? tx0: ty0);
    sum1 += ARG_1D(2,gid+32)*((lane_id < 30)? tx1: ty1);
    sum2 += ARG_1D(2,gid+64)*((lane_id < 30)? tx2: ty2);
    sum3 += ARG_1D(2,gid+96)*((lane_id < 30)? tx3: ty3);

    OUT_1D(gid   ) = sum0; 
    OUT_1D(gid+32) = sum1; 
    OUT_1D(gid+64) = sum2; 
    OUT_1D(gid+96) = sum3; 
}

__global__ void Stencil_Cuda_Shfl8(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int halo) 
{
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % warpSize;
    unsigned int gid = (((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<8) + lane_id + halo;  
    int warp_id_x = ((((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<8) + lane_id)>>5;

    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5,
              threadInput6, threadInput7, threadInput8;
    int new_i = (warp_id_x<<5) + lane_id%34;
    threadInput0 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+32)/34)*34 + (lane_id+32)%34;
    threadInput1 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+64)/34)*34 + (lane_id+64)%34;
    threadInput2 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+96)/34)*34 + (lane_id+96)%34;
    threadInput3 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+128)/34)*34 + (lane_id+128)%34;
    threadInput4 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+160)/34)*34 + (lane_id+160)%34;
    threadInput5 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+192)/34)*34 + (lane_id+192)%34;
    threadInput6 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+224)/34)*34 + (lane_id+224)%34;
    threadInput7 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+256)/34)*34 + (lane_id+256)%34;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    threadInput8 = IN_1D(new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    DATA_TYPE sum4 = 0.0;
    DATA_TYPE sum5 = 0.0;
    DATA_TYPE sum6 = 0.0;
    DATA_TYPE sum7 = 0.0;
    int friend_id0;
    // int friend_id1;
    DATA_TYPE tx0, ty0, tx1, ty1;
    DATA_TYPE tx2, ty2, tx3, ty3;
    DATA_TYPE tx4, ty4, tx5, ty5;
    DATA_TYPE tx6, ty6, tx7, ty7;

    friend_id0 = (lane_id+0 )&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    tx1 = __shfl(threadInput1, friend_id0);
    tx2 = __shfl(threadInput2, friend_id0);
    tx3 = __shfl(threadInput3, friend_id0);
    tx4 = __shfl(threadInput4, friend_id0);
    tx5 = __shfl(threadInput5, friend_id0);
    tx6 = __shfl(threadInput6, friend_id0);
    tx7 = __shfl(threadInput7, friend_id0);
    sum0 += ARG_1D(0,gid    )*tx0;
    sum1 += ARG_1D(0,gid+32 )*tx1;
    sum2 += ARG_1D(0,gid+64 )*tx2;
    sum3 += ARG_1D(0,gid+96 )*tx3;
    sum4 += ARG_1D(0,gid+128)*tx4;
    sum5 += ARG_1D(0,gid+160)*tx5;
    sum6 += ARG_1D(0,gid+192)*tx6;
    sum7 += ARG_1D(0,gid+224)*tx7;

    friend_id0 = (lane_id+1 )&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id0);
    ty1 = __shfl(threadInput2, friend_id0);
    tx2 = __shfl(threadInput2, friend_id0);
    ty2 = __shfl(threadInput3, friend_id0);
    tx3 = __shfl(threadInput3, friend_id0);
    ty3 = __shfl(threadInput4, friend_id0);
    tx4 = __shfl(threadInput4, friend_id0);
    ty4 = __shfl(threadInput5, friend_id0);
    tx5 = __shfl(threadInput5, friend_id0);
    ty5 = __shfl(threadInput6, friend_id0);
    tx6 = __shfl(threadInput6, friend_id0);
    ty6 = __shfl(threadInput7, friend_id0);
    tx7 = __shfl(threadInput7, friend_id0);
    ty7 = __shfl(threadInput8, friend_id0);
    sum0 += ARG_1D(1,gid    )*((lane_id < 31)? tx0: ty0);
    sum1 += ARG_1D(1,gid+32 )*((lane_id < 31)? tx1: ty1);
    sum2 += ARG_1D(1,gid+64 )*((lane_id < 31)? tx2: ty2);
    sum3 += ARG_1D(1,gid+96 )*((lane_id < 31)? tx3: ty3);
    sum4 += ARG_1D(1,gid+128)*((lane_id < 31)? tx4: ty4);
    sum5 += ARG_1D(1,gid+160)*((lane_id < 31)? tx5: ty5);
    sum6 += ARG_1D(1,gid+192)*((lane_id < 31)? tx6: ty6);
    sum7 += ARG_1D(1,gid+224)*((lane_id < 31)? tx7: ty7);

    friend_id0 = (lane_id+2 )&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id0);
    ty1 = __shfl(threadInput2, friend_id0);
    tx2 = __shfl(threadInput2, friend_id0);
    ty2 = __shfl(threadInput3, friend_id0);
    tx3 = __shfl(threadInput3, friend_id0);
    ty3 = __shfl(threadInput4, friend_id0);
    tx4 = __shfl(threadInput4, friend_id0);
    ty4 = __shfl(threadInput5, friend_id0);
    tx5 = __shfl(threadInput5, friend_id0);
    ty5 = __shfl(threadInput6, friend_id0);
    tx6 = __shfl(threadInput6, friend_id0);
    ty6 = __shfl(threadInput7, friend_id0);
    tx7 = __shfl(threadInput7, friend_id0);
    ty7 = __shfl(threadInput8, friend_id0);
    sum0 += ARG_1D(2,gid    )*((lane_id < 30)? tx0: ty0);
    sum1 += ARG_1D(2,gid+32 )*((lane_id < 30)? tx1: ty1);
    sum2 += ARG_1D(2,gid+64 )*((lane_id < 30)? tx2: ty2);
    sum3 += ARG_1D(2,gid+96 )*((lane_id < 30)? tx3: ty3);
    sum4 += ARG_1D(2,gid+128)*((lane_id < 30)? tx4: ty4);
    sum5 += ARG_1D(2,gid+160)*((lane_id < 30)? tx5: ty5);
    sum6 += ARG_1D(2,gid+192)*((lane_id < 30)? tx6: ty6);
    sum7 += ARG_1D(2,gid+224)*((lane_id < 30)? tx7: ty7);

    OUT_1D(gid    ) = sum0; 
    OUT_1D(gid+32 ) = sum1; 
    OUT_1D(gid+64 ) = sum2; 
    OUT_1D(gid+96 ) = sum3; 
    OUT_1D(gid+128) = sum4; 
    OUT_1D(gid+160) = sum5; 
    OUT_1D(gid+192) = sum6; 
    OUT_1D(gid+224) = sum7; 
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
    int K = total*3;
    DATA_TYPE *args = new DATA_TYPE[K];
#ifdef __DEBUG
    Init_Args_1D(args, 3, n, halo, 1.0);
#else
    Init_Args_1D(args, 3, n, halo, 0.33);
#endif
    DATA_TYPE *in = new DATA_TYPE[total];
    DATA_TYPE *out_ref = new DATA_TYPE[total];
    unsigned int seed = time(NULL);
    Init_Input_1D(in, n, halo, seed);

    // Show_Me(in, n, halo, "Input:");
    for(int i=0; i< ITER; i++)
    {
        Stencil_Seq(in, out_ref, args, n, halo);
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
    DATA_TYPE *args_d;
    DATA_TYPE *out_d;
    DATA_TYPE *out = new DATA_TYPE[total];
    cudaMalloc((void**)&in_d, total*sizeof(DATA_TYPE));
    cudaMalloc((void**)&out_d, total*sizeof(DATA_TYPE));
    cudaMalloc((void**)&args_d, (K)*sizeof(DATA_TYPE));
    cudaMemcpy(args_d, args, (K)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    
    // Cuda version
    /////////////////////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid((n)/256, 1, 1);
    dim3 dimBlock(256, 1, 1);

    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda<<<dimGrid, dimBlock>>>(in_d, out_d, args_d, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, n, halo,  "Output(Device):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(3, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shared Memory version 
    /////////////////////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid1((n)/256, 1, 1);
    dim3 dimBlock1(256, 1, 1);

    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Sm<<<dimGrid1, dimBlock1>>>(in_d, out_d, args_d, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, n, halo,  "Output(Device):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sm: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(3, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl version 
    /////////////////////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid2((n)/256, 1, 1);
    dim3 dimBlock2(256, 1, 1);

    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl<<<dimGrid2, dimBlock2>>>(in_d, out_d, args_d, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, n, halo,  "Output(Device):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(3, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl2 version 
    /////////////////////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid3((n)/(256*2), 1, 1);
    dim3 dimBlock3(256, 1, 1);

    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl2<<<dimGrid3, dimBlock3>>>(in_d, out_d, args_d, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, n, halo,  "Output(Device):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl2: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(3, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl4 version 
    /////////////////////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid4((n)/(256*4), 1, 1);
    dim3 dimBlock4(256, 1, 1);

    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl4<<<dimGrid4, dimBlock4>>>(in_d, out_d, args_d, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, n, halo,  "Output(Device):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl4: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(3, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl8 version 
    /////////////////////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid5((n)/(256*8), 1, 1);
    dim3 dimBlock5(256, 1, 1);

    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl8<<<dimGrid5, dimBlock5>>>(in_d, out_d, args_d, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, n, halo,  "Output(Device):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl8: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(3, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));


    cudaFree(in_d);
    cudaFree(out_d);

    delete[] in;
    delete[] out;
    delete[] out_ref;
}

