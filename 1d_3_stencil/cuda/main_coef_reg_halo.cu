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

__global__ void Stencil_Cuda(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;  
    OUT_1D(i) = a0*IN_1D(i-1) + 
                a1*IN_1D(i  ) + 
                a2*IN_1D(i+1) ;
}

__global__ void Stencil_Cuda_Sm_Branch(DATA_TYPE *in, DATA_TYPE *out, 
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

__global__ void Stencil_Cuda_Sm_Cyclic(DATA_TYPE *in, DATA_TYPE *out, 
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

__global__ void Stencil_Cuda_Shfl(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, int n, int halo) 
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

    /*
    friend_id0 = (lane_id+0 )&(warpSize-1);
    tx0 = threadInput0;
    ty0 = __shfl(tx0, friend_id0);
    sum0 += a0*ty0;

    friend_id0 = (lane_id+1 )&(warpSize-1);
    tx0 = (lane_id > 0)? threadInput0: threadInput1;
    ty0 = __shfl(tx0, friend_id0);
    sum0 += a1*ty0;

    friend_id0 = (lane_id+2 )&(warpSize-1);
    tx0 = (lane_id > 1)? threadInput0: threadInput1;
    ty0 = __shfl(tx0, friend_id0);
    sum0 += a2*ty0;
    */

    friend_id0 = (lane_id+0 )&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    sum0 += a0*tx0;

    friend_id0 = (lane_id+1 )&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    sum0 += a1*((lane_id < 31)? tx0: ty0);

    friend_id0 = (lane_id+2 )&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    sum0 += a2*((lane_id < 30)? tx0: ty0);

    OUT_1D(gid) = sum0; 
}

__global__ void Stencil_Cuda_Shfl2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, int n, int halo) 
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

    /*
    friend_id0 = (lane_id+0 )&(warpSize-1);
    tx0 = threadInput0;
    tx1 = threadInput1;
    ty0 = __shfl(tx0, friend_id0);
    ty1 = __shfl(tx1, friend_id0);
    sum0 += a0*ty0;
    sum1 += a0*ty1;

    friend_id0 = (lane_id+1 )&(warpSize-1);
    tx0 = (lane_id > 0)? threadInput0: threadInput1;
    tx1 = (lane_id > 0)? threadInput1: threadInput2;
    ty0 = __shfl(tx0, friend_id0);
    ty1 = __shfl(tx1, friend_id0);
    sum0 += a1*ty0;
    sum1 += a1*ty1;

    friend_id0 = (lane_id+2 )&(warpSize-1);
    tx0 = (lane_id > 1)? threadInput0: threadInput1;
    tx1 = (lane_id > 1)? threadInput1: threadInput2;
    ty0 = __shfl(tx0, friend_id0);
    ty1 = __shfl(tx1, friend_id0);
    sum0 += a2*ty0;
    sum1 += a2*ty1;
    */

    friend_id0 = (lane_id+0 )&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    tx1 = __shfl(threadInput1, friend_id0);
    sum0 += a0*tx0;
    sum1 += a0*tx1;

    friend_id0 = (lane_id+1 )&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id0);
    ty1 = __shfl(threadInput2, friend_id0);
    sum0 += a1*((lane_id < 31)? tx0: ty0);
    sum1 += a1*((lane_id < 31)? tx1: ty1);

    friend_id0 = (lane_id+2 )&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id0);
    ty1 = __shfl(threadInput2, friend_id0);
    sum0 += a2*((lane_id < 30)? tx0: ty0);
    sum1 += a2*((lane_id < 30)? tx1: ty1);

    OUT_1D(gid   ) = sum0; 
    OUT_1D(gid+32) = sum1; 
}

__global__ void Stencil_Cuda_Shfl4(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, int n, int halo) 
{
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % warpSize;
    unsigned int gid = (((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<7) + lane_id + halo;  
    int warp_id_x = ((((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<7) + lane_id)>>5;

    /*
    DATA_TYPE reg[5]; 
    int new_i = (warp_id_x<<5) + lane_id%34;
    reg[0] = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+32)/34)*34 + (lane_id+32)%34;
    reg[1] = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+64)/34)*34 + (lane_id+64)%34;
    reg[2] = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+96)/34)*34 + (lane_id+96)%34;
    reg[3] = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+128)/34)*34 + (lane_id+128)%34;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    reg[4] = IN_1D(new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    int friend_id0;
    // int friend_id1;
    DATA_TYPE tx0, ty0, tx1, ty1;
    DATA_TYPE tx2, ty2, tx3, ty3;
    int reg_id0;
    int reg_id1;
    int reg_id2;
    int reg_id3;

    friend_id0 = (lane_id+0 )&(warpSize-1);
    tx0 = __shfl(reg[0], friend_id0);
    tx1 = __shfl(reg[1], friend_id0);
    tx2 = __shfl(reg[2], friend_id0);
    tx3 = __shfl(reg[3], friend_id0);
    sum0 += a0*tx0;
    sum1 += a0*tx1;
    sum2 += a0*tx2;
    sum3 += a0*tx3;

    friend_id0 = (lane_id+1 )&(warpSize-1);
    reg_id0 = (lane_id > 0)? 0: 1;
    reg_id1 = (lane_id > 0)? 1: 2;
    reg_id2 = (lane_id > 0)? 2: 3;
    reg_id3 = (lane_id > 0)? 3: 4;
    tx0 = __shfl(reg[reg_id0], friend_id0);
    tx1 = __shfl(reg[reg_id1], friend_id0);
    tx2 = __shfl(reg[reg_id2], friend_id0);
    tx3 = __shfl(reg[reg_id3], friend_id0);
    sum0 += a1*tx0;
    sum1 += a1*tx1;
    sum2 += a1*tx2;
    sum3 += a1*tx3;

    friend_id0 = (lane_id+2 )&(warpSize-1);
    reg_id0 = (lane_id > 1)? 0: 1;
    reg_id1 = (lane_id > 1)? 1: 2;
    reg_id2 = (lane_id > 1)? 2: 3;
    reg_id3 = (lane_id > 1)? 3: 4;
    tx0 = __shfl(reg[reg_id0], friend_id0);
    tx1 = __shfl(reg[reg_id1], friend_id0);
    tx2 = __shfl(reg[reg_id2], friend_id0);
    tx3 = __shfl(reg[reg_id3], friend_id0);
    sum0 += a2*tx0;
    sum1 += a2*tx1;
    sum2 += a2*tx2;
    sum3 += a2*tx3;
    */

    /*
    DATA_TYPE reg0, reg1, reg2, reg3, reg4; 
    int new_i = (warp_id_x<<5) + lane_id%34;
    reg0 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+32)/34)*34 + (lane_id+32)%34;
    reg1 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+64)/34)*34 + (lane_id+64)%34;
    reg2 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+96)/34)*34 + (lane_id+96)%34;
    reg3 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+128)/34)*34 + (lane_id+128)%34;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    reg4 = IN_1D(new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    int friend_id0;
    // int friend_id1;
    DATA_TYPE tx0, ty0, tx1, ty1;
    DATA_TYPE tx2, ty2, tx3, ty3;
    int reg_id0;
    int reg_id1;
    int reg_id2;
    int reg_id3;

    friend_id0 = (lane_id+0 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    tx2 = __shfl(reg2, friend_id0);
    tx3 = __shfl(reg3, friend_id0);
    sum0 += a0*tx0;
    sum1 += a0*tx1;
    sum2 += a0*tx2;
    sum3 += a0*tx3;

    friend_id0 = (lane_id+1 )&(warpSize-1);
    reg_id0 = (lane_id > 0)? reg0: reg1;
    reg_id1 = (lane_id > 0)? reg1: reg2;
    reg_id2 = (lane_id > 0)? reg2: reg3;
    reg_id3 = (lane_id > 0)? reg3: reg4;
    tx0 = __shfl(reg_id0, friend_id0);
    tx1 = __shfl(reg_id1, friend_id0);
    tx2 = __shfl(reg_id2, friend_id0);
    tx3 = __shfl(reg_id3, friend_id0);
    sum0 += a1*tx0;
    sum1 += a1*tx1;
    sum2 += a1*tx2;
    sum3 += a1*tx3;

    friend_id0 = (lane_id+2 )&(warpSize-1);
    reg_id0 = (lane_id > 1)? reg0: reg1;
    reg_id1 = (lane_id > 1)? reg1: reg2;
    reg_id2 = (lane_id > 1)? reg2: reg3;
    reg_id3 = (lane_id > 1)? reg3: reg4;
    tx0 = __shfl(reg_id0, friend_id0);
    tx1 = __shfl(reg_id1, friend_id0);
    tx2 = __shfl(reg_id2, friend_id0);
    tx3 = __shfl(reg_id3, friend_id0);
    sum0 += a2*tx0;
    sum1 += a2*tx1;
    sum2 += a2*tx2;
    sum3 += a2*tx3;
    */

    DATA_TYPE reg0, reg1, reg2, reg3, reg4; 
    int new_i = (warp_id_x<<5) + lane_id%34;
    reg0 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+32)/34)*34 + (lane_id+32)%34;
    reg1 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+64)/34)*34 + (lane_id+64)%34;
    reg2 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+96)/34)*34 + (lane_id+96)%34;
    reg3 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+128)/34)*34 + (lane_id+128)%34;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    reg4 = IN_1D(new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    int friend_id0;
    // int friend_id1;
    DATA_TYPE tx0, ty0, tx1, ty1;
    DATA_TYPE tx2, ty2, tx3, ty3;

    friend_id0 = (lane_id+0 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    tx2 = __shfl(reg2, friend_id0);
    tx3 = __shfl(reg3, friend_id0);
    sum0 += a0*tx0;
    sum1 += a0*tx1;
    sum2 += a0*tx2;
    sum3 += a0*tx3;

    friend_id0 = (lane_id+1 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    ty1 = __shfl(reg2, friend_id0);
    tx2 = __shfl(reg2, friend_id0);
    ty2 = __shfl(reg3, friend_id0);
    tx3 = __shfl(reg3, friend_id0);
    ty3 = __shfl(reg4, friend_id0);
    sum0 += a1*((lane_id < 31)? tx0: ty0);
    sum1 += a1*((lane_id < 31)? tx1: ty1);
    sum2 += a1*((lane_id < 31)? tx2: ty2);
    sum3 += a1*((lane_id < 31)? tx3: ty3);

    friend_id0 = (lane_id+2 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    ty1 = __shfl(reg2, friend_id0);
    tx2 = __shfl(reg2, friend_id0);
    ty2 = __shfl(reg3, friend_id0);
    tx3 = __shfl(reg3, friend_id0);
    ty3 = __shfl(reg4, friend_id0);
    sum0 += a2*((lane_id < 30)? tx0: ty0);
    sum1 += a2*((lane_id < 30)? tx1: ty1);
    sum2 += a2*((lane_id < 30)? tx2: ty2);
    sum3 += a2*((lane_id < 30)? tx3: ty3);

    OUT_1D(gid   ) = sum0; 
    OUT_1D(gid+32) = sum1; 
    OUT_1D(gid+64) = sum2; 
    OUT_1D(gid+96) = sum3; 
}

__global__ void Stencil_Cuda_Shfl8(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, int n, int halo) 
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
    sum0 += a0*tx0;
    sum1 += a0*tx1;
    sum2 += a0*tx2;
    sum3 += a0*tx3;
    sum4 += a0*tx4;
    sum5 += a0*tx5;
    sum6 += a0*tx6;
    sum7 += a0*tx7;

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
    sum0 += a1*((lane_id < 31)? tx0: ty0);
    sum1 += a1*((lane_id < 31)? tx1: ty1);
    sum2 += a1*((lane_id < 31)? tx2: ty2);
    sum3 += a1*((lane_id < 31)? tx3: ty3);
    sum4 += a1*((lane_id < 31)? tx4: ty4);
    sum5 += a1*((lane_id < 31)? tx5: ty5);
    sum6 += a1*((lane_id < 31)? tx6: ty6);
    sum7 += a1*((lane_id < 31)? tx7: ty7);

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
    sum0 += a2*((lane_id < 30)? tx0: ty0);
    sum1 += a2*((lane_id < 30)? tx1: ty1);
    sum2 += a2*((lane_id < 30)? tx2: ty2);
    sum3 += a2*((lane_id < 30)? tx3: ty3);
    sum4 += a2*((lane_id < 30)? tx4: ty4);
    sum5 += a2*((lane_id < 30)? tx5: ty5);
    sum6 += a2*((lane_id < 30)? tx6: ty6);
    sum7 += a2*((lane_id < 30)? tx7: ty7);

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
        Stencil_Cuda<<<dimGrid, dimBlock>>>(in_d, out_d, args[0], args[1], args[2] , n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D3, time_wo_pci));
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
        Stencil_Cuda_Sm_Branch<<<dimGrid, dimBlock>>>(in_d, out_d, args[0], args[1], args[2] , n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sm_Branch: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sm_Branch Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D3, time_wo_pci));
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
        Stencil_Cuda_Sm_Cyclic<<<dimGrid, dimBlock>>>(in_d, out_d, args[0], args[1], args[2] , n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sm_Cyclic: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sm_Cyclic Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D3, time_wo_pci));
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
        Stencil_Cuda_Shfl<<<dimGrid, dimBlock>>>(in_d, out_d, args[0], args[1], args[2] , n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Shfl Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D3, time_wo_pci));
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
        Stencil_Cuda_Shfl2<<<dimGrid, dimBlock>>>(in_d, out_d, args[0], args[1], args[2] , n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl2: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Shfl2 Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D3, time_wo_pci));
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
        Stencil_Cuda_Shfl4<<<dimGrid, dimBlock>>>(in_d, out_d, args[0], args[1], args[2] , n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl4: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Shfl4 Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    /*
    // Cuda Shfl8 1D-Warp 
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
        Stencil_Cuda_Shfl8<<<dimGrid5, dimBlock5>>>(in_d, out_d, args[0], args[1], args[2] , n, halo);
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
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));
    */


    cudaFree(in_d);
    cudaFree(out_d);

    delete[] in;
    delete[] out;
    delete[] out_ref;
}

