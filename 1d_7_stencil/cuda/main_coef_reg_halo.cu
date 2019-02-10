#include <iostream>
#include "metrics.h"
using namespace std;
// #define DATA_TYPE float
// #define DATA_TYPE double
#define warpSize 32 

#define  IN_1D(_x)  in[_x]
#define OUT_1D(_x) out[_x]

// #define __DEBUG

#ifdef __DEBUG
#define ITER 1
#else
#define ITER 100
#endif

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


void Stencil_Seq(DATA_TYPE *in, DATA_TYPE *out, 
        DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2,
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int n, int halo)
{
#pragma omp parallel for
    for(int i = halo; i < n+halo; i++)
    {
        OUT_1D(i) = a0*IN_1D(i-3) + 
                    a1*IN_1D(i-2) + 
                    a2*IN_1D(i-1) + 
                    a3*IN_1D(i  ) + 
                    a4*IN_1D(i+1) + 
                    a5*IN_1D(i+2) + 
                    a6*IN_1D(i+3) ;
    }
}

__global__ void Stencil_Cuda(DATA_TYPE *in, DATA_TYPE *out, 
        DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2,
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;  
    OUT_1D(i) = a0*IN_1D(i-3) + 
                a1*IN_1D(i-2) + 
                a2*IN_1D(i-1) + 
                a3*IN_1D(i  ) + 
                a4*IN_1D(i+1) + 
                a5*IN_1D(i+2) + 
                a6*IN_1D(i+3) ;
}

__global__ void Stencil_Cuda_Sm_Branch(DATA_TYPE *in, DATA_TYPE *out, 
        DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2,
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int n, int halo) 
{
    __shared__ DATA_TYPE local[256+2*3];
    unsigned int tid = threadIdx.x;
    unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x + halo;  
    int local_id = tid + halo;
    local[local_id] = IN_1D(gid);
    if(tid == 0)
    {
        local[local_id-1] = IN_1D(gid-1);
        local[local_id-2] = IN_1D(gid-2);
        local[local_id-3] = IN_1D(gid-3);
    }
    if(tid == blockDim.x - 1)
    {
        local[local_id+1] = IN_1D(gid+1);
        local[local_id+2] = IN_1D(gid+2);
        local[local_id+3] = IN_1D(gid+3);
    }
    __syncthreads();

    OUT_1D(gid) = a0*local[local_id-3] + 
                  a1*local[local_id-2] + 
                  a2*local[local_id-1] + 
                  a3*local[local_id  ] + 
                  a4*local[local_id+1] + 
                  a5*local[local_id+2] + 
                  a6*local[local_id+3] ;
}

__global__ void Stencil_Cuda_Sm_Cyclic(DATA_TYPE *in, DATA_TYPE *out, 
        DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2,
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int n, int halo) 
{
    __shared__ DATA_TYPE local[256+2*3];
    unsigned int tid = threadIdx.x;
    unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x + halo;  
    int local_id = tid + halo;

    unsigned int lane_id = threadIdx.x;
    int lane_id_it = lane_id;
    int blk_id_x = blockIdx.x;
    int new_i  = (blk_id_x<<8) + lane_id_it%262;
    int new_li = lane_id_it%262;
    local[new_li] = IN_1D(new_i);
    lane_id_it += 256;
    new_i  = (blk_id_x<<8) + (lane_id_it/262)*262 + lane_id_it%262;
    new_li = (lane_id_it/262)*262 + lane_id_it%262;
    if(new_li < 262)
        local[new_li] = IN_1D(new_i);
    
    __syncthreads();

    OUT_1D(gid) = a0*local[local_id-3] + 
                  a1*local[local_id-2] + 
                  a2*local[local_id-1] + 
                  a3*local[local_id  ] + 
                  a4*local[local_id+1] + 
                  a5*local[local_id+2] + 
                  a6*local[local_id+3] ;
}

__global__ void Stencil_Cuda_Shfl(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2,
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6 , int n, int halo) 
{
    unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x + halo;  
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % warpSize;

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>5;

    DATA_TYPE reg0, reg1;
    int lane_id_it = lane_id;
    int new_i = (warp_id_x<<5) + lane_id_it%38;
    reg0 = IN_1D(new_i);
    lane_id_it += warpSize;
    new_i = (warp_id_x<<5) + (lane_id_it/38)*38 + lane_id_it%38;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    reg1 = IN_1D(new_i);

    DATA_TYPE sum0 = 0.0;
    int friend_id0;
    DATA_TYPE tx0, ty0;

    friend_id0 = (lane_id+0 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    sum0 += a0*tx0;

    friend_id0 = (lane_id+1 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a1*((lane_id < 31)? tx0: ty0);

    friend_id0 = (lane_id+2 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a2*((lane_id < 30)? tx0: ty0);

    friend_id0 = (lane_id+3 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a3*((lane_id < 29)? tx0: ty0);

    friend_id0 = (lane_id+4 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a4*((lane_id < 28)? tx0: ty0);

    friend_id0 = (lane_id+5 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a5*((lane_id < 27)? tx0: ty0);

    friend_id0 = (lane_id+6 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a6*((lane_id < 26)? tx0: ty0);

    OUT_1D(gid) = sum0; 
}

__global__ void Stencil_Cuda_Shfl2(DATA_TYPE *in, DATA_TYPE *out, 
        DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2,
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int n, int halo) 
{
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % warpSize;
    unsigned int gid = (((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<6) + lane_id + halo;  
    int warp_id_x = ((((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<6) + lane_id)>>5;

    DATA_TYPE reg0, reg1, reg2;
    int lane_id_it = lane_id;
    int new_i = (warp_id_x<<5) + lane_id_it%38;
    reg0 = IN_1D(new_i);
    lane_id_it += warpSize;
    new_i = (warp_id_x<<5) + (lane_id_it/38)*38 + lane_id_it%38;
    reg1 = IN_1D(new_i);
    lane_id_it += warpSize;
    new_i = (warp_id_x<<5) + (lane_id_it/38)*38 + lane_id_it%38;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    reg2 = IN_1D(new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    int friend_id0;
    // int friend_id1;
    DATA_TYPE tx0, ty0, tx1, ty1;

    friend_id0 = (lane_id+0 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    sum0 += a0*tx0;
    sum1 += a0*tx1;

    friend_id0 = (lane_id+1 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    ty1 = __shfl(reg2, friend_id0);
    sum0 += a1*((lane_id < 31)? tx0: ty0);
    sum1 += a1*((lane_id < 31)? tx1: ty1);

    friend_id0 = (lane_id+2 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    ty1 = __shfl(reg2, friend_id0);
    sum0 += a2*((lane_id < 30)? tx0: ty0);
    sum1 += a2*((lane_id < 30)? tx1: ty1);

    friend_id0 = (lane_id+3 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    ty1 = __shfl(reg2, friend_id0);
    sum0 += a3*((lane_id < 29)? tx0: ty0);
    sum1 += a3*((lane_id < 29)? tx1: ty1);

    friend_id0 = (lane_id+4 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    ty1 = __shfl(reg2, friend_id0);
    sum0 += a4*((lane_id < 28)? tx0: ty0);
    sum1 += a4*((lane_id < 28)? tx1: ty1);

    friend_id0 = (lane_id+5 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    ty1 = __shfl(reg2, friend_id0);
    sum0 += a5*((lane_id < 27)? tx0: ty0);
    sum1 += a5*((lane_id < 27)? tx1: ty1);

    friend_id0 = (lane_id+6 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    ty1 = __shfl(reg2, friend_id0);
    sum0 += a6*((lane_id < 26)? tx0: ty0);
    sum1 += a6*((lane_id < 26)? tx1: ty1);

    OUT_1D(gid   ) = sum0; 
    OUT_1D(gid+32) = sum1; 
}

__global__ void Stencil_Cuda_Shfl4(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2,
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6 , int n, int halo) 
{
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % warpSize;
    unsigned int gid = (((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<7) + lane_id + halo;  
    int warp_id_x = ((((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<7) + lane_id)>>5;

    DATA_TYPE reg0, reg1, reg2, reg3, reg4;
    int lane_id_it = lane_id;
    int new_i = (warp_id_x<<5) + lane_id_it%38;
    reg0 = IN_1D(new_i);
    lane_id_it += warpSize;
    new_i = (warp_id_x<<5) + (lane_id_it/38)*38 + lane_id_it%38;
    reg1 = IN_1D(new_i);
    lane_id_it += warpSize;
    new_i = (warp_id_x<<5) + (lane_id_it/38)*38 + lane_id_it%38;
    reg2 = IN_1D(new_i);
    lane_id_it += warpSize;
    new_i = (warp_id_x<<5) + (lane_id_it/38)*38 + lane_id_it%38;
    reg3 = IN_1D(new_i);
    lane_id_it += warpSize;
    new_i = (warp_id_x<<5) + (lane_id_it/38)*38 + lane_id_it%38;
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

    friend_id0 = (lane_id+3 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    ty1 = __shfl(reg2, friend_id0);
    tx2 = __shfl(reg2, friend_id0);
    ty2 = __shfl(reg3, friend_id0);
    tx3 = __shfl(reg3, friend_id0);
    ty3 = __shfl(reg4, friend_id0);
    sum0 += a3*((lane_id < 29)? tx0: ty0);
    sum1 += a3*((lane_id < 29)? tx1: ty1);
    sum2 += a3*((lane_id < 29)? tx2: ty2);
    sum3 += a3*((lane_id < 29)? tx3: ty3);

    friend_id0 = (lane_id+4 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    ty1 = __shfl(reg2, friend_id0);
    tx2 = __shfl(reg2, friend_id0);
    ty2 = __shfl(reg3, friend_id0);
    tx3 = __shfl(reg3, friend_id0);
    ty3 = __shfl(reg4, friend_id0);
    sum0 += a4*((lane_id < 28)? tx0: ty0);
    sum1 += a4*((lane_id < 28)? tx1: ty1);
    sum2 += a4*((lane_id < 28)? tx2: ty2);
    sum3 += a4*((lane_id < 28)? tx3: ty3);

    friend_id0 = (lane_id+5 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    ty1 = __shfl(reg2, friend_id0);
    tx2 = __shfl(reg2, friend_id0);
    ty2 = __shfl(reg3, friend_id0);
    tx3 = __shfl(reg3, friend_id0);
    ty3 = __shfl(reg4, friend_id0);
    sum0 += a5*((lane_id < 27)? tx0: ty0);
    sum1 += a5*((lane_id < 27)? tx1: ty1);
    sum2 += a5*((lane_id < 27)? tx2: ty2);
    sum3 += a5*((lane_id < 27)? tx3: ty3);

    friend_id0 = (lane_id+6 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    ty1 = __shfl(reg2, friend_id0);
    tx2 = __shfl(reg2, friend_id0);
    ty2 = __shfl(reg3, friend_id0);
    tx3 = __shfl(reg3, friend_id0);
    ty3 = __shfl(reg4, friend_id0);
    sum0 += a6*((lane_id < 26)? tx0: ty0);
    sum1 += a6*((lane_id < 26)? tx1: ty1);
    sum2 += a6*((lane_id < 26)? tx2: ty2);
    sum3 += a6*((lane_id < 26)? tx3: ty3);

    OUT_1D(gid   ) = sum0; 
    OUT_1D(gid+32) = sum1; 
    OUT_1D(gid+64) = sum2; 
    OUT_1D(gid+96) = sum3; 
}

__global__ void Stencil_Cuda_Shfl8(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2,
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6 , int n, int halo) 
{
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % warpSize;
    unsigned int gid = (((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<8) + lane_id + halo;  
    int warp_id_x = ((((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<8) + lane_id)>>5;

    DATA_TYPE reg0, reg1, reg2, reg3, reg4, reg5,
              reg6, reg7, reg8;
    int new_i = (warp_id_x<<5) + lane_id%34;
    reg0 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+32)/34)*34 + (lane_id+32)%34;
    reg1 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+64)/34)*34 + (lane_id+64)%34;
    reg2 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+96)/34)*34 + (lane_id+96)%34;
    reg3 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+128)/34)*34 + (lane_id+128)%34;
    reg4 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+160)/34)*34 + (lane_id+160)%34;
    reg5 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+192)/34)*34 + (lane_id+192)%34;
    reg6 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+224)/34)*34 + (lane_id+224)%34;
    reg7 = IN_1D(new_i);
    new_i = (warp_id_x<<5) + ((lane_id+256)/34)*34 + (lane_id+256)%34;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    reg8 = IN_1D(new_i);

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
    tx0 = __shfl(reg0, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    tx2 = __shfl(reg2, friend_id0);
    tx3 = __shfl(reg3, friend_id0);
    tx4 = __shfl(reg4, friend_id0);
    tx5 = __shfl(reg5, friend_id0);
    tx6 = __shfl(reg6, friend_id0);
    tx7 = __shfl(reg7, friend_id0);
    sum0 += a0*tx0;
    sum1 += a0*tx1;
    sum2 += a0*tx2;
    sum3 += a0*tx3;
    sum4 += a0*tx4;
    sum5 += a0*tx5;
    sum6 += a0*tx6;
    sum7 += a0*tx7;

    friend_id0 = (lane_id+1 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    ty1 = __shfl(reg2, friend_id0);
    tx2 = __shfl(reg2, friend_id0);
    ty2 = __shfl(reg3, friend_id0);
    tx3 = __shfl(reg3, friend_id0);
    ty3 = __shfl(reg4, friend_id0);
    tx4 = __shfl(reg4, friend_id0);
    ty4 = __shfl(reg5, friend_id0);
    tx5 = __shfl(reg5, friend_id0);
    ty5 = __shfl(reg6, friend_id0);
    tx6 = __shfl(reg6, friend_id0);
    ty6 = __shfl(reg7, friend_id0);
    tx7 = __shfl(reg7, friend_id0);
    ty7 = __shfl(reg8, friend_id0);
    sum0 += a1*((lane_id < 31)? tx0: ty0);
    sum1 += a1*((lane_id < 31)? tx1: ty1);
    sum2 += a1*((lane_id < 31)? tx2: ty2);
    sum3 += a1*((lane_id < 31)? tx3: ty3);
    sum4 += a1*((lane_id < 31)? tx4: ty4);
    sum5 += a1*((lane_id < 31)? tx5: ty5);
    sum6 += a1*((lane_id < 31)? tx6: ty6);
    sum7 += a1*((lane_id < 31)? tx7: ty7);

    friend_id0 = (lane_id+2 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    ty1 = __shfl(reg2, friend_id0);
    tx2 = __shfl(reg2, friend_id0);
    ty2 = __shfl(reg3, friend_id0);
    tx3 = __shfl(reg3, friend_id0);
    ty3 = __shfl(reg4, friend_id0);
    tx4 = __shfl(reg4, friend_id0);
    ty4 = __shfl(reg5, friend_id0);
    tx5 = __shfl(reg5, friend_id0);
    ty5 = __shfl(reg6, friend_id0);
    tx6 = __shfl(reg6, friend_id0);
    ty6 = __shfl(reg7, friend_id0);
    tx7 = __shfl(reg7, friend_id0);
    ty7 = __shfl(reg8, friend_id0);
    sum0 += a2*((lane_id < 30)? tx0: ty0);
    sum1 += a2*((lane_id < 30)? tx1: ty1);
    sum2 += a2*((lane_id < 30)? tx2: ty2);
    sum3 += a2*((lane_id < 30)? tx3: ty3);
    sum4 += a2*((lane_id < 30)? tx4: ty4);
    sum5 += a2*((lane_id < 30)? tx5: ty5);
    sum6 += a2*((lane_id < 30)? tx6: ty6);
    sum7 += a2*((lane_id < 30)? tx7: ty7);

    friend_id0 = (lane_id+3 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    ty1 = __shfl(reg2, friend_id0);
    tx2 = __shfl(reg2, friend_id0);
    ty2 = __shfl(reg3, friend_id0);
    tx3 = __shfl(reg3, friend_id0);
    ty3 = __shfl(reg4, friend_id0);
    tx4 = __shfl(reg4, friend_id0);
    ty4 = __shfl(reg5, friend_id0);
    tx5 = __shfl(reg5, friend_id0);
    ty5 = __shfl(reg6, friend_id0);
    tx6 = __shfl(reg6, friend_id0);
    ty6 = __shfl(reg7, friend_id0);
    tx7 = __shfl(reg7, friend_id0);
    ty7 = __shfl(reg8, friend_id0);
    sum0 += a3*((lane_id < 29)? tx0: ty0);
    sum1 += a3*((lane_id < 29)? tx1: ty1);
    sum2 += a3*((lane_id < 29)? tx2: ty2);
    sum3 += a3*((lane_id < 29)? tx3: ty3);
    sum4 += a3*((lane_id < 29)? tx4: ty4);
    sum5 += a3*((lane_id < 29)? tx5: ty5);
    sum6 += a3*((lane_id < 29)? tx6: ty6);
    sum7 += a3*((lane_id < 29)? tx7: ty7);

    friend_id0 = (lane_id+4 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    ty1 = __shfl(reg2, friend_id0);
    tx2 = __shfl(reg2, friend_id0);
    ty2 = __shfl(reg3, friend_id0);
    tx3 = __shfl(reg3, friend_id0);
    ty3 = __shfl(reg4, friend_id0);
    tx4 = __shfl(reg4, friend_id0);
    ty4 = __shfl(reg5, friend_id0);
    tx5 = __shfl(reg5, friend_id0);
    ty5 = __shfl(reg6, friend_id0);
    tx6 = __shfl(reg6, friend_id0);
    ty6 = __shfl(reg7, friend_id0);
    tx7 = __shfl(reg7, friend_id0);
    ty7 = __shfl(reg8, friend_id0);
    sum0 += a4*((lane_id < 28)? tx0: ty0);
    sum1 += a4*((lane_id < 28)? tx1: ty1);
    sum2 += a4*((lane_id < 28)? tx2: ty2);
    sum3 += a4*((lane_id < 28)? tx3: ty3);
    sum4 += a4*((lane_id < 28)? tx4: ty4);
    sum5 += a4*((lane_id < 28)? tx5: ty5);
    sum6 += a4*((lane_id < 28)? tx6: ty6);
    sum7 += a4*((lane_id < 28)? tx7: ty7);

    friend_id0 = (lane_id+5 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    ty1 = __shfl(reg2, friend_id0);
    tx2 = __shfl(reg2, friend_id0);
    ty2 = __shfl(reg3, friend_id0);
    tx3 = __shfl(reg3, friend_id0);
    ty3 = __shfl(reg4, friend_id0);
    tx4 = __shfl(reg4, friend_id0);
    ty4 = __shfl(reg5, friend_id0);
    tx5 = __shfl(reg5, friend_id0);
    ty5 = __shfl(reg6, friend_id0);
    tx6 = __shfl(reg6, friend_id0);
    ty6 = __shfl(reg7, friend_id0);
    tx7 = __shfl(reg7, friend_id0);
    ty7 = __shfl(reg8, friend_id0);
    sum0 += a5*((lane_id < 27)? tx0: ty0);
    sum1 += a5*((lane_id < 27)? tx1: ty1);
    sum2 += a5*((lane_id < 27)? tx2: ty2);
    sum3 += a5*((lane_id < 27)? tx3: ty3);
    sum4 += a5*((lane_id < 27)? tx4: ty4);
    sum5 += a5*((lane_id < 27)? tx5: ty5);
    sum6 += a5*((lane_id < 27)? tx6: ty6);
    sum7 += a5*((lane_id < 27)? tx7: ty7);

    friend_id0 = (lane_id+6 )&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id0);
    ty1 = __shfl(reg2, friend_id0);
    tx2 = __shfl(reg2, friend_id0);
    ty2 = __shfl(reg3, friend_id0);
    tx3 = __shfl(reg3, friend_id0);
    ty3 = __shfl(reg4, friend_id0);
    tx4 = __shfl(reg4, friend_id0);
    ty4 = __shfl(reg5, friend_id0);
    tx5 = __shfl(reg5, friend_id0);
    ty5 = __shfl(reg6, friend_id0);
    tx6 = __shfl(reg6, friend_id0);
    ty6 = __shfl(reg7, friend_id0);
    tx7 = __shfl(reg7, friend_id0);
    ty7 = __shfl(reg8, friend_id0);
    sum0 += a6*((lane_id < 26)? tx0: ty0);
    sum1 += a6*((lane_id < 26)? tx1: ty1);
    sum2 += a6*((lane_id < 26)? tx2: ty2);
    sum3 += a6*((lane_id < 26)? tx3: ty3);
    sum4 += a6*((lane_id < 26)? tx4: ty4);
    sum5 += a6*((lane_id < 26)? tx5: ty5);
    sum6 += a6*((lane_id < 26)? tx6: ty6);
    sum7 += a6*((lane_id < 26)? tx7: ty7);



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
    int halo = 3; 
    int total = (n+2*halo);
    const int K = 7;
#ifdef __DEBUG
    DATA_TYPE args[K] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
#else
    DATA_TYPE args[K] = {0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14};
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
        Stencil_Seq(in, out_ref, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                n, halo);
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
        Stencil_Cuda<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                n, halo);
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
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D7, time_wo_pci));
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
        Stencil_Cuda_Sm_Branch<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                n, halo);
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
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D7, time_wo_pci));
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
        Stencil_Cuda_Sm_Cyclic<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                n, halo);
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
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D7, time_wo_pci));
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
        Stencil_Cuda_Shfl<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                n, halo);
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
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl2 version 
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
        Stencil_Cuda_Shfl2<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                n, halo);
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
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl4 version 
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
        Stencil_Cuda_Shfl4<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                n, halo);
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
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    /*

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
        Stencil_Cuda_Shfl8<<<dimGrid5, dimBlock5>>>(in_d, out_d, args[0], args[1], args[2], args[3], args[4], args[5], 
                args[6] , n, halo);
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
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));
    */


    cudaFree(in_d);
    cudaFree(out_d);

    delete[] in;
    delete[] out;
    delete[] out_ref;
}

