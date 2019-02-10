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

void Init_Input_2D(DATA_TYPE *in, int m, int n, int halo, unsigned int seed)
{
    srand(seed);
    for(int i = 0; i < m+2*halo; i++)
    {
        for(int j = 0; j < n+2*halo; j++)
        {
            if(i < halo || j < halo || i >= m+halo || j >= n+halo)
                IN_2D(i,j) = 0.0;
            else
#ifdef __DEBUG
                IN_2D(i,j) = 1.0; 
                // IN_2D(i,j) = (DATA_TYPE)rand()*100.0 / ((long)RAND_MAX);
#else
                IN_2D(i,j) = (DATA_TYPE)rand()*100.0 / ((long)RAND_MAX);
#endif
        }
    }
}

void Clear_Output_2D(DATA_TYPE *in, int m, int n, int halo)
{
    for(int i = 0; i < m+2*halo; i++)
    {
        for(int j = 0; j < n+2*halo; j++)
        {
            IN_2D(i,j) = 0.0;
        }
    }
}

void Fill_Halo_2D(DATA_TYPE *in, int m, int n, int halo)
{
    for(int i = 0; i < m+2*halo; i++)
    {
        for(int j = 0; j < n+2*halo; j++)
        {
            if(i<halo || j<halo || i>=m+halo || j>=n+halo)
                IN_2D(i,j) = 0.0;
        }
    }
}

void Show_Me(DATA_TYPE *in, int m, int n, int halo, string prompt)
{
    cout << prompt << endl;
    for(int i = 0; i < m+2*halo; i++)
    {
        for(int j = 0; j < n+2*halo; j++)
        {
            std::cout << IN_2D(i,j) << ",";
        }
        std::cout << std::endl;
    }
}

void Stencil_Seq(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, 
        DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, DATA_TYPE a8,
        int m, int n, int halo)
{
#pragma omp parallel for
    for(int j = halo; j < m+halo; j++)
    {
        for(int i = halo; i < n+halo; i++)
        {
            OUT_2D(j,i) = a0 * IN_2D(j-1,i-1) +
                          a1 * IN_2D(j  ,i-1) +
                          a2 * IN_2D(j+1,i-1) +
                          a3 * IN_2D(j-1,i  ) +
                          a4 * IN_2D(j  ,i  ) +
                          a5 * IN_2D(j+1,i  ) +
                          a6 * IN_2D(j-1,i+1) +
                          a7 * IN_2D(j  ,i+1) +
                          a8 * IN_2D(j+1,i+1) ;
        }
    }
}

__global__ void Stencil_Cuda_L1_2Blk(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, 
        DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, DATA_TYPE a8,
        int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    OUT_2D(j,i) = a0*IN_2D(j-1,i-1) + 
                  a1*IN_2D(j  ,i-1) + 
                  a2*IN_2D(j+1,i-1) +
                  a3*IN_2D(j-1,i  ) + 
                  a4*IN_2D(j  ,i  ) + 
                  a5*IN_2D(j+1,i  ) +
                  a6*IN_2D(j-1,i+1) + 
                  a7*IN_2D(j  ,i+1) + 
                  a8*IN_2D(j+1,i+1) ;
}


__global__ void Stencil_Cuda_Reg1_2Blk2Wf(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, 
        DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, DATA_TYPE a8,
        int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;

    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;

    int warp_id0 = (threadIdx.x + blockIdx.x * blockDim.x)>>3;
    int warp_id1 = (threadIdx.y + blockIdx.y * blockDim.y)>>2;

    DATA_TYPE reg0, reg1;
    int lane_id_it = lane_id;
    int new_id0 ;
    int new_id1 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10 ;
    reg0 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10 ;
    new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
    new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
    reg1 = IN_2D(new_id1, new_id0) ;

    
    DATA_TYPE sum0 = 0.0;
    int friend_id0;
    DATA_TYPE tx0, ty0;

    // process (0, 0, 0)
    friend_id0 = (lane_id+ 0+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a0 *((lane_id < 26 )? tx0: ty0);
    // process (1, 0, 0)
    friend_id0 = (lane_id+ 1+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a1 *((lane_id < 25 )? tx0: ty0);
    // process (2, 0, 0)
    friend_id0 = (lane_id+ 2+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a2 *((lane_id < 24 )? tx0: ty0);
    // process (0, 1, 0)
    friend_id0 = (lane_id+10+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a3 *((lane_id < 18 )? tx0: ty0);
    // process (1, 1, 0)
    friend_id0 = (lane_id+11+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a4 *((lane_id < 17 )? tx0: ty0);
    // process (2, 1, 0)
    friend_id0 = (lane_id+12+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a5 *((lane_id < 16 )? tx0: ty0);
    // process (0, 2, 0)
    friend_id0 = (lane_id+20+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a6 *((lane_id < 10 )? tx0: ty0);
    // process (1, 2, 0)
    friend_id0 = (lane_id+21+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a7 *((lane_id < 9 )? tx0: ty0);
    // process (2, 2, 0)
    friend_id0 = (lane_id+22+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a8 *((lane_id < 8 )? tx0: ty0);

    OUT_2D(j,i) = sum0; 
}

__global__ void Stencil_Cuda_Reg2_2Blk2Wf(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, 
        DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, DATA_TYPE a8,
        int m, int n, int halo) 
{
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<3) + (lane_id>>3)  + halo;

    int warp_id0 = (threadIdx.x + blockIdx.x * blockDim.x)>>3;
    int warp_id1 = ((((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<3) + (lane_id>>3))>>2;

    DATA_TYPE reg0, reg1, reg2, reg3;
    int lane_id_it = lane_id;
    int new_id0 ;
    int new_id1 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10 ;
    reg0 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10 ;
    reg1 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10 ;
    reg2 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10 ;
    new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
    new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
    reg3 = IN_2D(new_id1, new_id0) ;
    
    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    int friend_id0, friend_id1;
    DATA_TYPE tx0, ty0, tx1, ty1, tz1;

    // process (0, 0, 0)
    friend_id0 = (lane_id+ 0+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a0 *((lane_id < 26 )? tx0: ty0);
    friend_id1 = (lane_id+ 8+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a0 *((lane_id < 20 )? tx1: ty1);
    // process (1, 0, 0)
    friend_id0 = (lane_id+ 1+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a1 *((lane_id < 25 )? tx0: ty0);
    friend_id1 = (lane_id+ 9+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a1 *((lane_id < 19 )? tx1: ty1);
    // process (2, 0, 0)
    friend_id0 = (lane_id+ 2+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a2 *((lane_id < 24 )? tx0: ty0);
    friend_id1 = (lane_id+10+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a2 *((lane_id < 18 )? tx1: ty1);
    // process (0, 1, 0)
    friend_id0 = (lane_id+10+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a3 *((lane_id < 18 )? tx0: ty0);
    friend_id1 = (lane_id+18+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a3 *((lane_id < 12 )? tx1: ty1);
    // process (1, 1, 0)
    friend_id0 = (lane_id+11+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a4 *((lane_id < 17 )? tx0: ty0);
    friend_id1 = (lane_id+19+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a4 *((lane_id < 11 )? tx1: ty1);
    // process (2, 1, 0)
    friend_id0 = (lane_id+12+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a5 *((lane_id < 16 )? tx0: ty0);
    friend_id1 = (lane_id+20+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a5 *((lane_id < 10 )? tx1: ty1);
    // process (0, 2, 0)
    friend_id0 = (lane_id+20+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a6 *((lane_id < 10 )? tx0: ty0);
    friend_id1 = (lane_id+28+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    tz1 = __shfl(reg3, friend_id1);
    sum1 += a6 *((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
    // process (1, 2, 0)
    friend_id0 = (lane_id+21+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a7 *((lane_id < 9 )? tx0: ty0);
    friend_id1 = (lane_id+29+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    tz1 = __shfl(reg3, friend_id1);
    sum1 += a7 *((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
    // process (2, 2, 0)
    friend_id0 = (lane_id+22+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a8 *((lane_id < 8 )? tx0: ty0);
    friend_id1 = (lane_id+30+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    tz1 = __shfl(reg3, friend_id1);
    sum1 += a8 *((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
    
    OUT_2D(j  ,i) = sum0; 
    OUT_2D(j+4,i) = sum1; 
}

__global__ void Stencil_Cuda_Reg1_2Blk1Wf(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, 
        DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, DATA_TYPE a8,
        int m, int n, int halo) 
{
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<0) + (lane_id>>5)  + halo;

    int warp_id0 = (threadIdx.x + blockIdx.x * blockDim.x)>>5;
    int warp_id1 = ((((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<0) + (lane_id>>5))>>0;

    DATA_TYPE reg0, reg1, reg2, reg3;
    int lane_id_it = lane_id;
    int new_id0 ;
    int new_id1 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    reg0 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    reg1 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    reg2 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
    new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
    reg3 = IN_2D(new_id1, new_id0) ;

    DATA_TYPE sum0 = 0.0;
    int friend_id0;
    DATA_TYPE tx0, ty0;

    // process (0, 0, 0)
    friend_id0 = (lane_id+ 0)&31 ;
    tx0 = __shfl(reg0, friend_id0);
    sum0 += a0 *(tx0);
    // process (1, 0, 0)
    friend_id0 = (lane_id+ 1)&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a1 *((lane_id < 31 )? tx0: ty0);
    // process (2, 0, 0)
    friend_id0 = (lane_id+ 2)&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a2 *((lane_id < 30 )? tx0: ty0);
    // process (0, 1, 0)
    friend_id0 = (lane_id+ 2)&31 ;
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    sum0 += a3 *((lane_id < 30 )? tx0: ty0);
    // process (1, 1, 0)
    friend_id0 = (lane_id+ 3)&31 ;
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    sum0 += a4 *((lane_id < 29 )? tx0: ty0);
    // process (2, 1, 0)
    friend_id0 = (lane_id+ 4)&31 ;
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    sum0 += a5 *((lane_id < 28 )? tx0: ty0);
    // process (0, 2, 0)
    friend_id0 = (lane_id+ 4)&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a6 *((lane_id < 28 )? tx0: ty0);
    // process (1, 2, 0)
    friend_id0 = (lane_id+ 5)&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a7 *((lane_id < 27 )? tx0: ty0);
    // process (2, 2, 0)
    friend_id0 = (lane_id+ 6)&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a8 *((lane_id < 26 )? tx0: ty0);
    
    OUT_2D(j  ,i) = sum0; 
}

__global__ void Stencil_Cuda_Reg2_2Blk1Wf(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, 
        DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, DATA_TYPE a8,
        int m, int n, int halo) 
{
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<1) + (lane_id>>5)  + halo;

    int warp_id0 = (threadIdx.x + blockIdx.x * blockDim.x)>>5;
    int warp_id1 = ((((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<1) + (lane_id>>5))>>0;

    DATA_TYPE reg0, reg1, reg2, reg3, reg4;
    int lane_id_it = lane_id;
    int new_id0 ;
    int new_id1 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    reg0 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    reg1 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    reg2 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    reg3 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
    new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
    reg4 = IN_2D(new_id1, new_id0) ;

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    int friend_id0, friend_id1;
    DATA_TYPE tx0, ty0, tx1, ty1, tz1;

    // process (0, 0, 0)
    friend_id0 = (lane_id+ 0)&31 ;
    tx0 = __shfl(reg0, friend_id0);
    sum0 += a0 *(tx0);
    friend_id1 = (lane_id+ 2)&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a0 *((lane_id < 30 )? tx1: ty1);
    // process (1, 0, 0)
    friend_id0 = (lane_id+ 1)&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a1 *((lane_id < 31 )? tx0: ty0);
    friend_id1 = (lane_id+ 3)&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a1 *((lane_id < 29 )? tx1: ty1);
    // process (2, 0, 0)
    friend_id0 = (lane_id+ 2)&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a2 *((lane_id < 30 )? tx0: ty0);
    friend_id1 = (lane_id+ 4)&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a2 *((lane_id < 28 )? tx1: ty1);
    // process (0, 1, 0)
    friend_id0 = (lane_id+ 2)&31 ;
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    sum0 += a3 *((lane_id < 30 )? tx0: ty0);
    friend_id1 = (lane_id+ 4)&31 ;
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    sum1 += a3 *((lane_id < 28 )? tx1: ty1);
    // process (1, 1, 0)
    friend_id0 = (lane_id+ 3)&31 ;
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    sum0 += a4 *((lane_id < 29 )? tx0: ty0);
    friend_id1 = (lane_id+ 5)&31 ;
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    sum1 += a4 *((lane_id < 27 )? tx1: ty1);
    // process (2, 1, 0)
    friend_id0 = (lane_id+ 4)&31 ;
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    sum0 += a5 *((lane_id < 28 )? tx0: ty0);
    friend_id1 = (lane_id+ 6)&31 ;
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    sum1 += a5 *((lane_id < 26 )? tx1: ty1);
    // process (0, 2, 0)
    friend_id0 = (lane_id+ 4)&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a6 *((lane_id < 28 )? tx0: ty0);
    friend_id1 = (lane_id+ 6)&31 ;
    tx1 = __shfl(reg3, friend_id1);
    ty1 = __shfl(reg4, friend_id1);
    sum1 += a6 *((lane_id < 26 )? tx1: ty1);
    // process (1, 2, 0)
    friend_id0 = (lane_id+ 5)&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a7 *((lane_id < 27 )? tx0: ty0);
    friend_id1 = (lane_id+ 7)&31 ;
    tx1 = __shfl(reg3, friend_id1);
    ty1 = __shfl(reg4, friend_id1);
    sum1 += a7 *((lane_id < 25 )? tx1: ty1);
    // process (2, 2, 0)
    friend_id0 = (lane_id+ 6)&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a8 *((lane_id < 26 )? tx0: ty0);
    friend_id1 = (lane_id+ 8)&31 ;
    tx1 = __shfl(reg3, friend_id1);
    ty1 = __shfl(reg4, friend_id1);
    sum1 += a8 *((lane_id < 24 )? tx1: ty1);
    
    OUT_2D(j  ,i) = sum0; 
    OUT_2D(j+1,i) = sum1; 
}

__global__ void Stencil_Cuda_Reg4_2Blk1Wf(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, 
        DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, DATA_TYPE a8,
        int m, int n, int halo) 
{
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<2) + (lane_id>>5)  + halo;

    int warp_id0 = (threadIdx.x + blockIdx.x * blockDim.x)>>5;
    int warp_id1 = ((((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<2) + (lane_id>>5))>>0;

    DATA_TYPE reg0, reg1, reg2, reg3, reg4, reg5, reg6;
    int lane_id_it = lane_id;
    int new_id0 ;
    int new_id1 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    reg0 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    reg1 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    reg2 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    reg3 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    reg4 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    reg5 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
    new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
    reg6 = IN_2D(new_id1, new_id0) ;

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    int friend_id0, friend_id1;
    int friend_id2, friend_id3;
    DATA_TYPE tx0, ty0, tx1, ty1, tz1, tx2, ty2, tz2, tx3, ty3, tz3;
    // process (0, 0, 0)
    friend_id0 = (lane_id+ 0)&31 ;
    tx0 = __shfl(reg0, friend_id0);
    sum0 += a0 *(tx0);
    friend_id1 = (lane_id+ 2)&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a0 *((lane_id < 30 )? tx1: ty1);
    friend_id2 = (lane_id+ 4)&31 ;
    tx2 = __shfl(reg2, friend_id2);
    ty2 = __shfl(reg3, friend_id2);
    sum2 += a0 *((lane_id < 28 )? tx2: ty2);
    friend_id3 = (lane_id+ 6)&31 ;
    tx3 = __shfl(reg3, friend_id3);
    ty3 = __shfl(reg4, friend_id3);
    sum3 += a0 *((lane_id < 26 )? tx3: ty3);
    // process (1, 0, 0)
    friend_id0 = (lane_id+ 1)&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a1 *((lane_id < 31 )? tx0: ty0);
    friend_id1 = (lane_id+ 3)&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a1 *((lane_id < 29 )? tx1: ty1);
    friend_id2 = (lane_id+ 5)&31 ;
    tx2 = __shfl(reg2, friend_id2);
    ty2 = __shfl(reg3, friend_id2);
    sum2 += a1 *((lane_id < 27 )? tx2: ty2);
    friend_id3 = (lane_id+ 7)&31 ;
    tx3 = __shfl(reg3, friend_id3);
    ty3 = __shfl(reg4, friend_id3);
    sum3 += a1 *((lane_id < 25 )? tx3: ty3);
    // process (2, 0, 0)
    friend_id0 = (lane_id+ 2)&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a2 *((lane_id < 30 )? tx0: ty0);
    friend_id1 = (lane_id+ 4)&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a2 *((lane_id < 28 )? tx1: ty1);
    friend_id2 = (lane_id+ 6)&31 ;
    tx2 = __shfl(reg2, friend_id2);
    ty2 = __shfl(reg3, friend_id2);
    sum2 += a2 *((lane_id < 26 )? tx2: ty2);
    friend_id3 = (lane_id+ 8)&31 ;
    tx3 = __shfl(reg3, friend_id3);
    ty3 = __shfl(reg4, friend_id3);
    sum3 += a2 *((lane_id < 24 )? tx3: ty3);
    // process (0, 1, 0)
    friend_id0 = (lane_id+ 2)&31 ;
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    sum0 += a3 *((lane_id < 30 )? tx0: ty0);
    friend_id1 = (lane_id+ 4)&31 ;
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    sum1 += a3 *((lane_id < 28 )? tx1: ty1);
    friend_id2 = (lane_id+ 6)&31 ;
    tx2 = __shfl(reg3, friend_id2);
    ty2 = __shfl(reg4, friend_id2);
    sum2 += a3 *((lane_id < 26 )? tx2: ty2);
    friend_id3 = (lane_id+ 8)&31 ;
    tx3 = __shfl(reg4, friend_id3);
    ty3 = __shfl(reg5, friend_id3);
    sum3 += a3 *((lane_id < 24 )? tx3: ty3);
    // process (1, 1, 0)
    friend_id0 = (lane_id+ 3)&31 ;
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    sum0 += a4 *((lane_id < 29 )? tx0: ty0);
    friend_id1 = (lane_id+ 5)&31 ;
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    sum1 += a4 *((lane_id < 27 )? tx1: ty1);
    friend_id2 = (lane_id+ 7)&31 ;
    tx2 = __shfl(reg3, friend_id2);
    ty2 = __shfl(reg4, friend_id2);
    sum2 += a4 *((lane_id < 25 )? tx2: ty2);
    friend_id3 = (lane_id+ 9)&31 ;
    tx3 = __shfl(reg4, friend_id3);
    ty3 = __shfl(reg5, friend_id3);
    sum3 += a4 *((lane_id < 23 )? tx3: ty3);
    // process (2, 1, 0)
    friend_id0 = (lane_id+ 4)&31 ;
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    sum0 += a5 *((lane_id < 28 )? tx0: ty0);
    friend_id1 = (lane_id+ 6)&31 ;
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    sum1 += a5 *((lane_id < 26 )? tx1: ty1);
    friend_id2 = (lane_id+ 8)&31 ;
    tx2 = __shfl(reg3, friend_id2);
    ty2 = __shfl(reg4, friend_id2);
    sum2 += a5 *((lane_id < 24 )? tx2: ty2);
    friend_id3 = (lane_id+10)&31 ;
    tx3 = __shfl(reg4, friend_id3);
    ty3 = __shfl(reg5, friend_id3);
    sum3 += a5 *((lane_id < 22 )? tx3: ty3);
    // process (0, 2, 0)
    friend_id0 = (lane_id+ 4)&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a6 *((lane_id < 28 )? tx0: ty0);
    friend_id1 = (lane_id+ 6)&31 ;
    tx1 = __shfl(reg3, friend_id1);
    ty1 = __shfl(reg4, friend_id1);
    sum1 += a6 *((lane_id < 26 )? tx1: ty1);
    friend_id2 = (lane_id+ 8)&31 ;
    tx2 = __shfl(reg4, friend_id2);
    ty2 = __shfl(reg5, friend_id2);
    sum2 += a6 *((lane_id < 24 )? tx2: ty2);
    friend_id3 = (lane_id+10)&31 ;
    tx3 = __shfl(reg5, friend_id3);
    ty3 = __shfl(reg6, friend_id3);
    sum3 += a6 *((lane_id < 22 )? tx3: ty3);
    // process (1, 2, 0)
    friend_id0 = (lane_id+ 5)&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a7 *((lane_id < 27 )? tx0: ty0);
    friend_id1 = (lane_id+ 7)&31 ;
    tx1 = __shfl(reg3, friend_id1);
    ty1 = __shfl(reg4, friend_id1);
    sum1 += a7 *((lane_id < 25 )? tx1: ty1);
    friend_id2 = (lane_id+ 9)&31 ;
    tx2 = __shfl(reg4, friend_id2);
    ty2 = __shfl(reg5, friend_id2);
    sum2 += a7 *((lane_id < 23 )? tx2: ty2);
    friend_id3 = (lane_id+11)&31 ;
    tx3 = __shfl(reg5, friend_id3);
    ty3 = __shfl(reg6, friend_id3);
    sum3 += a7 *((lane_id < 21 )? tx3: ty3);
    // process (2, 2, 0)
    friend_id0 = (lane_id+ 6)&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a8 *((lane_id < 26 )? tx0: ty0);
    friend_id1 = (lane_id+ 8)&31 ;
    tx1 = __shfl(reg3, friend_id1);
    ty1 = __shfl(reg4, friend_id1);
    sum1 += a8 *((lane_id < 24 )? tx1: ty1);
    friend_id2 = (lane_id+10)&31 ;
    tx2 = __shfl(reg4, friend_id2);
    ty2 = __shfl(reg5, friend_id2);
    sum2 += a8 *((lane_id < 22 )? tx2: ty2);
    friend_id3 = (lane_id+12)&31 ;
    tx3 = __shfl(reg5, friend_id3);
    ty3 = __shfl(reg6, friend_id3);
    sum3 += a8 *((lane_id < 20 )? tx3: ty3);
    
    OUT_2D(j  ,i) = sum0; 
    OUT_2D(j+1,i) = sum1; 
    OUT_2D(j+2,i) = sum2; 
    OUT_2D(j+3,i) = sum3; 
}
// div 4 and plus lane_id>>3 (determined by the warp dimension, eg. warp dim is 4 by 8)
__global__ void Stencil_Cuda_Reg4_2Blk2Wf(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, 
        DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, DATA_TYPE a8,
        int m, int n, int halo) 
{
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<4) + (lane_id>>3)  + halo;

    int warp_id0 = (threadIdx.x + blockIdx.x * blockDim.x)>>3;
    int warp_id1 = ((((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<4) + (lane_id>>3))>>2;

    DATA_TYPE reg0, reg1, reg2, reg3, reg4, reg5;
    int lane_id_it = lane_id;
    int new_id0 ;
    int new_id1 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10 ;
    reg0 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10 ;
    reg1 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10 ;
    reg2 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10 ;
    reg3 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10 ;
    reg4 = IN_2D(new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10 ;
    new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
    new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
    reg5 = IN_2D(new_id1, new_id0) ;

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    int friend_id0, friend_id1;
    int friend_id2, friend_id3;
    DATA_TYPE tx0, ty0, tx1, ty1, tz1, tx2, ty2, tz2, tx3, ty3, tz3;

    // process (0, 0, 0)
    friend_id0 = (lane_id+ 0+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a0 *((lane_id < 26 )? tx0: ty0);
    friend_id1 = (lane_id+ 8+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a0 *((lane_id < 20 )? tx1: ty1);
    friend_id2 = (lane_id+16+((lane_id>>3)*2))&31 ;
    tx2 = __shfl(reg2, friend_id2);
    ty2 = __shfl(reg3, friend_id2);
    sum2 += a0 *((lane_id < 14 )? tx2: ty2);
    friend_id3 = (lane_id+24+((lane_id>>3)*2))&31 ;
    tx3 = __shfl(reg3, friend_id3);
    ty3 = __shfl(reg4, friend_id3);
    sum3 += a0 *((lane_id < 8 )? tx3: ty3);
    // process (1, 0, 0)
    friend_id0 = (lane_id+ 1+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a1 *((lane_id < 25 )? tx0: ty0);
    friend_id1 = (lane_id+ 9+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a1 *((lane_id < 19 )? tx1: ty1);
    friend_id2 = (lane_id+17+((lane_id>>3)*2))&31 ;
    tx2 = __shfl(reg2, friend_id2);
    ty2 = __shfl(reg3, friend_id2);
    sum2 += a1 *((lane_id < 13 )? tx2: ty2);
    friend_id3 = (lane_id+25+((lane_id>>3)*2))&31 ;
    tx3 = __shfl(reg3, friend_id3);
    ty3 = __shfl(reg4, friend_id3);
    sum3 += a1 *((lane_id < 7 )? tx3: ty3);
    // process (2, 0, 0)
    friend_id0 = (lane_id+ 2+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a2 *((lane_id < 24 )? tx0: ty0);
    friend_id1 = (lane_id+10+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a2 *((lane_id < 18 )? tx1: ty1);
    friend_id2 = (lane_id+18+((lane_id>>3)*2))&31 ;
    tx2 = __shfl(reg2, friend_id2);
    ty2 = __shfl(reg3, friend_id2);
    sum2 += a2 *((lane_id < 12 )? tx2: ty2);
    friend_id3 = (lane_id+26+((lane_id>>3)*2))&31 ;
    tx3 = __shfl(reg3, friend_id3);
    ty3 = __shfl(reg4, friend_id3);
    sum3 += a2 *((lane_id < 6 )? tx3: ty3);
    // process (0, 1, 0)
    friend_id0 = (lane_id+10+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a3 *((lane_id < 18 )? tx0: ty0);
    friend_id1 = (lane_id+18+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a3 *((lane_id < 12 )? tx1: ty1);
    friend_id2 = (lane_id+26+((lane_id>>3)*2))&31 ;
    tx2 = __shfl(reg2, friend_id2);
    ty2 = __shfl(reg3, friend_id2);
    sum2 += a3 *((lane_id < 6 )? tx2: ty2);
    friend_id3 = (lane_id+ 2+((lane_id>>3)*2))&31 ;
    tx3 = __shfl(reg4, friend_id3);
    ty3 = __shfl(reg5, friend_id3);
    sum3 += a3 *((lane_id < 24 )? tx3: ty3);
    // process (1, 1, 0)
    friend_id0 = (lane_id+11+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a4 *((lane_id < 17 )? tx0: ty0);
    friend_id1 = (lane_id+19+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a4 *((lane_id < 11 )? tx1: ty1);
    friend_id2 = (lane_id+27+((lane_id>>3)*2))&31 ;
    tx2 = __shfl(reg2, friend_id2);
    ty2 = __shfl(reg3, friend_id2);
    tz2 = __shfl(reg4, friend_id2);
    sum2 += a4 *((lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2));
    friend_id3 = (lane_id+ 3+((lane_id>>3)*2))&31 ;
    tx3 = __shfl(reg4, friend_id3);
    ty3 = __shfl(reg5, friend_id3);
    sum3 += a4 *((lane_id < 24 )? tx3: ty3);
    // process (2, 1, 0)
    friend_id0 = (lane_id+12+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a5 *((lane_id < 16 )? tx0: ty0);
    friend_id1 = (lane_id+20+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    sum1 += a5 *((lane_id < 10 )? tx1: ty1);
    friend_id2 = (lane_id+28+((lane_id>>3)*2))&31 ;
    tx2 = __shfl(reg2, friend_id2);
    ty2 = __shfl(reg3, friend_id2);
    tz2 = __shfl(reg4, friend_id2);
    sum2 += a5 *((lane_id < 4 )? tx2: ((lane_id < 30)? ty2: tz2));
    friend_id3 = (lane_id+ 4+((lane_id>>3)*2))&31 ;
    tx3 = __shfl(reg4, friend_id3);
    ty3 = __shfl(reg5, friend_id3);
    sum3 += a5 *((lane_id < 24 )? tx3: ty3);
    // process (0, 2, 0)
    friend_id0 = (lane_id+20+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a6 *((lane_id < 10 )? tx0: ty0);
    friend_id1 = (lane_id+28+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    tz1 = __shfl(reg3, friend_id1);
    sum1 += a6 *((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
    friend_id2 = (lane_id+ 4+((lane_id>>3)*2))&31 ;
    tx2 = __shfl(reg3, friend_id2);
    ty2 = __shfl(reg4, friend_id2);
    sum2 += a6 *((lane_id < 24 )? tx2: ty2);
    friend_id3 = (lane_id+12+((lane_id>>3)*2))&31 ;
    tx3 = __shfl(reg4, friend_id3);
    ty3 = __shfl(reg5, friend_id3);
    sum3 += a6 *((lane_id < 16 )? tx3: ty3);
    // process (1, 2, 0)
    friend_id0 = (lane_id+21+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a7 *((lane_id < 9 )? tx0: ty0);
    friend_id1 = (lane_id+29+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    tz1 = __shfl(reg3, friend_id1);
    sum1 += a7 *((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
    friend_id2 = (lane_id+ 5+((lane_id>>3)*2))&31 ;
    tx2 = __shfl(reg3, friend_id2);
    ty2 = __shfl(reg4, friend_id2);
    sum2 += a7 *((lane_id < 23 )? tx2: ty2);
    friend_id3 = (lane_id+13+((lane_id>>3)*2))&31 ;
    tx3 = __shfl(reg4, friend_id3);
    ty3 = __shfl(reg5, friend_id3);
    sum3 += a7 *((lane_id < 16 )? tx3: ty3);
    // process (2, 2, 0)
    friend_id0 = (lane_id+22+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a8 *((lane_id < 8 )? tx0: ty0);
    friend_id1 = (lane_id+30+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    tz1 = __shfl(reg3, friend_id1);
    sum1 += a8 *((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
    friend_id2 = (lane_id+ 6+((lane_id>>3)*2))&31 ;
    tx2 = __shfl(reg3, friend_id2);
    ty2 = __shfl(reg4, friend_id2);
    sum2 += a8 *((lane_id < 22 )? tx2: ty2);
    friend_id3 = (lane_id+14+((lane_id>>3)*2))&31 ;
    tx3 = __shfl(reg4, friend_id3);
    ty3 = __shfl(reg5, friend_id3);
    sum3 += a8 *((lane_id < 16 )? tx3: ty3);

    

    OUT_2D(j   ,i) = sum0; 
    OUT_2D(j+4 ,i) = sum1; 
    OUT_2D(j+8 ,i) = sum2; 
    OUT_2D(j+12,i) = sum3; 
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
            std::cout << "difference: " << fabs(test[i]-ref[i])-precision << std::endl;
            std::cout << "wrong at " << i << " test:" << test[i] << " (ref: " << ref[i] << ")";
            std::cout << std::endl;
            flag = false;
            break;
        }
    }
    return flag;
}

__global__ void Stencil_Cuda_Lds_2BlkBrc(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, 
        DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, DATA_TYPE a8,
        int m, int n, int halo) 
{
    __shared__ DATA_TYPE local[8+2][32+2];
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo; 
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo; 
    int li = threadIdx.x + halo;
    int lj = threadIdx.y + halo;

    LOC_2D(lj,li) = IN_2D(j,i);
    if(li == halo)               LOC_2D(lj  ,li-1) = IN_2D(j  ,i-1);
    if(li == 32  )               LOC_2D(lj  ,li+1) = IN_2D(j  ,i+1);
    if(lj == halo)               LOC_2D(lj-1,li  ) = IN_2D(j-1,i  );
    if(lj == 8   )               LOC_2D(lj+1,li  ) = IN_2D(j+1,i  );
    if(li == halo && lj == halo) LOC_2D(lj-1,li-1) = IN_2D(j-1,i-1);
    if(li == 32   && lj == halo) LOC_2D(lj-1,li+1) = IN_2D(j-1,i+1);
    if(li == halo && lj == 8   ) LOC_2D(lj+1,li-1) = IN_2D(j+1,i-1);
    if(li == 32   && lj == 8   ) LOC_2D(lj+1,li+1) = IN_2D(j+1,i+1);

    __syncthreads();

    OUT_2D(j,i) = a0 * LOC_2D(lj-1,li-1) +
                  a1 * LOC_2D(lj  ,li-1) +
                  a2 * LOC_2D(lj+1,li-1) +
                  a3 * LOC_2D(lj-1,li  ) +
                  a4 * LOC_2D(lj  ,li  ) +
                  a5 * LOC_2D(lj+1,li  ) +
                  a6 * LOC_2D(lj-1,li+1) +
                  a7 * LOC_2D(lj  ,li+1) +
                  a8 * LOC_2D(lj+1,li+1) ;
    
}

__global__ void Stencil_Cuda_Lds_2BlkCyc(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, 
        DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, DATA_TYPE a8,
        int m, int n, int halo) 
{
    __shared__ DATA_TYPE local[8+2][32+2];
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo; 
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo; 
    int li = threadIdx.x + halo;
    int lj = threadIdx.y + halo;

    unsigned int lane_id = threadIdx.x + threadIdx.y * blockDim.x;

    int blk_id_x = blockIdx.x;//(threadIdx.x + blockIdx.x * blockDim.x)>>4;
    int blk_id_y = blockIdx.y;//(threadIdx.y + blockIdx.y * blockDim.y)>>4;

    int new_i  = (blk_id_x<<5) + lane_id%34;
    int new_j  = (blk_id_y<<3) + lane_id/34;
    int new_li = lane_id%34;
    int new_lj = lane_id/34;
    LOC_2D(new_lj,new_li) = IN_2D(new_j,new_i);
    new_i  = (blk_id_x<<5) + (lane_id+256)%34;
    new_j  = (blk_id_y<<3) + (lane_id+256)/34;
    new_li = (lane_id+256)%34;
    new_lj = (lane_id+256)/34;
    new_i  = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j  = (new_j < m+2*halo)? new_j: m+2*halo-1;
    if(new_li < 34 &&  new_lj < 10)
        LOC_2D(new_lj,new_li) = IN_2D(new_j,new_i);

    __syncthreads();


    OUT_2D(j,i) = a0 * LOC_2D(lj-1,li-1) +
                  a1 * LOC_2D(lj  ,li-1) +
                  a2 * LOC_2D(lj+1,li-1) +
                  a3 * LOC_2D(lj-1,li  ) +
                  a4 * LOC_2D(lj  ,li  ) +
                  a5 * LOC_2D(lj+1,li  ) +
                  a6 * LOC_2D(lj-1,li+1) +
                  a7 * LOC_2D(lj  ,li+1) +
                  a8 * LOC_2D(lj+1,li+1) ;
}


int main(int argc, char **argv)
{
#ifdef __DEBUG
    int m = 64;
    int n = 64;
#else
    int m = 4096;
    int n = 4096;
#endif
    int halo = 1; 
    int total = (m+2*halo)*(n+2*halo);
    const int K = 9;
#ifdef __DEBUG
    DATA_TYPE args[K] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
#else
    DATA_TYPE args[K] = {0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11};
#endif

    DATA_TYPE *in = new DATA_TYPE[total];
    DATA_TYPE *out_ref = new DATA_TYPE[total];
    unsigned int seed = time(NULL);
    Clear_Output_2D(in, m, n, halo);
    Clear_Output_2D(out_ref, m, n, halo);
    Init_Input_2D(in, m, n, halo, seed);

    // Show_Me(in, m, n, halo, "Input:");
    for(int i=0; i< ITER; i++)
    {
        Stencil_Seq(in, out_ref, 
                args[0], args[1], args[2], args[3], args[4], 
                args[5], args[6], args[7], args[8],
                m, n, halo);
        swap(in, out_ref);
    }
    swap(in, out_ref);
    // Show_Me(out_ref, m, n, halo, "Output:");

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
    Init_Input_2D(in, m, n, halo, seed);
    Clear_Output_2D(out, m, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/32;
    dimGrid.y = (m)/8;
    dimGrid.z = 1;
    dimBlock.x = 32;
    dimBlock.y = 8;
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_L1_2Blk<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], 
                args[5], args[6], args[7], args[8],
                m, n, halo); 
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_L1_2Blk: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_L1_2Blk Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D9, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shared Memory with Branch
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, halo, seed);
    Clear_Output_2D(out, m, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/32;
    dimGrid.y = (m)/8;
    dimGrid.z = 1;
    dimBlock.x = 32;
    dimBlock.y = 8;
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Lds_2BlkBrc<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], 
                args[5], args[6], args[7], args[8],
                m, n, halo); 
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Lds_2BlkBrc: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Lds_2BlkBrc Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D9, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));


    // Cuda Shared Memory with Cyclic
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, halo, seed);
    Clear_Output_2D(out, m, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/32;
    dimGrid.y = (m)/8;
    dimGrid.z = 1;
    dimBlock.x = 32;
    dimBlock.y = 8;
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Lds_2BlkCyc<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], 
                args[5], args[6], args[7], args[8],
                m, n, halo); 
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Lds_2BlkCyc: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Lds_2BlkCyc Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D9, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl version 
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, halo, seed);
    Clear_Output_2D(out, m, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/8;
    dimGrid.y = (m)/32;
    dimGrid.z = 1;
    dimBlock.x = 8;
    dimBlock.y = 32;
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Reg1_2Blk2Wf<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], 
                args[5], args[6], args[7], args[8],
                m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Reg1_2Blk2Wf: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Reg1_2Blk2Wf Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D9, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl2 version 
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, halo, seed);
    Clear_Output_2D(out, m, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/8;
    dimGrid.y = (m)/(32*2);
    dimGrid.z = 1;
    dimBlock.x = 8;
    dimBlock.y = 32;
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Reg2_2Blk2Wf<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], 
                args[5], args[6], args[7], args[8],
                m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Reg2_2Blk2Wf: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Reg2_2Blk2Wf Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D9, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl4 version 
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, halo, seed);
    Clear_Output_2D(out, m, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/8;
    dimGrid.y = (m)/(32*4);
    dimGrid.z = 1;
    dimBlock.x = 8;
    dimBlock.y = 32;
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Reg4_2Blk2Wf<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], 
                args[5], args[6], args[7], args[8],
                m, n, halo); 
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Reg4_2Blk2Wf: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Reg4_2Blk2Wf Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D9, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl with 1D-Warp
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, halo, seed);
    Clear_Output_2D(out, m, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/32;
    dimGrid.y = (m)/8;
    dimGrid.z = 1;
    dimBlock.x = 32;
    dimBlock.y = 8;
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Reg1_2Blk1Wf<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], 
                args[5], args[6], args[7], args[8],
                m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Reg1_2Blk1Wf: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Reg1_2Blk1Wf Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D9, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl2 with 1D-Warp
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, halo, seed);
    Clear_Output_2D(out, m, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/32;
    dimGrid.y = (m)/(8*2);
    dimGrid.z = 1;
    dimBlock.x = 32;
    dimBlock.y = 8;
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Reg2_2Blk1Wf<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], 
                args[5], args[6], args[7], args[8],
                m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Reg2_2Blk1Wf: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Reg2_2Blk1Wf Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D9, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl4 with 1D-Warp
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, halo, seed);
    Clear_Output_2D(out, m, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/32;
    dimGrid.y = (m)/(8*4);
    dimGrid.z = 1;
    dimBlock.x = 32;
    dimBlock.y = 8;
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Reg4_2Blk1Wf<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], 
                args[5], args[6], args[7], args[8],
                m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Reg4_2Blk1Wf: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Reg4_2Blk1Wf Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D9, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    cudaFree(in_d);
    cudaFree(out_d);

    delete[] in;
    delete[] out;
    delete[] out_ref;
}


