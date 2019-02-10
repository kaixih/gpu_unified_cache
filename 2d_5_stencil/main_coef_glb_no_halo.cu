
#include <iostream>
#include <metrics.h>

using namespace std;
#define IN_2D(_x,_y) in[(_x)*(N)+(_y)]
#define OUT_2D(_x,_y) out[(_x)*(N)+(_y)]
#define LOC_2D(_x,_y) local[(_x)*(18)+(_y)]
#define ARG_2D(_l,_x,_y)   args[(_l)*(N)*(M)+(_x)*(N)+(_y)]

// #define __DEBUG

#ifdef __DEBUG
#define ITER 1
#else
#define ITER 100
#endif

// #define DATA_TYPE float
// #define DATA_TYPE double
#define warpSize 32 

__device__ __inline__ double shfl(double x, int lane)
{
    // Split the double number into 2 32b registers.
    int lo, hi;
    asm volatile( "mov.b32 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x));
    // Shuffle the two 32b registers.
    lo = __shfl(lo, lane);
    hi = __shfl(hi, lane);
    // Recreate the 64b number.
    asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(x) : "r"(lo), "r"(hi));
    return x;
}

void Init_Input_2D(DATA_TYPE *in, int M, int N, unsigned int seed)
{
    srand(seed);

    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
#ifdef __DEBUG
                IN_2D(i,j) = 1; 
#else
                IN_2D(i,j) = (DATA_TYPE)rand()*100.0 / ((long)RAND_MAX);
#endif
        }
    }
}

void Init_Args_2D(DATA_TYPE *args, int l, int M, int N, DATA_TYPE val)
{
    for(int k = 0; k < l; k++)
    {
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                ARG_2D(k,i,j) = val; 
            }
        }
    }
}

void Clear_Output_2D(DATA_TYPE *in, int M, int N)
{
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            IN_2D(i,j) = 0; 
        }
    }
}

__global__ void Stencil_Cuda(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int M, int N) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;
    int n = (j == 0)      ? j : j - 1;
    int s = (j == M-1)    ? j : j + 1;
    int w = (i == 0)      ? i : i - 1;
    int e = (i == N-1)    ? i : i + 1;
    OUT_2D(j,i) = ARG_2D(0,j,i)*IN_2D(n  ,i  ) + 
                  ARG_2D(1,j,i)*IN_2D(j  ,w  ) + 
                  ARG_2D(2,j,i)*IN_2D(s  ,i  ) +
                  ARG_2D(3,j,i)*IN_2D(j  ,e  ) + 
                  ARG_2D(4,j,i)*IN_2D(j  ,i  ) ;
}

void Stencil_Seq(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int M, int N)
{
    for(int j = 0; j < M; j++)
    {
        int n = (j == 0)      ? j : j - 1;
        int s = (j == M-1)    ? j : j + 1;
        for(int i = 0; i < N; i++)
        {
            int w = (i == 0)      ? i : i - 1;
            int e = (i == N-1)    ? i : i + 1;
            OUT_2D(j,i) = ARG_2D(0,j,i) * IN_2D(n  ,i  ) +
                          ARG_2D(1,j,i) * IN_2D(j  ,w  ) +
                          ARG_2D(2,j,i) * IN_2D(s  ,i  ) +
                          ARG_2D(3,j,i) * IN_2D(j  ,e  ) +
                          ARG_2D(4,j,i) * IN_2D(j  ,i  ) ;
        }
    }
}

void Show_Me(DATA_TYPE *in, int M, int N, std::string prompt)
{
    std::cout << prompt << std::endl;
    for(int j = 0; j < M; j++)
    {
        for(int i = 0; i < N; i++)
            std::cout << IN_2D(j,i) << ",";
        std::cout << std::endl;
    }
}

__global__ void Stencil_Cuda_Shfl4(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int M, int N) 
{
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<4) + (lane_id>>3) ;

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3;
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<4) + (lane_id>>3))>>2;

    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5;
    int new_i = (warp_id_x<<3) + lane_id%10-1;
    int new_j = (warp_id_y<<2) + lane_id/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput0 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+32)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput1 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+64)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput2 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+96)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput3 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+128)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput4 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+160)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput5 = IN_2D(new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    int friend_id0, friend_id1;
    int friend_id2, friend_id3;
    DATA_TYPE tx0, ty0, tx1, ty1, tz1, tx2, ty2, tz2, tx3, ty3, tz3;
    
    friend_id0 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tx2 = __shfl(threadInput2, friend_id2);
    ty2 = __shfl(threadInput3, friend_id2);
    tx3 = __shfl(threadInput3, friend_id3);
    ty3 = __shfl(threadInput4, friend_id3);
    sum0 += ARG_2D(0,j   ,i)*((lane_id < 25)? tx0: ty0);
    sum1 += ARG_2D(0,j+4 ,i)*((lane_id < 19)? tx1: ty1);
    sum2 += ARG_2D(0,j+8 ,i)*((lane_id < 13)? tx2: ty2);
    sum3 += ARG_2D(0,j+12,i)*((lane_id < 7 )? tx3: ty3);

   
    friend_id0 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tx2 = __shfl(threadInput2, friend_id2);
    ty2 = __shfl(threadInput3, friend_id2);
    ty3 = __shfl(threadInput4, friend_id3);
    tz3 = __shfl(threadInput5, friend_id3);
    sum0 += ARG_2D(1,j   ,i)*((lane_id < 18)? tx0: ty0);
    sum1 += ARG_2D(1,j+4 ,i)*((lane_id < 12)? tx1: ty1);
    sum2 += ARG_2D(1,j+8 ,i)*((lane_id < 6 )? tx2: ty2);
    sum3 += ARG_2D(1,j+12,i)*((lane_id < 24)? ty3: tz3);

  
    friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tx2 = __shfl(threadInput2, friend_id2);
    ty2 = __shfl(threadInput3, friend_id2);
    tz2 = __shfl(threadInput4, friend_id2);
    ty3 = __shfl(threadInput4, friend_id3);
    tz3 = __shfl(threadInput5, friend_id3);
    sum0 += ARG_2D(4,j   ,i)*((lane_id < 17)? tx0: ty0);
    sum1 += ARG_2D(4,j+4 ,i)*((lane_id < 11)? tx1: ty1);
    sum2 += ARG_2D(4,j+8 ,i)*((lane_id < 5)? tx2: ((lane_id < 31)? ty2: tz2));
    sum3 += ARG_2D(4,j+12,i)*((lane_id < 24)? ty3: tz3);

 
    friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tx2 = __shfl(threadInput2, friend_id2);
    ty2 = __shfl(threadInput3, friend_id2);
    tz2 = __shfl(threadInput4, friend_id2);
    ty3 = __shfl(threadInput4, friend_id3);
    tz3 = __shfl(threadInput5, friend_id3);
    sum0 += ARG_2D(3,j   ,i)*((lane_id < 16)? tx0: ty0);
    sum1 += ARG_2D(3,j+4 ,i)*((lane_id < 10)? tx1: ty1);
    sum2 += ARG_2D(3,j+8 ,i)*((lane_id < 4 )? tx2: ((lane_id < 30)? ty2: tz2));
    sum3 += ARG_2D(3,j+12,i)*((lane_id < 24)? ty3: tz3);


    friend_id0 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tz1 = __shfl(threadInput3, friend_id1);
    ty2 = __shfl(threadInput3, friend_id2);
    tz2 = __shfl(threadInput4, friend_id2);
    ty3 = __shfl(threadInput4, friend_id3);
    tz3 = __shfl(threadInput5, friend_id3);
    sum0 += ARG_2D(2,j   ,i)*((lane_id < 9 )? tx0: ty0);
    sum1 += ARG_2D(2,j+4 ,i)*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
    sum2 += ARG_2D(2,j+8 ,i)*((lane_id < 23)? ty2: tz2);
    sum3 += ARG_2D(2,j+12,i)*((lane_id < 16)? ty3: tz3);


    OUT_2D(j   ,i) = sum0; 
    OUT_2D(j+4 ,i) = sum1; 
    OUT_2D(j+8 ,i) = sum2; 
    OUT_2D(j+12,i) = sum3; 
}

__global__ void Stencil_Cuda_Shfl8(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int M, int N) 
{
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<5) + (lane_id>>3) ;

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3;
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<5) + (lane_id>>3))>>2;

    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5;
    DATA_TYPE threadInput6, threadInput7, threadInput8, threadInput9, threadInput10;
    int new_i = (warp_id_x<<3) + lane_id%10-1;
    int new_j = (warp_id_y<<2) + lane_id/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput0 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+32)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput1 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+64)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput2 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+96)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput3 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+128)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput4 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+160)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput5 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+192)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+192)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput6 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+224)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+224)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput7 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+256)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+256)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput8 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+288)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+288)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput9 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+320)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+320)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput10 = IN_2D(new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    DATA_TYPE sum4 = 0.0;
    DATA_TYPE sum5 = 0.0;
    DATA_TYPE sum6 = 0.0;
    DATA_TYPE sum7 = 0.0;
    int friend_id0, friend_id1, friend_id2, friend_id3;
    // int friend_id4, friend_id5, friend_id6, friend_id7;
    DATA_TYPE tx0, ty0, tx1, ty1, tz1, tx2, ty2, tz2, tx3, ty3, tz3;
    DATA_TYPE rx0, ry0, rx1, ry1, rz1, rx2, ry2, rz2, rx3, ry3, rz3;
    
    friend_id0 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tx2 = __shfl(threadInput2, friend_id2);
    ty2 = __shfl(threadInput3, friend_id2);
    tx3 = __shfl(threadInput3, friend_id3);
    ty3 = __shfl(threadInput4, friend_id3);
    rx0 = __shfl(threadInput5, friend_id0);
    ry0 = __shfl(threadInput6, friend_id0);
    rx1 = __shfl(threadInput6, friend_id1);
    ry1 = __shfl(threadInput7, friend_id1);
    rx2 = __shfl(threadInput7, friend_id2);
    ry2 = __shfl(threadInput8, friend_id2);
    rx3 = __shfl(threadInput8, friend_id3);
    ry3 = __shfl(threadInput9, friend_id3);
    sum0 += ARG_2D(0,j   ,i)*((lane_id < 25)? tx0: ty0);
    sum1 += ARG_2D(0,j+4 ,i)*((lane_id < 19)? tx1: ty1);
    sum2 += ARG_2D(0,j+8 ,i)*((lane_id < 13)? tx2: ty2);
    sum3 += ARG_2D(0,j+12,i)*((lane_id < 7 )? tx3: ty3);
    sum4 += ARG_2D(0,j+16,i)*((lane_id < 25)? rx0: ry0);
    sum5 += ARG_2D(0,j+20,i)*((lane_id < 19)? rx1: ry1);
    sum6 += ARG_2D(0,j+24,i)*((lane_id < 13)? rx2: ry2);
    sum7 += ARG_2D(0,j+28,i)*((lane_id < 7 )? rx3: ry3);

    friend_id0 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tx2 = __shfl(threadInput2, friend_id2);
    ty2 = __shfl(threadInput3, friend_id2);
    ty3 = __shfl(threadInput4, friend_id3);
    tz3 = __shfl(threadInput5, friend_id3);
    rx0 = __shfl(threadInput5, friend_id0);
    ry0 = __shfl(threadInput6, friend_id0);
    rx1 = __shfl(threadInput6, friend_id1);
    ry1 = __shfl(threadInput7, friend_id1);
    rx2 = __shfl(threadInput7, friend_id2);
    ry2 = __shfl(threadInput8, friend_id2);
    ry3 = __shfl(threadInput9, friend_id3);
    rz3 = __shfl(threadInput10, friend_id3);
    sum0 += ARG_2D(1,j   ,i)*((lane_id < 18)? tx0: ty0);
    sum1 += ARG_2D(1,j+4 ,i)*((lane_id < 12)? tx1: ty1);
    sum2 += ARG_2D(1,j+8 ,i)*((lane_id < 6 )? tx2: ty2);
    sum3 += ARG_2D(1,j+12,i)*((lane_id < 24)? ty3: tz3);
    sum4 += ARG_2D(1,j+16,i)*((lane_id < 18)? rx0: ry0);
    sum5 += ARG_2D(1,j+20,i)*((lane_id < 12)? rx1: ry1);
    sum6 += ARG_2D(1,j+24,i)*((lane_id < 6 )? rx2: ry2);
    sum7 += ARG_2D(1,j+28,i)*((lane_id < 24)? ry3: rz3);

    friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0 , friend_id0);
    ty0 = __shfl(threadInput1 , friend_id0);
    tx1 = __shfl(threadInput1 , friend_id1);
    ty1 = __shfl(threadInput2 , friend_id1);
    tx2 = __shfl(threadInput2 , friend_id2);
    ty2 = __shfl(threadInput3 , friend_id2);
    tz2 = __shfl(threadInput4 , friend_id2);
    ty3 = __shfl(threadInput4 , friend_id3);
    tz3 = __shfl(threadInput5 , friend_id3);
    rx0 = __shfl(threadInput5 , friend_id0);
    ry0 = __shfl(threadInput6 , friend_id0);
    rx1 = __shfl(threadInput6 , friend_id1);
    ry1 = __shfl(threadInput7 , friend_id1);
    rx2 = __shfl(threadInput7 , friend_id2);
    ry2 = __shfl(threadInput8 , friend_id2);
    rz2 = __shfl(threadInput9 , friend_id2);
    ry3 = __shfl(threadInput9 , friend_id3);
    rz3 = __shfl(threadInput10, friend_id3);
    sum0 += ARG_2D(4,j   ,i)*((lane_id < 17)? tx0: ty0);
    sum1 += ARG_2D(4,j+4 ,i)*((lane_id < 11)? tx1: ty1);
    sum2 += ARG_2D(4,j+8 ,i)*((lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2));
    sum3 += ARG_2D(4,j+12,i)*((lane_id < 24)? ty3: tz3);
    sum4 += ARG_2D(4,j+16,i)*((lane_id < 17)? rx0: ry0);
    sum5 += ARG_2D(4,j+20,i)*((lane_id < 11)? rx1: ry1);
    sum6 += ARG_2D(4,j+24,i)*((lane_id < 5 )? rx2: ((lane_id < 31)? ry2: rz2));
    sum7 += ARG_2D(4,j+28,i)*((lane_id < 24)? ry3: rz3);

    friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0 , friend_id0);
    ty0 = __shfl(threadInput1 , friend_id0);
    tx1 = __shfl(threadInput1 , friend_id1);
    ty1 = __shfl(threadInput2 , friend_id1);
    tx2 = __shfl(threadInput2 , friend_id2);
    ty2 = __shfl(threadInput3 , friend_id2);
    tz2 = __shfl(threadInput4 , friend_id2);
    ty3 = __shfl(threadInput4 , friend_id3);
    tz3 = __shfl(threadInput5 , friend_id3);
    rx0 = __shfl(threadInput5 , friend_id0);
    ry0 = __shfl(threadInput6 , friend_id0);
    rx1 = __shfl(threadInput6 , friend_id1);
    ry1 = __shfl(threadInput7 , friend_id1);
    rx2 = __shfl(threadInput7 , friend_id2);
    ry2 = __shfl(threadInput8 , friend_id2);
    rz2 = __shfl(threadInput9 , friend_id2);
    ry3 = __shfl(threadInput9 , friend_id3);
    rz3 = __shfl(threadInput10, friend_id3);
    sum0 += ARG_2D(3,j   ,i)*((lane_id < 16)? tx0: ty0);
    sum1 += ARG_2D(3,j+4 ,i)*((lane_id < 10)? tx1: ty1);
    sum2 += ARG_2D(3,j+8 ,i)*((lane_id < 4 )? tx2: ((lane_id < 30)? ty2: tz2));
    sum3 += ARG_2D(3,j+12,i)*((lane_id < 24)? ty3: tz3);
    sum4 += ARG_2D(3,j+16,i)*((lane_id < 16)? rx0: ry0);
    sum5 += ARG_2D(3,j+20,i)*((lane_id < 10)? rx1: ry1);
    sum6 += ARG_2D(3,j+24,i)*((lane_id < 4 )? rx2: ((lane_id < 30)? ry2: rz2));
    sum7 += ARG_2D(3,j+28,i)*((lane_id < 24)? ry3: rz3);

    friend_id0 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0 , friend_id0);
    ty0 = __shfl(threadInput1 , friend_id0);
    tx1 = __shfl(threadInput1 , friend_id1);
    ty1 = __shfl(threadInput2 , friend_id1);
    tz1 = __shfl(threadInput3 , friend_id1);
    ty2 = __shfl(threadInput3 , friend_id2);
    tz2 = __shfl(threadInput4 , friend_id2);
    ty3 = __shfl(threadInput4 , friend_id3);
    tz3 = __shfl(threadInput5 , friend_id3);
    rx0 = __shfl(threadInput5 , friend_id0);
    ry0 = __shfl(threadInput6 , friend_id0);
    rx1 = __shfl(threadInput6 , friend_id1);
    ry1 = __shfl(threadInput7 , friend_id1);
    rz1 = __shfl(threadInput8 , friend_id1);
    ry2 = __shfl(threadInput8 , friend_id2);
    rz2 = __shfl(threadInput9 , friend_id2);
    ry3 = __shfl(threadInput9 , friend_id3);
    rz3 = __shfl(threadInput10, friend_id3);
    sum0 += ARG_2D(2,j   ,i)*((lane_id < 9 )? tx0: ty0);
    sum1 += ARG_2D(2,j+4 ,i)*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
    sum2 += ARG_2D(2,j+8 ,i)*((lane_id < 23)? ty2: tz2);
    sum3 += ARG_2D(2,j+12,i)*((lane_id < 16)? ty3: tz3);
    sum4 += ARG_2D(2,j+16,i)*((lane_id < 9 )? rx0: ry0);
    sum5 += ARG_2D(2,j+20,i)*((lane_id < 3 )? rx1: ((lane_id < 29)? ry1: rz1));
    sum6 += ARG_2D(2,j+24,i)*((lane_id < 23)? ry2: rz2);
    sum7 += ARG_2D(2,j+28,i)*((lane_id < 16)? ry3: rz3);

    OUT_2D(j   ,i) = sum0; 
    OUT_2D(j+4 ,i) = sum1; 
    OUT_2D(j+8 ,i) = sum2; 
    OUT_2D(j+12,i) = sum3; 
    OUT_2D(j+16,i) = sum4; 
    OUT_2D(j+20,i) = sum5; 
    OUT_2D(j+24,i) = sum6; 
    OUT_2D(j+28,i) = sum7; 
}

__global__ void Stencil_Cuda_Shfl4_2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int M, int N) 
{
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;

    int i = (((threadIdx.x + blockIdx.x * blockDim.x)>>3)<<4) + (lane_id&7)  ;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<3) + (lane_id>>3) ;

    int warp_id_x = ((((threadIdx.x + blockIdx.x * blockDim.x)>>3)<<4) + (lane_id&7) )>>3;
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<3) + (lane_id>>3))>>2;

    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5;
    int new_i = (warp_id_x<<3) + lane_id%18-1;
    int new_j = (warp_id_y<<2) + lane_id/18-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput0 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%18-1;
    new_j = (warp_id_y<<2) + (lane_id+32)/18-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput1 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%18-1;
    new_j = (warp_id_y<<2) + (lane_id+64)/18-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput2 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%18-1;
    new_j = (warp_id_y<<2) + (lane_id+96)/18-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput3 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%18-1;
    new_j = (warp_id_y<<2) + (lane_id+128)/18-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput4 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%18-1;
    new_j = (warp_id_y<<2) + (lane_id+160)/18-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput5 = IN_2D(new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    int friend_id0, friend_id1;
    int friend_id2, friend_id3;
    DATA_TYPE tx0, ty0, tz0, tx1, ty1, tz1, tx2, ty2, tz2, tx3, ty3, tz3;

    friend_id0 = (lane_id+1 +((lane_id>>3)*10))&(warpSize-1);
    friend_id1 = (lane_id+9 +((lane_id>>3)*10))&(warpSize-1);
    friend_id2 = (lane_id+9 +((lane_id>>3)*10))&(warpSize-1);
    friend_id3 = (lane_id+17+((lane_id>>3)*10))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput0, friend_id1);
    ty1 = __shfl(threadInput1, friend_id1);
    tz1 = __shfl(threadInput2, friend_id1);
    tx2 = __shfl(threadInput2, friend_id2);
    ty2 = __shfl(threadInput3, friend_id2);
    tz2 = __shfl(threadInput4, friend_id2);
    tx3 = __shfl(threadInput2, friend_id3);
    ty3 = __shfl(threadInput3, friend_id3);
    tz3 = __shfl(threadInput4, friend_id3);
    sum0 += ARG_2D(0,j  ,i  )*((lane_id < 16)? tx0: ty0);
    sum1 += ARG_2D(0,j  ,i+8)*((lane_id < 13)? tx1: ((lane_id < 25)? ty1: tz1));
    sum2 += ARG_2D(0,j+4,i  )*((lane_id < 13)? tx2: ((lane_id < 25)? ty2: tz2));
    sum3 += ARG_2D(0,j+4,i+8)*((lane_id < 8 )? tx3: ((lane_id < 24)? ty3: tz3));

    friend_id0 = (lane_id+18+((lane_id>>3)*10))&(warpSize-1);
    friend_id1 = (lane_id+26+((lane_id>>3)*10))&(warpSize-1);
    friend_id2 = (lane_id+26+((lane_id>>3)*10))&(warpSize-1);
    friend_id3 = (lane_id+2 +((lane_id>>3)*10))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tz0 = __shfl(threadInput2, friend_id0);
    tx1 = __shfl(threadInput0, friend_id1);
    ty1 = __shfl(threadInput1, friend_id1);
    tz1 = __shfl(threadInput2, friend_id1);
    tx2 = __shfl(threadInput2, friend_id2);
    ty2 = __shfl(threadInput3, friend_id2);
    tz2 = __shfl(threadInput4, friend_id2);
    tx3 = __shfl(threadInput3, friend_id3);
    ty3 = __shfl(threadInput4, friend_id3);
    sum0 += ARG_2D(1,j  ,i  )*((lane_id < 8 )? tx0: ((lane_id < 24)? ty0: tz0));
    sum1 += ARG_2D(1,j  ,i+8)*((lane_id < 6 )? tx1: ((lane_id < 18)? ty1: tz1));
    sum2 += ARG_2D(1,j+4,i  )*((lane_id < 6 )? tx2: ((lane_id < 18)? ty2: tz2));
    sum3 += ARG_2D(1,j+4,i+8)*((lane_id < 16)? tx3: ty3);

    friend_id0 = (lane_id+19+((lane_id>>3)*10))&(warpSize-1);
    friend_id1 = (lane_id+27+((lane_id>>3)*10))&(warpSize-1);
    friend_id2 = (lane_id+27+((lane_id>>3)*10))&(warpSize-1);
    friend_id3 = (lane_id+3 +((lane_id>>3)*10))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tz0 = __shfl(threadInput2, friend_id0);
    tx1 = __shfl(threadInput0, friend_id1);
    ty1 = __shfl(threadInput1, friend_id1);
    tz1 = __shfl(threadInput2, friend_id1);
    tx2 = __shfl(threadInput2, friend_id2);
    ty2 = __shfl(threadInput3, friend_id2);
    tz2 = __shfl(threadInput4, friend_id2);
    tx3 = __shfl(threadInput3, friend_id3);
    ty3 = __shfl(threadInput4, friend_id3);
    tz3 = __shfl(threadInput5, friend_id3);
    sum0 += ARG_2D(4,j  ,i  )*((lane_id < 8 )? tx0: ((lane_id < 24)? ty0: tz0));
    sum1 += ARG_2D(4,j  ,i+8)*((lane_id < 5 )? tx1: ((lane_id < 17)? ty1: tz1));
    sum2 += ARG_2D(4,j+4,i  )*((lane_id < 5 )? tx2: ((lane_id < 17)? ty2: tz2));
    sum3 += ARG_2D(4,j+4,i+8)*((lane_id < 16)? tx3: ((lane_id < 31)? ty3: tz3));

    friend_id0 = (lane_id+20+((lane_id>>3)*10))&(warpSize-1);
    friend_id1 = (lane_id+28+((lane_id>>3)*10))&(warpSize-1);
    friend_id2 = (lane_id+28+((lane_id>>3)*10))&(warpSize-1);
    friend_id3 = (lane_id+4 +((lane_id>>3)*10))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tz0 = __shfl(threadInput2, friend_id0);
    tx1 = __shfl(threadInput0, friend_id1);
    ty1 = __shfl(threadInput1, friend_id1);
    tz1 = __shfl(threadInput2, friend_id1);
    tx2 = __shfl(threadInput2, friend_id2);
    ty2 = __shfl(threadInput3, friend_id2);
    tz2 = __shfl(threadInput4, friend_id2);
    tx3 = __shfl(threadInput3, friend_id3);
    ty3 = __shfl(threadInput4, friend_id3);
    tz3 = __shfl(threadInput5, friend_id3);
    sum0 += ARG_2D(3,j  ,i  )*((lane_id < 8 )? tx0: ((lane_id < 24)? ty0: tz0));
    sum1 += ARG_2D(3,j  ,i+8)*((lane_id < 4 )? tx1: ((lane_id < 16)? ty1: tz1));
    sum2 += ARG_2D(3,j+4,i  )*((lane_id < 4 )? tx2: ((lane_id < 16)? ty2: tz2));
    sum3 += ARG_2D(3,j+4,i+8)*((lane_id < 16)? tx3: ((lane_id < 30)? ty3: tz3));


    friend_id0 = (lane_id+5 +((lane_id>>3)*10))&(warpSize-1);
    friend_id1 = (lane_id+13+((lane_id>>3)*10))&(warpSize-1);
    friend_id2 = (lane_id+13+((lane_id>>3)*10))&(warpSize-1);
    friend_id3 = (lane_id+21+((lane_id>>3)*10))&(warpSize-1);
    tx0 = __shfl(threadInput1, friend_id0);
    ty0 = __shfl(threadInput2, friend_id0);
    tz0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tz1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput3, friend_id2);
    ty2 = __shfl(threadInput4, friend_id2);
    tz2 = __shfl(threadInput5, friend_id2);
    tx3 = __shfl(threadInput3, friend_id3);
    ty3 = __shfl(threadInput4, friend_id3);
    tz3 = __shfl(threadInput5, friend_id3);
    sum0 += ARG_2D(2,j  ,i  )*((lane_id < 16)? tx0: ((lane_id < 29)? ty0: tz0));
    sum1 += ARG_2D(2,j  ,i+8)*((lane_id < 9 )? tx1: ((lane_id < 24)? ty1: tz1));
    sum2 += ARG_2D(2,j+4,i  )*((lane_id < 9 )? tx2: ((lane_id < 24)? ty2: tz2));
    sum3 += ARG_2D(2,j+4,i+8)*((lane_id < 8 )? tx3: ((lane_id < 23)? ty3: tz3));

    OUT_2D(j  ,i  ) = sum0; 
    OUT_2D(j  ,i+8) = sum1; 
    OUT_2D(j+4,i  ) = sum2; 
    OUT_2D(j+4,i+8) = sum3; 
}

__global__ void Stencil_Cuda_Shfl8_2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int M, int N) 
{
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;

    int i = (((threadIdx.x + blockIdx.x * blockDim.x)>>3)<<4) + (lane_id&7)  ;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<4) + (lane_id>>3) ;

    int warp_id_x = ((((threadIdx.x + blockIdx.x * blockDim.x)>>3)<<4) + (lane_id&7) )>>3;
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<4) + (lane_id>>3))>>2;

    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5;
    DATA_TYPE threadInput6, threadInput7, threadInput8, threadInput9, threadInput10;
    int new_i = (warp_id_x<<3) + lane_id%18-1;
    int new_j = (warp_id_y<<2) + lane_id/18-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput0 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%18-1;
    new_j = (warp_id_y<<2) + (lane_id+32)/18-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput1 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%18-1;
    new_j = (warp_id_y<<2) + (lane_id+64)/18-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput2 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%18-1;
    new_j = (warp_id_y<<2) + (lane_id+96)/18-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput3 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%18-1;
    new_j = (warp_id_y<<2) + (lane_id+128)/18-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput4 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%18-1;
    new_j = (warp_id_y<<2) + (lane_id+160)/18-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput5 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+192)%18-1;
    new_j = (warp_id_y<<2) + (lane_id+192)/18-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput6 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+224)%18-1;
    new_j = (warp_id_y<<2) + (lane_id+224)/18-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput7 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+256)%18-1;
    new_j = (warp_id_y<<2) + (lane_id+256)/18-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput8 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+288)%18-1;
    new_j = (warp_id_y<<2) + (lane_id+288)/18-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput9 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+320)%18-1;
    new_j = (warp_id_y<<2) + (lane_id+320)/18-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput10 = IN_2D(new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    DATA_TYPE sum4 = 0.0;
    DATA_TYPE sum5 = 0.0;
    DATA_TYPE sum6 = 0.0;
    DATA_TYPE sum7 = 0.0;
    int friend_id0, friend_id1, friend_id2, friend_id3;
    int friend_id4, friend_id5, friend_id6, friend_id7;
    DATA_TYPE tx0, ty0, tz0, tx1, ty1, tz1, tx2, ty2, tz2, tx3, ty3, tz3;
    DATA_TYPE rx0, ry0, rz0, rx1, ry1, rz1, rx2, ry2, rz2, rx3, ry3, rz3;

    friend_id0 = (lane_id+1 +((lane_id>>3)*10))&(warpSize-1);
    friend_id1 = (lane_id+9 +((lane_id>>3)*10))&(warpSize-1);
    friend_id2 = (lane_id+9 +((lane_id>>3)*10))&(warpSize-1);
    friend_id3 = (lane_id+17+((lane_id>>3)*10))&(warpSize-1);
    friend_id4 = (lane_id+17+((lane_id>>3)*10))&(warpSize-1);
    friend_id5 = (lane_id+25+((lane_id>>3)*10))&(warpSize-1);
    friend_id6 = (lane_id+25+((lane_id>>3)*10))&(warpSize-1);
    friend_id7 = (lane_id+1 +((lane_id>>3)*10))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput0, friend_id1);
    ty1 = __shfl(threadInput1, friend_id1);
    tz1 = __shfl(threadInput2, friend_id1);
    tx2 = __shfl(threadInput2, friend_id2);
    ty2 = __shfl(threadInput3, friend_id2);
    tz2 = __shfl(threadInput4, friend_id2);
    tx3 = __shfl(threadInput2, friend_id3);
    ty3 = __shfl(threadInput3, friend_id3);
    tz3 = __shfl(threadInput4, friend_id3);
    rx0 = __shfl(threadInput4, friend_id4);
    ry0 = __shfl(threadInput5, friend_id4);
    rz0 = __shfl(threadInput6, friend_id4);
    rx1 = __shfl(threadInput4, friend_id5);
    ry1 = __shfl(threadInput5, friend_id5);
    rz1 = __shfl(threadInput6, friend_id5);
    rx2 = __shfl(threadInput6, friend_id6);
    ry2 = __shfl(threadInput7, friend_id6);
    rz2 = __shfl(threadInput8, friend_id6);
    rx3 = __shfl(threadInput7, friend_id7);
    ry3 = __shfl(threadInput8, friend_id7);
    sum0 += ARG_2D(0,j   ,i  )*((lane_id < 16)? tx0: ty0);
    sum1 += ARG_2D(0,j   ,i+8)*((lane_id < 13)? tx1: ((lane_id < 25)? ty1: tz1));
    sum2 += ARG_2D(0,j+4 ,i  )*((lane_id < 13)? tx2: ((lane_id < 25)? ty2: tz2));
    sum3 += ARG_2D(0,j+4 ,i+8)*((lane_id < 8 )? tx3: ((lane_id < 24)? ty3: tz3));
    sum4 += ARG_2D(0,j+8 ,i  )*((lane_id < 8 )? rx0: ((lane_id < 24)? ry0: rz0));
    sum5 += ARG_2D(0,j+8 ,i+8)*((lane_id < 7 )? rx1: ((lane_id < 19)? ry1: rz1));
    sum6 += ARG_2D(0,j+12,i  )*((lane_id < 7 )? rx2: ((lane_id < 19)? ry2: rz2));
    sum7 += ARG_2D(0,j+12,i+8)*((lane_id < 16)? rx3: ry3);

    friend_id0 = (lane_id+18+((lane_id>>3)*10))&(warpSize-1);
    friend_id1 = (lane_id+26+((lane_id>>3)*10))&(warpSize-1);
    friend_id2 = (lane_id+26+((lane_id>>3)*10))&(warpSize-1);
    friend_id3 = (lane_id+2 +((lane_id>>3)*10))&(warpSize-1);
    friend_id4 = (lane_id+2 +((lane_id>>3)*10))&(warpSize-1);
    friend_id5 = (lane_id+10+((lane_id>>3)*10))&(warpSize-1);
    friend_id6 = (lane_id+10+((lane_id>>3)*10))&(warpSize-1);
    friend_id7 = (lane_id+18+((lane_id>>3)*10))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tz0 = __shfl(threadInput2, friend_id0);
    tx1 = __shfl(threadInput0, friend_id1);
    ty1 = __shfl(threadInput1, friend_id1);
    tz1 = __shfl(threadInput2, friend_id1);
    tx2 = __shfl(threadInput2, friend_id2);
    ty2 = __shfl(threadInput3, friend_id2);
    tz2 = __shfl(threadInput4, friend_id2);
    tx3 = __shfl(threadInput3, friend_id3);
    ty3 = __shfl(threadInput4, friend_id3);
    rx0 = __shfl(threadInput5, friend_id4);
    ry0 = __shfl(threadInput6, friend_id4);
    rx1 = __shfl(threadInput5, friend_id5);
    ry1 = __shfl(threadInput6, friend_id5);
    rz1 = __shfl(threadInput7, friend_id5);
    rx2 = __shfl(threadInput7, friend_id6);
    ry2 = __shfl(threadInput8, friend_id6);
    rz2 = __shfl(threadInput9, friend_id6);
    rx3 = __shfl(threadInput7, friend_id7);
    ry3 = __shfl(threadInput8, friend_id7);
    rz3 = __shfl(threadInput9, friend_id7);
    sum0 += ARG_2D(1,j   ,i  )*((lane_id < 8 )? tx0: ((lane_id < 24)? ty0: tz0));
    sum1 += ARG_2D(1,j   ,i+8)*((lane_id < 6 )? tx1: ((lane_id < 18)? ty1: tz1));
    sum2 += ARG_2D(1,j+4 ,i  )*((lane_id < 6 )? tx2: ((lane_id < 18)? ty2: tz2));
    sum3 += ARG_2D(1,j+4 ,i+8)*((lane_id < 16)? tx3: ty3);
    sum4 += ARG_2D(1,j+8 ,i  )*((lane_id < 16)? rx0: ry0);
    sum5 += ARG_2D(1,j+8 ,i+8)*((lane_id < 12)? rx1: ((lane_id < 24)? ry1: rz1));
    sum6 += ARG_2D(1,j+12,i  )*((lane_id < 12)? rx2: ((lane_id < 24)? ry2: rz2));
    sum7 += ARG_2D(1,j+12,i+8)*((lane_id < 8 )? rx3: ((lane_id < 24)? ry3: rz3));

    friend_id0 = (lane_id+19+((lane_id>>3)*10))&(warpSize-1);
    friend_id1 = (lane_id+27+((lane_id>>3)*10))&(warpSize-1);
    friend_id2 = (lane_id+27+((lane_id>>3)*10))&(warpSize-1);
    friend_id3 = (lane_id+3 +((lane_id>>3)*10))&(warpSize-1);
    friend_id4 = (lane_id+3 +((lane_id>>3)*10))&(warpSize-1);
    friend_id5 = (lane_id+11+((lane_id>>3)*10))&(warpSize-1);
    friend_id6 = (lane_id+11+((lane_id>>3)*10))&(warpSize-1);
    friend_id7 = (lane_id+19+((lane_id>>3)*10))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tz0 = __shfl(threadInput2, friend_id0);
    tx1 = __shfl(threadInput0, friend_id1);
    ty1 = __shfl(threadInput1, friend_id1);
    tz1 = __shfl(threadInput2, friend_id1);
    tx2 = __shfl(threadInput2, friend_id2);
    ty2 = __shfl(threadInput3, friend_id2);
    tz2 = __shfl(threadInput4, friend_id2);
    tx3 = __shfl(threadInput3, friend_id3);
    ty3 = __shfl(threadInput4, friend_id3);
    tz3 = __shfl(threadInput5, friend_id3);
    rx0 = __shfl(threadInput5, friend_id4);
    ry0 = __shfl(threadInput6, friend_id4);
    rz0 = __shfl(threadInput7, friend_id4);
    rx1 = __shfl(threadInput5, friend_id5);
    ry1 = __shfl(threadInput6, friend_id5);
    rz1 = __shfl(threadInput7, friend_id5);
    rx2 = __shfl(threadInput7, friend_id6);
    ry2 = __shfl(threadInput8, friend_id6);
    rz2 = __shfl(threadInput9, friend_id6);
    rx3 = __shfl(threadInput7, friend_id7);
    ry3 = __shfl(threadInput8, friend_id7);
    rz3 = __shfl(threadInput9, friend_id7);
    sum0 += ARG_2D(4,j   ,i  )*((lane_id < 8 )? tx0: ((lane_id < 24)? ty0: tz0));
    sum1 += ARG_2D(4,j   ,i+8)*((lane_id < 5 )? tx1: ((lane_id < 17)? ty1: tz1));
    sum2 += ARG_2D(4,j+4 ,i  )*((lane_id < 5 )? tx2: ((lane_id < 17)? ty2: tz2));
    sum3 += ARG_2D(4,j+4 ,i+8)*((lane_id < 16)? tx3: ((lane_id < 31)? ty3: tz3));
    sum4 += ARG_2D(4,j+8 ,i  )*((lane_id < 16)? rx0: ((lane_id < 31)? ry0: rz0));
    sum5 += ARG_2D(4,j+8 ,i+8)*((lane_id < 11)? rx1: ((lane_id < 24)? ry1: rz1));
    sum6 += ARG_2D(4,j+12,i  )*((lane_id < 11)? rx2: ((lane_id < 24)? ry2: rz2));
    sum7 += ARG_2D(4,j+12,i+8)*((lane_id < 8 )? rx3: ((lane_id < 24)? ry3: rz3));

    friend_id0 = (lane_id+20+((lane_id>>3)*10))&(warpSize-1);
    friend_id1 = (lane_id+28+((lane_id>>3)*10))&(warpSize-1);
    friend_id2 = (lane_id+28+((lane_id>>3)*10))&(warpSize-1);
    friend_id3 = (lane_id+4 +((lane_id>>3)*10))&(warpSize-1);
    friend_id4 = (lane_id+4 +((lane_id>>3)*10))&(warpSize-1);
    friend_id5 = (lane_id+12+((lane_id>>3)*10))&(warpSize-1);
    friend_id6 = (lane_id+12+((lane_id>>3)*10))&(warpSize-1);
    friend_id7 = (lane_id+20+((lane_id>>3)*10))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tz0 = __shfl(threadInput2, friend_id0);
    tx1 = __shfl(threadInput0, friend_id1);
    ty1 = __shfl(threadInput1, friend_id1);
    tz1 = __shfl(threadInput2, friend_id1);
    tx2 = __shfl(threadInput2, friend_id2);
    ty2 = __shfl(threadInput3, friend_id2);
    tz2 = __shfl(threadInput4, friend_id2);
    tx3 = __shfl(threadInput3, friend_id3);
    ty3 = __shfl(threadInput4, friend_id3);
    tz3 = __shfl(threadInput5, friend_id3);
    rx0 = __shfl(threadInput5, friend_id4);
    ry0 = __shfl(threadInput6, friend_id4);
    rz0 = __shfl(threadInput7, friend_id4);
    rx1 = __shfl(threadInput5, friend_id5);
    ry1 = __shfl(threadInput6, friend_id5);
    rz1 = __shfl(threadInput7, friend_id5);
    rx2 = __shfl(threadInput7, friend_id6);
    ry2 = __shfl(threadInput8, friend_id6);
    rz2 = __shfl(threadInput9, friend_id6);
    rx3 = __shfl(threadInput7, friend_id7);
    ry3 = __shfl(threadInput8, friend_id7);
    rz3 = __shfl(threadInput9, friend_id7);
    sum0 += ARG_2D(3,j   ,i  )*((lane_id < 8 )? tx0: ((lane_id < 24)? ty0: tz0));
    sum1 += ARG_2D(3,j   ,i+8)*((lane_id < 4 )? tx1: ((lane_id < 16)? ty1: tz1));
    sum2 += ARG_2D(3,j+4 ,i  )*((lane_id < 4 )? tx2: ((lane_id < 16)? ty2: tz2));
    sum3 += ARG_2D(3,j+4 ,i+8)*((lane_id < 16)? tx3: ((lane_id < 30)? ty3: tz3));
    sum4 += ARG_2D(3,j+8 ,i  )*((lane_id < 16)? rx0: ((lane_id < 30)? ry0: rz0));
    sum5 += ARG_2D(3,j+8 ,i+8)*((lane_id < 10)? rx1: ((lane_id < 24)? ry1: rz1));
    sum6 += ARG_2D(3,j+12,i  )*((lane_id < 10)? rx2: ((lane_id < 24)? ry2: rz2));
    sum7 += ARG_2D(3,j+12,i+8)*((lane_id < 8 )? rx3: ((lane_id < 24)? ry3: rz3));

    friend_id0 = (lane_id+5 +((lane_id>>3)*10))&(warpSize-1);
    friend_id1 = (lane_id+13+((lane_id>>3)*10))&(warpSize-1);
    friend_id2 = (lane_id+13+((lane_id>>3)*10))&(warpSize-1);
    friend_id3 = (lane_id+21+((lane_id>>3)*10))&(warpSize-1);
    friend_id4 = (lane_id+21+((lane_id>>3)*10))&(warpSize-1);
    friend_id5 = (lane_id+29+((lane_id>>3)*10))&(warpSize-1);
    friend_id6 = (lane_id+29+((lane_id>>3)*10))&(warpSize-1);
    friend_id7 = (lane_id+5 +((lane_id>>3)*10))&(warpSize-1);
    tx0 = __shfl(threadInput1, friend_id0);
    ty0 = __shfl(threadInput2, friend_id0);
    tz0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tz1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput3, friend_id2);
    ty2 = __shfl(threadInput4, friend_id2);
    tz2 = __shfl(threadInput5, friend_id2);
    tx3 = __shfl(threadInput3, friend_id3);
    ty3 = __shfl(threadInput4, friend_id3);
    tz3 = __shfl(threadInput5, friend_id3);
    rx0 = __shfl(threadInput5 , friend_id4);
    ry0 = __shfl(threadInput6 , friend_id4);
    rz0 = __shfl(threadInput7 , friend_id4);
    rx1 = __shfl(threadInput5 , friend_id5);
    ry1 = __shfl(threadInput6 , friend_id5);
    rz1 = __shfl(threadInput7 , friend_id5);
    rx2 = __shfl(threadInput7 , friend_id6);
    ry2 = __shfl(threadInput8 , friend_id6);
    rz2 = __shfl(threadInput9 , friend_id6);
    rx3 = __shfl(threadInput8 , friend_id7);
    ry3 = __shfl(threadInput9 , friend_id7);
    rz3 = __shfl(threadInput10, friend_id7);
    sum0 += ARG_2D(2,j   ,i  )*((lane_id < 16)? tx0: ((lane_id < 29)? ty0: tz0));
    sum1 += ARG_2D(2,j   ,i+8)*((lane_id < 9 )? tx1: ((lane_id < 24)? ty1: tz1));
    sum2 += ARG_2D(2,j+4 ,i  )*((lane_id < 9 )? tx2: ((lane_id < 24)? ty2: tz2));
    sum3 += ARG_2D(2,j+4 ,i+8)*((lane_id < 8 )? tx3: ((lane_id < 23)? ty3: tz3));
    sum4 += ARG_2D(2,j+8 ,i  )*((lane_id < 8 )? rx0: ((lane_id < 23)? ry0: rz0));
    sum5 += ARG_2D(2,j+8 ,i+8)*((lane_id < 3 )? rx1: ((lane_id < 16)? ry1: rz1));
    sum6 += ARG_2D(2,j+12,i  )*((lane_id < 3 )? rx2: ((lane_id < 16)? ry2: rz2));
    sum7 += ARG_2D(2,j+12,i+8)*((lane_id < 16)? rx3: ((lane_id < 29)? ry3: rz3));


    OUT_2D(j   ,i  ) = sum0; 
    OUT_2D(j   ,i+8) = sum1; 
    OUT_2D(j+4 ,i  ) = sum2; 
    OUT_2D(j+4 ,i+8) = sum3; 
    OUT_2D(j+8 ,i  ) = sum4; 
    OUT_2D(j+8 ,i+8) = sum5; 
    OUT_2D(j+12,i  ) = sum6; 
    OUT_2D(j+12,i+8) = sum7; 
}

__global__ void Stencil_Cuda_Sm(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, int M, int N) 
{
    __shared__ DATA_TYPE local[18*18];
    int i = threadIdx.x + blockIdx.x * blockDim.x ; 
    int j = threadIdx.y + blockIdx.y * blockDim.y ; 
    int li = threadIdx.x + 1;
    int lj = threadIdx.y + 1;

    LOC_2D(lj,li) = IN_2D(j,i);
    int n = (j == 0)      ? j : j - 1;
    int s = (j == M-1)    ? j : j + 1;
    int w = (i == 0)      ? i : i - 1;
    int e = (i == N-1)    ? i : i + 1;

    if(threadIdx.x == 0)              LOC_2D(lj,li-1) = IN_2D(j,w);

    if(threadIdx.x == blockDim.x-1)   LOC_2D(lj,li+1) = IN_2D(j,e);
            
    if(threadIdx.y == 0)              LOC_2D(lj-1,li) = IN_2D(n,i);
                                                
    if(threadIdx.y == blockDim.y-1)   LOC_2D(lj+1,li) = IN_2D(s,i);

    __syncthreads();

    OUT_2D(j,i) = a0 *LOC_2D(lj-1,li  ) + 
                  a1 *LOC_2D(lj  ,li-1) + 
                  a2 *LOC_2D(lj+1,li  ) +
                  a3 *LOC_2D(lj  ,li+1) + 
                  a4 *LOC_2D(lj  ,li  ) ;
}

__global__ void Stencil_Cuda_SmX(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int M, int N) 
{
    __shared__ DATA_TYPE local[18*18];
    int i = threadIdx.x + blockIdx.x * blockDim.x ; 
    int j = threadIdx.y + blockIdx.y * blockDim.y ; 
    int li = threadIdx.x + 1;
    int lj = threadIdx.y + 1;

    unsigned int lane_id = threadIdx.x + threadIdx.y * blockDim.x;

    int blk_id_x = blockIdx.x;//(threadIdx.x + blockIdx.x * blockDim.x)>>4;
    int blk_id_y = blockIdx.y;//(threadIdx.y + blockIdx.y * blockDim.y)>>4;

    int new_i  = (blk_id_x<<4) + lane_id%18-1;
    int new_j  = (blk_id_y<<4) + lane_id/18-1;
    int new_li = lane_id%18;
    int new_lj = lane_id/18;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    LOC_2D(new_lj,new_li) = IN_2D(new_j,new_i);
    new_i  = (blk_id_x<<4) + (lane_id+256)%18-1;
    new_j  = (blk_id_y<<4) + (lane_id+256)/18-1;
    new_li = (lane_id+256)%18;
    new_lj = (lane_id+256)/18;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    if(new_li < 18 &&  new_lj < 18)
        LOC_2D(new_lj,new_li) = IN_2D(new_j,new_i);

    __syncthreads();

    OUT_2D(j,i) = ARG_2D(0,j,i) *LOC_2D(lj-1,li  ) + 
                  ARG_2D(1,j,i) *LOC_2D(lj  ,li-1) + 
                  ARG_2D(2,j,i) *LOC_2D(lj+1,li  ) +
                  ARG_2D(3,j,i) *LOC_2D(lj  ,li+1) + 
                  ARG_2D(4,j,i) *LOC_2D(lj  ,li  ) ;
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

__global__ void Stencil_Cuda_Shfl(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int M, int N) 
{
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3;
    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2;

    DATA_TYPE threadInput0, threadInput1;
    int new_i = (warp_id_x<<3) + lane_id%10-1;
    int new_j = (warp_id_y<<2) + lane_id/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput0 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+32)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput1 = IN_2D(new_j, new_i);

    DATA_TYPE sum = 0.0;
    int friend_id;
    DATA_TYPE tx, ty;
    
    friend_id = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput0, friend_id);
    ty = __shfl(threadInput1, friend_id);
    sum += ARG_2D(0,j,i)*((lane_id < 25)? tx: ty);

    friend_id = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput0, friend_id);
    ty = __shfl(threadInput1, friend_id);
    sum += ARG_2D(1,j,i)*((lane_id < 18)? tx: ty);

    friend_id = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput0, friend_id);
    ty = __shfl(threadInput1, friend_id);
    sum += ARG_2D(4,j,i)*((lane_id < 17)? tx: ty);

    friend_id = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput0, friend_id);
    ty = __shfl(threadInput1, friend_id);
    sum += ARG_2D(3,j,i)*((lane_id < 16)? tx: ty);

    friend_id = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput0, friend_id);
    ty = __shfl(threadInput1, friend_id);
    sum += ARG_2D(2,j,i)*((lane_id < 9)? tx: ty);

    OUT_2D(j,i) = sum; 
}

__global__ void Stencil_Cuda_Shfl2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int M, int N) 
{
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<3) + (lane_id>>3) ;

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3;
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<3) + (lane_id>>3))>>2;

    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3;
    int new_i = (warp_id_x<<3) + lane_id%10-1;
    int new_j = (warp_id_y<<2) + lane_id/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput0 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+32)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput1 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+64)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput2 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+96)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput3 = IN_2D(new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    int friend_id0, friend_id1;
    DATA_TYPE tx0, ty0, tx1, ty1, tz1;

    friend_id0 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    sum0 += ARG_2D(0,j  ,i)*((lane_id < 25)? tx0: ty0);
    sum1 += ARG_2D(0,j+4,i)*((lane_id < 19)? tx1: ty1);

    friend_id0 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    sum0 += ARG_2D(1,j  ,i)*((lane_id < 18)? tx0: ty0);
    sum1 += ARG_2D(1,j+4,i)*((lane_id < 12)? tx1: ty1);

    friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    sum0 += ARG_2D(4,j  ,i)*((lane_id < 17)? tx0: ty0);
    sum1 += ARG_2D(4,j+4,i)*((lane_id < 11)? tx1: ty1);

    friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    sum0 += ARG_2D(3,j  ,i)*((lane_id < 16)? tx0: ty0);
    sum1 += ARG_2D(3,j+4,i)*((lane_id < 10)? tx1: ty1);

    friend_id0 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tz1 = __shfl(threadInput3, friend_id1);
    sum0 += ARG_2D(2,j  ,i)*((lane_id < 9)? tx0: ty0);
    sum1 += ARG_2D(2,j+4,i)*((lane_id < 3)? tx1: ((lane_id < 29)? ty1: tz1));

    OUT_2D(j  ,i) = sum0; 
    OUT_2D(j+4,i) = sum1; 
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
    int total = (m)*(n);
    int K = total*5;
    DATA_TYPE *args = new DATA_TYPE[K];
#ifdef __DEBUG
    Init_Args_2D(args, 5, m, n, 1.0);
#else
    Init_Args_2D(args, 5, m, n, 0.20);
#endif
    DATA_TYPE *in = new DATA_TYPE[total];
    DATA_TYPE *out_ref = new DATA_TYPE[total];
    unsigned int seed = time(NULL);
    Init_Input_2D(in, m, n, seed);

    // Show_Me(in, m, n, "Input:");
    for(int i=0; i< ITER; i++)
    {
        Stencil_Seq(in, out_ref, args, m, n);
        swap(in, out_ref);
    }
    swap(in, out_ref);
    // Show_Me(out_ref, m, n, "Output:");

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
    Init_Input_2D(in, m, n, seed);
    Clear_Output_2D(out, m, n);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid((n)/16, (m)/16, 1);
    dim3 dimBlock(16, 16, 1);
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda<<<dimGrid, dimBlock>>>(in_d, out_d, args_d, m, n);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, m, n, "Output:");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D5, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(ARGC_2D5, 1, m, n, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shared Memory version 
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, seed);
    Clear_Output_2D(out, m, n);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid1((n)/16, (m)/16, 1);
    dim3 dimBlock1(16, 16, 1);
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_SmX<<<dimGrid1, dimBlock1>>>(in_d, out_d, args_d, m, n);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, m, n, "Output(SM):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sm: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sm Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D5, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(ARGC_2D5, 1, m, n, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl version 
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, seed);
    Clear_Output_2D(out, m, n);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid2((n)/8, (m)/32, 1);
    dim3 dimBlock2(8, 32, 1);
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl<<<dimGrid2, dimBlock2>>>(in_d, out_d, args_d, m, n);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, m, n, "Output(Shfl):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Shfl Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D5, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(ARGC_2D5, 1, m, n, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl2 version 
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, seed);
    Clear_Output_2D(out, m, n);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid3((n)/8, (m)/(32*2), 1);
    dim3 dimBlock3(8, 32, 1);
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl2<<<dimGrid3, dimBlock3>>>(in_d, out_d, args_d, m, n);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, m, n, "Output(Shfl2):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl2: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Shfl2 Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D5, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(ARGC_2D5, 1, m, n, ITER, time_wo_pci, sizeof(DATA_TYPE)));


    // Cuda Shfl4 version 
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, seed);
    Clear_Output_2D(out, m, n);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid4((n)/8, (m)/(32*4), 1);
    dim3 dimBlock4(8, 32, 1);
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl4<<<dimGrid4, dimBlock4>>>(in_d, out_d, args_d, m, n);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, m, n, "Output:");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl4: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Shfl4 Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D5, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(ARGC_2D5, 1, m, n, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl8 version 
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, seed);
    Clear_Output_2D(out, m, n);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid5((n)/8, (m)/(32*8), 1);
    dim3 dimBlock5(8, 32, 1);
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl8<<<dimGrid5, dimBlock5>>>(in_d, out_d, args_d, m, n);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, m, n, "Output:");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl8: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Shfl8 Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D5, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(ARGC_2D5, 1, m, n, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl4_2 version 
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, seed);
    Clear_Output_2D(out, m, n);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid6((n)/(8*2), (m)/(32*2), 1);
    dim3 dimBlock6(8, 32, 1);
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl4_2<<<dimGrid6, dimBlock6>>>(in_d, out_d, args_d, m, n);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, m, n, "Output:");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl4_2: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Shfl4_2 Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D5, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(ARGC_2D5, 1, m, n, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Shfl8_2 version 
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, seed);
    Clear_Output_2D(out, m, n);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid7((n)/(8*2), (m)/(32*4), 1);
    dim3 dimBlock7(8, 32, 1);
    cudaEventRecord(start);
    for(int i=0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl8_2<<<dimGrid7, dimBlock7>>>(in_d, out_d, args_d, m, n);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, m, n, "Output:");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl8_2: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Shfl8_2 Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D5, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(ARGC_2D5, 1, m, n, ITER, time_wo_pci, sizeof(DATA_TYPE)));
   
    cudaFree(in_d);
    cudaFree(out_d);

    delete[] in;
    delete[] out;
    delete[] out_ref;
}

