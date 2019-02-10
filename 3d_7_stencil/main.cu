#include <iostream>
using namespace std;
#define IN_3D(_z,_y,_x)   in[(_z)*(M)*(N)+(_y)*(N)+(_x)]
#define OUT_3D(_z,_y,_x) out[(_z)*(M)*(N)+(_y)*(N)+(_x)]

#define SM_Z (8+2)
#define SM_M (4+2)
#define SM_N (8+2)
#define LOC_3D(_z,_y,_x) local[(_z)*(SM_M)*(SM_N)+(_y)*(SM_N)+(_x)]

#define SM_2D_M (4+2)
#define SM_2D_N (64+2)
#define LOC_2D(_y,_x) local[(_y)*(SM_2D_N)+(_x)]
// #define LOC_2D2(_y,_x) local[(_y)*(SM_2D_N2+2*halo)+(_x)]
// #define LOC_L_2D(_z,_y,_x) local[(_z)*(SM_2D_M*SM_2D_N)+(_y)*(SM_2D_N)+(_x)]

#define DATA_TYPE float
#define warpSize 32 

// #define __DEBUG

#ifdef __DEBUG
#define ITER 1
#else
#define ITER 100
#endif


// #define TEMP
#define SPAC1

float GetGFLOPS(int z, int m, int n, int count, int ops, float time) 
{
    float f = (z*m*n)*(float)(ops)*(float)(count)/time * 1.0e-09;
    return f;
}

float GetThroughput(int z, int m, int n, int count, float time) 
{
    return (z*m*n) * sizeof(DATA_TYPE) * 2.0 * ((float)count)
            / time * 1.0e-09;    
          
}

void Init_Input_3D(DATA_TYPE *in, int Z, int M, int N)
{
    srand(time(NULL));

    for(int k = 0; k < Z; k++)
        for(int j = 0; j < M; j++)
            for(int i = 0; i < N; i++)
#ifdef __DEBUG
                IN_3D(k,j,i) = 1; //(DATA_TYPE)rand() * 100.0 / RAND_MAX;
#else
                IN_3D(k,j,i) = (DATA_TYPE)rand()*100.0 / RAND_MAX;
#endif
}

void Clear_Output_3D(DATA_TYPE *in, int Z, int M, int N)
{
    for(int k = 0; k < Z; k++)
        for(int j = 0; j < M; j++)
            for(int i = 0; i < N; i++)
                IN_3D(k,j,i) = 0;
}

void Show_Me(DATA_TYPE *in, int Z, int M, int N, std::string prompt)
{
    std::cout << prompt << std::endl;
    for(int k = 0; k < Z; k++)
    {
        for(int j = 0; j < M; j++)
        {
            for(int i = 0; i < N; i++)
                std::cout << IN_3D(k,j,i) << ",";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void Stencil_Seq(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int Z, int M, int N)
{
    for(int k = 0; k < Z; k++)
    {
        int b = (k == 0)      ? k : k - 1;
        int t = (k == Z-1)    ? k : k + 1;
        for(int j = 0; j < M; j++)
        {
            int n = (j == 0)      ? j : j - 1;
            int s = (j == M-1)    ? j : j + 1;
            for(int i = 0; i < N; i++)
            {
                int w = (i == 0)      ? i : i - 1;
                int e = (i == N-1)    ? i : i + 1;
                OUT_3D(k,j,i) = a0 * IN_3D(b  ,j  ,i  ) +
                                a1 * IN_3D(k  ,n  ,i  ) +
                                a2 * IN_3D(k  ,j  ,w  ) +
                                a3 * IN_3D(k  ,j  ,i  ) +
                                a4 * IN_3D(k  ,j  ,e  ) +
                                a5 * IN_3D(k  ,s  ,i  ) +
                                a6 * IN_3D(t  ,j  ,i  ) ;
            }
        }
    }
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

__global__ void Stencil_Cuda(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int Z, int M, int N) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;
    int k = threadIdx.z + blockIdx.z * blockDim.z ;

    int b = (k == 0)      ? k : k - 1;
    int t = (k == Z-1)    ? k : k + 1;
    int n = (j == 0)      ? j : j - 1;
    int s = (j == M-1)    ? j : j + 1;
    int w = (i == 0)      ? i : i - 1;
    int e = (i == N-1)    ? i : i + 1;
    OUT_3D(k,j,i) = a0 * IN_3D(b  ,j  ,i  ) +
                    a1 * IN_3D(k  ,n  ,i  ) +
                    a2 * IN_3D(k  ,j  ,w  ) +
                    a3 * IN_3D(k  ,j  ,i  ) +
                    a4 * IN_3D(k  ,j  ,e  ) +
                    a5 * IN_3D(k  ,s  ,i  ) +
                    a6 * IN_3D(t  ,j  ,i  ) ;
}

__global__ void Stencil_Cuda_Sweep(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int Z, int M, int N) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;

    const int block_z = Z / gridDim.z;
    int k = block_z * blockIdx.z ;
    const int k_end = k + block_z;

    int n = (j == 0)      ? j : j - 1;
    int s = (j == M-1)    ? j : j + 1;
    int w = (i == 0)      ? i : i - 1;
    int e = (i == N-1)    ? i : i + 1;
#pragma unroll // it seems the loop-unroll is useless to performance
    for(; k < k_end; ++k)
    {
        int b = (k == 0)      ? k : k - 1;
        int t = (k == Z-1)    ? k : k + 1;
        OUT_3D(k,j,i) = a0 * IN_3D(b  ,j  ,i  ) +
                        a1 * IN_3D(k  ,n  ,i  ) +
                        a2 * IN_3D(k  ,j  ,w  ) +
                        a3 * IN_3D(k  ,j  ,i  ) +
                        a4 * IN_3D(k  ,j  ,e  ) +
                        a5 * IN_3D(k  ,s  ,i  ) +
                        a6 * IN_3D(t  ,j  ,i  ) ;
    }
}

__global__ void Stencil_Cuda_Sweep_Sm(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int Z, int M, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;
    const int block_z = Z / gridDim.z;
    int k = block_z * blockIdx.z ;
    const int k_end = k + block_z;
    int li = threadIdx.x + 1;
    int lj = threadIdx.y + 1;

    extern __shared__ DATA_TYPE local[];

    DATA_TYPE t1, t2, t3;
    DATA_TYPE r1, r2, r3, r4;
    DATA_TYPE sum = 0.0;
    int n = (j == 0)      ? j : j - 1;
    int s = (j == M-1)    ? j : j + 1;
    int w = (i == 0)      ? i : i - 1;
    int e = (i == N-1)    ? i : i + 1;
    // load current layer
    t3 = IN_3D(k, j, i); 
    if(threadIdx.x == 0)            r1 = IN_3D(k,j,w);
    if(threadIdx.x == blockDim.x-1) r2 = IN_3D(k,j,e);
    if(threadIdx.y == 0)            r3 = IN_3D(k,n,i);
    if(threadIdx.y == blockDim.y-1) r4 = IN_3D(k,s,i);

    // load previous layer (same with k)
    t2 = IN_3D(k, j, i);

#pragma unroll // it seems the loop-unroll is useless to performance
    for(; k < k_end; ++k)
    {
        sum = 0.0;
        t1 = t2;
        t2 = t3;
        LOC_2D(lj,li) = t2;
        if(threadIdx.x == 0)            LOC_2D(lj  ,li-1) = r1; 
        if(threadIdx.x == blockDim.x-1) LOC_2D(lj  ,li+1) = r2; 
        if(threadIdx.y == 0)            LOC_2D(lj-1,li  ) = r3; 
        if(threadIdx.y == blockDim.y-1) LOC_2D(lj+1,li  ) = r4; 

        // load next layer
        int t = (k == Z-1)    ? k : k+1;
        t3 = IN_3D(t, j, i); 
        if(threadIdx.x == 0)            r1 = IN_3D(t,j,w);
        if(threadIdx.x == blockDim.x-1) r2 = IN_3D(t,j,e);
        if(threadIdx.y == 0)            r3 = IN_3D(t,n,i);
        if(threadIdx.y == blockDim.y-1) r4 = IN_3D(t,s,i);

        sum += a0 * t1 + a3 * t2 + a5 * t3;
        __syncthreads();
        sum += a1 * LOC_2D(lj-1,li  );
        sum += a2 * LOC_2D(lj  ,li-1);
        sum += a4 * LOC_2D(lj  ,li+1);
        sum += a6 * LOC_2D(lj+1,li  );

        OUT_3D(k,j,i) = sum;
        __syncthreads();
    }
}

__global__ void Stencil_Cuda_Sm(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int Z, int M, int N) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;
    int k = threadIdx.z + blockIdx.z * blockDim.z ;

    int li = threadIdx.x + 1;
    int lj = threadIdx.y + 1;
    int lk = threadIdx.z + 1;

    __shared__ DATA_TYPE local[SM_Z*SM_M*SM_N];

    LOC_3D(lk,lj,li) = IN_3D(k,j,i);

    int b = (k == 0)      ? k : k - 1;
    int t = (k == Z-1)    ? k : k + 1;
    int n = (j == 0)      ? j : j - 1;
    int s = (j == M-1)    ? j : j + 1;
    int w = (i == 0)      ? i : i - 1;
    int e = (i == N-1)    ? i : i + 1;
    if(threadIdx.x == 0)            LOC_3D(lk  ,lj  ,li-1) = IN_3D(k,j,w);
    if(threadIdx.x == blockDim.x-1) LOC_3D(lk  ,lj  ,li+1) = IN_3D(k,j,e);
    if(threadIdx.y == 0)            LOC_3D(lk  ,lj-1,li  ) = IN_3D(k,n,i);
    if(threadIdx.y == blockDim.y-1) LOC_3D(lk  ,lj+1,li  ) = IN_3D(k,s,i);
    if(threadIdx.z == 0)            LOC_3D(lk-1,lj  ,li  ) = IN_3D(b,j,i);
    if(threadIdx.z == blockDim.z-1) LOC_3D(lk+1,lj  ,li  ) = IN_3D(t,j,i);
    __syncthreads();

    OUT_3D(k,j,i) = a0 * LOC_3D(lk-1,lj  ,li  ) +
                    a1 * LOC_3D(lk  ,lj-1,li  ) +
                    a2 * LOC_3D(lk  ,lj  ,li-1) +
                    a3 * LOC_3D(lk  ,lj  ,li  ) +
                    a4 * LOC_3D(lk  ,lj  ,li+1) +
                    a5 * LOC_3D(lk  ,lj+1,li  ) +
                    a6 * LOC_3D(lk+1,lj  ,li  ) ;
}

__global__ void Stencil_Cuda_Shfl(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int Z, int M, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;
    int k = threadIdx.z + blockIdx.z * blockDim.z ;
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    int lane_id = tid % warpSize;
    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
    int warp_id_z = (threadIdx.z + blockIdx.z * blockDim.z)>>0; // there numbers
    int new_i = (warp_id_x<<3) + lane_id%10 - 1;     // 10 is extended dimension of i
    int new_j = (warp_id_y<<2) + (lane_id/10)%6 - 1; // 6  is extended dimension of j 
    int new_k = (warp_id_z<<0) + lane_id/60 - 1;     // 60 is extended area of ixj = 10x6
    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5;
    
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput0 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10 -1;
    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+32)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput1 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+64)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput2 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+96)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput3 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+128)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput4 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+160)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput5 = IN_3D(new_k, new_j, new_i);

    DATA_TYPE sum = 0.0;
    int friend_id;
    DATA_TYPE tx, ty, tz;
    friend_id = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    tx = a0*__shfl(threadInput0, friend_id);
    ty = a0*__shfl(threadInput1, friend_id);
    sum += (lane_id < 17)? tx: ty;

    friend_id = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    tx = a1*__shfl(threadInput1, friend_id);
    ty = a1*__shfl(threadInput2, friend_id);
    tz = a1*__shfl(threadInput3, friend_id);
    sum += (lane_id < 3 )? tx: ((lane_id < 29)? ty: tz);

    friend_id = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    tx = a2*__shfl(threadInput2, friend_id);
    ty = a2*__shfl(threadInput3, friend_id);
    sum += (lane_id < 22)? tx: ty;

    friend_id = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    tx = a3*__shfl(threadInput2, friend_id);
    ty = a3*__shfl(threadInput3, friend_id);
    sum += (lane_id < 21)? tx: ty;

    friend_id = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    tx = a4*__shfl(threadInput2, friend_id);
    ty = a4*__shfl(threadInput3, friend_id);
    sum += (lane_id < 20)? tx: ty;

    friend_id = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    tx = a5*__shfl(threadInput2, friend_id);
    ty = a5*__shfl(threadInput3, friend_id);
    sum += (lane_id < 13)? tx: ty;

    friend_id = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    tx = a6*__shfl(threadInput4, friend_id);
    ty = a6*__shfl(threadInput5, friend_id);
    sum += (lane_id < 24)? tx: ty;

    OUT_3D(k,j,i) = sum;

}

__global__ void Stencil_Cuda_Shfl2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int Z, int M, int N)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;
    int k = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<1) + (lane_id>>5) ; 
    // thread coarsening: related to warp dimensions 1x4x8. 
    // We coarsen from dimension z from 2^0 to 2^1, also need to know there are how many values in dimension z

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
    int warp_id_z = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<1) + (lane_id>>5); // these numbers
    int new_i = (warp_id_x<<3) + lane_id%10-1;     // 10 is extended dimension of i
    int new_j = (warp_id_y<<2) + (lane_id/10)%6-1; // 6  is extended dimension of j 
    int new_k = (warp_id_z<<0) + lane_id/60-1;     // 60 is extended area of ixj = 10x6
    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5,
              threadInput6, threadInput7;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput0 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+32)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput1 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+64)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput2 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+96)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput3 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+128)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput4 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+160)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput5 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+192)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+192)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+192)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput6 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+224)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+224)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+224)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput7 = IN_3D(new_k, new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    int friend_id0, friend_id1;
    DATA_TYPE tx0, ty0, tz0, tx1, ty1, tz1;
    friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a0*__shfl(threadInput0, friend_id0);
    ty0 = a0*__shfl(threadInput1, friend_id0);
    tx1 = a0*__shfl(threadInput2, friend_id1);
    ty1 = a0*__shfl(threadInput3, friend_id1);
    sum0 += (lane_id < 17)? tx0: ty0;
    sum1 += (lane_id < 21)? tx1: ty1;

    friend_id0 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a1*__shfl(threadInput1, friend_id0);
    ty0 = a1*__shfl(threadInput2, friend_id0);
    tz0 = a1*__shfl(threadInput3, friend_id0);
    tx1 = a1*__shfl(threadInput3, friend_id1);
    ty1 = a1*__shfl(threadInput4, friend_id1);
    sum0 += (lane_id < 3 )? tx0: ((lane_id < 29)? ty0: tz0);
    sum1 += (lane_id < 7 )? tx1: ty1;

    friend_id0 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a2*__shfl(threadInput2, friend_id0);
    ty0 = a2*__shfl(threadInput3, friend_id0);
    tx1 = a2*__shfl(threadInput4, friend_id1);
    ty1 = a2*__shfl(threadInput5, friend_id1);
    sum0 += (lane_id < 22)? tx0: ty0;
    sum1 += (lane_id < 24)? tx1: ty1;

    friend_id0 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a3*__shfl(threadInput2, friend_id0);
    ty0 = a3*__shfl(threadInput3, friend_id0);
    tx1 = a3*__shfl(threadInput4, friend_id1);
    ty1 = a3*__shfl(threadInput5, friend_id1);
    sum0 += (lane_id < 21)? tx0: ty0;
    sum1 += (lane_id < 24)? tx1: ty1;

    friend_id0 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a4*__shfl(threadInput2, friend_id0);
    ty0 = a4*__shfl(threadInput3, friend_id0);
    tx1 = a4*__shfl(threadInput4, friend_id1);
    ty1 = a4*__shfl(threadInput5, friend_id1);
    sum0 += (lane_id < 20)? tx0: ty0;
    sum1 += (lane_id < 24)? tx1: ty1;

    friend_id0 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a5*__shfl(threadInput2, friend_id0);
    ty0 = a5*__shfl(threadInput3, friend_id0);
    tx1 = a5*__shfl(threadInput4, friend_id1);
    ty1 = a5*__shfl(threadInput5, friend_id1);
    sum0 += (lane_id < 13)? tx0: ty0;
    sum1 += (lane_id < 16)? tx1: ty1;

    friend_id0 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a6*__shfl(threadInput4, friend_id0);
    ty0 = a6*__shfl(threadInput5, friend_id0);
    tx1 = a6*__shfl(threadInput5, friend_id1);
    ty1 = a6*__shfl(threadInput6, friend_id1);
    tz1 = a6*__shfl(threadInput7, friend_id1);
    sum0 += (lane_id < 24)? tx0: ty0;
    sum1 += (lane_id < 1 )? tx1: ((lane_id < 27)? ty1: tz1);

    OUT_3D(k,j,i) = sum0;
    OUT_3D(k+1,j,i) = sum1;

}

__global__ void Stencil_Cuda_Shfl4(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int Z, int M, int N)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;
    int k = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<2) + (lane_id>>5) ; 
    // Thread coarsening: related to warp dimensions 1x4x8. 
    // We coarsen from dimension z from 2^0 to 2^2, also need to know there are how many values in dimension z,
    // which is (lane_id>>5) 

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
    int warp_id_z = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<2) + (lane_id>>5); // these numbers
    int new_i = (warp_id_x<<3) + lane_id%10-1;     // 10 is extended dimension of i
    int new_j = (warp_id_y<<2) + (lane_id/10)%6-1; // 6  is extended dimension of j 
    int new_k = (warp_id_z<<0) + lane_id/60-1;     // 60 is extended area of ixj = 10x6
    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5,
              threadInput6, threadInput7, threadInput8, threadInput9, threadInput10; //, threadInput11;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput0 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+32)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput1 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+64)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput2 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+96)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput3 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+128)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput4 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+160)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput5 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+192)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+192)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+192)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput6 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+224)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+224)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+224)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput7 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+256)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+256)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+256)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput8 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+288)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+288)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+288)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput9 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+320)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+320)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+320)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput10 = IN_3D(new_k, new_j, new_i);
    // new_i = (warp_id_x<<3) + (lane_id+352)%10-1;
    // new_j = (warp_id_y<<2) + ((lane_id+352)/10)%6-1;
    // new_k = (warp_id_z<<0) + (lane_id+352)/60-1;
    // new_k = (new_k == -1)   ? 0   : new_k;
    // new_k = (new_k >= Z)    ? Z-1 : new_k;
    // new_j = (new_j == -1)   ? 0   : new_j;
    // new_j = (new_j >= M)    ? M-1 : new_j;
    // new_i = (new_i == -1)   ? 0   : new_i;
    // new_i = (new_i >= N)    ? N-1 : new_i;
    // threadInput11 = IN_3D(new_k, new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    int friend_id0, friend_id1, friend_id2, friend_id3;
    DATA_TYPE tx0, ty0, tz0, tx1, ty1, tz1, tx2, ty2, tz2, tx3, ty3, tz3;
    friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a0*__shfl(threadInput0, friend_id0);
    ty0 = a0*__shfl(threadInput1, friend_id0);
    tx1 = a0*__shfl(threadInput2, friend_id1);
    ty1 = a0*__shfl(threadInput3, friend_id1);
    tx2 = a0*__shfl(threadInput4, friend_id2);
    ty2 = a0*__shfl(threadInput5, friend_id2);
    tx3 = a0*__shfl(threadInput5, friend_id3);
    ty3 = a0*__shfl(threadInput6, friend_id3);
    tz3 = a0*__shfl(threadInput7, friend_id3);
    sum0 += (lane_id < 17)? tx0: ty0;
    sum1 += (lane_id < 21)? tx1: ty1;
    sum2 += (lane_id < 24)? tx2: ty2;
    sum3 += (lane_id < 1 )? tx3: ((lane_id < 27)? ty3: tz3);

    friend_id0 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a1*__shfl(threadInput1, friend_id0);
    ty0 = a1*__shfl(threadInput2, friend_id0);
    tz0 = a1*__shfl(threadInput3, friend_id0);
    tx1 = a1*__shfl(threadInput3, friend_id1);
    ty1 = a1*__shfl(threadInput4, friend_id1);
    tx2 = a1*__shfl(threadInput5, friend_id2);
    ty2 = a1*__shfl(threadInput6, friend_id2);
    tx3 = a1*__shfl(threadInput7, friend_id3);
    ty3 = a1*__shfl(threadInput8, friend_id3);
    sum0 += (lane_id < 3 )? tx0: ((lane_id < 29)? ty0: tz0);
    sum1 += (lane_id < 7 )? tx1: ty1;
    sum2 += (lane_id < 9 )? tx2: ty2;
    sum3 += (lane_id < 13)? tx3: ty3;

    friend_id0 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a2*__shfl(threadInput2, friend_id0);
    ty0 = a2*__shfl(threadInput3, friend_id0);
    tx1 = a2*__shfl(threadInput4, friend_id1);
    ty1 = a2*__shfl(threadInput5, friend_id1);
    tx2 = a2*__shfl(threadInput5, friend_id2);
    ty2 = a2*__shfl(threadInput6, friend_id2);
    tz2 = a2*__shfl(threadInput7, friend_id2);
    tx3 = a2*__shfl(threadInput7, friend_id3);
    ty3 = a2*__shfl(threadInput8, friend_id3);
    sum0 += (lane_id < 22)? tx0: ty0;
    sum1 += (lane_id < 24)? tx1: ty1;
    sum2 += (lane_id < 2 )? tx2: ((lane_id < 28)? ty2: tz2);
    sum3 += (lane_id < 6 )? tx3: ty3;

    friend_id0 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a3*__shfl(threadInput2, friend_id0);
    ty0 = a3*__shfl(threadInput3, friend_id0);
    tx1 = a3*__shfl(threadInput4, friend_id1);
    ty1 = a3*__shfl(threadInput5, friend_id1);
    tx2 = a3*__shfl(threadInput5, friend_id2);
    ty2 = a3*__shfl(threadInput6, friend_id2);
    tz2 = a3*__shfl(threadInput7, friend_id2);
    tx3 = a3*__shfl(threadInput7, friend_id3);
    ty3 = a3*__shfl(threadInput8, friend_id3);
    tz3 = a3*__shfl(threadInput9, friend_id3);
    sum0 += (lane_id < 21)? tx0: ty0;
    sum1 += (lane_id < 24)? tx1: ty1;
    sum2 += (lane_id < 1 )? tx2: ((lane_id < 27)? ty2: tz2);
    sum3 += (lane_id < 5 )? tx3: ((lane_id < 31)? ty3: tz3);

    friend_id0 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a4*__shfl(threadInput2, friend_id0);
    ty0 = a4*__shfl(threadInput3, friend_id0);
    tx1 = a4*__shfl(threadInput4, friend_id1);
    ty1 = a4*__shfl(threadInput5, friend_id1);
    tx2 = a4*__shfl(threadInput6, friend_id2);
    ty2 = a4*__shfl(threadInput7, friend_id2);
    tx3 = a4*__shfl(threadInput7, friend_id3);
    ty3 = a4*__shfl(threadInput8, friend_id3);
    tz3 = a4*__shfl(threadInput9, friend_id3);
    sum0 += (lane_id < 20)? tx0: ty0;
    sum1 += (lane_id < 24)? tx1: ty1;
    sum2 += (lane_id < 26)? tx2: ty2;
    sum3 += (lane_id < 4 )? tx3: ((lane_id < 30)? ty3: tz3);

    friend_id0 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a5*__shfl(threadInput2, friend_id0);
    ty0 = a5*__shfl(threadInput3, friend_id0);
    tx1 = a5*__shfl(threadInput4, friend_id1);
    ty1 = a5*__shfl(threadInput5, friend_id1);
    tx2 = a5*__shfl(threadInput6, friend_id2);
    ty2 = a5*__shfl(threadInput7, friend_id2);
    tx3 = a5*__shfl(threadInput8, friend_id3);
    ty3 = a5*__shfl(threadInput9, friend_id3);
    sum0 += (lane_id < 13)? tx0: ty0;
    sum1 += (lane_id < 16)? tx1: ty1;
    sum2 += (lane_id < 19)? tx2: ty2;
    sum3 += (lane_id < 23)? tx3: ty3;

    friend_id0 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+23+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a6*__shfl(threadInput4 , friend_id0);
    ty0 = a6*__shfl(threadInput5 , friend_id0);
    tx1 = a6*__shfl(threadInput5 , friend_id1);
    ty1 = a6*__shfl(threadInput6 , friend_id1);
    tz1 = a6*__shfl(threadInput7 , friend_id1);
    tx2 = a6*__shfl(threadInput7 , friend_id2);
    ty2 = a6*__shfl(threadInput8 , friend_id2);
    tz2 = a6*__shfl(threadInput9 , friend_id2);
    tx3 = a6*__shfl(threadInput9 , friend_id3);
    ty3 = a6*__shfl(threadInput10, friend_id3);
    sum0 += (lane_id < 24)? tx0: ty0;
    sum1 += (lane_id < 1 )? tx1: ((lane_id < 27)? ty1: tz1);
    sum2 += (lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2);
    sum3 += (lane_id < 8 )? tx3: ty3;

    OUT_3D(k,j,i) = sum0;
    OUT_3D(k+1,j,i) = sum1;
    OUT_3D(k+2,j,i) = sum2;
    OUT_3D(k+3,j,i) = sum3;
}

__global__ void Stencil_Cuda_Shfl8(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int Z, int M, int N)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;
    int k = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<3) + (lane_id>>5) ; 
    // Thread coarsening: related to warp dimensions 1x4x8. 
    // We coarsen from dimension z from 2^0 to 2^3, also need to know there are how many values in dimension z,
    // which is (lane_id>>5) 

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
    int warp_id_z = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<3) + (lane_id>>5); // these numbers
    int new_i = (warp_id_x<<3) + lane_id%10-1;     // 10 is extended dimension of i
    int new_j = (warp_id_y<<2) + (lane_id/10)%6-1; // 6  is extended dimension of j 
    int new_k = (warp_id_z<<0) + lane_id/60-1;     // 60 is extended area of ixj = 10x6
    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5,
              threadInput6, threadInput7, threadInput8, threadInput9, threadInput10, threadInput11,
              threadInput12, threadInput13, threadInput14, threadInput15, threadInput16, threadInput17,
              threadInput18;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput0 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+32)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput1 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+64)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput2 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+96)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput3 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+128)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput4 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+160)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput5 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+192)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+192)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+192)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput6 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+224)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+224)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+224)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput7 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+256)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+256)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+256)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput8 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+288)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+288)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+288)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput9 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+320)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+320)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+320)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput10 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+352)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+352)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+352)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput11 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+384)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+384)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+384)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput12 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+416)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+416)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+416)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput13 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+448)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+448)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+448)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput14 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+480)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+480)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+480)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput15 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+512)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+512)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+512)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput16 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+544)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+544)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+544)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput17 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+576)%10-1;
    new_j = (warp_id_y<<2) + ((lane_id+576)/10)%6-1;
    new_k = (warp_id_z<<0) + (lane_id+576)/60-1;
    new_k = (new_k == -1)   ? 0   : new_k;
    new_k = (new_k >= Z)    ? Z-1 : new_k;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    threadInput18 = IN_3D(new_k, new_j, new_i);

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
    friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+23+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+15+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a0*__shfl(threadInput0, friend_id0);
    ty0 = a0*__shfl(threadInput1, friend_id0);
    tx1 = a0*__shfl(threadInput2, friend_id1);
    ty1 = a0*__shfl(threadInput3, friend_id1);
    tx2 = a0*__shfl(threadInput4, friend_id2);
    ty2 = a0*__shfl(threadInput5, friend_id2);
    tx3 = a0*__shfl(threadInput5, friend_id3);
    ty3 = a0*__shfl(threadInput6, friend_id3);
    tz3 = a0*__shfl(threadInput7, friend_id3);
    rx0 = a0*__shfl(threadInput7, friend_id4);
    ry0 = a0*__shfl(threadInput8, friend_id4);
    rz0 = a0*__shfl(threadInput9, friend_id4);
    rx1 = a0*__shfl(threadInput9, friend_id5);
    ry1 = a0*__shfl(threadInput10, friend_id5);
    rx2 = a0*__shfl(threadInput11, friend_id6);
    ry2 = a0*__shfl(threadInput12, friend_id6);
    rx3 = a0*__shfl(threadInput13, friend_id7);
    ry3 = a0*__shfl(threadInput14, friend_id7);

    sum0 += (lane_id < 17)? tx0: ty0;
    sum1 += (lane_id < 21)? tx1: ty1;
    sum2 += (lane_id < 24)? tx2: ty2;
    sum3 += (lane_id < 1 )? tx3: ((lane_id < 27)? ty3: tz3);
    sum4 += (lane_id < 5 )? rx0: ((lane_id < 31)? ry0: rz0);
    sum5 += (lane_id < 8 )? rx1: ry1;
    sum6 += (lane_id < 11)? rx2: ry2;
    sum7 += (lane_id < 15)? rx3: ry3;

    friend_id0 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a1*__shfl(threadInput1, friend_id0);
    ty0 = a1*__shfl(threadInput2, friend_id0);
    tz0 = a1*__shfl(threadInput3, friend_id0);
    tx1 = a1*__shfl(threadInput3, friend_id1);
    ty1 = a1*__shfl(threadInput4, friend_id1);
    tx2 = a1*__shfl(threadInput5, friend_id2);
    ty2 = a1*__shfl(threadInput6, friend_id2);
    tx3 = a1*__shfl(threadInput7, friend_id3);
    ty3 = a1*__shfl(threadInput8, friend_id3);
    rx0 = a1*__shfl(threadInput9, friend_id4);
    ry0 = a1*__shfl(threadInput10, friend_id4);
    rx1 = a1*__shfl(threadInput11, friend_id5);
    ry1 = a1*__shfl(threadInput12, friend_id5);
    rx2 = a1*__shfl(threadInput13, friend_id6);
    ry2 = a1*__shfl(threadInput14, friend_id6);
    rx3 = a1*__shfl(threadInput15, friend_id7);
    ry3 = a1*__shfl(threadInput16, friend_id7);
    sum0 += (lane_id < 3 )? tx0: ((lane_id < 29)? ty0: tz0);
    sum1 += (lane_id < 7 )? tx1: ty1;
    sum2 += (lane_id < 9 )? tx2: ty2;
    sum3 += (lane_id < 13)? tx3: ty3;
    sum4 += (lane_id < 16)? rx0: ry0;
    sum5 += (lane_id < 19)? rx1: ry1;
    sum6 += (lane_id < 23)? rx2: ry2;
    sum7 += (lane_id < 25)? rx3: ry3;

    friend_id0 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a2*__shfl(threadInput2, friend_id0);
    ty0 = a2*__shfl(threadInput3, friend_id0);
    tx1 = a2*__shfl(threadInput4, friend_id1);
    ty1 = a2*__shfl(threadInput5, friend_id1);
    tx2 = a2*__shfl(threadInput5, friend_id2);
    ty2 = a2*__shfl(threadInput6, friend_id2);
    tz2 = a2*__shfl(threadInput7, friend_id2);
    tx3 = a2*__shfl(threadInput7, friend_id3);
    ty3 = a2*__shfl(threadInput8, friend_id3);
    rx0 = a2*__shfl(threadInput9, friend_id4);
    ry0 = a2*__shfl(threadInput10, friend_id4);
    rx1 = a2*__shfl(threadInput11, friend_id5);
    ry1 = a2*__shfl(threadInput12, friend_id5);
    rx2 = a2*__shfl(threadInput13, friend_id6);
    ry2 = a2*__shfl(threadInput14, friend_id6);
    rx3 = a2*__shfl(threadInput15, friend_id7);
    ry3 = a2*__shfl(threadInput16, friend_id7);
    sum0 += (lane_id < 22)? tx0: ty0;
    sum1 += (lane_id < 24)? tx1: ty1;
    sum2 += (lane_id < 2 )? tx2: ((lane_id < 28)? ty2: tz2);
    sum3 += (lane_id < 6 )? tx3: ty3;
    sum4 += (lane_id < 8 )? rx0: ry0;
    sum5 += (lane_id < 12)? rx1: ry1;
    sum6 += (lane_id < 16)? rx2: ry2;
    sum7 += (lane_id < 18)? rx3: ry3;

    friend_id0 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+23+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+15+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a3*__shfl(threadInput2, friend_id0);
    ty0 = a3*__shfl(threadInput3, friend_id0);
    tx1 = a3*__shfl(threadInput4, friend_id1);
    ty1 = a3*__shfl(threadInput5, friend_id1);
    tx2 = a3*__shfl(threadInput5, friend_id2);
    ty2 = a3*__shfl(threadInput6, friend_id2);
    tz2 = a3*__shfl(threadInput7, friend_id2);
    tx3 = a3*__shfl(threadInput7, friend_id3);
    ty3 = a3*__shfl(threadInput8, friend_id3);
    tz3 = a3*__shfl(threadInput9, friend_id3);
    rx0 = a3*__shfl(threadInput9, friend_id4);
    ry0 = a3*__shfl(threadInput10, friend_id4);
    rx1 = a3*__shfl(threadInput11, friend_id5);
    ry1 = a3*__shfl(threadInput12, friend_id5);
    rx2 = a3*__shfl(threadInput13, friend_id6);
    ry2 = a3*__shfl(threadInput14, friend_id6);
    rx3 = a3*__shfl(threadInput15, friend_id7);
    ry3 = a3*__shfl(threadInput16, friend_id7);
    sum0 += (lane_id < 21)? tx0: ty0;
    sum1 += (lane_id < 24)? tx1: ty1;
    sum2 += (lane_id < 1 )? tx2: ((lane_id < 27)? ty2: tz2);
    sum3 += (lane_id < 5 )? tx3: ((lane_id < 31)? ty3: tz3);
    sum4 += (lane_id < 8 )? rx0: ry0;
    sum5 += (lane_id < 11)? rx1: ry1;
    sum6 += (lane_id < 15)? rx2: ry2;
    sum7 += (lane_id < 17)? rx3: ry3;

    friend_id0 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a4*__shfl(threadInput2, friend_id0);
    ty0 = a4*__shfl(threadInput3, friend_id0);
    tx1 = a4*__shfl(threadInput4, friend_id1);
    ty1 = a4*__shfl(threadInput5, friend_id1);
    tx2 = a4*__shfl(threadInput6, friend_id2);
    ty2 = a4*__shfl(threadInput7, friend_id2);
    tx3 = a4*__shfl(threadInput7, friend_id3);
    ty3 = a4*__shfl(threadInput8, friend_id3);
    tz3 = a4*__shfl(threadInput9, friend_id3);
    rx0 = a4*__shfl(threadInput9, friend_id4);
    ry0 = a4*__shfl(threadInput10, friend_id4);
    rx1 = a4*__shfl(threadInput11, friend_id5);
    ry1 = a4*__shfl(threadInput12, friend_id5);
    rx2 = a4*__shfl(threadInput13, friend_id6);
    ry2 = a4*__shfl(threadInput14, friend_id6);
    rx3 = a4*__shfl(threadInput15, friend_id7);
    ry3 = a4*__shfl(threadInput16, friend_id7);
    sum0 += (lane_id < 20)? tx0: ty0;
    sum1 += (lane_id < 24)? tx1: ty1;
    sum2 += (lane_id < 26)? tx2: ty2;
    sum3 += (lane_id < 4 )? tx3: ((lane_id < 30)? ty3: tz3);
    sum4 += (lane_id < 8 )? rx0: ry0;
    sum5 += (lane_id < 10)? rx1: ry1;
    sum6 += (lane_id < 14)? rx2: ry2;
    sum7 += (lane_id < 16)? rx3: ry3;

    friend_id0 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a5*__shfl(threadInput2, friend_id0);
    ty0 = a5*__shfl(threadInput3, friend_id0);
    tx1 = a5*__shfl(threadInput4, friend_id1);
    ty1 = a5*__shfl(threadInput5, friend_id1);
    tx2 = a5*__shfl(threadInput6, friend_id2);
    ty2 = a5*__shfl(threadInput7, friend_id2);
    tx3 = a5*__shfl(threadInput8, friend_id3);
    ty3 = a5*__shfl(threadInput9, friend_id3);
    rx0 = a5*__shfl(threadInput10, friend_id4);
    ry0 = a5*__shfl(threadInput11, friend_id4);
    rx1 = a5*__shfl(threadInput11, friend_id5);
    ry1 = a5*__shfl(threadInput12, friend_id5);
    rz1 = a5*__shfl(threadInput13, friend_id5);
    rx2 = a5*__shfl(threadInput13, friend_id6);
    ry2 = a5*__shfl(threadInput14, friend_id6);
    rx3 = a5*__shfl(threadInput15, friend_id7);
    ry3 = a5*__shfl(threadInput16, friend_id7);
    sum0 += (lane_id < 13)? tx0: ty0;
    sum1 += (lane_id < 16)? tx1: ty1;
    sum2 += (lane_id < 19)? tx2: ty2;
    sum3 += (lane_id < 23)? tx3: ty3;
    sum4 += (lane_id < 25)? rx0: ry0;
    sum5 += (lane_id < 3 )? rx1: ((lane_id < 29)? ry1: rz1);
    sum6 += (lane_id < 7 )? rx2: ry2;
    sum7 += (lane_id < 9 )? rx3: ry3;

    friend_id0 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+23+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+15+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = a6*__shfl(threadInput4 , friend_id0);
    ty0 = a6*__shfl(threadInput5 , friend_id0);
    tx1 = a6*__shfl(threadInput5 , friend_id1);
    ty1 = a6*__shfl(threadInput6 , friend_id1);
    tz1 = a6*__shfl(threadInput7 , friend_id1);
    tx2 = a6*__shfl(threadInput7 , friend_id2);
    ty2 = a6*__shfl(threadInput8 , friend_id2);
    tz2 = a6*__shfl(threadInput9 , friend_id2);
    tx3 = a6*__shfl(threadInput9 , friend_id3);
    ty3 = a6*__shfl(threadInput10, friend_id3);
    rx0 = a6*__shfl(threadInput11, friend_id4);
    ry0 = a6*__shfl(threadInput12, friend_id4);
    rx1 = a6*__shfl(threadInput13, friend_id5);
    ry1 = a6*__shfl(threadInput14, friend_id5);
    rx2 = a6*__shfl(threadInput15, friend_id6);
    ry2 = a6*__shfl(threadInput16, friend_id6);
    rx3 = a6*__shfl(threadInput17, friend_id7);
    ry3 = a6*__shfl(threadInput18, friend_id7);
    sum0 += (lane_id < 24)? tx0: ty0;
    sum1 += (lane_id < 1 )? tx1: ((lane_id < 27)? ty1: tz1);
    sum2 += (lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2);
    sum3 += (lane_id < 8 )? tx3: ty3;
    sum4 += (lane_id < 11)? rx0: ry0;
    sum5 += (lane_id < 15)? rx1: ry1;
    sum6 += (lane_id < 17)? rx2: ry2;
    sum7 += (lane_id < 21)? rx3: ry3;

    OUT_3D(k,j,i) = sum0;
    OUT_3D(k+1,j,i) = sum1;
    OUT_3D(k+2,j,i) = sum2;
    OUT_3D(k+3,j,i) = sum3;
    OUT_3D(k+4,j,i) = sum4;
    OUT_3D(k+5,j,i) = sum5;
    OUT_3D(k+6,j,i) = sum6;
    OUT_3D(k+7,j,i) = sum7;
}

__global__ void Stencil_Cuda_Sweep_Shfl(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int Z, int M, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;
    const int block_z = Z / gridDim.z;
    int k = block_z * blockIdx.z ;
    const int k_end = k + block_z;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % warpSize;
    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
    DATA_TYPE tx, ty;
    int friend_id;
    int new_i, new_j;
    DATA_TYPE t3_threadInput0, t3_threadInput1;
    DATA_TYPE t2_threadInput0, t2_threadInput1;
    DATA_TYPE t1_threadInput0, t1_threadInput1;

#define SM_2D_M2 32 
#define SM_2D_N2 8 

    DATA_TYPE sum = 0.0;

    int b = (k == 0)      ? k : k - 1;
    // t3 is current layer; t2 is previous layer
    new_i = (warp_id_x<<3) + lane_id%10-1;     // 10 is extended dimension of i
    new_j = (warp_id_y<<2) + lane_id/10-1;     
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    t3_threadInput0 = IN_3D(k  , new_j, new_i);
    t2_threadInput0 = IN_3D(b  , new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+32)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    t3_threadInput1 = IN_3D(k  , new_j, new_i);
    t2_threadInput1 = IN_3D(b  , new_j, new_i);

#pragma unroll // it seems the loop-unroll is useless to performance
    for(; k < k_end; ++k)
    {
        sum = 0.0;
        // move the current storage down 
        t1_threadInput0 = t2_threadInput0;
        t1_threadInput1 = t2_threadInput1;
        t2_threadInput0 = t3_threadInput0;
        t2_threadInput1 = t3_threadInput1;

        int t = (k == Z-1)    ? k : k + 1;
        new_i = (warp_id_x<<3) + lane_id%10-1;  
        new_j = (warp_id_y<<2) + lane_id/10-1;     
        new_j = (new_j == -1)   ? 0   : new_j;
        new_j = (new_j >= M)    ? M-1 : new_j;
        new_i = (new_i == -1)   ? 0   : new_i;
        new_i = (new_i >= N)    ? N-1 : new_i;
        t3_threadInput0 = IN_3D(t, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+32)%10-1;
        new_j = (warp_id_y<<2) + (lane_id+32)/10-1;
        new_j = (new_j == -1)   ? 0   : new_j;
        new_j = (new_j >= M)    ? M-1 : new_j;
        new_i = (new_i == -1)   ? 0   : new_i;
        new_i = (new_i >= N)    ? N-1 : new_i;
        t3_threadInput1 = IN_3D(t, new_j, new_i);

        friend_id = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
        tx = a0*__shfl(t1_threadInput0, friend_id);
        ty = a0*__shfl(t1_threadInput1, friend_id);
        sum += (lane_id < 17)? tx: ty;
        tx = a3*__shfl(t2_threadInput0, friend_id);
        ty = a3*__shfl(t2_threadInput1, friend_id);
        sum += (lane_id < 17)? tx: ty;
        tx = a5*__shfl(t3_threadInput0, friend_id);
        ty = a5*__shfl(t3_threadInput1, friend_id);
        sum += (lane_id < 17)? tx: ty;
        friend_id = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
        tx = a1*__shfl(t2_threadInput0, friend_id);
        ty = a1*__shfl(t2_threadInput1, friend_id);
        sum += (lane_id < 25)? tx: ty;
        friend_id = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
        tx = a2*__shfl(t2_threadInput0, friend_id);
        ty = a2*__shfl(t2_threadInput1, friend_id);
        sum += (lane_id < 18)? tx: ty;
        friend_id = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
        tx = a4*__shfl(t2_threadInput0, friend_id);
        ty = a4*__shfl(t2_threadInput1, friend_id);
        sum += (lane_id < 16)? tx: ty;
        friend_id = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
        tx = a6*__shfl(t2_threadInput0, friend_id);
        ty = a6*__shfl(t2_threadInput1, friend_id);
        sum += (lane_id < 9 )? tx: ty;

        OUT_3D(k,j,i) = sum;
    }
}

__global__ void Stencil_Cuda_Sweep_Shfl2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int Z, int M, int N)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<3) + (lane_id>>3) ;

    const int block_z = Z / gridDim.z;
    int k = block_z * blockIdx.z ;
    const int k_end = k + block_z;
    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<3) + (lane_id>>3))>>2; // 1x4x8, warp_ids are division of 
    DATA_TYPE tx0, ty0;
    DATA_TYPE tx1, ty1, tz1;
    int friend_id0, friend_id1;
    int new_i, new_j;
    DATA_TYPE t3_threadInput0, t3_threadInput1, t3_threadInput2, t3_threadInput3;
    DATA_TYPE t2_threadInput0, t2_threadInput1, t2_threadInput2, t2_threadInput3;
    DATA_TYPE t1_threadInput0, t1_threadInput1, t1_threadInput2, t1_threadInput3;

#define SM_2D_M2 32 
#define SM_2D_N2 8 

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;

    // t3 is current layer; t2 is previous layer
    int b = (k == 0)      ? k : k - 1;
    new_i = (warp_id_x<<3) + lane_id%10-1;     // 10 is extended dimension of i
    new_j = (warp_id_y<<2) + lane_id/10-1;     
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    t3_threadInput0 = IN_3D(k  , new_j, new_i);
    t2_threadInput0 = IN_3D(b  , new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+32)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    t3_threadInput1 = IN_3D(k  , new_j, new_i);
    t2_threadInput1 = IN_3D(b  , new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10;
    new_j = (warp_id_y<<2) + (lane_id+64)/10;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    t3_threadInput2 = IN_3D(k  , new_j, new_i);
    t2_threadInput2 = IN_3D(b  , new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10;
    new_j = (warp_id_y<<2) + (lane_id+96)/10;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    t3_threadInput3 = IN_3D(k  , new_j, new_i);
    t2_threadInput3 = IN_3D(b  , new_j, new_i);

#pragma unroll // it seems the loop-unroll is useless to performance
    for(; k < k_end; ++k)
    {
        sum0 = 0.0;
        sum1 = 0.0;
        // move the current storage down 
        t1_threadInput0 = t2_threadInput0;
        t1_threadInput1 = t2_threadInput1;
        t1_threadInput2 = t2_threadInput2;
        t1_threadInput3 = t2_threadInput3;
        t2_threadInput0 = t3_threadInput0;
        t2_threadInput1 = t3_threadInput1;
        t2_threadInput2 = t3_threadInput2;
        t2_threadInput3 = t3_threadInput3;

        int t = (k == Z-1)    ? k : k + 1;
        new_i = (warp_id_x<<3) + lane_id%10-1;  
        new_j = (warp_id_y<<2) + lane_id/10-1;     
        new_j = (new_j == -1)   ? 0   : new_j;
        new_j = (new_j >= M)    ? M-1 : new_j;
        new_i = (new_i == -1)   ? 0   : new_i;
        new_i = (new_i >= N)    ? N-1 : new_i;
        t3_threadInput0 = IN_3D(t  , new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+32)%10-1;
        new_j = (warp_id_y<<2) + (lane_id+32)/10-1;
        new_j = (new_j == -1)   ? 0   : new_j;
        new_j = (new_j >= M)    ? M-1 : new_j;
        new_i = (new_i == -1)   ? 0   : new_i;
        new_i = (new_i >= N)    ? N-1 : new_i;
        t3_threadInput1 = IN_3D(t  , new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+64)%10-1;
        new_j = (warp_id_y<<2) + (lane_id+64)/10-1;
        new_j = (new_j == -1)   ? 0   : new_j;
        new_j = (new_j >= M)    ? M-1 : new_j;
        new_i = (new_i == -1)   ? 0   : new_i;
        new_i = (new_i >= N)    ? N-1 : new_i;
        t3_threadInput2 = IN_3D(t  , new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+96)%10-1;
        new_j = (warp_id_y<<2) + (lane_id+96)/10-1;
        new_j = (new_j == -1)   ? 0   : new_j;
        new_j = (new_j >= M)    ? M-1 : new_j;
        new_i = (new_i == -1)   ? 0   : new_i;
        new_i = (new_i >= N)    ? N-1 : new_i;
        t3_threadInput3 = IN_3D(t  , new_j, new_i);

        friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = a0*__shfl(t1_threadInput0, friend_id0);
        ty0 = a0*__shfl(t1_threadInput1, friend_id0);
        tx1 = a0*__shfl(t1_threadInput1, friend_id1);
        ty1 = a0*__shfl(t1_threadInput2, friend_id1);
        sum0 += (lane_id < 17)? tx0: ty0;
        sum1 += (lane_id < 11)? tx1: ty1;
        tx0 = a3*__shfl(t2_threadInput0, friend_id0);
        ty0 = a3*__shfl(t2_threadInput1, friend_id0);
        tx1 = a3*__shfl(t2_threadInput1, friend_id1);
        ty1 = a3*__shfl(t2_threadInput2, friend_id1);
        sum0 += (lane_id < 17)? tx0: ty0;
        sum1 += (lane_id < 11)? tx1: ty1;
        tx0 = a5*__shfl(t3_threadInput0, friend_id0);
        ty0 = a5*__shfl(t3_threadInput1, friend_id0);
        tx1 = a5*__shfl(t3_threadInput1, friend_id1);
        ty1 = a5*__shfl(t3_threadInput2, friend_id1);
        sum0 += (lane_id < 17)? tx0: ty0;
        sum1 += (lane_id < 11)? tx1: ty1;

        friend_id0 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
        tx0 = a1*__shfl(t2_threadInput0, friend_id0);
        ty0 = a1*__shfl(t2_threadInput1, friend_id0);
        tx1 = a1*__shfl(t2_threadInput1, friend_id1);
        ty1 = a1*__shfl(t2_threadInput2, friend_id1);
        sum0 += (lane_id < 25)? tx0: ty0;
        sum1 += (lane_id < 19)? tx1: ty1;
        friend_id0 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = a2*__shfl(t2_threadInput0, friend_id0);
        ty0 = a2*__shfl(t2_threadInput1, friend_id0);
        tx1 = a2*__shfl(t2_threadInput1, friend_id1);
        ty1 = a2*__shfl(t2_threadInput2, friend_id1);
        sum0 += (lane_id < 18)? tx0: ty0;
        sum1 += (lane_id < 12)? tx1: ty1;
        friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = a4*__shfl(t2_threadInput0, friend_id0);
        ty0 = a4*__shfl(t2_threadInput1, friend_id0);
        tx1 = a4*__shfl(t2_threadInput1, friend_id1);
        ty1 = a4*__shfl(t2_threadInput2, friend_id1);
        sum0 += (lane_id < 16)? tx0: ty0;
        sum1 += (lane_id < 10)? tx1: ty1;
        friend_id0 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = a5*__shfl(t2_threadInput0, friend_id0);
        ty0 = a5*__shfl(t2_threadInput1, friend_id0);
        tx1 = a5*__shfl(t2_threadInput1, friend_id1);
        ty1 = a5*__shfl(t2_threadInput2, friend_id1);
        tz1 = a5*__shfl(t2_threadInput3, friend_id1);
        sum0 += (lane_id < 9 )? tx0: ty0;
        sum1 += (lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1);

        OUT_3D(k,j  ,i) = sum0;
        OUT_3D(k,j+4,i) = sum1;
    }
}

__global__ void Stencil_Cuda_Sweep_Shfl4(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int Z, int M, int N)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<4) + (lane_id>>3) ;

    const int block_z = Z / gridDim.z;
    int k = block_z * blockIdx.z ;
    const int k_end = k + block_z;
    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<4) + (lane_id>>3))>>2; // 1x4x8, warp_ids are division of 
    DATA_TYPE tx0, ty0;
    DATA_TYPE tx1, ty1, tz1;
    DATA_TYPE tx2, ty2, tz2;
    DATA_TYPE tx3, ty3, tz3;
    int friend_id0, friend_id1;
    int friend_id2, friend_id3;
    int new_i, new_j;
    DATA_TYPE t3_threadInput0, t3_threadInput1, t3_threadInput2, t3_threadInput3, t3_threadInput4, t3_threadInput5;
    DATA_TYPE t2_threadInput0, t2_threadInput1, t2_threadInput2, t2_threadInput3, t2_threadInput4, t2_threadInput5;
    DATA_TYPE t1_threadInput0, t1_threadInput1, t1_threadInput2, t1_threadInput3, t1_threadInput4, t1_threadInput5;

#define SM_2D_M2 32 
#define SM_2D_N2 8 

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;

    // t3 is current layer; t2 is previous layer
    int b = (k == 0)      ? k : k - 1;
    new_i = (warp_id_x<<3) + lane_id%10-1;     // 10 is extended dimension of i
    new_j = (warp_id_y<<2) + lane_id/10-1;     
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    t3_threadInput0 = IN_3D(k  , new_j, new_i);
    t2_threadInput0 = IN_3D(b  , new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+32)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    t3_threadInput1 = IN_3D(k  , new_j, new_i);
    t2_threadInput1 = IN_3D(b  , new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+64)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    t3_threadInput2 = IN_3D(k  , new_j, new_i);
    t2_threadInput2 = IN_3D(b  , new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+96)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    t3_threadInput3 = IN_3D(k  , new_j, new_i);
    t2_threadInput3 = IN_3D(b  , new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+128)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    t3_threadInput4 = IN_3D(k  , new_j, new_i);
    t2_threadInput4 = IN_3D(b  , new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10-1;
    new_j = (warp_id_y<<2) + (lane_id+160)/10-1;
    new_j = (new_j == -1)   ? 0   : new_j;
    new_j = (new_j >= M)    ? M-1 : new_j;
    new_i = (new_i == -1)   ? 0   : new_i;
    new_i = (new_i >= N)    ? N-1 : new_i;
    t3_threadInput5 = IN_3D(k  , new_j, new_i);
    t2_threadInput5 = IN_3D(b  , new_j, new_i);

#pragma unroll // it seems the loop-unroll is useless to performance
    for(; k < k_end; ++k)
    {
        sum0 = 0.0;
        sum1 = 0.0;
        sum2 = 0.0;
        sum3 = 0.0;
        // move the current storage down 
        t1_threadInput0 = t2_threadInput0;
        t1_threadInput1 = t2_threadInput1;
        t1_threadInput2 = t2_threadInput2;
        t1_threadInput3 = t2_threadInput3;
        t1_threadInput4 = t2_threadInput4;
        t1_threadInput5 = t2_threadInput5;

        t2_threadInput0 = t3_threadInput0;
        t2_threadInput1 = t3_threadInput1;
        t2_threadInput2 = t3_threadInput2;
        t2_threadInput3 = t3_threadInput3;
        t2_threadInput4 = t3_threadInput4;
        t2_threadInput5 = t3_threadInput5;

        int t = (k == Z-1)    ? k : k + 1;
        new_i = (warp_id_x<<3) + lane_id%10-1;  
        new_j = (warp_id_y<<2) + lane_id/10-1;     
        new_j = (new_j == -1)   ? 0   : new_j;
        new_j = (new_j >= M)    ? M-1 : new_j;
        new_i = (new_i == -1)   ? 0   : new_i;
        new_i = (new_i >= N)    ? N-1 : new_i;
        t3_threadInput0 = IN_3D(t  , new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+32)%10-1;
        new_j = (warp_id_y<<2) + (lane_id+32)/10-1;
        new_j = (new_j == -1)   ? 0   : new_j;
        new_j = (new_j >= M)    ? M-1 : new_j;
        new_i = (new_i == -1)   ? 0   : new_i;
        new_i = (new_i >= N)    ? N-1 : new_i;
        t3_threadInput1 = IN_3D(t  , new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+64)%10-1;
        new_j = (warp_id_y<<2) + (lane_id+64)/10-1;
        new_j = (new_j == -1)   ? 0   : new_j;
        new_j = (new_j >= M)    ? M-1 : new_j;
        new_i = (new_i == -1)   ? 0   : new_i;
        new_i = (new_i >= N)    ? N-1 : new_i;
        t3_threadInput2 = IN_3D(t  , new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+96)%10-1;
        new_j = (warp_id_y<<2) + (lane_id+96)/10-1;
        new_j = (new_j == -1)   ? 0   : new_j;
        new_j = (new_j >= M)    ? M-1 : new_j;
        new_i = (new_i == -1)   ? 0   : new_i;
        new_i = (new_i >= N)    ? N-1 : new_i;
        t3_threadInput3 = IN_3D(t  , new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+128)%10-1;
        new_j = (warp_id_y<<2) + (lane_id+128)/10-1;
        new_j = (new_j == -1)   ? 0   : new_j;
        new_j = (new_j >= M)    ? M-1 : new_j;
        new_i = (new_i == -1)   ? 0   : new_i;
        new_i = (new_i >= N)    ? N-1 : new_i;
        t3_threadInput4 = IN_3D(t  , new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+160)%10-1;
        new_j = (warp_id_y<<2) + (lane_id+160)/10-1;
        new_j = (new_j == -1)   ? 0   : new_j;
        new_j = (new_j >= M)    ? M-1 : new_j;
        new_i = (new_i == -1)   ? 0   : new_i;
        new_i = (new_i >= N)    ? N-1 : new_i;
        t3_threadInput5 = IN_3D(t  , new_j, new_i);

        friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
        tx0 = a0*__shfl(t1_threadInput0, friend_id0);
        ty0 = a0*__shfl(t1_threadInput1, friend_id0);
        tx1 = a0*__shfl(t1_threadInput1, friend_id1);
        ty1 = a0*__shfl(t1_threadInput2, friend_id1);
        tx2 = a0*__shfl(t1_threadInput2, friend_id2);
        ty2 = a0*__shfl(t1_threadInput3, friend_id2);
        tz2 = a0*__shfl(t1_threadInput4, friend_id2);
        tx3 = a0*__shfl(t1_threadInput4, friend_id3);
        ty3 = a0*__shfl(t1_threadInput5, friend_id3);
        sum0 += (lane_id < 17)? tx0: ty0;
        sum1 += (lane_id < 11)? tx1: ty1;
        sum2 += (lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2);
        sum3 += (lane_id < 24)? tx3: ty3;
        tx0 = a3*__shfl(t2_threadInput0, friend_id0);
        ty0 = a3*__shfl(t2_threadInput1, friend_id0);
        tx1 = a3*__shfl(t2_threadInput1, friend_id1);
        ty1 = a3*__shfl(t2_threadInput2, friend_id1);
        tx2 = a3*__shfl(t2_threadInput2, friend_id2);
        ty2 = a3*__shfl(t2_threadInput3, friend_id2);
        tz2 = a3*__shfl(t2_threadInput4, friend_id2);
        tx3 = a3*__shfl(t2_threadInput4, friend_id3);
        ty3 = a3*__shfl(t2_threadInput5, friend_id3);
        sum0 += (lane_id < 17)? tx0: ty0;
        sum1 += (lane_id < 11)? tx1: ty1;
        sum2 += (lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2);
        sum3 += (lane_id < 24)? tx3: ty3;
        tx0 = a5*__shfl(t3_threadInput0, friend_id0);
        ty0 = a5*__shfl(t3_threadInput1, friend_id0);
        tx1 = a5*__shfl(t3_threadInput1, friend_id1);
        ty1 = a5*__shfl(t3_threadInput2, friend_id1);
        tx2 = a5*__shfl(t3_threadInput2, friend_id2);
        ty2 = a5*__shfl(t3_threadInput3, friend_id2);
        tz2 = a5*__shfl(t3_threadInput4, friend_id2);
        tx3 = a5*__shfl(t3_threadInput4, friend_id3);
        ty3 = a5*__shfl(t3_threadInput5, friend_id3);
        sum0 += (lane_id < 17)? tx0: ty0;
        sum1 += (lane_id < 11)? tx1: ty1;
        sum2 += (lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2);
        sum3 += (lane_id < 24)? tx3: ty3;

        friend_id0 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = a1*__shfl(t2_threadInput0, friend_id0);
        ty0 = a1*__shfl(t2_threadInput1, friend_id0);
        tx1 = a1*__shfl(t2_threadInput1, friend_id1);
        ty1 = a1*__shfl(t2_threadInput2, friend_id1);
        tx2 = a1*__shfl(t3_threadInput2, friend_id2);
        ty2 = a1*__shfl(t3_threadInput3, friend_id2);
        tx3 = a1*__shfl(t3_threadInput3, friend_id3);
        ty3 = a1*__shfl(t3_threadInput4, friend_id3);
        sum0 += (lane_id < 25)? tx0: ty0;
        sum1 += (lane_id < 19)? tx1: ty1;
        sum2 += (lane_id < 13)? tx2: ty2;
        sum3 += (lane_id < 7 )? tx3: ty3;
        friend_id0 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
        tx0 = a2*__shfl(t2_threadInput0, friend_id0);
        ty0 = a2*__shfl(t2_threadInput1, friend_id0);
        tx1 = a2*__shfl(t2_threadInput1, friend_id1);
        ty1 = a2*__shfl(t2_threadInput2, friend_id1);
        tx2 = a2*__shfl(t3_threadInput2, friend_id2);
        ty2 = a2*__shfl(t3_threadInput3, friend_id2);
        tx3 = a2*__shfl(t3_threadInput3, friend_id3);
        ty3 = a2*__shfl(t3_threadInput4, friend_id3);
        sum0 += (lane_id < 18)? tx0: ty0;
        sum1 += (lane_id < 12)? tx1: ty1;
        sum2 += (lane_id < 6 )? tx2: ty2;
        sum3 += (lane_id < 24)? tx3: ty3;
        friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
        tx0 = a4*__shfl(t2_threadInput0, friend_id0);
        ty0 = a4*__shfl(t2_threadInput1, friend_id0);
        tx1 = a4*__shfl(t2_threadInput1, friend_id1);
        ty1 = a4*__shfl(t2_threadInput2, friend_id1);
        tx2 = a4*__shfl(t3_threadInput2, friend_id2);
        ty2 = a4*__shfl(t3_threadInput3, friend_id2);
        tz2 = a4*__shfl(t3_threadInput4, friend_id2);
        tx3 = a4*__shfl(t3_threadInput4, friend_id3);
        ty3 = a4*__shfl(t3_threadInput5, friend_id3);
        sum0 += (lane_id < 16)? tx0: ty0;
        sum1 += (lane_id < 10)? tx1: ty1;
        sum2 += (lane_id < 4 )? tx2: ((lane_id < 30)? ty2: tz2);
        sum3 += (lane_id < 24)? tx3: ty3;
        friend_id0 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = a5*__shfl(t2_threadInput0, friend_id0);
        ty0 = a5*__shfl(t2_threadInput1, friend_id0);
        tx1 = a5*__shfl(t2_threadInput1, friend_id1);
        ty1 = a5*__shfl(t2_threadInput2, friend_id1);
        tz1 = a5*__shfl(t2_threadInput3, friend_id1);
        tx2 = a5*__shfl(t3_threadInput3, friend_id2);
        ty2 = a5*__shfl(t3_threadInput4, friend_id2);
        tx3 = a5*__shfl(t3_threadInput4, friend_id3);
        ty3 = a5*__shfl(t3_threadInput5, friend_id3);
        sum0 += (lane_id < 9 )? tx0: ty0;
        sum1 += (lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1);
        sum2 += (lane_id < 23)? tx2: ty2;
        sum3 += (lane_id < 16)? tx3: ty3;

        OUT_3D(k,j  ,i) = sum0;
        OUT_3D(k,j+4,i) = sum1;
        OUT_3D(k,j+8,i) = sum2;
        OUT_3D(k,j+12,i) = sum3;
    }
}

int main(int argc, char **argv)
{
    // int z = 192; // need to be multiple of 64
    // int m = 160;
    // int n = 1600; // need to be mutiple of 64 
#ifdef __DEBUG
    int z = 8;
    int m = 8;
    int n = 8;
#else
    int z = 256; 
    int m = 256;
    int n = 256; 
#endif
    // int z = 192;
    // int m = 160;
    // int n = 1612;
    // int halo = 1;
    // int total = (z+2*halo)*(m+2*halo)*(n+2*halo);
    int total = (z)*(m)*(n);
    const int K = 7;
    DATA_TYPE args[K] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    DATA_TYPE *in = new DATA_TYPE[total];
    DATA_TYPE *out_ref = new DATA_TYPE[total];
    Init_Input_3D(in, z, m, n);

    // Show_Me(in, z, m, n, "Input:");
    for(int i =0; i< ITER; i++)
    {
        Stencil_Seq(in, out_ref, args[0], args[1], args[2], 
            args[3], args[4], args[5], args[6], z, m, n);
        swap(in, out_ref);
    }
    swap(in, out_ref);
    // Show_Me(out_ref, z, m, n, "Output:");


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
    Init_Input_3D(in, z, m, n);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid(n/8, m/4, z/8);
    dim3 dimBlock(8, 4, 8);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda<<<dimGrid, dimBlock>>>(in_d, out_d, args[0], args[1], args[2], 
            args[3], args[4], args[5], args[6], z, m, n); 
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);

    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, z, m, n, "Output(Cuda):");

    cout << "Verify Cuda: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, 13, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(z, m, n, ITER, time_wo_pci));


    Init_Input_3D(in, z, m, n); // reset input
    Clear_Output_3D(out, z, m, n); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid1((n)/64, (m)/4, 4);
    dim3 dimBlock1(64, 4, 1);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sweep<<<dimGrid1, dimBlock1>>>(in_d, out_d, args[0], args[1], args[2], 
            args[3], args[4], args[5], args[6], z, m, n);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
#ifdef __DEBUG_
    Show_Me(out, z, m, n, "Output(Sweep):");
#endif
    cout << "Verify Cuda_Sweep: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Sweep Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, 13, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(z, m, n, ITER, time_wo_pci));


    Init_Input_3D(in, z, m, n); // reset input
    Clear_Output_3D(out, z, m, n); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid2((n)/8, (m)/4, (z)/8);
    dim3 dimBlock2(8, 4, 8);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sm<<<dimGrid2, dimBlock2, ((SM_Z)*(SM_M)*(SM_N))*sizeof(DATA_TYPE)>>>(
            in_d, out_d, args[0], args[1], args[2], args[3], args[4], args[5], args[6], z, m, n);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
#ifdef __DEBUG_
    Show_Me(out, z, m, n, "Output(Cuda_Sm):");
#endif
    cout << "Verify Cuda_Sm: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Sm Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, 13, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(z, m, n, ITER, time_wo_pci));


    Init_Input_3D(in, z, m, n); // reset input
    Clear_Output_3D(out, z, m, n); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid3((n)/64, (m)/4, 4);
    dim3 dimBlock3(64, 4, 1);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sweep_Sm<<<dimGrid3, dimBlock3, ((SM_2D_M)*(SM_2D_N)*sizeof(DATA_TYPE))>>>(
            in_d, out_d, args[0], args[1], args[2], args[3], args[4], args[5], args[6], z, m, n);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
#ifdef __DEBUG_
    Show_Me(out, z, m, n, "Output(Cuda_Sweep_Sm):");
#endif
    cout << "Verify Cuda_Sweep_Sm: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Sweep_Sm Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, 13, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(z, m, n, ITER, time_wo_pci));


    Init_Input_3D(in, z, m, n); // reset input
    Clear_Output_3D(out, z, m, n); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid4((n)/8, (m)/4, (z)/8);
    dim3 dimBlock4(8, 4, 8);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl<<<dimGrid4, dimBlock4>>>(
            in_d, out_d, args[0], args[1], args[2], args[3], args[4], args[5], args[6], z, m, n);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
#ifdef __DEBUG
    Show_Me(out, z, m, n, "Output(Cuda_Shfl):");
#endif
    cout << "Verify Cuda_Shfl: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Shfl Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, 13, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(z, m, n, ITER, time_wo_pci));


    Init_Input_3D(in, z, m, n); // reset input
    Clear_Output_3D(out, z, m, n); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid5((n)/8, (m)/4, (z)/(8*2));
    dim3 dimBlock5(8, 4, 8);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl2<<<dimGrid5, dimBlock5>>>(
            in_d, out_d, args[0], args[1], args[2], args[3], args[4], args[5], args[6], z, m, n);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
#ifdef __DEBUG_
    Show_Me(out, z, m, n, "Output(Cuda_Shfl2):");
#endif
    cout << "Verify Cuda_Shfl2: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Shfl2 Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, 13, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(z, m, n, ITER, time_wo_pci));

    Init_Input_3D(in, z, m, n); // reset input
    Clear_Output_3D(out, z, m, n); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid6((n)/8, (m)/4, (z)/(8*4));
    dim3 dimBlock6(8, 4, 8);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl4<<<dimGrid6, dimBlock6>>>(
            in_d, out_d, args[0], args[1], args[2], args[3], args[4], args[5], args[6], z, m, n);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
#ifdef __DEBUG_
    Show_Me(out, z, m, n, "Output(Cuda_Shfl4):");
#endif
    cout << "Verify Cuda_Shfl4: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Shfl4 Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, 13, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(z, m, n, ITER, time_wo_pci));


    Init_Input_3D(in, z, m, n); // reset input
    Clear_Output_3D(out, z, m, n); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid7((n)/8, (m)/4, (z)/(8*8));
    dim3 dimBlock7(8, 4, 8);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl8<<<dimGrid7, dimBlock7>>>(
            in_d, out_d, args[0], args[1], args[2], args[3], args[4], args[5], args[6], z, m, n);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
#ifdef __DEBUG_
    Show_Me(out, z, m, n, "Output(Cuda_Shfl8):");
#endif
    cout << "Verify Cuda_Shfl8: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Shfl8 Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, 13, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(z, m, n, ITER, time_wo_pci));


    Init_Input_3D(in, z, m, n); // reset input
    Clear_Output_3D(out, z, m, n); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid8((n)/8, (m)/32, 4);
    dim3 dimBlock8(8, 32, 1);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sweep_Shfl<<<dimGrid8, dimBlock8>>>(
            in_d, out_d, args[0], args[1], args[2], args[3], args[4], args[5], args[6], z, m, n);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
#ifdef __DEBUG_
    Show_Me(out, z, m, n, "Output(Cuda_Sweep_Shfl):");
#endif
    cout << "Verify Cuda_Sweep_Shfl: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Sweep_Shfl Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, 13, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(z, m, n, ITER, time_wo_pci));


    Init_Input_3D(in, z, m, n); // reset input
    Clear_Output_3D(out, z, m, n); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid9((n)/8, (m)/(32*2), 4);
    dim3 dimBlock9(8, 32, 1);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sweep_Shfl2<<<dimGrid9, dimBlock9>>>(
            in_d, out_d, args[0], args[1], args[2], args[3], args[4], args[5], args[6], z, m, n);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
#ifdef __DEBUG_
    Show_Me(out, z, m, n, "Output(Cuda_Sweep_Shfl2):");
#endif
    cout << "Verify Cuda_Sweep_Shfl2: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Sweep_Shfl2 Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, 13, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(z, m, n, ITER, time_wo_pci));

    Init_Input_3D(in, z, m, n); // reset input
    Clear_Output_3D(out, z, m, n); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid10((n)/8, (m)/(32*4), 4);
    dim3 dimBlock10(8, 32, 1);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sweep_Shfl4<<<dimGrid10, dimBlock10>>>(
            in_d, out_d, args[0], args[1], args[2], args[3], args[4], args[5], args[6], z, m, n);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
#ifdef __DEBUG_
    Show_Me(out, z, m, n, "Output(Cuda_Sweep_Shfl4):");
#endif
    cout << "Verify Cuda_Sweep_Shfl4: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Sweep_Shfl4 Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, 13, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(z, m, n, ITER, time_wo_pci));

    cudaFree(in_d);
    cudaFree(out_d);


    delete[] in;
    delete[] out;
    delete[] out_ref;

}
