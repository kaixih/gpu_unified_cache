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

void Init_Input_3D(DATA_TYPE *in, int z, int m, int n, int halo, unsigned int seed)
{
    srand(seed);

    for(int k = halo; k < z+halo; k++)
        for(int j = halo; j < m+halo; j++)
            for(int i = halo; i < n+halo; i++)
#ifdef __DEBUG
                IN_3D(k,j,i) = 1; 
                // IN_3D(k,j,i) = (DATA_TYPE)rand()*100.0 / ((long)RAND_MAX);
#else
                IN_3D(k,j,i) = (DATA_TYPE)rand()*100.0 / ((long)RAND_MAX);
#endif
}

void Clear_Output_3D(DATA_TYPE *in, int z, int m, int n, int halo)
{
    for(int k = 0; k < z+2*halo; k++)
        for(int j = 0; j < m+2*halo; j++)
            for(int i = 0; i < n+2*halo; i++)
                IN_3D(k,j,i) = 0;
}

void Show_Me(DATA_TYPE *in, int z, int m, int n, int halo, std::string prompt)
{
    std::cout << prompt << std::endl;
    for(int k = 0; k < z+2*halo; k++)
    {
        for(int j = 0; j < m+2*halo; j++)
        {
            for(int i = 0; i < n+2*halo; i++)
                std::cout << IN_3D(k,j,i) << ",";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void Stencil_Seq(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int z, int m, int n, int halo)
{

#pragma omp parallel for 
    for(int k = halo; k < z+halo; k++)
    {
        for(int j = halo; j < m+halo; j++)
        {
            for(int i = halo; i < n+halo; i++)
            {
                OUT_3D(k,j,i) = a0 * IN_3D(k-1,j  ,i  ) +
                                a1 * IN_3D(k  ,j-1,i  ) +
                                a2 * IN_3D(k  ,j  ,i-1) +
                                a3 * IN_3D(k  ,j  ,i  ) +
                                a4 * IN_3D(k  ,j  ,i+1) +
                                a5 * IN_3D(k  ,j+1,i  ) +
                                a6 * IN_3D(k+1,j  ,i  ) ;
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

__global__ void Stencil_Cuda_L1_3Blk(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int z, int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = threadIdx.z + blockIdx.z * blockDim.z + halo;

    OUT_3D(k,j,i) = a0 * IN_3D(k-1,j  ,i  ) +
                    a1 * IN_3D(k  ,j-1,i  ) +
                    a2 * IN_3D(k  ,j  ,i-1) +
                    a3 * IN_3D(k  ,j  ,i  ) +
                    a4 * IN_3D(k  ,j  ,i+1) +
                    a5 * IN_3D(k  ,j+1,i  ) +
                    a6 * IN_3D(k+1,j  ,i  ) ;
}

__global__ void Stencil_Cuda_L1_25Blk(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int z, int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;

    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;

#pragma unroll // it seems the loop-unroll is useless to performance
    for(; k < k_end; ++k)
    {
        OUT_3D(k,j,i) = a0 * IN_3D(k-1,j  ,i  ) +
                        a1 * IN_3D(k  ,j-1,i  ) +
                        a2 * IN_3D(k  ,j  ,i-1) +
                        a3 * IN_3D(k  ,j  ,i  ) +
                        a4 * IN_3D(k  ,j  ,i+1) +
                        a5 * IN_3D(k  ,j+1,i  ) +
                        a6 * IN_3D(k+1,j  ,i  ) ;
    }
}

__global__ void Stencil_Cuda_Lds_25BlkBrc(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int z, int m, int n, int halo)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;
    int li = threadIdx.x + 1;
    int lj = threadIdx.y + 1;

    __shared__ DATA_TYPE local[3][8+2][32+2];

    int t1, t2, t3;
    t3 = 2; t2 = 1;
    
    LOC_3D(t3,lj,li) = IN_3D(k  ,j,i);
    LOC_3D(t2,lj,li) = IN_3D(k-1,j,i);
    if(li == halo)                                   
    {
        LOC_3D(t3,lj  ,li-1) = IN_3D(k  ,j  ,i-1); 
        LOC_3D(t2,lj  ,li-1) = IN_3D(k-1,j  ,i-1);
    }
    if(li == 32  )                         
    {
        LOC_3D(t3,lj  ,li+1) = IN_3D(k  ,j  ,i+1); 
        LOC_3D(t2,lj  ,li+1) = IN_3D(k-1,j  ,i+1);
    }
    if(lj == halo)                                   
    {
        LOC_3D(t3,lj-1,li  ) = IN_3D(k  ,j-1,i  ); 
        LOC_3D(t2,lj-1,li  ) = IN_3D(k-1,j-1,i  );
    }
    if(lj == 8   )                         
    {
        LOC_3D(t3,lj+1,li  ) = IN_3D(k  ,j+1,i  ); 
        LOC_3D(t2,lj+1,li  ) = IN_3D(k-1,j+1,i  );
    }

#pragma unroll // it seems the loop-unroll is useless to performance
    for(; k < k_end; ++k)
    {
        t1 = t2;
        t2 = t3;
        t3 = (t3+1)%3;

        LOC_3D(t3,lj,li) = IN_3D(k+1,j,i);
        if(li == halo)               {LOC_3D(t3,lj  ,li-1) = IN_3D(k+1,j  ,i-1);}
        if(li == 32  )               {LOC_3D(t3,lj  ,li+1) = IN_3D(k+1,j  ,i+1);}
        if(lj == halo)               {LOC_3D(t3,lj-1,li  ) = IN_3D(k+1,j-1,i  );}
        if(lj == 8   )               {LOC_3D(t3,lj+1,li  ) = IN_3D(k+1,j+1,i  );}
        __syncthreads();

        OUT_3D(k,j,i) = a0 * LOC_3D(t1,lj  ,li  ) +
                        a1 * LOC_3D(t2,lj-1,li  ) + 
                        a2 * LOC_3D(t2,lj  ,li-1) + 
                        a3 * LOC_3D(t2,lj  ,li  ) + 
                        a4 * LOC_3D(t2,lj  ,li+1) + 
                        a5 * LOC_3D(t2,lj+1,li  ) + 
                        a6 * LOC_3D(t3,lj  ,li  ) ;
        
        __syncthreads();
    }
}

__global__ void Stencil_Cuda_Lds_25BlkCyc(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int z, int m, int n, int halo)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;

    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;

    int li = threadIdx.x + 1;
    int lj = threadIdx.y + 1;

    __shared__ DATA_TYPE local[3][8+2][32+2];

    int t1, t2, t3;
    t3 = 2; t2 = 1;

    unsigned int lane_id = threadIdx.x + threadIdx.y * blockDim.x;
    int blk_id_x = blockIdx.x;
    int blk_id_y = blockIdx.y;
    int new_i, new_j, new_li, new_lj;
    new_i  = (blk_id_x<<5) + lane_id%34;
    new_j  = (blk_id_y<<3) + lane_id/34;
    new_li = lane_id%34;
    new_lj = lane_id/34;
    LOC_3D(t3,new_lj,new_li) = IN_3D(k  ,new_j,new_i);
    LOC_3D(t2,new_lj,new_li) = IN_3D(k-1,new_j,new_i);
    new_i  = (blk_id_x<<5) + (lane_id+256)%34;
    new_j  = (blk_id_y<<3) + (lane_id+256)/34;
    new_li = (lane_id+256)%34;
    new_lj = (lane_id+256)/34;
    new_i  = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j  = (new_j < m+2*halo)? new_j: m+2*halo-1;
    if(new_li < 34 &&  new_lj < 10)
    {
        LOC_3D(t3,new_lj,new_li) = IN_3D(k  ,new_j,new_i);
        LOC_3D(t2,new_lj,new_li) = IN_3D(k-1,new_j,new_i);
    }


#pragma unroll // it seems the loop-unroll is useless to performance
    for(; k < k_end; ++k)
    {
        t1 = t2;
        t2 = t3;
        t3 = (t3+1)%3;

        new_i  = (blk_id_x<<5) + lane_id%34;
        new_j  = (blk_id_y<<3) + lane_id/34;
        new_li = lane_id%34;
        new_lj = lane_id/34;
        LOC_3D(t3,new_lj,new_li) = IN_3D(k+1,new_j,new_i);
        new_i  = (blk_id_x<<5) + (lane_id+256)%34;
        new_j  = (blk_id_y<<3) + (lane_id+256)/34;
        new_li = (lane_id+256)%34;
        new_lj = (lane_id+256)/34;
        new_i  = (new_i < n+2*halo)? new_i: n+2*halo-1;
        new_j  = (new_j < m+2*halo)? new_j: m+2*halo-1;
        if(new_li < 34 &&  new_lj < 10)
        {
            LOC_3D(t3,new_lj,new_li) = IN_3D(k+1,new_j,new_i);
        }
        __syncthreads();

        OUT_3D(k,j,i) = a0 * LOC_3D(t1,lj  ,li  ) +
                        a1 * LOC_3D(t2,lj-1,li  ) + 
                        a2 * LOC_3D(t2,lj  ,li-1) + 
                        a3 * LOC_3D(t2,lj  ,li  ) + 
                        a4 * LOC_3D(t2,lj  ,li+1) + 
                        a5 * LOC_3D(t2,lj+1,li  ) + 
                        a6 * LOC_3D(t3,lj  ,li  ) ;
        
        __syncthreads();
    }
}


__global__ void Stencil_Cuda_Lds_3BlkBrc(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int z, int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = threadIdx.z + blockIdx.z * blockDim.z + halo;

    int li = threadIdx.x + 1;
    int lj = threadIdx.y + 1;
    int lk = threadIdx.z + 1;

    __shared__ DATA_TYPE local[8+2][4+2][8+2];
    LOC_3D(lk,lj,li) = IN_3D(k,j,i);

    if(li == halo) LOC_3D(lk,lj,li-1) = IN_3D(k,j,i-1);
    if(li == 8   ) LOC_3D(lk,lj,li+1) = IN_3D(k,j,i+1);
    if(lj == halo) LOC_3D(lk,lj-1,li) = IN_3D(k,j-1,i);
    if(lj == 4   ) LOC_3D(lk,lj+1,li) = IN_3D(k,j+1,i);
    if(lk == halo) LOC_3D(lk-1,lj,li) = IN_3D(k-1,j,i);
    if(lk == 8   ) LOC_3D(lk+1,lj,li) = IN_3D(k+1,j,i);
    
    __syncthreads();

    OUT_3D(k,j,i) = a0 * LOC_3D(lk-1,lj  ,li  ) +
                    a1 * LOC_3D(lk  ,lj-1,li  ) +
                    a2 * LOC_3D(lk  ,lj  ,li-1) +
                    a3 * LOC_3D(lk  ,lj  ,li  ) +
                    a4 * LOC_3D(lk  ,lj  ,li+1) +
                    a5 * LOC_3D(lk  ,lj+1,li  ) +
                    a6 * LOC_3D(lk+1,lj  ,li  ) ;
}

__global__ void Stencil_Cuda_Lds_3BlkCyc(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int z, int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = threadIdx.z + blockIdx.z * blockDim.z + halo;

    int li = threadIdx.x + 1;
    int lj = threadIdx.y + 1;
    int lk = threadIdx.z + 1;

    __shared__ DATA_TYPE local[8+2][4+2][8+2];

    int lane_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;

    int blk_id_x = blockIdx.x;
    int blk_id_y = blockIdx.y;
    int blk_id_z = blockIdx.z;
    int new_i, new_j, new_k, new_li, new_lj, new_lk;
    new_i  = (blk_id_x<<3) + lane_id%10 ;     
    new_j  = (blk_id_y<<2) + (lane_id/10)%6 ; 
    new_k  = (blk_id_z<<3) + lane_id/60 ;     
    new_li = lane_id%10;
    new_lj = (lane_id/10)%6;
    new_lk = lane_id/60;
    LOC_3D(new_lk,new_lj,new_li) = IN_3D(new_k,new_j,new_i);
    new_i  = (blk_id_x<<3) + (lane_id+256)%10 ;
    new_j  = (blk_id_y<<2) + ((lane_id+256)/10)%6 ;
    new_k  = (blk_id_z<<3) + (lane_id+256)/60 ;
    new_li = (lane_id+256)%10;
    new_lj = ((lane_id+256)/10)%6;
    new_lk = (lane_id+256)/60; 
    LOC_3D(new_lk,new_lj,new_li) = IN_3D(new_k,new_j,new_i);
    new_i  = (blk_id_x<<3) + (lane_id+512)%10 ;
    new_j  = (blk_id_y<<2) + ((lane_id+512)/10)%6 ;
    new_k  = (blk_id_z<<3) + (lane_id+512)/60 ;
    new_li = (lane_id+512)%10;
    new_lj = ((lane_id+512)/10)%6;
    new_lk = (lane_id+512)/60; 
    if(new_li < 10 &&  new_lj < 6 && new_lk < 10 )
        LOC_3D(new_lk,new_lj,new_li) = IN_3D(new_k,new_j,new_i);

    __syncthreads();

    OUT_3D(k,j,i) = a0 * LOC_3D(lk-1,lj  ,li  ) +
                    a1 * LOC_3D(lk  ,lj-1,li  ) +
                    a2 * LOC_3D(lk  ,lj  ,li-1) +
                    a3 * LOC_3D(lk  ,lj  ,li  ) +
                    a4 * LOC_3D(lk  ,lj  ,li+1) +
                    a5 * LOC_3D(lk  ,lj+1,li  ) +
                    a6 * LOC_3D(lk+1,lj  ,li  ) ;
}


__global__ void Stencil_Cuda_Reg1_3Blk2Wf(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int z, int m, int n, int halo)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = threadIdx.z + blockIdx.z * blockDim.z + halo;
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    int lane_id = tid % 32;

    int warp_id0 = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id1 = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
    int warp_id2 = (threadIdx.z + blockIdx.z * blockDim.z)>>0; // there numbers
    DATA_TYPE reg0, reg1, reg2, reg3, reg4, reg5;

    int lane_id_it = lane_id;
    // load to regs: 
    int new_id0 ;
    int new_id1 ;
    int new_id2 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg0 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg1 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg2 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg3 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg4 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
    new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
    new_id2 = (new_id2 < z+2)? new_id2 : z+1 ;
    reg5 = IN_3D(new_id2, new_id1, new_id0) ;
    
    
    DATA_TYPE sum0 = 0.0;
    int friend_id0;
    DATA_TYPE tx0, ty0, tz0;

    // process (1, 1, 0)
    friend_id0 = (lane_id+11+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a0 *((lane_id < 17 )? tx0: ty0);
    // process (1, 0, 1)
    friend_id0 = (lane_id+29+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    tz0 = __shfl(reg3, friend_id0);
    sum0 += a1 *((lane_id < 3 )? tx0: ((lane_id < 29)? ty0: tz0));
    // process (0, 1, 1)
    friend_id0 = (lane_id+ 6+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a2 *((lane_id < 22 )? tx0: ty0);
    // process (1, 1, 1)
    friend_id0 = (lane_id+ 7+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a3 *((lane_id < 21 )? tx0: ty0);
    // process (2, 1, 1)
    friend_id0 = (lane_id+ 8+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a4 *((lane_id < 20 )? tx0: ty0);
    // process (1, 2, 1)
    friend_id0 = (lane_id+17+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a5 *((lane_id < 13 )? tx0: ty0);
    // process (1, 1, 2)
    friend_id0 = (lane_id+ 3+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    sum0 += a6 *((lane_id < 24 )? tx0: ty0);
    
    OUT_3D(k,j,i) = sum0;

}

__global__ void Stencil_Cuda_Reg2_3Blk2Wf(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int z, int m, int n, int halo)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<1) + (lane_id>>5) + halo; 
    // thread coarsening: related to warp dimensions 1x4x8. 
    // We coarsen from dimension z from 2^0 to 2^1, also need to know there are how many values in dimension z

    int warp_id0 = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id1 = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
    int warp_id2 = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<1) + (lane_id>>5); // these numbers
    DATA_TYPE reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
    int lane_id_it = lane_id;
    // load to regs: 
    int new_id0 ;
    int new_id1 ;
    int new_id2 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg0 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg1 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg2 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg3 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg4 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg5 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg6 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
    new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
    new_id2 = (new_id2 < z+2)? new_id2 : z+1 ;
    reg7 = IN_3D(new_id2, new_id1, new_id0) ;


    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    int friend_id0, friend_id1;
    DATA_TYPE tx0, ty0, tz0, tx1, ty1, tz1;

    // process (1, 1, 0)
    friend_id0 = (lane_id+11+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a0 *((lane_id < 17 )? tx0: ty0);
    friend_id1 = (lane_id+ 7+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    sum1 += a0 *((lane_id < 21 )? tx1: ty1);
    // process (1, 0, 1)
    friend_id0 = (lane_id+29+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    tz0 = __shfl(reg3, friend_id0);
    sum0 += a1*((lane_id < 3 )? tx0: ((lane_id < 29)? ty0: tz0));
    friend_id1 = (lane_id+25+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg3, friend_id1);
    ty1 = __shfl(reg4, friend_id1);
    sum1 += a1*((lane_id < 7 )? tx1: ty1);
    // process (0, 1, 1)
    friend_id0 = (lane_id+ 6+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a2*((lane_id < 22 )? tx0: ty0);
    friend_id1 = (lane_id+ 2+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    sum1 += a2*((lane_id < 24 )? tx1: ty1);
    // process (1, 1, 1)
    friend_id0 = (lane_id+ 7+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a3*((lane_id < 21 )? tx0: ty0);
    friend_id1 = (lane_id+ 3+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    sum1 += a3*((lane_id < 24 )? tx1: ty1);
    // process (2, 1, 1)
    friend_id0 = (lane_id+ 8+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a4*((lane_id < 20 )? tx0: ty0);
    friend_id1 = (lane_id+ 4+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    sum1 += a4*((lane_id < 24 )? tx1: ty1);
    // process (1, 2, 1)
    friend_id0 = (lane_id+17+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a5*((lane_id < 13 )? tx0: ty0);
    friend_id1 = (lane_id+13+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    sum1 += a5*((lane_id < 16 )? tx1: ty1);
    // process (1, 1, 2)
    friend_id0 = (lane_id+ 3+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    sum0 += a6*((lane_id < 24 )? tx0: ty0);
    friend_id1 = (lane_id+31+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg5, friend_id1);
    ty1 = __shfl(reg6, friend_id1);
    tz1 = __shfl(reg7, friend_id1);
    sum1 += a6*((lane_id < 1 )? tx1: ((lane_id < 27)? ty1: tz1));

    OUT_3D(k  ,j,i) = sum0;
    OUT_3D(k+1,j,i) = sum1;

}

__global__ void Stencil_Cuda_Reg4_3Blk2Wf(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int z, int m, int n, int halo)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<2) + (lane_id>>5) + halo; 
    // Thread coarsening: related to warp dimensions 1x4x8. 
    // We coarsen from dimension z from 2^0 to 2^2, also need to know there are how many values in dimension z,
    // which is (lane_id>>5) 

    int warp_id0 = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id1 = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
    int warp_id2 = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<2) + (lane_id>>5); // these numbers
    DATA_TYPE reg0, reg1, reg2, reg3, reg4, reg5,
              reg6, reg7, reg8, reg9, reg10, reg11;
    int lane_id_it = lane_id;
    // load to regs: 
    int new_id0 ;
    int new_id1 ;
    int new_id2 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg0 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg1 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg2 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg3 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg4 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg5 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg6 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg7 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg8 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg9 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    reg10 = IN_3D(new_id2, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<3) + lane_id_it%10 ;
    new_id1 = (warp_id1<<2) + lane_id_it/10%6 ;
    new_id2 = (warp_id2<<0) + lane_id_it/60 ;
    new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
    new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
    new_id2 = (new_id2 < z+2)? new_id2 : z+1 ;
    reg11 = IN_3D(new_id2, new_id1, new_id0) ;


    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    int friend_id0, friend_id1, friend_id2, friend_id3;
    DATA_TYPE tx0, ty0, tz0, tx1, ty1, tz1, tx2, ty2, tz2, tx3, ty3, tz3;

    // process (1, 1, 0)
    friend_id0 = (lane_id+11+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    sum0 += a0 *((lane_id < 17 )? tx0: ty0);
    friend_id1 = (lane_id+ 7+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    sum1 += a0 *((lane_id < 21 )? tx1: ty1);
    friend_id2 = (lane_id+ 3+((lane_id>>3)*2))&31 ;
    tx2 = __shfl(reg4, friend_id2);
    ty2 = __shfl(reg5, friend_id2);
    sum2 += a0 *((lane_id < 24 )? tx2: ty2);
    friend_id3 = (lane_id+31+((lane_id>>3)*2))&31 ;
    tx3 = __shfl(reg5, friend_id3);
    ty3 = __shfl(reg6, friend_id3);
    tz3 = __shfl(reg7, friend_id3);
    sum3 += a0 *((lane_id < 1 )? tx3: ((lane_id < 27)? ty3: tz3));
    // process (1, 0, 1)
    friend_id0 = (lane_id+29+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    tz0 = __shfl(reg3, friend_id0);
    sum0 += a1*((lane_id < 3 )? tx0: ((lane_id < 29)? ty0: tz0));
    friend_id1 = (lane_id+25+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg3, friend_id1);
    ty1 = __shfl(reg4, friend_id1);
    sum1 += a1*((lane_id < 7 )? tx1: ty1);
    friend_id2 = (lane_id+21+((lane_id>>3)*2))&31 ;
    tx2 = __shfl(reg5, friend_id2);
    ty2 = __shfl(reg6, friend_id2);
    sum2 += a1*((lane_id < 9 )? tx2: ty2);
    friend_id3 = (lane_id+17+((lane_id>>3)*2))&31 ;
    tx3 = __shfl(reg7, friend_id3);
    ty3 = __shfl(reg8, friend_id3);
    sum3 += a1*((lane_id < 13 )? tx3: ty3);
    // process (0, 1, 1)
    friend_id0 = (lane_id+ 6+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a2*((lane_id < 22 )? tx0: ty0);
    friend_id1 = (lane_id+ 2+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    sum1 += a2*((lane_id < 24 )? tx1: ty1);
    friend_id2 = (lane_id+30+((lane_id>>3)*2))&31 ;
    tx2 = __shfl(reg5, friend_id2);
    ty2 = __shfl(reg6, friend_id2);
    tz2 = __shfl(reg7, friend_id2);
    sum2 += a2*((lane_id < 2 )? tx2: ((lane_id < 28)? ty2: tz2));
    friend_id3 = (lane_id+26+((lane_id>>3)*2))&31 ;
    tx3 = __shfl(reg7, friend_id3);
    ty3 = __shfl(reg8, friend_id3);
    sum3 += a2*((lane_id < 6 )? tx3: ty3);
    // process (1, 1, 1)
    friend_id0 = (lane_id+ 7+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a3*((lane_id < 21 )? tx0: ty0);
    friend_id1 = (lane_id+ 3+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    sum1 += a3*((lane_id < 24 )? tx1: ty1);
    friend_id2 = (lane_id+31+((lane_id>>3)*2))&31 ;
    tx2 = __shfl(reg5, friend_id2);
    ty2 = __shfl(reg6, friend_id2);
    tz2 = __shfl(reg7, friend_id2);
    sum2 += a3*((lane_id < 1 )? tx2: ((lane_id < 27)? ty2: tz2));
    friend_id3 = (lane_id+27+((lane_id>>3)*2))&31 ;
    tx3 = __shfl(reg7, friend_id3);
    ty3 = __shfl(reg8, friend_id3);
    tz3 = __shfl(reg9, friend_id3);
    sum3 += a3*((lane_id < 5 )? tx3: ((lane_id < 31)? ty3: tz3));
    // process (2, 1, 1)
    friend_id0 = (lane_id+ 8+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a4*((lane_id < 20 )? tx0: ty0);
    friend_id1 = (lane_id+ 4+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    sum1 += a4*((lane_id < 24 )? tx1: ty1);
    friend_id2 = (lane_id+ 0+((lane_id>>3)*2))&31 ;
    tx2 = __shfl(reg6, friend_id2);
    ty2 = __shfl(reg7, friend_id2);
    sum2 += a4*((lane_id < 26 )? tx2: ty2);
    friend_id3 = (lane_id+28+((lane_id>>3)*2))&31 ;
    tx3 = __shfl(reg7, friend_id3);
    ty3 = __shfl(reg8, friend_id3);
    tz3 = __shfl(reg9, friend_id3);
    sum3 += a4*((lane_id < 4 )? tx3: ((lane_id < 30)? ty3: tz3));
    // process (1, 2, 1)
    friend_id0 = (lane_id+17+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    sum0 += a5*((lane_id < 13 )? tx0: ty0);
    friend_id1 = (lane_id+13+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    sum1 += a5*((lane_id < 16 )? tx1: ty1);
    friend_id2 = (lane_id+ 9+((lane_id>>3)*2))&31 ;
    tx2 = __shfl(reg6, friend_id2);
    ty2 = __shfl(reg7, friend_id2);
    sum2 += a5*((lane_id < 19 )? tx2: ty2);
    friend_id3 = (lane_id+ 5+((lane_id>>3)*2))&31 ;
    tx3 = __shfl(reg8, friend_id3);
    ty3 = __shfl(reg9, friend_id3);
    sum3 += a5*((lane_id < 23 )? tx3: ty3);
    // process (1, 1, 2)
    friend_id0 = (lane_id+ 3+((lane_id>>3)*2))&31 ;
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    sum0 += a6*((lane_id < 24 )? tx0: ty0);
    friend_id1 = (lane_id+31+((lane_id>>3)*2))&31 ;
    tx1 = __shfl(reg5, friend_id1);
    ty1 = __shfl(reg6, friend_id1);
    tz1 = __shfl(reg7, friend_id1);
    sum1 += a6*((lane_id < 1 )? tx1: ((lane_id < 27)? ty1: tz1));
    friend_id2 = (lane_id+27+((lane_id>>3)*2))&31 ;
    tx2 = __shfl(reg7, friend_id2);
    ty2 = __shfl(reg8, friend_id2);
    tz2 = __shfl(reg9, friend_id2);
    sum2 += a6*((lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2));
    friend_id3 = (lane_id+23+((lane_id>>3)*2))&31 ;
    tx3 = __shfl(reg9, friend_id3);
    ty3 = __shfl(reg10, friend_id3);
    sum3 += a6*((lane_id < 8 )? tx3: ty3);

    OUT_3D(k  ,j,i) = sum0;
    OUT_3D(k+1,j,i) = sum1;
    OUT_3D(k+2,j,i) = sum2;
    OUT_3D(k+3,j,i) = sum3;
}

__global__ void Stencil_Cuda_Reg1_25Blk1Wf(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6,  
        int z, int m, int n, int halo)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<0) + (lane_id>>5) + halo;

    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;

    int warp_id0 = (threadIdx.x + blockIdx.x * blockDim.x)>>5; // because the warp dimensions are 
    int warp_id1 = ((((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<0) + (lane_id>>5))>>0; // 1x1x32, warp_ids are division of 

    DATA_TYPE tx0, ty0, tz0;
    int friend_id0;
    int new_i, new_j;
    DATA_TYPE t3_reg0, t3_reg1, t3_reg2, t3_reg3;
    DATA_TYPE t2_reg0, t2_reg1, t2_reg2, t2_reg3;
    DATA_TYPE t1_reg0, t1_reg1, t1_reg2, t1_reg3;

    DATA_TYPE sum0 = 0.0;
    int lane_id_it = lane_id;
    // load to regs: 
    int new_id0 ;
    int new_id1 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    t3_reg0 = IN_3D(k  , new_id1, new_id0) ;
    t2_reg0 = IN_3D(k-1, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    t3_reg1 = IN_3D(k  , new_id1, new_id0) ;
    t2_reg1 = IN_3D(k-1, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    t3_reg2 = IN_3D(k  , new_id1, new_id0) ;
    t2_reg2 = IN_3D(k-1, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
    new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
    t3_reg3 = IN_3D(k  , new_id1, new_id0) ;
    t2_reg3 = IN_3D(k-1, new_id1, new_id0) ;
        
#pragma unroll // it seems the loop-unroll is useless to performance
    for(; k < k_end; ++k)
    {
        sum0 = 0.0;
        // move the current storage down 
        t1_reg0 = t2_reg0;
        t1_reg1 = t2_reg1;
        t1_reg2 = t2_reg2;
        t1_reg3 = t2_reg3;

        t2_reg0 = t3_reg0;
        t2_reg1 = t3_reg1;
        t2_reg2 = t3_reg2;
        t2_reg3 = t3_reg3;

        lane_id_it = lane_id;
        new_id0 = (warp_id0<<5) + lane_id_it%34 ;
        new_id1 = (warp_id1<<0) + lane_id_it/34 ;
        t3_reg0 = IN_3D(k+1, new_id1, new_id0) ;
        lane_id_it += 32 ;
        new_id0 = (warp_id0<<5) + lane_id_it%34 ;
        new_id1 = (warp_id1<<0) + lane_id_it/34 ;
        t3_reg1 = IN_3D(k+1, new_id1, new_id0) ;
        lane_id_it += 32 ;
        new_id0 = (warp_id0<<5) + lane_id_it%34 ;
        new_id1 = (warp_id1<<0) + lane_id_it/34 ;
        t3_reg2 = IN_3D(k+1, new_id1, new_id0) ;
        lane_id_it += 32 ;
        new_id0 = (warp_id0<<5) + lane_id_it%34 ;
        new_id1 = (warp_id1<<0) + lane_id_it/34 ;
        new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
        new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
        t3_reg3 = IN_3D(k+1, new_id1, new_id0) ;

        // process (1, 0, 0)
        friend_id0 = (lane_id+ 1)&31 ;
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        sum0 += a1*((lane_id < 31 )? tx0: ty0);
        // process (0, 1, 0)
        friend_id0 = (lane_id+ 2)&31 ;
        tx0 = __shfl(t2_reg1, friend_id0);
        ty0 = __shfl(t2_reg2, friend_id0);
        sum0 += a2*((lane_id < 30 )? tx0: ty0);
        // process (1, 1, 0)
        friend_id0 = (lane_id+ 3)&31 ;
        tx0 = __shfl(t1_reg1, friend_id0);
        ty0 = __shfl(t1_reg2, friend_id0);
        sum0 += a0 *((lane_id < 29 )? tx0: ty0);
        tx0 = __shfl(t2_reg1, friend_id0);
        ty0 = __shfl(t2_reg2, friend_id0);
        sum0 += a3*((lane_id < 29 )? tx0: ty0);
        tx0 = __shfl(t3_reg1, friend_id0);
        ty0 = __shfl(t3_reg2, friend_id0);
        sum0 += a6*((lane_id < 29 )? tx0: ty0);
        // process (2, 1, 0)
        friend_id0 = (lane_id+ 4)&31 ;
        tx0 = __shfl(t2_reg1, friend_id0);
        ty0 = __shfl(t2_reg2, friend_id0);
        sum0 += a4*((lane_id < 28 )? tx0: ty0);
        // process (1, 2, 0)
        friend_id0 = (lane_id+ 5)&31 ;
        tx0 = __shfl(t2_reg2, friend_id0);
        ty0 = __shfl(t2_reg3, friend_id0);
        sum0 += a5*((lane_id < 27 )? tx0: ty0);

        OUT_3D(k,j  ,i) = sum0;
    }
}

__global__ void Stencil_Cuda_Reg2_25Blk1Wf(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6,  
        int z, int m, int n, int halo)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<1) + (lane_id>>5) + halo;

    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;

    int warp_id0 = (threadIdx.x + blockIdx.x * blockDim.x)>>5; // because the warp dimensions are 
    int warp_id1 = ((((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<1) + (lane_id>>5))>>0; // 1x1x32, warp_ids are division of 
    DATA_TYPE tx0, ty0, tz0;
    DATA_TYPE tx1, ty1, tz1;

    int friend_id0, friend_id1;
    int new_i, new_j;
    DATA_TYPE t3_reg0, t3_reg1, t3_reg2, t3_reg3, t3_reg4;
    DATA_TYPE t2_reg0, t2_reg1, t2_reg2, t2_reg3, t2_reg4;
    DATA_TYPE t1_reg0, t1_reg1, t1_reg2, t1_reg3, t1_reg4;
    int lane_id_it = lane_id;

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    // load to regs: 
    int new_id0 ;
    int new_id1 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    t3_reg0 = IN_3D(k  , new_id1, new_id0) ;
    t2_reg0 = IN_3D(k-1, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    t3_reg1 = IN_3D(k  , new_id1, new_id0) ;
    t2_reg1 = IN_3D(k-1, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    t3_reg2 = IN_3D(k  , new_id1, new_id0) ;
    t2_reg2 = IN_3D(k-1, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    t3_reg3 = IN_3D(k  , new_id1, new_id0) ;
    t2_reg3 = IN_3D(k-1, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
    new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
    t3_reg4 = IN_3D(k  , new_id1, new_id0) ;
    t2_reg4 = IN_3D(k-1, new_id1, new_id0) ;


#pragma unroll // it seems the loop-unroll is useless to performance
    for(; k < k_end; ++k)
    {
        sum0 = 0.0;
        sum1 = 0.0;
        // move the current storage down 
        t1_reg0 = t2_reg0;
        t1_reg1 = t2_reg1;
        t1_reg2 = t2_reg2;
        t1_reg3 = t2_reg3;
        t1_reg4 = t2_reg4;

        t2_reg0 = t3_reg0;
        t2_reg1 = t3_reg1;
        t2_reg2 = t3_reg2;
        t2_reg3 = t3_reg3;
        t2_reg4 = t3_reg4;

        lane_id_it = lane_id;
        new_id0 = (warp_id0<<5) + lane_id_it%34 ;
        new_id1 = (warp_id1<<0) + lane_id_it/34 ;
        t3_reg0 = IN_3D(k+1, new_id1, new_id0) ;
        lane_id_it += 32 ;
        new_id0 = (warp_id0<<5) + lane_id_it%34 ;
        new_id1 = (warp_id1<<0) + lane_id_it/34 ;
        t3_reg1 = IN_3D(k+1, new_id1, new_id0) ;
        lane_id_it += 32 ;
        new_id0 = (warp_id0<<5) + lane_id_it%34 ;
        new_id1 = (warp_id1<<0) + lane_id_it/34 ;
        t3_reg2 = IN_3D(k+1, new_id1, new_id0) ;
        lane_id_it += 32 ;
        new_id0 = (warp_id0<<5) + lane_id_it%34 ;
        new_id1 = (warp_id1<<0) + lane_id_it/34 ;
        t3_reg3 = IN_3D(k+1, new_id1, new_id0) ;
        lane_id_it += 32 ;
        new_id0 = (warp_id0<<5) + lane_id_it%34 ;
        new_id1 = (warp_id1<<0) + lane_id_it/34 ;
        new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
        new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
        t3_reg4 = IN_3D(k+1, new_id1, new_id0) ;

        // process (1, 0, 0)
        friend_id0 = (lane_id+ 1)&31 ;
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        sum0 += a1*((lane_id < 31 )? tx0: ty0);
        friend_id1 = (lane_id+ 3)&31 ;
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        sum1 += a1*((lane_id < 29 )? tx1: ty1);
        // process (0, 1, 0)
        friend_id0 = (lane_id+ 2)&31 ;
        tx0 = __shfl(t2_reg1, friend_id0);
        ty0 = __shfl(t2_reg2, friend_id0);
        sum0 += a2*((lane_id < 30 )? tx0: ty0);
        friend_id1 = (lane_id+ 4)&31 ;
        tx1 = __shfl(t2_reg2, friend_id1);
        ty1 = __shfl(t2_reg3, friend_id1);
        sum1 += a1*((lane_id < 28 )? tx1: ty1);
        // process (1, 1, 0)
        friend_id0 = (lane_id+ 3)&31 ;
        tx0 = __shfl(t1_reg1, friend_id0);
        ty0 = __shfl(t1_reg2, friend_id0);
        sum0 += a0*((lane_id < 29 )? tx0: ty0);
        tx0 = __shfl(t2_reg1, friend_id0);
        ty0 = __shfl(t2_reg2, friend_id0);
        sum0 += a3*((lane_id < 29 )? tx0: ty0);
        tx0 = __shfl(t3_reg1, friend_id0);
        ty0 = __shfl(t3_reg2, friend_id0);
        sum0 += a6*((lane_id < 29 )? tx0: ty0);
        friend_id1 = (lane_id+ 5)&31 ;
        tx1 = __shfl(t1_reg2, friend_id1);
        ty1 = __shfl(t1_reg3, friend_id1);
        sum1 += a0*((lane_id < 27 )? tx1: ty1);
        tx1 = __shfl(t2_reg2, friend_id1);
        ty1 = __shfl(t2_reg3, friend_id1);
        sum1 += a3*((lane_id < 27 )? tx1: ty1);
        tx1 = __shfl(t3_reg2, friend_id1);
        ty1 = __shfl(t3_reg3, friend_id1);
        sum1 += a6*((lane_id < 27 )? tx1: ty1);
        // process (2, 1, 0)
        friend_id0 = (lane_id+ 4)&31 ;
        tx0 = __shfl(t2_reg1, friend_id0);
        ty0 = __shfl(t2_reg2, friend_id0);
        sum0 += a4*((lane_id < 28 )? tx0: ty0);
        friend_id1 = (lane_id+ 6)&31 ;
        tx1 = __shfl(t2_reg2, friend_id1);
        ty1 = __shfl(t2_reg3, friend_id1);
        sum1 += a4*((lane_id < 26 )? tx1: ty1);
        // process (1, 2, 0)
        friend_id0 = (lane_id+ 5)&31 ;
        tx0 = __shfl(t2_reg2, friend_id0);
        ty0 = __shfl(t2_reg3, friend_id0);
        sum0 += a5*((lane_id < 27 )? tx0: ty0);
        friend_id1 = (lane_id+ 7)&31 ;
        tx1 = __shfl(t2_reg3, friend_id1);
        ty1 = __shfl(t2_reg4, friend_id1);
        sum1 += a5*((lane_id < 25 )? tx1: ty1);
        
        OUT_3D(k,j  ,i) = sum0;
        OUT_3D(k,j+1,i) = sum1;
    }
}

__global__ void Stencil_Cuda_Reg4_25Blk1Wf(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6,  
        int z, int m, int n, int halo)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<2) + (lane_id>>5) + halo;

    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;

    int warp_id0 = (threadIdx.x + blockIdx.x * blockDim.x)>>5; // because the warp dimensions are 
    int warp_id1 = ((((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<2) + (lane_id>>5))>>0; // 1x1x32, warp_ids are division of 
    DATA_TYPE tx0, ty0, tz0;
    DATA_TYPE tx1, ty1, tz1;
    DATA_TYPE tx2, ty2, tz2;
    DATA_TYPE tx3, ty3, tz3;
    int friend_id0, friend_id1;
    int friend_id2, friend_id3;
    DATA_TYPE t3_reg0, t3_reg1, t3_reg2, t3_reg3, t3_reg4, t3_reg5, t3_reg6;
    DATA_TYPE t2_reg0, t2_reg1, t2_reg2, t2_reg3, t2_reg4, t2_reg5, t2_reg6;
    DATA_TYPE t1_reg0, t1_reg1, t1_reg2, t1_reg3, t1_reg4, t1_reg5, t1_reg6;
    int lane_id_it = lane_id;

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    // load to regs: 
    int new_id0 ;
    int new_id1 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    t3_reg0 = IN_3D(k  , new_id1, new_id0) ;
    t2_reg0 = IN_3D(k-1, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    t3_reg1 = IN_3D(k  , new_id1, new_id0) ;
    t2_reg1 = IN_3D(k-1, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    t3_reg2 = IN_3D(k  , new_id1, new_id0) ;
    t2_reg2 = IN_3D(k-1, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    t3_reg3 = IN_3D(k  , new_id1, new_id0) ;
    t2_reg3 = IN_3D(k-1, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    t3_reg4 = IN_3D(k  , new_id1, new_id0) ;
    t2_reg4 = IN_3D(k-1, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    t3_reg5 = IN_3D(k  , new_id1, new_id0) ;
    t2_reg5 = IN_3D(k-1, new_id1, new_id0) ;
    lane_id_it += 32 ;
    new_id0 = (warp_id0<<5) + lane_id_it%34 ;
    new_id1 = (warp_id1<<0) + lane_id_it/34 ;
    new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
    new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
    t3_reg6 = IN_3D(k  , new_id1, new_id0) ;
    t2_reg6 = IN_3D(k-1, new_id1, new_id0) ;

#pragma unroll // it seems the loop-unroll is useless to performance
    for(; k < k_end; ++k)
    {
        sum0 = 0.0;
        sum1 = 0.0;
        sum2 = 0.0;
        sum3 = 0.0;
        // move the current storage down 
        t1_reg0 = t2_reg0;
        t1_reg1 = t2_reg1;
        t1_reg2 = t2_reg2;
        t1_reg3 = t2_reg3;
        t1_reg4 = t2_reg4;
        t1_reg5 = t2_reg5;
        t1_reg6 = t2_reg6;

        t2_reg0 = t3_reg0;
        t2_reg1 = t3_reg1;
        t2_reg2 = t3_reg2;
        t2_reg3 = t3_reg3;
        t2_reg4 = t3_reg4;
        t2_reg5 = t3_reg5;
        t2_reg6 = t3_reg6;

        lane_id_it = lane_id;
        new_id0 = (warp_id0<<5) + lane_id_it%34 ;
        new_id1 = (warp_id1<<0) + lane_id_it/34 ;
        t3_reg0 = IN_3D(k+1, new_id1, new_id0) ;
        lane_id_it += 32 ;
        new_id0 = (warp_id0<<5) + lane_id_it%34 ;
        new_id1 = (warp_id1<<0) + lane_id_it/34 ;
        t3_reg1 = IN_3D(k+1, new_id1, new_id0) ;
        lane_id_it += 32 ;
        new_id0 = (warp_id0<<5) + lane_id_it%34 ;
        new_id1 = (warp_id1<<0) + lane_id_it/34 ;
        t3_reg2 = IN_3D(k+1, new_id1, new_id0) ;
        lane_id_it += 32 ;
        new_id0 = (warp_id0<<5) + lane_id_it%34 ;
        new_id1 = (warp_id1<<0) + lane_id_it/34 ;
        t3_reg3 = IN_3D(k+1, new_id1, new_id0) ;
        lane_id_it += 32 ;
        new_id0 = (warp_id0<<5) + lane_id_it%34 ;
        new_id1 = (warp_id1<<0) + lane_id_it/34 ;
        t3_reg4 = IN_3D(k+1, new_id1, new_id0) ;
        lane_id_it += 32 ;
        new_id0 = (warp_id0<<5) + lane_id_it%34 ;
        new_id1 = (warp_id1<<0) + lane_id_it/34 ;
        t3_reg5 = IN_3D(k+1, new_id1, new_id0) ;
        lane_id_it += 32 ;
        new_id0 = (warp_id0<<5) + lane_id_it%34 ;
        new_id1 = (warp_id1<<0) + lane_id_it/34 ;
        new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
        new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
        t3_reg6 = IN_3D(k+1, new_id1, new_id0) ;

        // process (1, 0, 0)
        friend_id0 = (lane_id+ 1)&31 ;
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        sum0 += a1*((lane_id < 31 )? tx0: ty0);
        friend_id1 = (lane_id+ 3)&31 ;
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        sum1 += a1*((lane_id < 29 )? tx1: ty1);
        friend_id2 = (lane_id+ 5)&31 ;
        tx2 = __shfl(t2_reg2, friend_id2);
        ty2 = __shfl(t2_reg3, friend_id2);
        sum2 += a1*((lane_id < 27 )? tx2: ty2);
        friend_id3 = (lane_id+ 7)&31 ;
        tx3 = __shfl(t2_reg3, friend_id3);
        ty3 = __shfl(t2_reg4, friend_id3);
        sum3 += a1*((lane_id < 25 )? tx3: ty3);
        // process (0, 1, 0)
        friend_id0 = (lane_id+ 2)&31 ;
        tx0 = __shfl(t2_reg1, friend_id0);
        ty0 = __shfl(t2_reg2, friend_id0);
        sum0 += a2*((lane_id < 30 )? tx0: ty0);
        friend_id1 = (lane_id+ 4)&31 ;
        tx1 = __shfl(t2_reg2, friend_id1);
        ty1 = __shfl(t2_reg3, friend_id1);
        sum1 += a2*((lane_id < 28 )? tx1: ty1);
        friend_id2 = (lane_id+ 6)&31 ;
        tx2 = __shfl(t2_reg3, friend_id2);
        ty2 = __shfl(t2_reg4, friend_id2);
        sum2 += a2*((lane_id < 26 )? tx2: ty2);
        friend_id3 = (lane_id+ 8)&31 ;
        tx3 = __shfl(t2_reg4, friend_id3);
        ty3 = __shfl(t2_reg5, friend_id3);
        sum3 += a2*((lane_id < 24 )? tx3: ty3);
        // process (1, 1, 0)
        friend_id0 = (lane_id+ 3)&31 ;
        tx0 = __shfl(t1_reg1, friend_id0);
        ty0 = __shfl(t1_reg2, friend_id0);
        sum0 += a0*((lane_id < 29 )? tx0: ty0);
        tx0 = __shfl(t2_reg1, friend_id0);
        ty0 = __shfl(t2_reg2, friend_id0);
        sum0 += a3*((lane_id < 29 )? tx0: ty0);
        tx0 = __shfl(t3_reg1, friend_id0);
        ty0 = __shfl(t3_reg2, friend_id0);
        sum0 += a6*((lane_id < 29 )? tx0: ty0);
        friend_id1 = (lane_id+ 5)&31 ;
        tx1 = __shfl(t1_reg2, friend_id1);
        ty1 = __shfl(t1_reg3, friend_id1);
        sum1 += a0*((lane_id < 27 )? tx1: ty1);
        tx1 = __shfl(t2_reg2, friend_id1);
        ty1 = __shfl(t2_reg3, friend_id1);
        sum1 += a3*((lane_id < 27 )? tx1: ty1);
        tx1 = __shfl(t3_reg2, friend_id1);
        ty1 = __shfl(t3_reg3, friend_id1);
        sum1 += a6*((lane_id < 27 )? tx1: ty1);
        friend_id2 = (lane_id+ 7)&31 ;
        tx2 = __shfl(t1_reg3, friend_id2);
        ty2 = __shfl(t1_reg4, friend_id2);
        sum2 += a0*((lane_id < 25 )? tx2: ty2);
        tx2 = __shfl(t2_reg3, friend_id2);
        ty2 = __shfl(t2_reg4, friend_id2);
        sum2 += a3*((lane_id < 25 )? tx2: ty2);
        tx2 = __shfl(t3_reg3, friend_id2);
        ty2 = __shfl(t3_reg4, friend_id2);
        sum2 += a6*((lane_id < 25 )? tx2: ty2);
        friend_id3 = (lane_id+ 9)&31 ;
        tx3 = __shfl(t1_reg4, friend_id3);
        ty3 = __shfl(t1_reg5, friend_id3);
        sum3 += a0*((lane_id < 23 )? tx3: ty3);
        tx3 = __shfl(t2_reg4, friend_id3);
        ty3 = __shfl(t2_reg5, friend_id3);
        sum3 += a3*((lane_id < 23 )? tx3: ty3);
        tx3 = __shfl(t3_reg4, friend_id3);
        ty3 = __shfl(t3_reg5, friend_id3);
        sum3 += a6*((lane_id < 23 )? tx3: ty3);
        // process (2, 1, 0)
        friend_id0 = (lane_id+ 4)&31 ;
        tx0 = __shfl(t2_reg1, friend_id0);
        ty0 = __shfl(t2_reg2, friend_id0);
        sum0 += a4*((lane_id < 28 )? tx0: ty0);
        friend_id1 = (lane_id+ 6)&31 ;
        tx1 = __shfl(t2_reg2, friend_id1);
        ty1 = __shfl(t2_reg3, friend_id1);
        sum1 += a4*((lane_id < 26 )? tx1: ty1);
        friend_id2 = (lane_id+ 8)&31 ;
        tx2 = __shfl(t2_reg3, friend_id2);
        ty2 = __shfl(t2_reg4, friend_id2);
        sum2 += a4*((lane_id < 24 )? tx2: ty2);
        friend_id3 = (lane_id+10)&31 ;
        tx3 = __shfl(t2_reg4, friend_id3);
        ty3 = __shfl(t2_reg5, friend_id3);
        sum3 += a4*((lane_id < 22 )? tx3: ty3);
        // process (1, 2, 0)
        friend_id0 = (lane_id+ 5)&31 ;
        tx0 = __shfl(t2_reg2, friend_id0);
        ty0 = __shfl(t2_reg3, friend_id0);
        sum0 += a5*((lane_id < 27 )? tx0: ty0);
        friend_id1 = (lane_id+ 7)&31 ;
        tx1 = __shfl(t2_reg3, friend_id1);
        ty1 = __shfl(t2_reg4, friend_id1);
        sum1 += a5*((lane_id < 25 )? tx1: ty1);
        friend_id2 = (lane_id+ 9)&31 ;
        tx2 = __shfl(t2_reg4, friend_id2);
        ty2 = __shfl(t2_reg5, friend_id2);
        sum2 += a5*((lane_id < 23 )? tx2: ty2);
        friend_id3 = (lane_id+11)&31 ;
        tx3 = __shfl(t2_reg5, friend_id3);
        ty3 = __shfl(t2_reg6, friend_id3);
        sum3 += a5*((lane_id < 21 )? tx3: ty3);

        OUT_3D(k,j  ,i) = sum0;
        OUT_3D(k,j+1,i) = sum1;
        OUT_3D(k,j+2,i) = sum2;
        OUT_3D(k,j+3,i) = sum3;
    }
}

int main(int argc, char **argv)
{
#ifdef __DEBUG
    int z = 64;
    int m = 8;
    int n = 8;
#else
    int z = 256; 
    int m = 256;
    int n = 256; 
#endif
    int halo = 1;
    int total = (z+2*halo)*(m+2*halo)*(n+2*halo);
    const int K = 7;
#ifdef __DEBUG
    DATA_TYPE args[K] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
#else
    DATA_TYPE args[K] = {0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14};
#endif
    DATA_TYPE *in = new DATA_TYPE[total];
    DATA_TYPE *out_ref = new DATA_TYPE[total];
    unsigned int seed = time(NULL);
    Clear_Output_3D(in, z, m, n, halo);
    Clear_Output_3D(out_ref, z, m, n, halo);
    Init_Input_3D(in, z, m, n, halo, seed);

    // Show_Me(in, z, m, n, halo, "Input:");
    for(int i = 0; i < ITER; i++)
    {
        Stencil_Seq(in, out_ref, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo);
        swap(in, out_ref);
    }
    swap(in, out_ref);
    // Show_Me(out_ref, z, m, n, halo, "Output:");

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
    Init_Input_3D(in, z, m, n, halo, seed);
    Clear_Output_3D(out, z, m, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/8; 
    dimGrid.y = (m)/4; 
    dimGrid.z = (z)/8;
    dimBlock.x = 8; 
    dimBlock.y = 4; 
    dimBlock.z = 8;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_L1_3Blk<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo); 
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_L1_3Blk: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_L1_3Blk Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Sweep version
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/32; 
    dimGrid.y = (m)/8; 
    dimGrid.z = 4;
    dimBlock.x = 32; 
    dimBlock.y = 8; 
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_L1_25Blk<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_L1_25Blk: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_L1_25Blk Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda 3D-Block with SM_Branch
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/8; 
    dimGrid.y = (m)/4; 
    dimGrid.z = (z)/8;
    dimBlock.x = 8; 
    dimBlock.y = 4; 
    dimBlock.z = 8;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Lds_3BlkBrc<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Lds_3BlkBrc: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Lds_3BlkBrc Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda 3D-Block with SM_Cyclic
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/8; 
    dimGrid.y = (m)/4; 
    dimGrid.z = (z)/8;
    dimBlock.x = 8; 
    dimBlock.y = 4; 
    dimBlock.z = 8;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Lds_3BlkCyc<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Lds_3BlkCyc: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Lds_3BlkCyc Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda 2.5D-Block (Sweep) with SM_Branch
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/32; 
    dimGrid.y = (m)/8; 
    dimGrid.z = 4;
    dimBlock.x = 32; 
    dimBlock.y = 8; 
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Lds_25BlkBrc<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Lds_25BlkBrc: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Lds_25BlkBrc Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda 2.5D-Block (Sweep) with SM_Cyclic
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/32; 
    dimGrid.y = (m)/8; 
    dimGrid.z = 4;
    dimBlock.x = 32; 
    dimBlock.y = 8; 
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Lds_25BlkCyc<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Lds_25BlkCyc: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Lds_25BlkCyc Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda 3D-Block with Shfl 1-Point (2D-Warp)
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/8; 
    dimGrid.y = (m)/4; 
    dimGrid.z = (z)/8;
    dimBlock.x = 8; 
    dimBlock.y = 4; 
    dimBlock.z = 8;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Reg1_3Blk2Wf<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Reg1_3Blk2Wf: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Reg1_3Blk2Wf Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda 3D-Block with Shfl 2-Point (2D-Warp)
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/8; 
    dimGrid.y = (m)/4; 
    dimGrid.z = (z)/(8*2);
    dimBlock.x = 8; 
    dimBlock.y = 4; 
    dimBlock.z = 8;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Reg2_3Blk2Wf<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Reg2_3Blk2Wf: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Reg2_3Blk2Wf Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda 3D-Block with Shfl 4-Point (2D-Warp)
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/8; 
    dimGrid.y = (m)/4; 
    dimGrid.z = (z)/(8*4);
    dimBlock.x = 8; 
    dimBlock.y = 4; 
    dimBlock.z = 8;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Reg4_3Blk2Wf<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Reg4_3Blk2Wf: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Reg4_3Blk2Wf Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda 2.5D-Block with Shfl 1-Point (1D-Warp)
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/32; 
    dimGrid.y = (m)/8; 
    dimGrid.z = 4;
    dimBlock.x = 32; 
    dimBlock.y = 8; 
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Reg1_25Blk1Wf<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Reg1_25Blk1Wf: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Reg1_25Blk1Wf Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda 2.5D-Block with Shfl 2-Point (1D-Warp)
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/32; 
    dimGrid.y = (m)/(8*2); 
    dimGrid.z = 4;
    dimBlock.x = 32; 
    dimBlock.y = 8; 
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Reg2_25Blk1Wf<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Reg2_25Blk1Wf: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Reg2_25Blk1Wf Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda 2.5D-Block with Shfl 4-Point (1D-Warp)
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/32; 
    dimGrid.y = (m)/(8*4); 
    dimGrid.z = 4;
    dimBlock.x = 32; 
    dimBlock.y = 8; 
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Reg4_25Blk1Wf<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Reg4_25Blk1Wf: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Reg4_25Blk1Wf Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));


    cudaFree(in_d);
    cudaFree(out_d);

    delete[] in;
    delete[] out;
    delete[] out_ref;

}
