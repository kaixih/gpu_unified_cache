#include <iostream>
#include <metrics.h>
using namespace std;
#define  IN_3D(_z,_y,_x)  in[(_z)*(m+2*halo)*(n+2*halo)+(_y)*(n+2*halo)+(_x)]
#define OUT_3D(_z,_y,_x) out[(_z)*(m+2*halo)*(n+2*halo)+(_y)*(n+2*halo)+(_x)]

#define LOC_3D(_z,_y,_x) local[(_z)][(_y)][(_x)]

// #define DATA_TYPE float
// #define DATA_TYPE double
#define warpSize 32 

// #define __DEBUG

#ifdef __DEBUG
#define ITER 1
#else
#define ITER 100
#endif

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
                IN_3D(k,j,i) = (DATA_TYPE)rand()*10.0 / ((long)RAND_MAX);
#endif
}

/*
void Init_Args_3D(DATA_TYPE *args, int l, int z, int m, int n, int halo, DATA_TYPE val)
{
    for(int k = 0; k < l; k++)
    {
        for(int w = 0; w < z+2*halo; w++)
        {
            for(int i = 0; i < m+2*halo; i++)
            {
                for(int j = 0; j < n+2*halo; j++)
                {
                    ARG_3D(k,w,i,j) = val; 
                }
            }
        }
    }
}
*/

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

void Stencil_Seq(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, 
        DATA_TYPE a8, DATA_TYPE a9, DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, 
        DATA_TYPE a13, DATA_TYPE a14, DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, 
        DATA_TYPE a18, DATA_TYPE a19, DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, 
        DATA_TYPE a23, DATA_TYPE a24, DATA_TYPE a25, DATA_TYPE a26, 
        int z, int m, int n, int halo)
{

#pragma omp parallel for
    for(int k = halo; k < z+halo; k++)
    {
        for(int j = halo; j < m+halo; j++)
        {
            for(int i = halo; i < n+halo; i++)
            {
                OUT_3D(k,j,i) = a0  * IN_3D(k-1,j-1,i-1) +
                                a1  * IN_3D(k-1,j-1,i  ) +
                                a2  * IN_3D(k-1,j-1,i+1) +
                                a3  * IN_3D(k-1,j  ,i-1) +
                                a4  * IN_3D(k-1,j  ,i  ) +
                                a5  * IN_3D(k-1,j  ,i+1) +
                                a6  * IN_3D(k-1,j+1,i-1) + 
                                a7  * IN_3D(k-1,j+1,i  ) + 
                                a8  * IN_3D(k-1,j+1,i+1) + 
                                a9  * IN_3D(k  ,j-1,i-1) + 
                                a10 * IN_3D(k  ,j-1,i  ) + 
                                a11 * IN_3D(k  ,j-1,i+1) + 
                                a12 * IN_3D(k  ,j  ,i-1) + 
                                a13 * IN_3D(k  ,j  ,i  ) + 
                                a14 * IN_3D(k  ,j  ,i+1) + 
                                a15 * IN_3D(k  ,j+1,i-1) + 
                                a16 * IN_3D(k  ,j+1,i  ) + 
                                a17 * IN_3D(k  ,j+1,i+1) + 
                                a18 * IN_3D(k+1,j-1,i-1) + 
                                a19 * IN_3D(k+1,j-1,i  ) + 
                                a20 * IN_3D(k+1,j-1,i+1) + 
                                a21 * IN_3D(k+1,j  ,i-1) + 
                                a22 * IN_3D(k+1,j  ,i  ) + 
                                a23 * IN_3D(k+1,j  ,i+1) + 
                                a24 * IN_3D(k+1,j+1,i-1) + 
                                a25 * IN_3D(k+1,j+1,i  ) + 
                                a26 * IN_3D(k+1,j+1,i+1) ;
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

__global__ void Stencil_Cuda(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, 
        DATA_TYPE a8, DATA_TYPE a9, DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, 
        DATA_TYPE a13, DATA_TYPE a14, DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, 
        DATA_TYPE a18, DATA_TYPE a19, DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, 
        DATA_TYPE a23, DATA_TYPE a24, DATA_TYPE a25, DATA_TYPE a26 , 
        int z, int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = threadIdx.z + blockIdx.z * blockDim.z + halo;

    OUT_3D(k,j,i) = a0  * IN_3D(k-1,j-1,i-1) +
                    a1  * IN_3D(k-1,j-1,i  ) +
                    a2  * IN_3D(k-1,j-1,i+1) +
                    a3  * IN_3D(k-1,j  ,i-1) +
                    a4  * IN_3D(k-1,j  ,i  ) +
                    a5  * IN_3D(k-1,j  ,i+1) +
                    a6  * IN_3D(k-1,j+1,i-1) + 
                    a7  * IN_3D(k-1,j+1,i  ) + 
                    a8  * IN_3D(k-1,j+1,i+1) + 
                    a9  * IN_3D(k  ,j-1,i-1) + 
                    a10 * IN_3D(k  ,j-1,i  ) + 
                    a11 * IN_3D(k  ,j-1,i+1) + 
                    a12 * IN_3D(k  ,j  ,i-1) + 
                    a13 * IN_3D(k  ,j  ,i  ) + 
                    a14 * IN_3D(k  ,j  ,i+1) + 
                    a15 * IN_3D(k  ,j+1,i-1) + 
                    a16 * IN_3D(k  ,j+1,i  ) + 
                    a17 * IN_3D(k  ,j+1,i+1) + 
                    a18 * IN_3D(k+1,j-1,i-1) + 
                    a19 * IN_3D(k+1,j-1,i  ) + 
                    a20 * IN_3D(k+1,j-1,i+1) + 
                    a21 * IN_3D(k+1,j  ,i-1) + 
                    a22 * IN_3D(k+1,j  ,i  ) + 
                    a23 * IN_3D(k+1,j  ,i+1) + 
                    a24 * IN_3D(k+1,j+1,i-1) + 
                    a25 * IN_3D(k+1,j+1,i  ) + 
                    a26 * IN_3D(k+1,j+1,i+1) ;
}

__global__ void Stencil_Cuda_Sweep(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, 
        DATA_TYPE a8, DATA_TYPE a9, DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, 
        DATA_TYPE a13, DATA_TYPE a14, DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, 
        DATA_TYPE a18, DATA_TYPE a19, DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, 
        DATA_TYPE a23, DATA_TYPE a24, DATA_TYPE a25, DATA_TYPE a26 , 
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
        OUT_3D(k,j,i) = a0  * IN_3D(k-1,j-1,i-1) +
                        a1  * IN_3D(k-1,j-1,i  ) +
                        a2  * IN_3D(k-1,j-1,i+1) +
                        a3  * IN_3D(k-1,j  ,i-1) +
                        a4  * IN_3D(k-1,j  ,i  ) +
                        a5  * IN_3D(k-1,j  ,i+1) +
                        a6  * IN_3D(k-1,j+1,i-1) + 
                        a7  * IN_3D(k-1,j+1,i  ) + 
                        a8  * IN_3D(k-1,j+1,i+1) + 
                        a9  * IN_3D(k  ,j-1,i-1) + 
                        a10 * IN_3D(k  ,j-1,i  ) + 
                        a11 * IN_3D(k  ,j-1,i+1) + 
                        a12 * IN_3D(k  ,j  ,i-1) + 
                        a13 * IN_3D(k  ,j  ,i  ) + 
                        a14 * IN_3D(k  ,j  ,i+1) + 
                        a15 * IN_3D(k  ,j+1,i-1) + 
                        a16 * IN_3D(k  ,j+1,i  ) + 
                        a17 * IN_3D(k  ,j+1,i+1) + 
                        a18 * IN_3D(k+1,j-1,i-1) + 
                        a19 * IN_3D(k+1,j-1,i  ) + 
                        a20 * IN_3D(k+1,j-1,i+1) + 
                        a21 * IN_3D(k+1,j  ,i-1) + 
                        a22 * IN_3D(k+1,j  ,i  ) + 
                        a23 * IN_3D(k+1,j  ,i+1) + 
                        a24 * IN_3D(k+1,j+1,i-1) + 
                        a25 * IN_3D(k+1,j+1,i  ) + 
                        a26 * IN_3D(k+1,j+1,i+1) ;
    }
}

__global__ void Stencil_Cuda_Sweep_Sm_Branch(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, 
        DATA_TYPE a8, DATA_TYPE a9, DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, 
        DATA_TYPE a13, DATA_TYPE a14, DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, 
        DATA_TYPE a18, DATA_TYPE a19, DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, 
        DATA_TYPE a23, DATA_TYPE a24, DATA_TYPE a25, DATA_TYPE a26 , 
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
    if(li == halo && lj == halo)                     
    {
        LOC_3D(t3,lj-1,li-1) = IN_3D(k  ,j-1,i-1); 
        LOC_3D(t2,lj-1,li-1) = IN_3D(k-1,j-1,i-1);
    }
    if(li == 32   && lj == halo)           
    {
        LOC_3D(t3,lj-1,li+1) = IN_3D(k  ,j-1,i+1); 
        LOC_3D(t2,lj-1,li+1) = IN_3D(k-1,j-1,i+1);
    }
    if(li == halo && lj == 8   )           
    { 
        LOC_3D(t3,lj+1,li-1) = IN_3D(k  ,j+1,i-1); 
        LOC_3D(t2,lj+1,li-1) = IN_3D(k-1,j+1,i-1);
    }
    if(li == 32   && lj == 8   ) 
    {
        LOC_3D(t3,lj+1,li+1) = IN_3D(k  ,j+1,i+1); 
        LOC_3D(t2,lj+1,li+1) = IN_3D(k-1,j+1,i+1);
    }



#pragma unroll 
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
        if(li == halo && lj == halo) {LOC_3D(t3,lj-1,li-1) = IN_3D(k+1,j-1,i-1);}
        if(li == 32   && lj == halo) {LOC_3D(t3,lj-1,li+1) = IN_3D(k+1,j-1,i+1);}
        if(li == halo && lj == 8   ) {LOC_3D(t3,lj+1,li-1) = IN_3D(k+1,j+1,i-1);}
        if(li == 32   && lj == 8   ) {LOC_3D(t3,lj+1,li+1) = IN_3D(k+1,j+1,i+1);}
        __syncthreads();

        OUT_3D(k,j,i) = a0  * LOC_3D(t1,lj-1,li-1) +
                        a1  * LOC_3D(t1,lj-1,li  ) +
                        a2  * LOC_3D(t1,lj-1,li+1) +
                        a3  * LOC_3D(t1,lj  ,li-1) +
                        a4  * LOC_3D(t1,lj  ,li  ) +
                        a5  * LOC_3D(t1,lj  ,li+1) +
                        a6  * LOC_3D(t1,lj+1,li-1) + 
                        a7  * LOC_3D(t1,lj+1,li  ) + 
                        a8  * LOC_3D(t1,lj+1,li+1) + 
                        a9  * LOC_3D(t2,lj-1,li-1) + 
                        a10 * LOC_3D(t2,lj-1,li  ) + 
                        a11 * LOC_3D(t2,lj-1,li+1) + 
                        a12 * LOC_3D(t2,lj  ,li-1) + 
                        a13 * LOC_3D(t2,lj  ,li  ) + 
                        a14 * LOC_3D(t2,lj  ,li+1) + 
                        a15 * LOC_3D(t2,lj+1,li-1) + 
                        a16 * LOC_3D(t2,lj+1,li  ) + 
                        a17 * LOC_3D(t2,lj+1,li+1) + 
                        a18 * LOC_3D(t3,lj-1,li-1) + 
                        a19 * LOC_3D(t3,lj-1,li  ) + 
                        a20 * LOC_3D(t3,lj-1,li+1) + 
                        a21 * LOC_3D(t3,lj  ,li-1) + 
                        a22 * LOC_3D(t3,lj  ,li  ) + 
                        a23 * LOC_3D(t3,lj  ,li+1) + 
                        a24 * LOC_3D(t3,lj+1,li-1) + 
                        a25 * LOC_3D(t3,lj+1,li  ) + 
                        a26 * LOC_3D(t3,lj+1,li+1) ;
        
        __syncthreads();
    }

}

__global__ void Stencil_Cuda_Sweep_Sm_Cyclic(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, 
        DATA_TYPE a8, DATA_TYPE a9, DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, 
        DATA_TYPE a13, DATA_TYPE a14, DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, 
        DATA_TYPE a18, DATA_TYPE a19, DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, 
        DATA_TYPE a23, DATA_TYPE a24, DATA_TYPE a25, DATA_TYPE a26 , 
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

#pragma unroll
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

        OUT_3D(k,j,i) = a0  * LOC_3D(t1,lj-1,li-1) +
                        a1  * LOC_3D(t1,lj-1,li  ) +
                        a2  * LOC_3D(t1,lj-1,li+1) +
                        a3  * LOC_3D(t1,lj  ,li-1) +
                        a4  * LOC_3D(t1,lj  ,li  ) +
                        a5  * LOC_3D(t1,lj  ,li+1) +
                        a6  * LOC_3D(t1,lj+1,li-1) + 
                        a7  * LOC_3D(t1,lj+1,li  ) + 
                        a8  * LOC_3D(t1,lj+1,li+1) + 
                        a9  * LOC_3D(t2,lj-1,li-1) + 
                        a10 * LOC_3D(t2,lj-1,li  ) + 
                        a11 * LOC_3D(t2,lj-1,li+1) + 
                        a12 * LOC_3D(t2,lj  ,li-1) + 
                        a13 * LOC_3D(t2,lj  ,li  ) + 
                        a14 * LOC_3D(t2,lj  ,li+1) + 
                        a15 * LOC_3D(t2,lj+1,li-1) + 
                        a16 * LOC_3D(t2,lj+1,li  ) + 
                        a17 * LOC_3D(t2,lj+1,li+1) + 
                        a18 * LOC_3D(t3,lj-1,li-1) + 
                        a19 * LOC_3D(t3,lj-1,li  ) + 
                        a20 * LOC_3D(t3,lj-1,li+1) + 
                        a21 * LOC_3D(t3,lj  ,li-1) + 
                        a22 * LOC_3D(t3,lj  ,li  ) + 
                        a23 * LOC_3D(t3,lj  ,li+1) + 
                        a24 * LOC_3D(t3,lj+1,li-1) + 
                        a25 * LOC_3D(t3,lj+1,li  ) + 
                        a26 * LOC_3D(t3,lj+1,li+1) ;
        
        __syncthreads();
    }

}

__global__ void Stencil_Cuda_Sm_Branch(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, 
        DATA_TYPE a8, DATA_TYPE a9, DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, 
        DATA_TYPE a13, DATA_TYPE a14, DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, 
        DATA_TYPE a18, DATA_TYPE a19, DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, 
        DATA_TYPE a23, DATA_TYPE a24, DATA_TYPE a25, DATA_TYPE a26 , 
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
    if(li == halo && lj == halo) LOC_3D(lk  ,lj-1,li-1) = IN_3D(k  ,j-1,i-1);
    if(li == halo && lk == halo) LOC_3D(lk-1,lj  ,li-1) = IN_3D(k-1,j  ,i-1);
    if(lj == halo && lk == halo) LOC_3D(lk-1,lj-1,li  ) = IN_3D(k-1,j-1,i  );
    if(li == 8    && lj == halo) LOC_3D(lk  ,lj-1,li+1) = IN_3D(k  ,j-1,i+1);
    if(li == 8    && lk == halo) LOC_3D(lk-1,lj  ,li+1) = IN_3D(k-1,j  ,i+1);
    if(lj == 4    && lk == halo) LOC_3D(lk-1,lj+1,li  ) = IN_3D(k-1,j+1,i  );
    if(li == halo && lj == 4   ) LOC_3D(lk  ,lj+1,li-1) = IN_3D(k  ,j+1,i-1);
    if(li == halo && lk == 8   ) LOC_3D(lk+1,lj  ,li-1) = IN_3D(k+1,j  ,i-1);
    if(lj == halo && lk == 8   ) LOC_3D(lk+1,lj-1,li  ) = IN_3D(k+1,j-1,i  );
    if(li == 8    && lj == 4   ) LOC_3D(lk  ,lj+1,li+1) = IN_3D(k  ,j+1,i+1);
    if(li == 8    && lk == 8   ) LOC_3D(lk+1,lj  ,li+1) = IN_3D(k+1,j  ,i+1);
    if(lj == 4    && lk == 8   ) LOC_3D(lk+1,lj+1,li  ) = IN_3D(k+1,j+1,i  );
    if(li == halo && lj == halo && lk == halo) LOC_3D(lk-1,lj-1,li-1) = IN_3D(k-1,j-1,i-1);
    if(li == halo && lj == halo && lk == 8   ) LOC_3D(lk+1,lj-1,li-1) = IN_3D(k+1,j-1,i-1);
    if(li == halo && lj == 4    && lk == halo) LOC_3D(lk-1,lj+1,li-1) = IN_3D(k-1,j+1,i-1);
    if(li == halo && lj == 4    && lk == 8   ) LOC_3D(lk+1,lj+1,li-1) = IN_3D(k+1,j+1,i-1);
    if(li == 8    && lj == halo && lk == halo) LOC_3D(lk-1,lj-1,li+1) = IN_3D(k-1,j-1,i+1);
    if(li == 8    && lj == halo && lk == 8   ) LOC_3D(lk+1,lj-1,li+1) = IN_3D(k+1,j-1,i+1);
    if(li == 8    && lj == 4    && lk == halo) LOC_3D(lk-1,lj+1,li+1) = IN_3D(k-1,j+1,i+1);
    if(li == 8    && lj == 4    && lk == 8   ) LOC_3D(lk+1,lj+1,li+1) = IN_3D(k+1,j+1,i+1);
    __syncthreads();

    OUT_3D(k,j,i) = a0  * LOC_3D(lk-1,lj-1,li-1) +
                    a1  * LOC_3D(lk-1,lj-1,li  ) +
                    a2  * LOC_3D(lk-1,lj-1,li+1) +
                    a3  * LOC_3D(lk-1,lj  ,li-1) +
                    a4  * LOC_3D(lk-1,lj  ,li  ) +
                    a5  * LOC_3D(lk-1,lj  ,li+1) +
                    a6  * LOC_3D(lk-1,lj+1,li-1) + 
                    a7  * LOC_3D(lk-1,lj+1,li  ) + 
                    a8  * LOC_3D(lk-1,lj+1,li+1) + 
                    a9  * LOC_3D(lk  ,lj-1,li-1) + 
                    a10 * LOC_3D(lk  ,lj-1,li  ) + 
                    a11 * LOC_3D(lk  ,lj-1,li+1) + 
                    a12 * LOC_3D(lk  ,lj  ,li-1) + 
                    a13 * LOC_3D(lk  ,lj  ,li  ) + 
                    a14 * LOC_3D(lk  ,lj  ,li+1) + 
                    a15 * LOC_3D(lk  ,lj+1,li-1) + 
                    a16 * LOC_3D(lk  ,lj+1,li  ) + 
                    a17 * LOC_3D(lk  ,lj+1,li+1) + 
                    a18 * LOC_3D(lk+1,lj-1,li-1) + 
                    a19 * LOC_3D(lk+1,lj-1,li  ) + 
                    a20 * LOC_3D(lk+1,lj-1,li+1) + 
                    a21 * LOC_3D(lk+1,lj  ,li-1) + 
                    a22 * LOC_3D(lk+1,lj  ,li  ) + 
                    a23 * LOC_3D(lk+1,lj  ,li+1) + 
                    a24 * LOC_3D(lk+1,lj+1,li-1) + 
                    a25 * LOC_3D(lk+1,lj+1,li  ) + 
                    a26 * LOC_3D(lk+1,lj+1,li+1) ;
}

__global__ void Stencil_Cuda_Sm_Cyclic(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, 
        DATA_TYPE a8, DATA_TYPE a9, DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, 
        DATA_TYPE a13, DATA_TYPE a14, DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, 
        DATA_TYPE a18, DATA_TYPE a19, DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, 
        DATA_TYPE a23, DATA_TYPE a24, DATA_TYPE a25, DATA_TYPE a26 , 
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

    OUT_3D(k,j,i) = a0  * LOC_3D(lk-1,lj-1,li-1) +
                    a1  * LOC_3D(lk-1,lj-1,li  ) +
                    a2  * LOC_3D(lk-1,lj-1,li+1) +
                    a3  * LOC_3D(lk-1,lj  ,li-1) +
                    a4  * LOC_3D(lk-1,lj  ,li  ) +
                    a5  * LOC_3D(lk-1,lj  ,li+1) +
                    a6  * LOC_3D(lk-1,lj+1,li-1) + 
                    a7  * LOC_3D(lk-1,lj+1,li  ) + 
                    a8  * LOC_3D(lk-1,lj+1,li+1) + 
                    a9  * LOC_3D(lk  ,lj-1,li-1) + 
                    a10 * LOC_3D(lk  ,lj-1,li  ) + 
                    a11 * LOC_3D(lk  ,lj-1,li+1) + 
                    a12 * LOC_3D(lk  ,lj  ,li-1) + 
                    a13 * LOC_3D(lk  ,lj  ,li  ) + 
                    a14 * LOC_3D(lk  ,lj  ,li+1) + 
                    a15 * LOC_3D(lk  ,lj+1,li-1) + 
                    a16 * LOC_3D(lk  ,lj+1,li  ) + 
                    a17 * LOC_3D(lk  ,lj+1,li+1) + 
                    a18 * LOC_3D(lk+1,lj-1,li-1) + 
                    a19 * LOC_3D(lk+1,lj-1,li  ) + 
                    a20 * LOC_3D(lk+1,lj-1,li+1) + 
                    a21 * LOC_3D(lk+1,lj  ,li-1) + 
                    a22 * LOC_3D(lk+1,lj  ,li  ) + 
                    a23 * LOC_3D(lk+1,lj  ,li+1) + 
                    a24 * LOC_3D(lk+1,lj+1,li-1) + 
                    a25 * LOC_3D(lk+1,lj+1,li  ) + 
                    a26 * LOC_3D(lk+1,lj+1,li+1) ;

}


__global__ void Stencil_Cuda_Shfl_2DWarp(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, 
        DATA_TYPE a8, DATA_TYPE a9, DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, 
        DATA_TYPE a13, DATA_TYPE a14, DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, 
        DATA_TYPE a18, DATA_TYPE a19, DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, 
        DATA_TYPE a23, DATA_TYPE a24, DATA_TYPE a25, DATA_TYPE a26 , 
        int z, int m, int n, int halo)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = threadIdx.z + blockIdx.z * blockDim.z + halo;
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    int lane_id = tid % warpSize;
    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
    int warp_id_z = (threadIdx.z + blockIdx.z * blockDim.z)>>0; // there numbers
    int new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
    int new_j = (warp_id_y<<2) + (lane_id/10)%6; // 6  is extended dimension of j 
    int new_k = (warp_id_z<<0) + lane_id/60;     // 60 is extended area of ixj = 10x6
    DATA_TYPE reg0, reg1, reg2, reg3, reg4, reg5;
    reg0 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+32)/60;
    reg1 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10;
    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+64)/60;
    reg2 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10;
    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+96)/60;
    reg3 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10;
    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+128)/60;
    reg4 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10;
    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+160)/60;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    new_k = (new_k < z+2*halo)? new_k: z+2*halo-1;
    reg5 = IN_3D(new_k, new_j, new_i);

    DATA_TYPE sum = 0.0;
    int friend_id;
    DATA_TYPE tx, ty, tz;

    friend_id = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg0, friend_id);
    ty = __shfl(reg1, friend_id);
    sum += a0*((lane_id < 26)? tx: ty);

    friend_id = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg0, friend_id);
    ty = __shfl(reg1, friend_id);
    sum += a1*((lane_id < 25)? tx: ty);

    friend_id = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg0, friend_id);
    ty = __shfl(reg1, friend_id);
    sum += a2*((lane_id < 24)? tx: ty);

    friend_id = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg0, friend_id);
    ty = __shfl(reg1, friend_id);
    sum += a3*((lane_id < 18)? tx: ty);

    friend_id = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg0, friend_id);
    ty = __shfl(reg1, friend_id);
    sum += a4*((lane_id < 17)? tx: ty);

    friend_id = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg0, friend_id);
    ty = __shfl(reg1, friend_id);
    sum += a5*((lane_id < 16)? tx: ty);

    friend_id = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg0, friend_id);
    ty = __shfl(reg1, friend_id);
    sum += a6*((lane_id < 10)? tx: ty);

    friend_id = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg0, friend_id);
    ty = __shfl(reg1, friend_id);
    sum += a7*((lane_id < 9 )? tx: ty);

    friend_id = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg0, friend_id);
    ty = __shfl(reg1, friend_id);
    sum += a8*((lane_id < 8 )? tx: ty);

    friend_id = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg1, friend_id);
    ty = __shfl(reg2, friend_id);
    tz = __shfl(reg3, friend_id);
    sum += a9*((lane_id < 4 )? tx: ((lane_id < 30)? ty: tz));

    friend_id = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg1, friend_id);
    ty = __shfl(reg2, friend_id);
    tz = __shfl(reg3, friend_id);
    sum += a10*((lane_id < 3 )? tx: ((lane_id < 29)? ty: tz));

    friend_id = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg1, friend_id);
    ty = __shfl(reg2, friend_id);
    tz = __shfl(reg3, friend_id);
    sum += a11*((lane_id < 2 )? tx: ((lane_id < 28)? ty: tz));

    friend_id = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg2, friend_id);
    ty = __shfl(reg3, friend_id);
    sum += a12*((lane_id < 22)? tx: ty);

    friend_id = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg2, friend_id);
    ty = __shfl(reg3, friend_id);
    sum += a13*((lane_id < 21)? tx: ty);

    friend_id = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg2, friend_id);
    ty = __shfl(reg3, friend_id);
    sum += a14*((lane_id < 20)? tx: ty);

    friend_id = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg2, friend_id);
    ty = __shfl(reg3, friend_id);
    sum += a15*((lane_id < 14)? tx: ty);

    friend_id = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg2, friend_id);
    ty = __shfl(reg3, friend_id);
    sum += a16*((lane_id < 13)? tx: ty);

    friend_id = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg2, friend_id);
    ty = __shfl(reg3, friend_id);
    sum += a17*((lane_id < 12)? tx: ty);

    friend_id = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg3, friend_id);
    ty = __shfl(reg4, friend_id);
    sum += a18*((lane_id < 8 )? tx: ty);

    friend_id = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg3, friend_id);
    ty = __shfl(reg4, friend_id);
    sum += a19*((lane_id < 7 )? tx: ty);

    friend_id = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg3, friend_id);
    ty = __shfl(reg4, friend_id);
    sum += a20*((lane_id < 6 )? tx: ty);

    friend_id = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg4, friend_id);
    ty = __shfl(reg5, friend_id);
    sum += a21*((lane_id < 24)? tx: ty);

    friend_id = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg4, friend_id);
    ty = __shfl(reg5, friend_id);
    sum += a22*((lane_id < 24)? tx: ty);

    friend_id = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg4, friend_id);
    ty = __shfl(reg5, friend_id);
    sum += a23*((lane_id < 24)? tx: ty);

    friend_id = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg4, friend_id);
    ty = __shfl(reg5, friend_id);
    sum += a24*((lane_id < 16)? tx: ty);

    friend_id = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg4, friend_id);
    ty = __shfl(reg5, friend_id);
    sum += a25*((lane_id < 16)? tx: ty);

    friend_id = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(reg4, friend_id);
    ty = __shfl(reg5, friend_id);
    sum += a26*((lane_id < 16)? tx: ty);
    
    OUT_3D(k,j,i) = sum;
}

__global__ void Stencil_Cuda_Shfl2_2DWarp(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, 
        DATA_TYPE a8, DATA_TYPE a9, DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, 
        DATA_TYPE a13, DATA_TYPE a14, DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, 
        DATA_TYPE a18, DATA_TYPE a19, DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, 
        DATA_TYPE a23, DATA_TYPE a24, DATA_TYPE a25, DATA_TYPE a26 , 
        int z, int m, int n, int halo)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<1) + (lane_id>>5) + halo; 
    // thread coarsening: related to warp dimensions 1x4x8. 
    // We coarsen from dimension z from 2^0 to 2^1, also need to know there are how many values in dimension z

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
    int warp_id_z = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<1) + (lane_id>>5); // these numbers
    int new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
    int new_j = (warp_id_y<<2) + (lane_id/10)%6; // 6  is extended dimension of j 
    int new_k = (warp_id_z<<0) + lane_id/60;     // 60 is extended area of ixj = 10x6
    DATA_TYPE reg0, reg1, reg2, reg3, reg4, reg5,
              reg6, reg7;
    reg0 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+32)/60;
    reg1 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10;
    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+64)/60;
    reg2 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10;
    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+96)/60;
    reg3 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10;
    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+128)/60;
    reg4 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10;
    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+160)/60;
    reg5 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+192)%10;
    new_j = (warp_id_y<<2) + ((lane_id+192)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+192)/60;
    reg6 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+224)%10;
    new_j = (warp_id_y<<2) + ((lane_id+224)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+224)/60;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    new_k = (new_k < z+2*halo)? new_k: z+2*halo-1;
    reg7 = IN_3D(new_k, new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    int friend_id0;
    int friend_id1;
    DATA_TYPE tx0, ty0, tz0;
    DATA_TYPE tx1, ty1, tz1;

    friend_id0 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    tz1 = __shfl(reg3, friend_id1);
    sum0 += a0*((lane_id < 26)? tx0: ty0);
    sum1 += a0*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));

    friend_id0 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    tz1 = __shfl(reg3, friend_id1);
    sum0 += a1*((lane_id < 25)? tx0: ty0);
    sum1 += a1*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));

    friend_id0 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    tz1 = __shfl(reg3, friend_id1);
    sum0 += a2*((lane_id < 24)? tx0: ty0);
    sum1 += a2*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));

    friend_id0 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    sum0 += a3*((lane_id < 18)? tx0: ty0);
    sum1 += a3*((lane_id < 22)? tx1: ty1);

    friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    sum0 += a4*((lane_id < 17)? tx0: ty0);
    sum1 += a4*((lane_id < 21)? tx1: ty1);

    friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    sum0 += a5*((lane_id < 16)? tx0: ty0);
    sum1 += a5*((lane_id < 20)? tx1: ty1);

    friend_id0 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    sum0 += a6*((lane_id < 10)? tx0: ty0);
    sum1 += a6*((lane_id < 14)? tx1: ty1);

    friend_id0 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    sum0 += a7*((lane_id < 9 )? tx0: ty0);
    sum1 += a7*((lane_id < 13)? tx1: ty1);

    friend_id0 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    sum0 += a8*((lane_id < 8 )? tx0: ty0);
    sum1 += a8*((lane_id < 12)? tx1: ty1);

    friend_id0 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    tz0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg3, friend_id1);
    ty1 = __shfl(reg4, friend_id1);
    sum0 += a9*((lane_id < 4 )? tx0: ((lane_id < 30)? ty0: tz0));
    sum1 += a9*((lane_id < 8)? tx1: ty1);

    friend_id0 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    tz0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg3, friend_id1);
    ty1 = __shfl(reg4, friend_id1);
    sum0 += a10*((lane_id < 3 )? tx0: ((lane_id < 29)? ty0: tz0));
    sum1 += a10*((lane_id < 7)? tx1: ty1);

    friend_id0 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    tz0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg3, friend_id1);
    ty1 = __shfl(reg4, friend_id1);
    sum0 += a11*((lane_id < 2 )? tx0: ((lane_id < 28)? ty0: tz0));
    sum1 += a11*((lane_id < 6)? tx1: ty1);

    friend_id0 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    sum0 += a12*((lane_id < 22)? tx0: ty0);
    sum1 += a12*((lane_id < 24)? tx1: ty1);

    friend_id0 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    sum0 += a13*((lane_id < 21)? tx0: ty0);
    sum1 += a13*((lane_id < 24)? tx1: ty1);

    friend_id0 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    sum0 += a14*((lane_id < 20)? tx0: ty0);
    sum1 += a14*((lane_id < 24)? tx1: ty1);

    friend_id0 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    sum0 += a15*((lane_id < 14)? tx0: ty0);
    sum1 += a15*((lane_id < 16)? tx1: ty1);

    friend_id0 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    sum0 += a16*((lane_id < 13)? tx0: ty0);
    sum1 += a16*((lane_id < 16)? tx1: ty1);

    friend_id0 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    sum0 += a17*((lane_id < 12)? tx0: ty0);
    sum1 += a17*((lane_id < 16)? tx1: ty1);

    friend_id0 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg3, friend_id0);
    ty0 = __shfl(reg4, friend_id0);
    tx1 = __shfl(reg5, friend_id1);
    ty1 = __shfl(reg6, friend_id1);
    sum0 += a18*((lane_id < 8 )? tx0: ty0);
    sum1 += a18*((lane_id < 10)? tx1: ty1);

    friend_id0 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg3, friend_id0);
    ty0 = __shfl(reg4, friend_id0);
    tx1 = __shfl(reg5, friend_id1);
    ty1 = __shfl(reg6, friend_id1);
    sum0 += a19*((lane_id < 7 )? tx0: ty0);
    sum1 += a19*((lane_id < 9 )? tx1: ty1);

    friend_id0 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg3, friend_id0);
    ty0 = __shfl(reg4, friend_id0);
    tx1 = __shfl(reg5, friend_id1);
    ty1 = __shfl(reg6, friend_id1);
    sum0 += a20*((lane_id < 6 )? tx0: ty0);
    sum1 += a20*((lane_id < 8 )? tx1: ty1);

    friend_id0 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    tx1 = __shfl(reg5, friend_id1);
    ty1 = __shfl(reg6, friend_id1);
    tz1 = __shfl(reg7, friend_id1);
    sum0 += a21*((lane_id < 24)? tx0: ty0);
    sum1 += a21*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));

    friend_id0 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    tx1 = __shfl(reg5, friend_id1);
    ty1 = __shfl(reg6, friend_id1);
    tz1 = __shfl(reg7, friend_id1);
    sum0 += a22*((lane_id < 24)? tx0: ty0);
    sum1 += a22*((lane_id < 1 )? tx1: ((lane_id < 27)? ty1: tz1));

    friend_id0 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    tx1 = __shfl(reg6, friend_id1);
    ty1 = __shfl(reg7, friend_id1);
    sum0 += a23*((lane_id < 24)? tx0: ty0);
    sum1 += a23*((lane_id < 26)? tx1: ty1);

    friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    tx1 = __shfl(reg6, friend_id1);
    ty1 = __shfl(reg7, friend_id1);
    sum0 += a24*((lane_id < 16)? tx0: ty0);
    sum1 += a24*((lane_id < 20)? tx1: ty1);

    friend_id0 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    tx1 = __shfl(reg6, friend_id1);
    ty1 = __shfl(reg7, friend_id1);
    sum0 += a25*((lane_id < 16)? tx0: ty0);
    sum1 += a25*((lane_id < 19)? tx1: ty1);

    friend_id0 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    tx1 = __shfl(reg6, friend_id1);
    ty1 = __shfl(reg7, friend_id1);
    sum0 += a26*((lane_id < 16)? tx0: ty0);
    sum1 += a26*((lane_id < 18)? tx1: ty1);

    OUT_3D(k  ,j,i) = sum0;
    OUT_3D(k+1,j,i) = sum1;
}

__global__ void Stencil_Cuda_Shfl4_2DWarp(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, 
        DATA_TYPE a8, DATA_TYPE a9, DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, 
        DATA_TYPE a13, DATA_TYPE a14, DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, 
        DATA_TYPE a18, DATA_TYPE a19, DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, 
        DATA_TYPE a23, DATA_TYPE a24, DATA_TYPE a25, DATA_TYPE a26 ,
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

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
    int warp_id_z = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<2) + (lane_id>>5); // these numbers
    int new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
    int new_j = (warp_id_y<<2) + (lane_id/10)%6; // 6  is extended dimension of j 
    int new_k = (warp_id_z<<0) + lane_id/60;     // 60 is extended area of ixj = 10x6
    DATA_TYPE reg0, reg1, reg2, reg3, reg4, reg5,
              reg6, reg7, reg8, reg9, reg10, reg11;
    reg0 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+32)/60;
    reg1 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10;
    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+64)/60;
    reg2 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10;
    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+96)/60;
    reg3 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10;
    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+128)/60;
    reg4 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10;
    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+160)/60;
    reg5 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+192)%10;
    new_j = (warp_id_y<<2) + ((lane_id+192)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+192)/60;
    reg6 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+224)%10;
    new_j = (warp_id_y<<2) + ((lane_id+224)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+224)/60;
    reg7 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+256)%10;
    new_j = (warp_id_y<<2) + ((lane_id+256)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+256)/60;
    reg8 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+288)%10;
    new_j = (warp_id_y<<2) + ((lane_id+288)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+288)/60;
    reg9 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+320)%10;
    new_j = (warp_id_y<<2) + ((lane_id+320)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+320)/60;
    reg10 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+352)%10;
    new_j = (warp_id_y<<2) + ((lane_id+352)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+352)/60;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    new_k = (new_k < z+2*halo)? new_k: z+2*halo-1;
    reg11 = IN_3D(new_k, new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    int friend_id0;
    int friend_id1;
    int friend_id2;
    int friend_id3;
    DATA_TYPE tx0, ty0, tz0;
    DATA_TYPE tx1, ty1, tz1; 
    DATA_TYPE tx2, ty2, tz2; 
    DATA_TYPE tx3, ty3, tz3;

    friend_id0 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    tz1 = __shfl(reg3, friend_id1);
    tx2 = __shfl(reg3, friend_id2);
    ty2 = __shfl(reg4, friend_id2);
    tx3 = __shfl(reg5, friend_id3);
    ty3 = __shfl(reg6, friend_id3);
    sum0 += a0*((lane_id < 26)? tx0: ty0);
    sum1 += a0*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
    sum2 += a0*((lane_id < 8 )? tx2: ty2);
    sum3 += a0*((lane_id < 10)? tx3: ty3);

    friend_id0 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    tz1 = __shfl(reg3, friend_id1);
    tx2 = __shfl(reg3, friend_id2);
    ty2 = __shfl(reg4, friend_id2);
    tx3 = __shfl(reg5, friend_id3);
    ty3 = __shfl(reg6, friend_id3);
    sum0 += a1*((lane_id < 25)? tx0: ty0);
    sum1 += a1*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
    sum2 += a1*((lane_id < 7 )? tx2: ty2);
    sum3 += a1*((lane_id < 9 )? tx3: ty3);

    friend_id0 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    tz1 = __shfl(reg3, friend_id1);
    tx2 = __shfl(reg3, friend_id2);
    ty2 = __shfl(reg4, friend_id2);
    tx3 = __shfl(reg5, friend_id3);
    ty3 = __shfl(reg6, friend_id3);
    sum0 += a2*((lane_id < 24)? tx0: ty0);
    sum1 += a2*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
    sum2 += a2*((lane_id < 6 )? tx2: ty2);
    sum3 += a2*((lane_id < 8 )? tx3: ty3);

    friend_id0 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    tx2 = __shfl(reg4, friend_id2);
    ty2 = __shfl(reg5, friend_id2);
    tx3 = __shfl(reg5, friend_id3);
    ty3 = __shfl(reg6, friend_id3);
    tz3 = __shfl(reg7, friend_id3);
    sum0 += a3*((lane_id < 18)? tx0: ty0);
    sum1 += a3*((lane_id < 22)? tx1: ty1);
    sum2 += a3*((lane_id < 24)? tx2: ty2);
    sum3 += a3*((lane_id < 2 )? tx3: ((lane_id < 28)? ty3: tz3));

    friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    tx2 = __shfl(reg4, friend_id2);
    ty2 = __shfl(reg5, friend_id2);
    tx3 = __shfl(reg5, friend_id3);
    ty3 = __shfl(reg6, friend_id3);
    tz3 = __shfl(reg7, friend_id3);
    sum0 += a4*((lane_id < 17)? tx0: ty0);
    sum1 += a4*((lane_id < 21)? tx1: ty1);
    sum2 += a4*((lane_id < 24)? tx2: ty2);
    sum3 += a4*((lane_id < 1 )? tx3: ((lane_id < 27)? ty3: tz3));

    friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    tx2 = __shfl(reg4, friend_id2);
    ty2 = __shfl(reg5, friend_id2);
    tx3 = __shfl(reg6, friend_id3);
    ty3 = __shfl(reg7, friend_id3);
    sum0 += a5*((lane_id < 16)? tx0: ty0);
    sum1 += a5*((lane_id < 20)? tx1: ty1);
    sum2 += a5*((lane_id < 24)? tx2: ty2);
    sum3 += a5*((lane_id < 26)? tx3: ty3);

    friend_id0 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    tx2 = __shfl(reg4, friend_id2);
    ty2 = __shfl(reg5, friend_id2);
    tx3 = __shfl(reg6, friend_id3);
    ty3 = __shfl(reg7, friend_id3);
    sum0 += a6*((lane_id < 10)? tx0: ty0);
    sum1 += a6*((lane_id < 14)? tx1: ty1);
    sum2 += a6*((lane_id < 16)? tx2: ty2);
    sum3 += a6*((lane_id < 20)? tx3: ty3);

    friend_id0 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    tx2 = __shfl(reg4, friend_id2);
    ty2 = __shfl(reg5, friend_id2);
    tx3 = __shfl(reg6, friend_id3);
    ty3 = __shfl(reg7, friend_id3);
    sum0 += a7*((lane_id < 9 )? tx0: ty0);
    sum1 += a7*((lane_id < 13)? tx1: ty1);
    sum2 += a7*((lane_id < 16)? tx2: ty2);
    sum3 += a7*((lane_id < 19)? tx3: ty3);

    friend_id0 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    tx2 = __shfl(reg4, friend_id2);
    ty2 = __shfl(reg5, friend_id2);
    tx3 = __shfl(reg6, friend_id3);
    ty3 = __shfl(reg7, friend_id3);
    sum0 += a8*((lane_id < 8 )? tx0: ty0);
    sum1 += a8*((lane_id < 12)? tx1: ty1);
    sum2 += a8*((lane_id < 16)? tx2: ty2);
    sum3 += a8*((lane_id < 18)? tx3: ty3);

    friend_id0 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    tz0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg3, friend_id1);
    ty1 = __shfl(reg4, friend_id1);
    tx2 = __shfl(reg5, friend_id2);
    ty2 = __shfl(reg6, friend_id2);
    tx3 = __shfl(reg7, friend_id3);
    ty3 = __shfl(reg8, friend_id3);
    sum0 += a9*((lane_id < 4 )? tx0: ((lane_id < 30)? ty0: tz0));
    sum1 += a9*((lane_id < 8)? tx1: ty1);
    sum2 += a9*((lane_id < 10)? tx2: ty2);
    sum3 += a9*((lane_id < 14)? tx3: ty3);

    friend_id0 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    tz0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg3, friend_id1);
    ty1 = __shfl(reg4, friend_id1);
    tx2 = __shfl(reg5, friend_id2);
    ty2 = __shfl(reg6, friend_id2);
    tx3 = __shfl(reg7, friend_id3);
    ty3 = __shfl(reg8, friend_id3);
    sum0 += a10*((lane_id < 3 )? tx0: ((lane_id < 29)? ty0: tz0));
    sum1 += a10*((lane_id < 7 )? tx1: ty1);
    sum2 += a10*((lane_id < 9 )? tx2: ty2);
    sum3 += a10*((lane_id < 13)? tx3: ty3);

    friend_id0 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    tz0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg3, friend_id1);
    ty1 = __shfl(reg4, friend_id1);
    tx2 = __shfl(reg5, friend_id2);
    ty2 = __shfl(reg6, friend_id2);
    tx3 = __shfl(reg7, friend_id3);
    ty3 = __shfl(reg8, friend_id3);
    sum0 += a11*((lane_id < 2 )? tx0: ((lane_id < 28)? ty0: tz0));
    sum1 += a11*((lane_id < 6 )? tx1: ty1);
    sum2 += a11*((lane_id < 8 )? tx2: ty2);
    sum3 += a11*((lane_id < 12)? tx3: ty3);

    friend_id0 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    tx2 = __shfl(reg5, friend_id2);
    ty2 = __shfl(reg6, friend_id2);
    tz2 = __shfl(reg7, friend_id2);
    tx3 = __shfl(reg7, friend_id3);
    ty3 = __shfl(reg8, friend_id3);
    sum0 += a12*((lane_id < 22)? tx0: ty0);
    sum1 += a12*((lane_id < 24)? tx1: ty1);
    sum2 += a12*((lane_id < 2 )? tx2: ((lane_id < 28)? ty2: tz2));
    sum3 += a12*((lane_id < 6 )? tx3: ty3);

    friend_id0 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    tx2 = __shfl(reg5, friend_id2);
    ty2 = __shfl(reg6, friend_id2);
    tz2 = __shfl(reg7, friend_id2);
    tx3 = __shfl(reg7, friend_id3);
    ty3 = __shfl(reg8, friend_id3);
    tz3 = __shfl(reg9, friend_id3);
    sum0 += a13*((lane_id < 21)? tx0: ty0);
    sum1 += a13*((lane_id < 24)? tx1: ty1);
    sum2 += a13*((lane_id < 1 )? tx2: ((lane_id < 27)? ty2: tz2));
    sum3 += a13*((lane_id < 5 )? tx3: ((lane_id < 31)? ty3: tz3));

    friend_id0 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    tx2 = __shfl(reg6, friend_id2);
    ty2 = __shfl(reg7, friend_id2);
    tx3 = __shfl(reg7, friend_id3);
    ty3 = __shfl(reg8, friend_id3);
    tz3 = __shfl(reg9, friend_id3);
    sum0 += a14*((lane_id < 20)? tx0: ty0);
    sum1 += a14*((lane_id < 24)? tx1: ty1);
    sum2 += a14*((lane_id < 26)? tx2: ty2);
    sum3 += a14*((lane_id < 4 )? tx3: ((lane_id < 30)? ty3: tz3));

    friend_id0 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    tx2 = __shfl(reg6, friend_id2);
    ty2 = __shfl(reg7, friend_id2);
    tx3 = __shfl(reg8, friend_id3);
    ty3 = __shfl(reg9, friend_id3);
    sum0 += a15*((lane_id < 14)? tx0: ty0);
    sum1 += a15*((lane_id < 16)? tx1: ty1);
    sum2 += a15*((lane_id < 20)? tx2: ty2);
    sum3 += a15*((lane_id < 24)? tx3: ty3);

    friend_id0 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    tx2 = __shfl(reg6, friend_id2);
    ty2 = __shfl(reg7, friend_id2);
    tx3 = __shfl(reg8, friend_id3);
    ty3 = __shfl(reg9, friend_id3);
    sum0 += a16*((lane_id < 13)? tx0: ty0);
    sum1 += a16*((lane_id < 16)? tx1: ty1);
    sum2 += a16*((lane_id < 19)? tx2: ty2);
    sum3 += a16*((lane_id < 23)? tx3: ty3);

    friend_id0 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    tx2 = __shfl(reg6, friend_id2);
    ty2 = __shfl(reg7, friend_id2);
    tx3 = __shfl(reg8, friend_id3);
    ty3 = __shfl(reg9, friend_id3);
    sum0 += a17*((lane_id < 12)? tx0: ty0);
    sum1 += a17*((lane_id < 16)? tx1: ty1);
    sum2 += a17*((lane_id < 18)? tx2: ty2);
    sum3 += a17*((lane_id < 22)? tx3: ty3);

    friend_id0 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg3, friend_id0);
    ty0 = __shfl(reg4, friend_id0);
    tx1 = __shfl(reg5, friend_id1);
    ty1 = __shfl(reg6, friend_id1);
    tx2 = __shfl(reg7, friend_id2);
    ty2 = __shfl(reg8, friend_id2);
    tx3 = __shfl(reg9 , friend_id3);
    ty3 = __shfl(reg10, friend_id3);
    sum0 += a18*((lane_id < 8 )? tx0: ty0);
    sum1 += a18*((lane_id < 10)? tx1: ty1);
    sum2 += a18*((lane_id < 14)? tx2: ty2);
    sum3 += a18*((lane_id < 16)? tx3: ty3);

    friend_id0 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg3, friend_id0);
    ty0 = __shfl(reg4, friend_id0);
    tx1 = __shfl(reg5, friend_id1);
    ty1 = __shfl(reg6, friend_id1);
    tx2 = __shfl(reg7, friend_id2);
    ty2 = __shfl(reg8, friend_id2);
    tx3 = __shfl(reg9 , friend_id3);
    ty3 = __shfl(reg10, friend_id3);
    sum0 += a19*((lane_id < 7 )? tx0: ty0);
    sum1 += a19*((lane_id < 9 )? tx1: ty1);
    sum2 += a19*((lane_id < 13)? tx2: ty2);
    sum3 += a19*((lane_id < 16)? tx3: ty3);

    friend_id0 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg3, friend_id0);
    ty0 = __shfl(reg4, friend_id0);
    tx1 = __shfl(reg5, friend_id1);
    ty1 = __shfl(reg6, friend_id1);
    tx2 = __shfl(reg7, friend_id2);
    ty2 = __shfl(reg8, friend_id2);
    tx3 = __shfl(reg9 , friend_id3);
    ty3 = __shfl(reg10, friend_id3);
    sum0 += a20*((lane_id < 6 )? tx0: ty0);
    sum1 += a20*((lane_id < 8 )? tx1: ty1);
    sum2 += a20*((lane_id < 12)? tx2: ty2);
    sum3 += a20*((lane_id < 16)? tx3: ty3);

    friend_id0 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    tx1 = __shfl(reg5, friend_id1);
    ty1 = __shfl(reg6, friend_id1);
    tz1 = __shfl(reg7, friend_id1);
    tx2 = __shfl(reg7, friend_id2);
    ty2 = __shfl(reg8, friend_id2);
    tx3 = __shfl(reg9 , friend_id3);
    ty3 = __shfl(reg10, friend_id3);
    sum0 += a21*((lane_id < 24)? tx0: ty0);
    sum1 += a21*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
    sum2 += a21*((lane_id < 6 )? tx2: ty2);
    sum3 += a21*((lane_id < 8 )? tx3: ty3);

    friend_id0 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+23+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    tx1 = __shfl(reg5, friend_id1);
    ty1 = __shfl(reg6, friend_id1);
    tz1 = __shfl(reg7, friend_id1);
    tx2 = __shfl(reg7, friend_id2);
    ty2 = __shfl(reg8, friend_id2);
    tz2 = __shfl(reg9, friend_id2);
    tx3 = __shfl(reg9 , friend_id3);
    ty3 = __shfl(reg10, friend_id3);
    sum0 += a22*((lane_id < 24)? tx0: ty0);
    sum1 += a22*((lane_id < 1 )? tx1: ((lane_id < 27)? ty1: tz1));
    sum2 += a22*((lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2));
    sum3 += a22*((lane_id < 8 )? tx3: ty3);

    friend_id0 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    tx1 = __shfl(reg6, friend_id1);
    ty1 = __shfl(reg7, friend_id1);
    tx2 = __shfl(reg7, friend_id2);
    ty2 = __shfl(reg8, friend_id2);
    tz2 = __shfl(reg9, friend_id2);
    tx3 = __shfl(reg9 , friend_id3);
    ty3 = __shfl(reg10, friend_id3);
    sum0 += a23*((lane_id < 24)? tx0: ty0);
    sum1 += a23*((lane_id < 26)? tx1: ty1);
    sum2 += a23*((lane_id < 4 )? tx2: ((lane_id < 30)? ty2: tz2));
    sum3 += a23*((lane_id < 8 )? tx3: ty3);

    friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    tx1 = __shfl(reg6, friend_id1);
    ty1 = __shfl(reg7, friend_id1);
    tx2 = __shfl(reg8, friend_id2);
    ty2 = __shfl(reg9, friend_id2);
    tx3 = __shfl(reg10, friend_id3);
    ty3 = __shfl(reg11, friend_id3);
    sum0 += a24*((lane_id < 16)? tx0: ty0);
    sum1 += a24*((lane_id < 20)? tx1: ty1);
    sum2 += a24*((lane_id < 24)? tx2: ty2);
    sum3 += a24*((lane_id < 26)? tx3: ty3);

    friend_id0 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    tx1 = __shfl(reg6, friend_id1);
    ty1 = __shfl(reg7, friend_id1);
    tx2 = __shfl(reg8, friend_id2);
    ty2 = __shfl(reg9, friend_id2);
    tx3 = __shfl(reg10, friend_id3);
    ty3 = __shfl(reg11, friend_id3);
    sum0 += a25*((lane_id < 16)? tx0: ty0);
    sum1 += a25*((lane_id < 19)? tx1: ty1);
    sum2 += a25*((lane_id < 23)? tx2: ty2);
    sum3 += a25*((lane_id < 25)? tx3: ty3);

    friend_id0 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    tx1 = __shfl(reg6, friend_id1);
    ty1 = __shfl(reg7, friend_id1);
    tx2 = __shfl(reg8, friend_id2);
    ty2 = __shfl(reg9, friend_id2);
    tx3 = __shfl(reg10, friend_id3);
    ty3 = __shfl(reg11, friend_id3);
    sum0 += a26*((lane_id < 16)? tx0: ty0);
    sum1 += a26*((lane_id < 18)? tx1: ty1);
    sum2 += a26*((lane_id < 22)? tx2: ty2);
    sum3 += a26*((lane_id < 24)? tx3: ty3);

    OUT_3D(k  ,j,i) = sum0;
    OUT_3D(k+1,j,i) = sum1;
    OUT_3D(k+2,j,i) = sum2;
    OUT_3D(k+3,j,i) = sum3;
}

__global__ void Stencil_Cuda_Shfl8_2DWarp(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, 
        DATA_TYPE a8, DATA_TYPE a9, DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, 
        DATA_TYPE a13, DATA_TYPE a14, DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, 
        DATA_TYPE a18, DATA_TYPE a19, DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, 
        DATA_TYPE a23, DATA_TYPE a24, DATA_TYPE a25, DATA_TYPE a26 , 
        int z, int m, int n, int halo)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<3) + (lane_id>>5) + halo; 
    // Thread coarsening: related to warp dimensions 1x4x8. 
    // We coarsen from dimension z from 2^0 to 2^3, also need to know there are how many values in dimension z,
    // which is (lane_id>>5) 

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
    int warp_id_z = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<3) + (lane_id>>5); // these numbers
    int new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
    int new_j = (warp_id_y<<2) + (lane_id/10)%6; // 6  is extended dimension of j 
    int new_k = (warp_id_z<<0) + lane_id/60;     // 60 is extended area of ixj = 10x6
    DATA_TYPE reg0, reg1, reg2, reg3, reg4, reg5,
              reg6, reg7, reg8, reg9, reg10, reg11,
              reg12, reg13, reg14, reg15, reg16, reg17,
              reg18;
    reg0 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+32)/60;
    reg1 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10;
    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+64)/60;
    reg2 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10;
    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+96)/60;
    reg3 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10;
    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+128)/60;
    reg4 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10;
    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+160)/60;
    reg5 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+192)%10;
    new_j = (warp_id_y<<2) + ((lane_id+192)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+192)/60;
    reg6 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+224)%10;
    new_j = (warp_id_y<<2) + ((lane_id+224)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+224)/60;
    reg7 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+256)%10;
    new_j = (warp_id_y<<2) + ((lane_id+256)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+256)/60;
    reg8 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+288)%10;
    new_j = (warp_id_y<<2) + ((lane_id+288)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+288)/60;
    reg9 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+320)%10;
    new_j = (warp_id_y<<2) + ((lane_id+320)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+320)/60;
    reg10 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+352)%10;
    new_j = (warp_id_y<<2) + ((lane_id+352)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+352)/60;
    reg11 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+384)%10;
    new_j = (warp_id_y<<2) + ((lane_id+384)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+384)/60;
    reg12 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+416)%10;
    new_j = (warp_id_y<<2) + ((lane_id+416)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+416)/60;
    reg13 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+448)%10;
    new_j = (warp_id_y<<2) + ((lane_id+448)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+448)/60;
    reg14 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+480)%10;
    new_j = (warp_id_y<<2) + ((lane_id+480)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+480)/60;
    reg15 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+512)%10;
    new_j = (warp_id_y<<2) + ((lane_id+512)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+512)/60;
    reg16 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+544)%10;
    new_j = (warp_id_y<<2) + ((lane_id+544)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+544)/60;
    reg17 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+576)%10;
    new_j = (warp_id_y<<2) + ((lane_id+576)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+576)/60;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    new_k = (new_k < z+2*halo)? new_k: z+2*halo-1;
    reg18 = IN_3D(new_k, new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    DATA_TYPE sum4 = 0.0;
    DATA_TYPE sum5 = 0.0;
    DATA_TYPE sum6 = 0.0;
    DATA_TYPE sum7 = 0.0;
    int friend_id0; 
    int friend_id1; 
    int friend_id2; 
    int friend_id3;
    int friend_id4;
    int friend_id5; 
    int friend_id6;
    int friend_id7;
    DATA_TYPE tx0, ty0, tz0;
    DATA_TYPE tx1, ty1, tz1; 
    DATA_TYPE tx2, ty2, tz2;
    DATA_TYPE tx3, ty3, tz3;
    DATA_TYPE rx0, ry0, rz0; 
    DATA_TYPE rx1, ry1, rz1; 
    DATA_TYPE rx2, ry2, rz2; 
    DATA_TYPE rx3, ry3, rz3;

    friend_id0 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    tz1 = __shfl(reg3, friend_id1);
    tx2 = __shfl(reg3, friend_id2);
    ty2 = __shfl(reg4, friend_id2);
    tx3 = __shfl(reg5, friend_id3);
    ty3 = __shfl(reg6, friend_id3);
    rx0 = __shfl(reg7, friend_id4);
    ry0 = __shfl(reg8, friend_id4);
    rx1 = __shfl(reg9 , friend_id5);
    ry1 = __shfl(reg10, friend_id5);
    rx2 = __shfl(reg11, friend_id6);
    ry2 = __shfl(reg12, friend_id6);
    rx3 = __shfl(reg13, friend_id7);
    ry3 = __shfl(reg14, friend_id7);
    sum0 += a0*((lane_id < 26)? tx0: ty0);
    sum1 += a0*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
    sum2 += a0*((lane_id < 8 )? tx2: ty2);
    sum3 += a0*((lane_id < 10)? tx3: ty3);
    sum4 += a0*((lane_id < 14)? rx0: ry0);
    sum5 += a0*((lane_id < 16)? rx1: ry1);
    sum6 += a0*((lane_id < 20)? rx2: ry2);
    sum7 += a0*((lane_id < 24)? rx3: ry3);

    friend_id0 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    tz1 = __shfl(reg3, friend_id1);
    tx2 = __shfl(reg3, friend_id2);
    ty2 = __shfl(reg4, friend_id2);
    tx3 = __shfl(reg5, friend_id3);
    ty3 = __shfl(reg6, friend_id3);
    rx0 = __shfl(reg7, friend_id4);
    ry0 = __shfl(reg8, friend_id4);
    rx1 = __shfl(reg9 , friend_id5);
    ry1 = __shfl(reg10, friend_id5);
    rx2 = __shfl(reg11, friend_id6);
    ry2 = __shfl(reg12, friend_id6);
    rx3 = __shfl(reg13, friend_id7);
    ry3 = __shfl(reg14, friend_id7);
    sum0 += a1*((lane_id < 25)? tx0: ty0);
    sum1 += a1*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
    sum2 += a1*((lane_id < 7 )? tx2: ty2);
    sum3 += a1*((lane_id < 9 )? tx3: ty3);
    sum4 += a1*((lane_id < 13)? rx0: ry0);
    sum5 += a1*((lane_id < 16)? rx1: ry1);
    sum6 += a1*((lane_id < 19)? rx2: ry2);
    sum7 += a1*((lane_id < 23)? rx3: ry3);

    friend_id0 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg1, friend_id1);
    ty1 = __shfl(reg2, friend_id1);
    tz1 = __shfl(reg3, friend_id1);
    tx2 = __shfl(reg3, friend_id2);
    ty2 = __shfl(reg4, friend_id2);
    tx3 = __shfl(reg5, friend_id3);
    ty3 = __shfl(reg6, friend_id3);
    rx0 = __shfl(reg7, friend_id4);
    ry0 = __shfl(reg8, friend_id4);
    rx1 = __shfl(reg9 , friend_id5);
    ry1 = __shfl(reg10, friend_id5);
    rx2 = __shfl(reg11, friend_id6);
    ry2 = __shfl(reg12, friend_id6);
    rx3 = __shfl(reg13, friend_id7);
    ry3 = __shfl(reg14, friend_id7);
    sum0 += a2*((lane_id < 24)? tx0: ty0);
    sum1 += a2*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
    sum2 += a2*((lane_id < 6 )? tx2: ty2);
    sum3 += a2*((lane_id < 8 )? tx3: ty3);
    sum4 += a2*((lane_id < 12)? rx0: ry0);
    sum5 += a2*((lane_id < 16)? rx1: ry1);
    sum6 += a2*((lane_id < 18)? rx2: ry2);
    sum7 += a2*((lane_id < 22)? rx3: ry3);

    friend_id0 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    tx2 = __shfl(reg4, friend_id2);
    ty2 = __shfl(reg5, friend_id2);
    tx3 = __shfl(reg5, friend_id3);
    ty3 = __shfl(reg6, friend_id3);
    tz3 = __shfl(reg7, friend_id3);
    rx0 = __shfl(reg7, friend_id4);
    ry0 = __shfl(reg8, friend_id4);
    rx1 = __shfl(reg9 , friend_id5);
    ry1 = __shfl(reg10, friend_id5);
    rx2 = __shfl(reg11, friend_id6);
    ry2 = __shfl(reg12, friend_id6);
    rx3 = __shfl(reg13, friend_id7);
    ry3 = __shfl(reg14, friend_id7);
    sum0 += a3*((lane_id < 18)? tx0: ty0);
    sum1 += a3*((lane_id < 22)? tx1: ty1);
    sum2 += a3*((lane_id < 24)? tx2: ty2);
    sum3 += a3*((lane_id < 2 )? tx3: ((lane_id < 28)? ty3: tz3));
    sum4 += a3*((lane_id < 6 )? rx0: ry0);
    sum5 += a3*((lane_id < 8 )? rx1: ry1);
    sum6 += a3*((lane_id < 12)? rx2: ry2);
    sum7 += a3*((lane_id < 16)? rx3: ry3);

    friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+23+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+15+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    tx2 = __shfl(reg4, friend_id2);
    ty2 = __shfl(reg5, friend_id2);
    tx3 = __shfl(reg5, friend_id3);
    ty3 = __shfl(reg6, friend_id3);
    tz3 = __shfl(reg7, friend_id3);
    rx0 = __shfl(reg7, friend_id4);
    ry0 = __shfl(reg8, friend_id4);
    rz0 = __shfl(reg9, friend_id4);
    rx1 = __shfl(reg9 , friend_id5);
    ry1 = __shfl(reg10, friend_id5);
    rx2 = __shfl(reg11, friend_id6);
    ry2 = __shfl(reg12, friend_id6);
    rx3 = __shfl(reg13, friend_id7);
    ry3 = __shfl(reg14, friend_id7);
    sum0 += a4*((lane_id < 17)? tx0: ty0);
    sum1 += a4*((lane_id < 21)? tx1: ty1);
    sum2 += a4*((lane_id < 24)? tx2: ty2);
    sum3 += a4*((lane_id < 1 )? tx3: ((lane_id < 27)? ty3: tz3));
    sum4 += a4*((lane_id < 5 )? rx0: ((lane_id < 31)? ry0: rz0));
    sum5 += a4*((lane_id < 8 )? rx1: ry1);
    sum6 += a4*((lane_id < 11)? rx2: ry2);
    sum7 += a4*((lane_id < 15)? rx3: ry3);

    friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    tx2 = __shfl(reg4, friend_id2);
    ty2 = __shfl(reg5, friend_id2);
    tx3 = __shfl(reg6, friend_id3);
    ty3 = __shfl(reg7, friend_id3);
    rx0 = __shfl(reg7, friend_id4);
    ry0 = __shfl(reg8, friend_id4);
    rz0 = __shfl(reg9, friend_id4);
    rx1 = __shfl(reg9 , friend_id5);
    ry1 = __shfl(reg10, friend_id5);
    rx2 = __shfl(reg11, friend_id6);
    ry2 = __shfl(reg12, friend_id6);
    rx3 = __shfl(reg13, friend_id7);
    ry3 = __shfl(reg14, friend_id7);
    sum0 += a5*((lane_id < 16)? tx0: ty0);
    sum1 += a5*((lane_id < 20)? tx1: ty1);
    sum2 += a5*((lane_id < 24)? tx2: ty2);
    sum3 += a5*((lane_id < 26)? tx3: ty3);
    sum4 += a5*((lane_id < 4 )? rx0: ((lane_id < 30)? ry0: rz0));
    sum5 += a5*((lane_id < 8 )? rx1: ry1);
    sum6 += a5*((lane_id < 10)? rx2: ry2);
    sum7 += a5*((lane_id < 14)? rx3: ry3);

    friend_id0 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    tx2 = __shfl(reg4, friend_id2);
    ty2 = __shfl(reg5, friend_id2);
    tx3 = __shfl(reg6, friend_id3);
    ty3 = __shfl(reg7, friend_id3);
    rx0 = __shfl(reg8, friend_id4);
    ry0 = __shfl(reg9, friend_id4);
    rx1 = __shfl(reg10, friend_id5);
    ry1 = __shfl(reg11, friend_id5);
    rx2 = __shfl(reg11, friend_id6);
    ry2 = __shfl(reg12, friend_id6);
    rz2 = __shfl(reg13, friend_id6);
    rx3 = __shfl(reg13, friend_id7);
    ry3 = __shfl(reg14, friend_id7);
    sum0 += a6*((lane_id < 10)? tx0: ty0);
    sum1 += a6*((lane_id < 14)? tx1: ty1);
    sum2 += a6*((lane_id < 16)? tx2: ty2);
    sum3 += a6*((lane_id < 20)? tx3: ty3);
    sum4 += a6*((lane_id < 24)? rx0: ry0);
    sum5 += a6*((lane_id < 26)? rx1: ry1);
    sum6 += a6*((lane_id < 4 )? rx2: ((lane_id < 30)? ry2: rz2));
    sum7 += a6*((lane_id < 8 )? rx3: ry3);

    friend_id0 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    tx2 = __shfl(reg4, friend_id2);
    ty2 = __shfl(reg5, friend_id2);
    tx3 = __shfl(reg6, friend_id3);
    ty3 = __shfl(reg7, friend_id3);
    rx0 = __shfl(reg8, friend_id4);
    ry0 = __shfl(reg9, friend_id4);
    rx1 = __shfl(reg10, friend_id5);
    ry1 = __shfl(reg11, friend_id5);
    rx2 = __shfl(reg11, friend_id6);
    ry2 = __shfl(reg12, friend_id6);
    rz2 = __shfl(reg13, friend_id6);
    rx3 = __shfl(reg13, friend_id7);
    ry3 = __shfl(reg14, friend_id7);
    sum0 += a7*((lane_id < 9 )? tx0: ty0);
    sum1 += a7*((lane_id < 13)? tx1: ty1);
    sum2 += a7*((lane_id < 16)? tx2: ty2);
    sum3 += a7*((lane_id < 19)? tx3: ty3);
    sum4 += a7*((lane_id < 23)? rx0: ry0);
    sum5 += a7*((lane_id < 25)? rx1: ry1);
    sum6 += a7*((lane_id < 3 )? rx2: ((lane_id < 29)? ry2: rz2));
    sum7 += a7*((lane_id < 7 )? rx3: ry3);

    friend_id0 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg0, friend_id0);
    ty0 = __shfl(reg1, friend_id0);
    tx1 = __shfl(reg2, friend_id1);
    ty1 = __shfl(reg3, friend_id1);
    tx2 = __shfl(reg4, friend_id2);
    ty2 = __shfl(reg5, friend_id2);
    tx3 = __shfl(reg6, friend_id3);
    ty3 = __shfl(reg7, friend_id3);
    rx0 = __shfl(reg8, friend_id4);
    ry0 = __shfl(reg9, friend_id4);
    rx1 = __shfl(reg10, friend_id5);
    ry1 = __shfl(reg11, friend_id5);
    rx2 = __shfl(reg11, friend_id6);
    ry2 = __shfl(reg12, friend_id6);
    rz2 = __shfl(reg13, friend_id6);
    rx3 = __shfl(reg13, friend_id7);
    ry3 = __shfl(reg14, friend_id7);
    sum0 += a8*((lane_id < 8 )? tx0: ty0);
    sum1 += a8*((lane_id < 12)? tx1: ty1);
    sum2 += a8*((lane_id < 16)? tx2: ty2);
    sum3 += a8*((lane_id < 18)? tx3: ty3);
    sum4 += a8*((lane_id < 22)? rx0: ry0);
    sum5 += a8*((lane_id < 24)? rx1: ry1);
    sum6 += a8*((lane_id < 2 )? rx2: ((lane_id < 28)? ry2: rz2));
    sum7 += a8*((lane_id < 6 )? rx3: ry3);

    friend_id0 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    tz0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg3, friend_id1);
    ty1 = __shfl(reg4, friend_id1);
    tx2 = __shfl(reg5, friend_id2);
    ty2 = __shfl(reg6, friend_id2);
    tx3 = __shfl(reg7, friend_id3);
    ty3 = __shfl(reg8, friend_id3);
    rx0 = __shfl(reg9 , friend_id4);
    ry0 = __shfl(reg10, friend_id4);
    rx1 = __shfl(reg11, friend_id5);
    ry1 = __shfl(reg12, friend_id5);
    rx2 = __shfl(reg13, friend_id6);
    ry2 = __shfl(reg14, friend_id6);
    rx3 = __shfl(reg15, friend_id7);
    ry3 = __shfl(reg16, friend_id7);
    sum0 += a9*((lane_id < 4 )? tx0: ((lane_id < 30)? ty0: tz0));
    sum1 += a9*((lane_id < 8)? tx1: ty1);
    sum2 += a9*((lane_id < 10)? tx2: ty2);
    sum3 += a9*((lane_id < 14)? tx3: ty3);
    sum4 += a9*((lane_id < 16)? rx0: ry0);
    sum5 += a9*((lane_id < 20)? rx1: ry1);
    sum6 += a9*((lane_id < 24)? rx2: ry2);
    sum7 += a9*((lane_id < 26)? rx3: ry3);

    friend_id0 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    tz0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg3, friend_id1);
    ty1 = __shfl(reg4, friend_id1);
    tx2 = __shfl(reg5, friend_id2);
    ty2 = __shfl(reg6, friend_id2);
    tx3 = __shfl(reg7, friend_id3);
    ty3 = __shfl(reg8, friend_id3);
    rx0 = __shfl(reg9 , friend_id4);
    ry0 = __shfl(reg10, friend_id4);
    rx1 = __shfl(reg11, friend_id5);
    ry1 = __shfl(reg12, friend_id5);
    rx2 = __shfl(reg13, friend_id6);
    ry2 = __shfl(reg14, friend_id6);
    rx3 = __shfl(reg15, friend_id7);
    ry3 = __shfl(reg16, friend_id7);
    sum0 += a10*((lane_id < 3 )? tx0: ((lane_id < 29)? ty0: tz0));
    sum1 += a10*((lane_id < 7 )? tx1: ty1);
    sum2 += a10*((lane_id < 9 )? tx2: ty2);
    sum3 += a10*((lane_id < 13)? tx3: ty3);
    sum4 += a10*((lane_id < 16)? rx0: ry0);
    sum5 += a10*((lane_id < 19)? rx1: ry1);
    sum6 += a10*((lane_id < 23)? rx2: ry2);
    sum7 += a10*((lane_id < 25)? rx3: ry3);

    friend_id0 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg1, friend_id0);
    ty0 = __shfl(reg2, friend_id0);
    tz0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg3, friend_id1);
    ty1 = __shfl(reg4, friend_id1);
    tx2 = __shfl(reg5, friend_id2);
    ty2 = __shfl(reg6, friend_id2);
    tx3 = __shfl(reg7, friend_id3);
    ty3 = __shfl(reg8, friend_id3);
    rx0 = __shfl(reg9 , friend_id4);
    ry0 = __shfl(reg10, friend_id4);
    rx1 = __shfl(reg11, friend_id5);
    ry1 = __shfl(reg12, friend_id5);
    rx2 = __shfl(reg13, friend_id6);
    ry2 = __shfl(reg14, friend_id6);
    rx3 = __shfl(reg15, friend_id7);
    ry3 = __shfl(reg16, friend_id7);
    sum0 += a11*((lane_id < 2 )? tx0: ((lane_id < 28)? ty0: tz0));
    sum1 += a11*((lane_id < 6 )? tx1: ty1);
    sum2 += a11*((lane_id < 8 )? tx2: ty2);
    sum3 += a11*((lane_id < 12)? tx3: ty3);
    sum4 += a11*((lane_id < 16)? rx0: ry0);
    sum5 += a11*((lane_id < 18)? rx1: ry1);
    sum6 += a11*((lane_id < 22)? rx2: ry2);
    sum7 += a11*((lane_id < 24)? rx3: ry3);

    friend_id0 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    tx2 = __shfl(reg5, friend_id2);
    ty2 = __shfl(reg6, friend_id2);
    tz2 = __shfl(reg7, friend_id2);
    tx3 = __shfl(reg7, friend_id3);
    ty3 = __shfl(reg8, friend_id3);
    rx0 = __shfl(reg9 , friend_id4);
    ry0 = __shfl(reg10, friend_id4);
    rx1 = __shfl(reg11, friend_id5);
    ry1 = __shfl(reg12, friend_id5);
    rx2 = __shfl(reg13, friend_id6);
    ry2 = __shfl(reg14, friend_id6);
    rx3 = __shfl(reg15, friend_id7);
    ry3 = __shfl(reg16, friend_id7);
    sum0 += a12*((lane_id < 22)? tx0: ty0);
    sum1 += a12*((lane_id < 24)? tx1: ty1);
    sum2 += a12*((lane_id < 2 )? tx2: ((lane_id < 28)? ty2: tz2));
    sum3 += a12*((lane_id < 6 )? tx3: ty3);
    sum4 += a12*((lane_id < 8 )? rx0: ry0);
    sum5 += a12*((lane_id < 12)? rx1: ry1);
    sum6 += a12*((lane_id < 16)? rx2: ry2);
    sum7 += a12*((lane_id < 18)? rx3: ry3);

    friend_id0 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+23+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+15+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    tx2 = __shfl(reg5, friend_id2);
    ty2 = __shfl(reg6, friend_id2);
    tz2 = __shfl(reg7, friend_id2);
    tx3 = __shfl(reg7, friend_id3);
    ty3 = __shfl(reg8, friend_id3);
    tz3 = __shfl(reg9, friend_id3);
    rx0 = __shfl(reg9 , friend_id4);
    ry0 = __shfl(reg10, friend_id4);
    rx1 = __shfl(reg11, friend_id5);
    ry1 = __shfl(reg12, friend_id5);
    rx2 = __shfl(reg13, friend_id6);
    ry2 = __shfl(reg14, friend_id6);
    rx3 = __shfl(reg15, friend_id7);
    ry3 = __shfl(reg16, friend_id7);
    sum0 += a13*((lane_id < 21)? tx0: ty0);
    sum1 += a13*((lane_id < 24)? tx1: ty1);
    sum2 += a13*((lane_id < 1 )? tx2: ((lane_id < 27)? ty2: tz2));
    sum3 += a13*((lane_id < 5 )? tx3: ((lane_id < 31)? ty3: tz3));
    sum4 += a13*((lane_id < 8 )? rx0: ry0);
    sum5 += a13*((lane_id < 11)? rx1: ry1);
    sum6 += a13*((lane_id < 15)? rx2: ry2);
    sum7 += a13*((lane_id < 17)? rx3: ry3);

    friend_id0 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    tx2 = __shfl(reg6, friend_id2);
    ty2 = __shfl(reg7, friend_id2);
    tx3 = __shfl(reg7, friend_id3);
    ty3 = __shfl(reg8, friend_id3);
    tz3 = __shfl(reg9, friend_id3);
    rx0 = __shfl(reg9 , friend_id4);
    ry0 = __shfl(reg10, friend_id4);
    rx1 = __shfl(reg11, friend_id5);
    ry1 = __shfl(reg12, friend_id5);
    rx2 = __shfl(reg13, friend_id6);
    ry2 = __shfl(reg14, friend_id6);
    rx3 = __shfl(reg15, friend_id7);
    ry3 = __shfl(reg16, friend_id7);
    sum0 += a14*((lane_id < 20)? tx0: ty0);
    sum1 += a14*((lane_id < 24)? tx1: ty1);
    sum2 += a14*((lane_id < 26)? tx2: ty2);
    sum3 += a14*((lane_id < 4 )? tx3: ((lane_id < 30)? ty3: tz3));
    sum4 += a14*((lane_id < 8 )? rx0: ry0);
    sum5 += a14*((lane_id < 10)? rx1: ry1);
    sum6 += a14*((lane_id < 14)? rx2: ry2);
    sum7 += a14*((lane_id < 16)? rx3: ry3);

    friend_id0 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+0+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    tx2 = __shfl(reg6, friend_id2);
    ty2 = __shfl(reg7, friend_id2);
    tx3 = __shfl(reg8, friend_id3);
    ty3 = __shfl(reg9, friend_id3);
    rx0 = __shfl(reg10, friend_id4);
    ry0 = __shfl(reg11, friend_id4);
    rx1 = __shfl(reg11, friend_id5);
    ry1 = __shfl(reg12, friend_id5);
    rz1 = __shfl(reg13, friend_id5);
    rx2 = __shfl(reg13, friend_id6);
    ry2 = __shfl(reg14, friend_id6);
    rx3 = __shfl(reg15, friend_id7);
    ry3 = __shfl(reg16, friend_id7);
    sum0 += a15*((lane_id < 14)? tx0: ty0);
    sum1 += a15*((lane_id < 16)? tx1: ty1);
    sum2 += a15*((lane_id < 20)? tx2: ty2);
    sum3 += a15*((lane_id < 24)? tx3: ty3);
    sum4 += a15*((lane_id < 26)? rx0: ry0);
    sum5 += a15*((lane_id < 4 )? rx1: ((lane_id < 30)? ry1: rz1));
    sum6 += a15*((lane_id < 8 )? rx2: ry2);
    sum7 += a15*((lane_id < 10)? rx3: ry3);

    friend_id0 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+1+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    tx2 = __shfl(reg6, friend_id2);
    ty2 = __shfl(reg7, friend_id2);
    tx3 = __shfl(reg8, friend_id3);
    ty3 = __shfl(reg9, friend_id3);
    rx0 = __shfl(reg10, friend_id4);
    ry0 = __shfl(reg11, friend_id4);
    rx1 = __shfl(reg11, friend_id5);
    ry1 = __shfl(reg12, friend_id5);
    rz1 = __shfl(reg13, friend_id5);
    rx2 = __shfl(reg13, friend_id6);
    ry2 = __shfl(reg14, friend_id6);
    rx3 = __shfl(reg15, friend_id7);
    ry3 = __shfl(reg16, friend_id7);
    sum0 += a16*((lane_id < 13)? tx0: ty0);
    sum1 += a16*((lane_id < 16)? tx1: ty1);
    sum2 += a16*((lane_id < 19)? tx2: ty2);
    sum3 += a16*((lane_id < 23)? tx3: ty3);
    sum4 += a16*((lane_id < 25)? rx0: ry0);
    sum5 += a16*((lane_id < 3 )? rx1: ((lane_id < 29)? ry1: rz1));
    sum6 += a16*((lane_id < 7 )? rx2: ry2);
    sum7 += a16*((lane_id < 9 )? rx3: ry3);

    friend_id0 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+2+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg2, friend_id0);
    ty0 = __shfl(reg3, friend_id0);
    tx1 = __shfl(reg4, friend_id1);
    ty1 = __shfl(reg5, friend_id1);
    tx2 = __shfl(reg6, friend_id2);
    ty2 = __shfl(reg7, friend_id2);
    tx3 = __shfl(reg8, friend_id3);
    ty3 = __shfl(reg9, friend_id3);
    rx0 = __shfl(reg10, friend_id4);
    ry0 = __shfl(reg11, friend_id4);
    rx1 = __shfl(reg11, friend_id5);
    ry1 = __shfl(reg12, friend_id5);
    rz1 = __shfl(reg13, friend_id5);
    rx2 = __shfl(reg13, friend_id6);
    ry2 = __shfl(reg14, friend_id6);
    rx3 = __shfl(reg15, friend_id7);
    ry3 = __shfl(reg16, friend_id7);
    sum0 += a17*((lane_id < 12)? tx0: ty0);
    sum1 += a17*((lane_id < 16)? tx1: ty1);
    sum2 += a17*((lane_id < 18)? tx2: ty2);
    sum3 += a17*((lane_id < 22)? tx3: ty3);
    sum4 += a17*((lane_id < 24)? rx0: ry0);
    sum5 += a17*((lane_id < 2 )? rx1: ((lane_id < 28)? ry1: rz1));
    sum6 += a17*((lane_id < 6 )? rx2: ry2);
    sum7 += a17*((lane_id < 8 )? rx3: ry3);

    friend_id0 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg3, friend_id0);
    ty0 = __shfl(reg4, friend_id0);
    tx1 = __shfl(reg5, friend_id1);
    ty1 = __shfl(reg6, friend_id1);
    tx2 = __shfl(reg7, friend_id2);
    ty2 = __shfl(reg8, friend_id2);
    tx3 = __shfl(reg9 , friend_id3);
    ty3 = __shfl(reg10, friend_id3);
    rx0 = __shfl(reg11, friend_id4);
    ry0 = __shfl(reg12, friend_id4);
    rx1 = __shfl(reg13, friend_id5);
    ry1 = __shfl(reg14, friend_id5);
    rx2 = __shfl(reg15, friend_id6);
    ry2 = __shfl(reg16, friend_id6);
    rx3 = __shfl(reg16, friend_id7);
    ry3 = __shfl(reg17, friend_id7);
    rz3 = __shfl(reg18, friend_id7);
    sum0 += a18*((lane_id < 8 )? tx0: ty0);
    sum1 += a18*((lane_id < 10)? tx1: ty1);
    sum2 += a18*((lane_id < 14)? tx2: ty2);
    sum3 += a18*((lane_id < 16)? tx3: ty3);
    sum4 += a18*((lane_id < 20)? rx0: ry0);
    sum5 += a18*((lane_id < 24)? rx1: ry1);
    sum6 += a18*((lane_id < 26)? rx2: ry2);
    sum7 += a18*((lane_id < 4 )? rx3: ((lane_id < 30)? ry3: rz3));

    friend_id0 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg3, friend_id0);
    ty0 = __shfl(reg4, friend_id0);
    tx1 = __shfl(reg5, friend_id1);
    ty1 = __shfl(reg6, friend_id1);
    tx2 = __shfl(reg7, friend_id2);
    ty2 = __shfl(reg8, friend_id2);
    tx3 = __shfl(reg9 , friend_id3);
    ty3 = __shfl(reg10, friend_id3);
    rx0 = __shfl(reg11, friend_id4);
    ry0 = __shfl(reg12, friend_id4);
    rx1 = __shfl(reg13, friend_id5);
    ry1 = __shfl(reg14, friend_id5);
    rx2 = __shfl(reg15, friend_id6);
    ry2 = __shfl(reg16, friend_id6);
    rx3 = __shfl(reg16, friend_id7);
    ry3 = __shfl(reg17, friend_id7);
    rz3 = __shfl(reg18, friend_id7);
    sum0 += a19*((lane_id < 7 )? tx0: ty0);
    sum1 += a19*((lane_id < 9 )? tx1: ty1);
    sum2 += a19*((lane_id < 13)? tx2: ty2);
    sum3 += a19*((lane_id < 16)? tx3: ty3);
    sum4 += a19*((lane_id < 19)? rx0: ry0);
    sum5 += a19*((lane_id < 23)? rx1: ry1);
    sum6 += a19*((lane_id < 25)? rx2: ry2);
    sum7 += a19*((lane_id < 3 )? rx3: ((lane_id < 29)? ry3: rz3));

    friend_id0 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg3, friend_id0);
    ty0 = __shfl(reg4, friend_id0);
    tx1 = __shfl(reg5, friend_id1);
    ty1 = __shfl(reg6, friend_id1);
    tx2 = __shfl(reg7, friend_id2);
    ty2 = __shfl(reg8, friend_id2);
    tx3 = __shfl(reg9 , friend_id3);
    ty3 = __shfl(reg10, friend_id3);
    rx0 = __shfl(reg11, friend_id4);
    ry0 = __shfl(reg12, friend_id4);
    rx1 = __shfl(reg13, friend_id5);
    ry1 = __shfl(reg14, friend_id5);
    rx2 = __shfl(reg15, friend_id6);
    ry2 = __shfl(reg16, friend_id6);
    rx3 = __shfl(reg16, friend_id7);
    ry3 = __shfl(reg17, friend_id7);
    rz3 = __shfl(reg18, friend_id7);
    sum0 += a20*((lane_id < 6 )? tx0: ty0);
    sum1 += a20*((lane_id < 8 )? tx1: ty1);
    sum2 += a20*((lane_id < 12)? tx2: ty2);
    sum3 += a20*((lane_id < 16)? tx3: ty3);
    sum4 += a20*((lane_id < 18)? rx0: ry0);
    sum5 += a20*((lane_id < 22)? rx1: ry1);
    sum6 += a20*((lane_id < 24)? rx2: ry2);
    sum7 += a20*((lane_id < 2 )? rx3: ((lane_id < 28)? ry3: rz3));

    friend_id0 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    tx1 = __shfl(reg5, friend_id1);
    ty1 = __shfl(reg6, friend_id1);
    tz1 = __shfl(reg7, friend_id1);
    tx2 = __shfl(reg7, friend_id2);
    ty2 = __shfl(reg8, friend_id2);
    tx3 = __shfl(reg9 , friend_id3);
    ty3 = __shfl(reg10, friend_id3);
    rx0 = __shfl(reg11, friend_id4);
    ry0 = __shfl(reg12, friend_id4);
    rx1 = __shfl(reg13, friend_id5);
    ry1 = __shfl(reg14, friend_id5);
    rx2 = __shfl(reg15, friend_id6);
    ry2 = __shfl(reg16, friend_id6);
    rx3 = __shfl(reg17, friend_id7);
    ry3 = __shfl(reg18, friend_id7);
    sum0 += a21*((lane_id < 24)? tx0: ty0);
    sum1 += a21*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
    sum2 += a21*((lane_id < 6 )? tx2: ty2);
    sum3 += a21*((lane_id < 8 )? tx3: ty3);
    sum4 += a21*((lane_id < 12)? rx0: ry0);
    sum5 += a21*((lane_id < 16)? rx1: ry1);
    sum6 += a21*((lane_id < 18)? rx2: ry2);
    sum7 += a21*((lane_id < 22)? rx3: ry3);

    friend_id0 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+23+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+15+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    tx1 = __shfl(reg5, friend_id1);
    ty1 = __shfl(reg6, friend_id1);
    tz1 = __shfl(reg7, friend_id1);
    tx2 = __shfl(reg7, friend_id2);
    ty2 = __shfl(reg8, friend_id2);
    tz2 = __shfl(reg9, friend_id2);
    tx3 = __shfl(reg9 , friend_id3);
    ty3 = __shfl(reg10, friend_id3);
    rx0 = __shfl(reg11, friend_id4);
    ry0 = __shfl(reg12, friend_id4);
    rx1 = __shfl(reg13, friend_id5);
    ry1 = __shfl(reg14, friend_id5);
    rx2 = __shfl(reg15, friend_id6);
    ry2 = __shfl(reg16, friend_id6);
    rx3 = __shfl(reg17, friend_id7);
    ry3 = __shfl(reg18, friend_id7);
    sum0 += a22*((lane_id < 24)? tx0: ty0);
    sum1 += a22*((lane_id < 1 )? tx1: ((lane_id < 27)? ty1: tz1));
    sum2 += a22*((lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2));
    sum3 += a22*((lane_id < 8 )? tx3: ty3);
    sum4 += a22*((lane_id < 11)? rx0: ry0);
    sum5 += a22*((lane_id < 15)? rx1: ry1);
    sum6 += a22*((lane_id < 17)? rx2: ry2);
    sum7 += a22*((lane_id < 21)? rx3: ry3);

    friend_id0 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    tx1 = __shfl(reg6, friend_id1);
    ty1 = __shfl(reg7, friend_id1);
    tx2 = __shfl(reg7, friend_id2);
    ty2 = __shfl(reg8, friend_id2);
    tz2 = __shfl(reg9, friend_id2);
    tx3 = __shfl(reg9 , friend_id3);
    ty3 = __shfl(reg10, friend_id3);
    rx0 = __shfl(reg11, friend_id4);
    ry0 = __shfl(reg12, friend_id4);
    rx1 = __shfl(reg13, friend_id5);
    ry1 = __shfl(reg14, friend_id5);
    rx2 = __shfl(reg15, friend_id6);
    ry2 = __shfl(reg16, friend_id6);
    rx3 = __shfl(reg17, friend_id7);
    ry3 = __shfl(reg18, friend_id7);
    sum0 += a23*((lane_id < 24)? tx0: ty0);
    sum1 += a23*((lane_id < 26)? tx1: ty1);
    sum2 += a23*((lane_id < 4 )? tx2: ((lane_id < 30)? ty2: tz2));
    sum3 += a23*((lane_id < 8 )? tx3: ty3);
    sum4 += a23*((lane_id < 10)? rx0: ry0);
    sum5 += a23*((lane_id < 14)? rx1: ry1);
    sum6 += a23*((lane_id < 16)? rx2: ry2);
    sum7 += a23*((lane_id < 20)? rx3: ry3);

    friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    tx1 = __shfl(reg6, friend_id1);
    ty1 = __shfl(reg7, friend_id1);
    tx2 = __shfl(reg8, friend_id2);
    ty2 = __shfl(reg9, friend_id2);
    tx3 = __shfl(reg10, friend_id3);
    ty3 = __shfl(reg11, friend_id3);
    rx0 = __shfl(reg11, friend_id4);
    ry0 = __shfl(reg12, friend_id4);
    rz0 = __shfl(reg13, friend_id4);
    rx1 = __shfl(reg13, friend_id5);
    ry1 = __shfl(reg14, friend_id5);
    rx2 = __shfl(reg15, friend_id6);
    ry2 = __shfl(reg16, friend_id6);
    rx3 = __shfl(reg17, friend_id7);
    ry3 = __shfl(reg18, friend_id7);
    sum0 += a24*((lane_id < 16)? tx0: ty0);
    sum1 += a24*((lane_id < 20)? tx1: ty1);
    sum2 += a24*((lane_id < 24)? tx2: ty2);
    sum3 += a24*((lane_id < 26)? tx3: ty3);
    sum4 += a24*((lane_id < 4 )? rx0: ((lane_id < 30)? ry0: rz0));
    sum5 += a24*((lane_id < 8 )? rx1: ry1);
    sum6 += a24*((lane_id < 10)? rx2: ry2);
    sum7 += a24*((lane_id < 14)? rx3: ry3);

    friend_id0 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    tx1 = __shfl(reg6, friend_id1);
    ty1 = __shfl(reg7, friend_id1);
    tx2 = __shfl(reg8, friend_id2);
    ty2 = __shfl(reg9, friend_id2);
    tx3 = __shfl(reg10, friend_id3);
    ty3 = __shfl(reg11, friend_id3);
    rx0 = __shfl(reg11, friend_id4);
    ry0 = __shfl(reg12, friend_id4);
    rz0 = __shfl(reg13, friend_id4);
    rx1 = __shfl(reg13, friend_id5);
    ry1 = __shfl(reg14, friend_id5);
    rx2 = __shfl(reg15, friend_id6);
    ry2 = __shfl(reg16, friend_id6);
    rx3 = __shfl(reg17, friend_id7);
    ry3 = __shfl(reg18, friend_id7);
    sum0 += a25*((lane_id < 16)? tx0: ty0);
    sum1 += a25*((lane_id < 19)? tx1: ty1);
    sum2 += a25*((lane_id < 23)? tx2: ty2);
    sum3 += a25*((lane_id < 25)? tx3: ty3);
    sum4 += a25*((lane_id < 3 )? rx0: ((lane_id < 29)? ry0: rz0));
    sum5 += a25*((lane_id < 7 )? rx1: ry1);
    sum6 += a25*((lane_id < 9 )? rx2: ry2);
    sum7 += a25*((lane_id < 13)? rx3: ry3);

    friend_id0 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(reg4, friend_id0);
    ty0 = __shfl(reg5, friend_id0);
    tx1 = __shfl(reg6, friend_id1);
    ty1 = __shfl(reg7, friend_id1);
    tx2 = __shfl(reg8, friend_id2);
    ty2 = __shfl(reg9, friend_id2);
    tx3 = __shfl(reg10, friend_id3);
    ty3 = __shfl(reg11, friend_id3);
    rx0 = __shfl(reg11, friend_id4);
    ry0 = __shfl(reg12, friend_id4);
    rz0 = __shfl(reg13, friend_id4);
    rx1 = __shfl(reg13, friend_id5);
    ry1 = __shfl(reg14, friend_id5);
    rx2 = __shfl(reg15, friend_id6);
    ry2 = __shfl(reg16, friend_id6);
    rx3 = __shfl(reg17, friend_id7);
    ry3 = __shfl(reg18, friend_id7);
    sum0 += a26*((lane_id < 16)? tx0: ty0);
    sum1 += a26*((lane_id < 18)? tx1: ty1);
    sum2 += a26*((lane_id < 22)? tx2: ty2);
    sum3 += a26*((lane_id < 24)? tx3: ty3);
    sum4 += a26*((lane_id < 2 )? rx0: ((lane_id < 28)? ry0: rz0));
    sum5 += a26*((lane_id < 6 )? rx1: ry1);
    sum6 += a26*((lane_id < 8 )? rx2: ry2);
    sum7 += a26*((lane_id < 12)? rx3: ry3);


    OUT_3D(k  ,j,i) = sum0;
    OUT_3D(k+1,j,i) = sum1;
    OUT_3D(k+2,j,i) = sum2;
    OUT_3D(k+3,j,i) = sum3;
    OUT_3D(k+4,j,i) = sum4;
    OUT_3D(k+5,j,i) = sum5;
    OUT_3D(k+6,j,i) = sum6;
    OUT_3D(k+7,j,i) = sum7;
}

__global__ void Stencil_Cuda_Sweep_Shfl_2DWarp(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, 
        DATA_TYPE a8, DATA_TYPE a9, DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, 
        DATA_TYPE a13, DATA_TYPE a14, DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, 
        DATA_TYPE a18, DATA_TYPE a19, DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, 
        DATA_TYPE a23, DATA_TYPE a24, DATA_TYPE a25, DATA_TYPE a26 , 
        int z, int m, int n, int halo)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % warpSize;
    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
    DATA_TYPE tx, ty;
    int friend_id;
    int new_i, new_j;
    DATA_TYPE t3_reg0, t3_reg1;
    DATA_TYPE t2_reg0, t2_reg1;
    DATA_TYPE t1_reg0, t1_reg1;

#define SM_2D_M2 32 
#define SM_2D_N2 8 

    DATA_TYPE sum = 0.0;

    // t3 is current layer; t2 is previous layer
    new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
    new_j = (warp_id_y<<2) + lane_id/10;     
    t3_reg0 = IN_3D(k  , new_j, new_i);
    t2_reg0 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + (lane_id+32)/10;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    t3_reg1 = IN_3D(k  , new_j, new_i);
    t2_reg1 = IN_3D(k-1, new_j, new_i);

#pragma unroll // it seems the loop-unroll is useless to performance
    for(; k < k_end; ++k)
    {
        sum = 0.0;
        // move the current storage down 
        t1_reg0 = t2_reg0;
        t1_reg1 = t2_reg1;
        t2_reg0 = t3_reg0;
        t2_reg1 = t3_reg1;

        new_i = (warp_id_x<<3) + lane_id%10;  
        new_j = (warp_id_y<<2) + lane_id/10;     
        t3_reg0 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+32)%10;
        new_j = (warp_id_y<<2) + (lane_id+32)/10;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
        t3_reg1 = IN_3D(k+1, new_j, new_i);

        friend_id = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
        tx = __shfl(t1_reg0, friend_id);
        ty = __shfl(t1_reg1, friend_id);
        sum += a0*((lane_id < 26)? tx: ty);
        tx = __shfl(t2_reg0, friend_id);
        ty = __shfl(t2_reg1, friend_id);
        sum += a9*((lane_id < 26)? tx: ty);
        tx = __shfl(t3_reg0, friend_id);
        ty = __shfl(t3_reg1, friend_id);
        sum += a18*((lane_id < 26)? tx: ty);

        friend_id = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
        tx = __shfl(t1_reg0, friend_id);
        ty = __shfl(t1_reg1, friend_id);
        sum += a1*((lane_id < 25)? tx: ty);
        tx = __shfl(t2_reg0, friend_id);
        ty = __shfl(t2_reg1, friend_id);
        sum += a10*((lane_id < 25)? tx: ty);
        tx = __shfl(t3_reg0, friend_id);
        ty = __shfl(t3_reg1, friend_id);
        sum += a19*((lane_id < 25)? tx: ty);

        friend_id = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
        tx = __shfl(t1_reg0, friend_id);
        ty = __shfl(t1_reg1, friend_id);
        sum += a2*((lane_id < 24)? tx: ty);
        tx = __shfl(t2_reg0, friend_id);
        ty = __shfl(t2_reg1, friend_id);
        sum += a11*((lane_id < 24)? tx: ty);
        tx = __shfl(t3_reg0, friend_id);
        ty = __shfl(t3_reg1, friend_id);
        sum += a20*((lane_id < 24)? tx: ty);

        friend_id = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
        tx = __shfl(t1_reg0, friend_id);
        ty = __shfl(t1_reg1, friend_id);
        sum += a3*((lane_id < 18)? tx: ty);
        tx = __shfl(t2_reg0, friend_id);
        ty = __shfl(t2_reg1, friend_id);
        sum += a12*((lane_id < 18)? tx: ty);
        tx = __shfl(t3_reg0, friend_id);
        ty = __shfl(t3_reg1, friend_id);
        sum += a21*((lane_id < 18)? tx: ty);

        friend_id = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
        tx = __shfl(t1_reg0, friend_id);
        ty = __shfl(t1_reg1, friend_id);
        sum += a4*((lane_id < 17)? tx: ty);
        tx = __shfl(t2_reg0, friend_id);
        ty = __shfl(t2_reg1, friend_id);
        sum += a13*((lane_id < 17)? tx: ty);
        tx = __shfl(t3_reg0, friend_id);
        ty = __shfl(t3_reg1, friend_id);
        sum += a22*((lane_id < 17)? tx: ty);

        friend_id = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
        tx = __shfl(t1_reg0, friend_id);
        ty = __shfl(t1_reg1, friend_id);
        sum += a5*((lane_id < 16)? tx: ty);
        tx = __shfl(t2_reg0, friend_id);
        ty = __shfl(t2_reg1, friend_id);
        sum += a14*((lane_id < 16)? tx: ty);
        tx = __shfl(t3_reg0, friend_id);
        ty = __shfl(t3_reg1, friend_id);
        sum += a23*((lane_id < 16)? tx: ty);

        friend_id = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
        tx = __shfl(t1_reg0, friend_id);
        ty = __shfl(t1_reg1, friend_id);
        sum += a6*((lane_id < 10)? tx: ty);
        tx = __shfl(t2_reg0, friend_id);
        ty = __shfl(t2_reg1, friend_id);
        sum += a15*((lane_id < 10)? tx: ty);
        tx = __shfl(t3_reg0, friend_id);
        ty = __shfl(t3_reg1, friend_id);
        sum += a24*((lane_id < 10)? tx: ty);

        friend_id = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
        tx = __shfl(t1_reg0, friend_id);
        ty = __shfl(t1_reg1, friend_id);
        sum += a7*((lane_id < 9 )? tx: ty);
        tx = __shfl(t2_reg0, friend_id);
        ty = __shfl(t2_reg1, friend_id);
        sum += a16*((lane_id < 9 )? tx: ty);
        tx = __shfl(t3_reg0, friend_id);
        ty = __shfl(t3_reg1, friend_id);
        sum += a25*((lane_id < 9 )? tx: ty);
        
        friend_id = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
        tx = __shfl(t1_reg0, friend_id);
        ty = __shfl(t1_reg1, friend_id);
        sum += a8*((lane_id < 8 )? tx: ty);
        tx = __shfl(t2_reg0, friend_id);
        ty = __shfl(t2_reg1, friend_id);
        sum += a17*((lane_id < 8 )? tx: ty);
        tx = __shfl(t3_reg0, friend_id);
        ty = __shfl(t3_reg1, friend_id);
        sum += a26*((lane_id < 8 )? tx: ty);

        OUT_3D(k,j,i) = sum;
    }
}

__global__ void Stencil_Cuda_Sweep_Shfl2_2DWarp(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, 
        DATA_TYPE a8, DATA_TYPE a9, DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, 
        DATA_TYPE a13, DATA_TYPE a14, DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, 
        DATA_TYPE a18, DATA_TYPE a19, DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, 
        DATA_TYPE a23, DATA_TYPE a24, DATA_TYPE a25, DATA_TYPE a26 , 
        int z, int m, int n, int halo)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<3) + (lane_id>>3) + halo;

    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;
    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<3) + (lane_id>>3))>>2; // 1x4x8, warp_ids are division of 
    DATA_TYPE tx0, ty0;
    DATA_TYPE tx1, ty1, tz1;
    int friend_id0, friend_id1;
    int new_i, new_j;
    DATA_TYPE t3_reg0, t3_reg1, t3_reg2, t3_reg3;
    DATA_TYPE t2_reg0, t2_reg1, t2_reg2, t2_reg3;
    DATA_TYPE t1_reg0, t1_reg1, t1_reg2, t1_reg3;

#define SM_2D_M2 32 
#define SM_2D_N2 8 

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;

    // t3 is current layer; t2 is previous layer
    new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
    new_j = (warp_id_y<<2) + lane_id/10;     
    t3_reg0 = IN_3D(k  , new_j, new_i);
    t2_reg0 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + (lane_id+32)/10;
    t3_reg1 = IN_3D(k  , new_j, new_i);
    t2_reg1 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10;
    new_j = (warp_id_y<<2) + (lane_id+64)/10;
    t3_reg2 = IN_3D(k  , new_j, new_i);
    t2_reg2 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10;
    new_j = (warp_id_y<<2) + (lane_id+96)/10;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    t3_reg3 = IN_3D(k  , new_j, new_i);
    t2_reg3 = IN_3D(k-1, new_j, new_i);

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
        t2_reg0 = t3_reg0;
        t2_reg1 = t3_reg1;
        t2_reg2 = t3_reg2;
        t2_reg3 = t3_reg3;

        new_i = (warp_id_x<<3) + lane_id%10;  
        new_j = (warp_id_y<<2) + lane_id/10;     
        t3_reg0 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+32)%10;
        new_j = (warp_id_y<<2) + (lane_id+32)/10;
        t3_reg1 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+64)%10;
        new_j = (warp_id_y<<2) + (lane_id+64)/10;
        t3_reg2 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+96)%10;
        new_j = (warp_id_y<<2) + (lane_id+96)/10;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
        t3_reg3 = IN_3D(k+1, new_j, new_i);

        friend_id0 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        sum0 += a0*((lane_id < 26)? tx0: ty0);
        sum1 += a0*((lane_id < 20)? tx1: ty1);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        sum0 += a9*((lane_id < 26)? tx0: ty0);
        sum1 += a9*((lane_id < 20)? tx1: ty1);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        sum0 += a18*((lane_id < 26)? tx0: ty0);
        sum1 += a18*((lane_id < 20)? tx1: ty1);

        friend_id0 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        sum0 += a1*((lane_id < 25)? tx0: ty0);
        sum1 += a1*((lane_id < 19)? tx1: ty1);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        sum0 += a10*((lane_id < 25)? tx0: ty0);
        sum1 += a10*((lane_id < 19)? tx1: ty1);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        sum0 += a19*((lane_id < 25)? tx0: ty0);
        sum1 += a19*((lane_id < 19)? tx1: ty1);

        friend_id0 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        sum0 += a2*((lane_id < 24)? tx0: ty0);
        sum1 += a2*((lane_id < 18)? tx1: ty1);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        sum0 += a11*((lane_id < 24)? tx0: ty0);
        sum1 += a11*((lane_id < 18)? tx1: ty1);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        sum0 += a20*((lane_id < 24)? tx0: ty0);
        sum1 += a20*((lane_id < 18)? tx1: ty1);

        friend_id0 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        sum0 += a3*((lane_id < 18)? tx0: ty0);
        sum1 += a3*((lane_id < 12)? tx1: ty1);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        sum0 += a12*((lane_id < 18)? tx0: ty0);
        sum1 += a12*((lane_id < 12)? tx1: ty1);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        sum0 += a21*((lane_id < 18)? tx0: ty0);
        sum1 += a21*((lane_id < 12)? tx1: ty1);

        friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        sum0 += a4*((lane_id < 17)? tx0: ty0);
        sum1 += a4*((lane_id < 11)? tx1: ty1);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        sum0 += a13*((lane_id < 17)? tx0: ty0);
        sum1 += a13*((lane_id < 11)? tx1: ty1);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        sum0 += a22*((lane_id < 17)? tx0: ty0);
        sum1 += a22*((lane_id < 11)? tx1: ty1);

        friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        sum0 += a5*((lane_id < 16)? tx0: ty0);
        sum1 += a5*((lane_id < 10)? tx1: ty1);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        sum0 += a14*((lane_id < 16)? tx0: ty0);
        sum1 += a14*((lane_id < 10)? tx1: ty1);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        sum0 += a23*((lane_id < 16)? tx0: ty0);
        sum1 += a23*((lane_id < 10)? tx1: ty1);

        friend_id0 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        tz1 = __shfl(t1_reg3, friend_id1);
        sum0 += a6*((lane_id < 10)? tx0: ty0);
        sum1 += a6*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        tz1 = __shfl(t2_reg3, friend_id1);
        sum0 += a15*((lane_id < 10)? tx0: ty0);
        sum1 += a15*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        tz1 = __shfl(t3_reg3, friend_id1);
        sum0 += a24*((lane_id < 10)? tx0: ty0);
        sum1 += a24*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));

        friend_id0 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        tz1 = __shfl(t1_reg3, friend_id1);
        sum0 += a7*((lane_id < 9 )? tx0: ty0);
        sum1 += a7*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        tz1 = __shfl(t2_reg3, friend_id1);
        sum0 += a16*((lane_id < 9 )? tx0: ty0);
        sum1 += a16*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        tz1 = __shfl(t3_reg3, friend_id1);
        sum0 += a25*((lane_id < 9 )? tx0: ty0);
        sum1 += a25*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
        
        friend_id0 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        tz1 = __shfl(t1_reg3, friend_id1);
        sum0 += a8*((lane_id < 8 )? tx0: ty0);
        sum1 += a8*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        tz1 = __shfl(t2_reg3, friend_id1);
        sum0 += a17*((lane_id < 8 )? tx0: ty0);
        sum1 += a17*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        tz1 = __shfl(t3_reg3, friend_id1);
        sum0 += a26*((lane_id < 8 )? tx0: ty0);
        sum1 += a26*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
        

        OUT_3D(k,j  ,i) = sum0;
        OUT_3D(k,j+4,i) = sum1;
    }
}

__global__ void Stencil_Cuda_Sweep_Shfl4_2DWarp(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, 
        DATA_TYPE a8, DATA_TYPE a9, DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, 
        DATA_TYPE a13, DATA_TYPE a14, DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, 
        DATA_TYPE a18, DATA_TYPE a19, DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, 
        DATA_TYPE a23, DATA_TYPE a24, DATA_TYPE a25, DATA_TYPE a26 , 
        int z, int m, int n, int halo)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<4) + (lane_id>>3) + halo;

    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
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
    DATA_TYPE t3_reg0, t3_reg1, t3_reg2, t3_reg3, t3_reg4, t3_reg5;
    DATA_TYPE t2_reg0, t2_reg1, t2_reg2, t2_reg3, t2_reg4, t2_reg5;
    DATA_TYPE t1_reg0, t1_reg1, t1_reg2, t1_reg3, t1_reg4, t1_reg5;

#define SM_2D_M2 32 
#define SM_2D_N2 8 

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;

    // t3 is current layer; t2 is previous layer
    new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
    new_j = (warp_id_y<<2) + lane_id/10;     
    t3_reg0 = IN_3D(k  , new_j, new_i);
    t2_reg0 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + (lane_id+32)/10;
    t3_reg1 = IN_3D(k  , new_j, new_i);
    t2_reg1 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10;
    new_j = (warp_id_y<<2) + (lane_id+64)/10;
    t3_reg2 = IN_3D(k  , new_j, new_i);
    t2_reg2 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10;
    new_j = (warp_id_y<<2) + (lane_id+96)/10;
    t3_reg3 = IN_3D(k  , new_j, new_i);
    t2_reg3 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10;
    new_j = (warp_id_y<<2) + (lane_id+128)/10;
    t3_reg4 = IN_3D(k  , new_j, new_i);
    t2_reg4 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10;
    new_j = (warp_id_y<<2) + (lane_id+160)/10;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    t3_reg5 = IN_3D(k  , new_j, new_i);
    t2_reg5 = IN_3D(k-1, new_j, new_i);

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

        t2_reg0 = t3_reg0;
        t2_reg1 = t3_reg1;
        t2_reg2 = t3_reg2;
        t2_reg3 = t3_reg3;
        t2_reg4 = t3_reg4;
        t2_reg5 = t3_reg5;

        new_i = (warp_id_x<<3) + lane_id%10;  
        new_j = (warp_id_y<<2) + lane_id/10;     
        t3_reg0 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+32)%10;
        new_j = (warp_id_y<<2) + (lane_id+32)/10;
        t3_reg1 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+64)%10;
        new_j = (warp_id_y<<2) + (lane_id+64)/10;
        t3_reg2 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+96)%10;
        new_j = (warp_id_y<<2) + (lane_id+96)/10;
        t3_reg3 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+128)%10;
        new_j = (warp_id_y<<2) + (lane_id+128)/10;
        t3_reg4 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+160)%10;
        new_j = (warp_id_y<<2) + (lane_id+160)/10;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo;
        t3_reg5 = IN_3D(k+1, new_j, new_i);


        friend_id0 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        tx2 = __shfl(t1_reg2, friend_id2);
        ty2 = __shfl(t1_reg3, friend_id2);
        tx3 = __shfl(t1_reg3, friend_id3);
        ty3 = __shfl(t1_reg4, friend_id3);
        sum0 += a0*((lane_id < 26)? tx0: ty0);
        sum1 += a0*((lane_id < 20)? tx1: ty1);
        sum2 += a0*((lane_id < 14)? tx2: ty2);
        sum3 += a0*((lane_id < 8 )? tx3: ty3);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        tx2 = __shfl(t2_reg2, friend_id2);
        ty2 = __shfl(t2_reg3, friend_id2);
        tx3 = __shfl(t2_reg3, friend_id3);
        ty3 = __shfl(t2_reg4, friend_id3);
        sum0 += a9*((lane_id < 26)? tx0: ty0);
        sum1 += a9*((lane_id < 20)? tx1: ty1);
        sum2 += a9*((lane_id < 14)? tx2: ty2);
        sum3 += a9*((lane_id < 8 )? tx3: ty3);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        tx2 = __shfl(t3_reg2, friend_id2);
        ty2 = __shfl(t3_reg3, friend_id2);
        tx3 = __shfl(t3_reg3, friend_id3);
        ty3 = __shfl(t3_reg4, friend_id3);
        sum0 += a18*((lane_id < 26)? tx0: ty0);
        sum1 += a18*((lane_id < 20)? tx1: ty1);
        sum2 += a18*((lane_id < 14)? tx2: ty2);
        sum3 += a18*((lane_id < 8 )? tx3: ty3);

        friend_id0 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        tx2 = __shfl(t1_reg2, friend_id2);
        ty2 = __shfl(t1_reg3, friend_id2);
        tx3 = __shfl(t1_reg3, friend_id3);
        ty3 = __shfl(t1_reg4, friend_id3);
        sum0 += a1*((lane_id < 25)? tx0: ty0);
        sum1 += a1*((lane_id < 19)? tx1: ty1);
        sum2 += a1*((lane_id < 13)? tx2: ty2);
        sum3 += a1*((lane_id < 7 )? tx3: ty3);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        tx2 = __shfl(t2_reg2, friend_id2);
        ty2 = __shfl(t2_reg3, friend_id2);
        tx3 = __shfl(t2_reg3, friend_id3);
        ty3 = __shfl(t2_reg4, friend_id3);
        sum0 += a10*((lane_id < 25)? tx0: ty0);
        sum1 += a10*((lane_id < 19)? tx1: ty1);
        sum2 += a10*((lane_id < 13)? tx2: ty2);
        sum3 += a10*((lane_id < 7 )? tx3: ty3);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        tx2 = __shfl(t3_reg2, friend_id2);
        ty2 = __shfl(t3_reg3, friend_id2);
        tx3 = __shfl(t3_reg3, friend_id3);
        ty3 = __shfl(t3_reg4, friend_id3);
        sum0 += a19*((lane_id < 25)? tx0: ty0);
        sum1 += a19*((lane_id < 19)? tx1: ty1);
        sum2 += a19*((lane_id < 13)? tx2: ty2);
        sum3 += a19*((lane_id < 7 )? tx3: ty3);

        friend_id0 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        tx2 = __shfl(t1_reg2, friend_id2);
        ty2 = __shfl(t1_reg3, friend_id2);
        tx3 = __shfl(t1_reg3, friend_id3);
        ty3 = __shfl(t1_reg4, friend_id3);
        sum0 += a2*((lane_id < 24)? tx0: ty0);
        sum1 += a2*((lane_id < 18)? tx1: ty1);
        sum2 += a2*((lane_id < 12)? tx2: ty2);
        sum3 += a2*((lane_id < 6 )? tx3: ty3);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        tx2 = __shfl(t2_reg2, friend_id2);
        ty2 = __shfl(t2_reg3, friend_id2);
        tx3 = __shfl(t2_reg3, friend_id3);
        ty3 = __shfl(t2_reg4, friend_id3);
        sum0 += a11*((lane_id < 24)? tx0: ty0);
        sum1 += a11*((lane_id < 18)? tx1: ty1);
        sum2 += a11*((lane_id < 12)? tx2: ty2);
        sum3 += a11*((lane_id < 6 )? tx3: ty3);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        tx2 = __shfl(t3_reg2, friend_id2);
        ty2 = __shfl(t3_reg3, friend_id2);
        tx3 = __shfl(t3_reg3, friend_id3);
        ty3 = __shfl(t3_reg4, friend_id3);
        sum0 += a20*((lane_id < 24)? tx0: ty0);
        sum1 += a20*((lane_id < 18)? tx1: ty1);
        sum2 += a20*((lane_id < 12)? tx2: ty2);
        sum3 += a20*((lane_id < 6 )? tx3: ty3);

        friend_id0 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        tx2 = __shfl(t1_reg2, friend_id2);
        ty2 = __shfl(t1_reg3, friend_id2);
        ty3 = __shfl(t1_reg4, friend_id3);
        tz3 = __shfl(t1_reg5, friend_id3);
        sum0 += a3*((lane_id < 18)? tx0: ty0);
        sum1 += a3*((lane_id < 12)? tx1: ty1);
        sum2 += a3*((lane_id < 6 )? tx2: ty2);
        sum3 += a3*((lane_id < 24)? ty3: tz3);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        tx2 = __shfl(t2_reg2, friend_id2);
        ty2 = __shfl(t2_reg3, friend_id2);
        ty3 = __shfl(t2_reg4, friend_id3);
        tz3 = __shfl(t2_reg5, friend_id3);
        sum0 += a12*((lane_id < 18)? tx0: ty0);
        sum1 += a12*((lane_id < 12)? tx1: ty1);
        sum2 += a12*((lane_id < 6 )? tx2: ty2);
        sum3 += a12*((lane_id < 24)? ty3: tz3);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        tx2 = __shfl(t3_reg2, friend_id2);
        ty2 = __shfl(t3_reg3, friend_id2);
        ty3 = __shfl(t3_reg4, friend_id3);
        tz3 = __shfl(t3_reg5, friend_id3);
        sum0 += a21*((lane_id < 18)? tx0: ty0);
        sum1 += a21*((lane_id < 12)? tx1: ty1);
        sum2 += a21*((lane_id < 6 )? tx2: ty2);
        sum3 += a21*((lane_id < 24)? ty3: tz3);

        friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        tx2 = __shfl(t1_reg2, friend_id2);
        ty2 = __shfl(t1_reg3, friend_id2);
        tz2 = __shfl(t1_reg4, friend_id2);
        ty3 = __shfl(t1_reg4, friend_id3);
        tz3 = __shfl(t1_reg5, friend_id3);
        sum0 += a4*((lane_id < 17)? tx0: ty0);
        sum1 += a4*((lane_id < 11)? tx1: ty1);
        sum2 += a4*((lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2));
        sum3 += a4*((lane_id < 24)? ty3: tz3);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        tx2 = __shfl(t2_reg2, friend_id2);
        ty2 = __shfl(t2_reg3, friend_id2);
        tz2 = __shfl(t2_reg4, friend_id2);
        ty3 = __shfl(t2_reg4, friend_id3);
        tz3 = __shfl(t2_reg5, friend_id3);
        sum0 += a13*((lane_id < 17)? tx0: ty0);
        sum1 += a13*((lane_id < 11)? tx1: ty1);
        sum2 += a13*((lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2));
        sum3 += a13*((lane_id < 24)? ty3: tz3);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        tx2 = __shfl(t3_reg2, friend_id2);
        ty2 = __shfl(t3_reg3, friend_id2);
        tz2 = __shfl(t3_reg4, friend_id2);
        ty3 = __shfl(t3_reg4, friend_id3);
        tz3 = __shfl(t3_reg5, friend_id3);
        sum0 += a22*((lane_id < 17)? tx0: ty0);
        sum1 += a22*((lane_id < 11)? tx1: ty1);
        sum2 += a22*((lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2));
        sum3 += a22*((lane_id < 24)? ty3: tz3);

        friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        tx2 = __shfl(t1_reg2, friend_id2);
        ty2 = __shfl(t1_reg3, friend_id2);
        tz2 = __shfl(t1_reg4, friend_id2);
        ty3 = __shfl(t1_reg4, friend_id3);
        tz3 = __shfl(t1_reg5, friend_id3);
        sum0 += a5*((lane_id < 16)? tx0: ty0);
        sum1 += a5*((lane_id < 10)? tx1: ty1);
        sum2 += a5*((lane_id < 4 )? tx2: ((lane_id < 30)? ty2: tz2));
        sum3 += a5*((lane_id < 24)? ty3: tz3);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        tx2 = __shfl(t2_reg2, friend_id2);
        ty2 = __shfl(t2_reg3, friend_id2);
        tz2 = __shfl(t2_reg4, friend_id2);
        ty3 = __shfl(t2_reg4, friend_id3);
        tz3 = __shfl(t2_reg5, friend_id3);
        sum0 += a14*((lane_id < 16)? tx0: ty0);
        sum1 += a14*((lane_id < 10)? tx1: ty1);
        sum2 += a14*((lane_id < 4 )? tx2: ((lane_id < 30)? ty2: tz2));
        sum3 += a14*((lane_id < 24)? ty3: tz3);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        tx2 = __shfl(t3_reg2, friend_id2);
        ty2 = __shfl(t3_reg3, friend_id2);
        tz2 = __shfl(t3_reg4, friend_id2);
        ty3 = __shfl(t3_reg4, friend_id3);
        tz3 = __shfl(t3_reg5, friend_id3);
        sum0 += a23*((lane_id < 16)? tx0: ty0);
        sum1 += a23*((lane_id < 10)? tx1: ty1);
        sum2 += a23*((lane_id < 4 )? tx2: ((lane_id < 30)? ty2: tz2));
        sum3 += a23*((lane_id < 24)? ty3: tz3);

        friend_id0 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        tz1 = __shfl(t1_reg3, friend_id1);
        ty2 = __shfl(t1_reg3, friend_id2);
        tz2 = __shfl(t1_reg4, friend_id2);
        ty3 = __shfl(t1_reg4, friend_id3);
        tz3 = __shfl(t1_reg5, friend_id3);
        sum0 += a6*((lane_id < 10)? tx0: ty0);
        sum1 += a6*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
        sum2 += a6*((lane_id < 24)? ty2: tz2);
        sum3 += a6*((lane_id < 16)? ty3: tz3);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        tz1 = __shfl(t2_reg3, friend_id1);
        ty2 = __shfl(t2_reg3, friend_id2);
        tz2 = __shfl(t2_reg4, friend_id2);
        ty3 = __shfl(t2_reg4, friend_id3);
        tz3 = __shfl(t2_reg5, friend_id3);
        sum0 += a15*((lane_id < 10)? tx0: ty0);
        sum1 += a15*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
        sum2 += a15*((lane_id < 24)? ty2: tz2);
        sum3 += a15*((lane_id < 16)? ty3: tz3);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        tz1 = __shfl(t3_reg3, friend_id1);
        ty2 = __shfl(t3_reg3, friend_id2);
        tz2 = __shfl(t3_reg4, friend_id2);
        ty3 = __shfl(t3_reg4, friend_id3);
        tz3 = __shfl(t3_reg5, friend_id3);
        sum0 += a24*((lane_id < 10)? tx0: ty0);
        sum1 += a24*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
        sum2 += a24*((lane_id < 24)? ty2: tz2);
        sum3 += a24*((lane_id < 16)? ty3: tz3);

        friend_id0 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        tz1 = __shfl(t1_reg3, friend_id1);
        ty2 = __shfl(t1_reg3, friend_id2);
        tz2 = __shfl(t1_reg4, friend_id2);
        ty3 = __shfl(t1_reg4, friend_id3);
        tz3 = __shfl(t1_reg5, friend_id3);
        sum0 += a7*((lane_id < 9 )? tx0: ty0);
        sum1 += a7*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
        sum2 += a7*((lane_id < 23)? ty2: tz2);
        sum3 += a7*((lane_id < 16)? ty3: tz3);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        tz1 = __shfl(t2_reg3, friend_id1);
        ty2 = __shfl(t2_reg3, friend_id2);
        tz2 = __shfl(t2_reg4, friend_id2);
        ty3 = __shfl(t2_reg4, friend_id3);
        tz3 = __shfl(t2_reg5, friend_id3);
        sum0 += a16*((lane_id < 9 )? tx0: ty0);
        sum1 += a16*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
        sum2 += a16*((lane_id < 23)? ty2: tz2);
        sum3 += a16*((lane_id < 16)? ty3: tz3);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        tz1 = __shfl(t3_reg3, friend_id1);
        ty2 = __shfl(t3_reg3, friend_id2);
        tz2 = __shfl(t3_reg4, friend_id2);
        ty3 = __shfl(t3_reg4, friend_id3);
        tz3 = __shfl(t3_reg5, friend_id3);
        sum0 += a25*((lane_id < 9 )? tx0: ty0);
        sum1 += a25*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
        sum2 += a25*((lane_id < 23)? ty2: tz2);
        sum3 += a25*((lane_id < 16)? ty3: tz3);
        
        friend_id0 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        tz1 = __shfl(t1_reg3, friend_id1);
        ty2 = __shfl(t1_reg3, friend_id2);
        tz2 = __shfl(t1_reg4, friend_id2);
        ty3 = __shfl(t1_reg4, friend_id3);
        tz3 = __shfl(t1_reg5, friend_id3);
        sum0 += a8*((lane_id < 8 )? tx0: ty0);
        sum1 += a8*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
        sum2 += a8*((lane_id < 22)? ty2: tz2);
        sum3 += a8*((lane_id < 16)? ty3: tz3);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        tz1 = __shfl(t2_reg3, friend_id1);
        ty2 = __shfl(t2_reg3, friend_id2);
        tz2 = __shfl(t2_reg4, friend_id2);
        ty3 = __shfl(t2_reg4, friend_id3);
        tz3 = __shfl(t2_reg5, friend_id3);
        sum0 += a17*((lane_id < 8 )? tx0: ty0);
        sum1 += a17*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
        sum2 += a17*((lane_id < 22)? ty2: tz2);
        sum3 += a17*((lane_id < 16)? ty3: tz3);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        tz1 = __shfl(t3_reg3, friend_id1);
        ty2 = __shfl(t3_reg3, friend_id2);
        tz2 = __shfl(t3_reg4, friend_id2);
        ty3 = __shfl(t3_reg4, friend_id3);
        tz3 = __shfl(t3_reg5, friend_id3);
        sum0 += a26*((lane_id < 8 )? tx0: ty0);
        sum1 += a26*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
        sum2 += a26*((lane_id < 22)? ty2: tz2);
        sum3 += a26*((lane_id < 16)? ty3: tz3);

        
        OUT_3D(k,j   ,i) = sum0;
        OUT_3D(k,j+4 ,i) = sum1;
        OUT_3D(k,j+8 ,i) = sum2;
        OUT_3D(k,j+12,i) = sum3;
    }
}

__global__ void Stencil_Cuda_Sweep_Shfl_1DWarp(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, 
        DATA_TYPE a8, DATA_TYPE a9, DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, 
        DATA_TYPE a13, DATA_TYPE a14, DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, 
        DATA_TYPE a18, DATA_TYPE a19, DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, 
        DATA_TYPE a23, DATA_TYPE a24, DATA_TYPE a25, DATA_TYPE a26 , 
        int z, int m, int n, int halo)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<0) + (lane_id>>5) + halo;

    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;
    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>5; // because the warp dimensions are 
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<0) + (lane_id>>5))>>0; // 1x1x32, warp_ids are division of 
    DATA_TYPE tx0, ty0, tz0;
    int friend_id0;
    int new_i, new_j;
    DATA_TYPE t3_reg0, t3_reg1, t3_reg2, t3_reg3;
    DATA_TYPE t2_reg0, t2_reg1, t2_reg2, t2_reg3;
    DATA_TYPE t1_reg0, t1_reg1, t1_reg2, t1_reg3;

    DATA_TYPE sum0 = 0.0;

    // t3 is current layer; t2 is previous layer
    new_i = (warp_id_x<<5) + lane_id%34;     // 10 is extended dimension of i
    new_j = (warp_id_y<<0) + lane_id/34;     
    t3_reg0 = IN_3D(k  , new_j, new_i);
    t2_reg0 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+32)%34;
    new_j = (warp_id_y<<0) + (lane_id+32)/34;
    t3_reg1 = IN_3D(k  , new_j, new_i);
    t2_reg1 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+64)%34;
    new_j = (warp_id_y<<0) + (lane_id+64)/34;
    t3_reg2 = IN_3D(k  , new_j, new_i);
    t2_reg2 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+96)%34;
    new_j = (warp_id_y<<0) + (lane_id+96)/34;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    t3_reg3 = IN_3D(k  , new_j, new_i);
    t2_reg3 = IN_3D(k-1, new_j, new_i);
    
    
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

        new_i = (warp_id_x<<5) + lane_id%34;  
        new_j = (warp_id_y<<0) + lane_id/34;     
        t3_reg0 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+32)%34;
        new_j = (warp_id_y<<0) + (lane_id+32)/34;
        t3_reg1 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+64)%34;
        new_j = (warp_id_y<<0) + (lane_id+64)/34;
        t3_reg2 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+96)%34;
        new_j = (warp_id_y<<0) + (lane_id+96)/34;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo;
        t3_reg3 = IN_3D(k+1, new_j, new_i);

        friend_id0 = (lane_id+0 )&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        sum0 += a0*tx0;
        tx0 = __shfl(t2_reg0, friend_id0);
        sum0 += a9*tx0;
        tx0 = __shfl(t3_reg0, friend_id0);
        sum0 += a18*tx0;

        
        friend_id0 = (lane_id+1 )&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        sum0 += a1*((lane_id < 31)? tx0: ty0);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        sum0 += a10*((lane_id < 31)? tx0: ty0);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        sum0 += a19*((lane_id < 31)? tx0: ty0);

        friend_id0 = (lane_id+2 )&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        sum0 += a2*((lane_id < 30)? tx0: ty0);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        sum0 += a11*((lane_id < 30)? tx0: ty0);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        sum0 += a20*((lane_id < 30)? tx0: ty0);

        friend_id0 = (lane_id+2 )&(warpSize-1);
        tx0 = __shfl(t1_reg1, friend_id0);
        ty0 = __shfl(t1_reg2, friend_id0);
        sum0 += a3*((lane_id < 30)? tx0: ty0);
        tx0 = __shfl(t2_reg1, friend_id0);
        ty0 = __shfl(t2_reg2, friend_id0);
        sum0 += a12*((lane_id < 30)? tx0: ty0);
        tx0 = __shfl(t3_reg1, friend_id0);
        ty0 = __shfl(t3_reg2, friend_id0);
        sum0 += a21*((lane_id < 30)? tx0: ty0);
        
        friend_id0 = (lane_id+3 )&(warpSize-1);
        tx0 = __shfl(t1_reg1, friend_id0);
        ty0 = __shfl(t1_reg2, friend_id0);
        sum0 += a4*((lane_id < 29)? tx0: ty0);
        tx0 = __shfl(t2_reg1, friend_id0);
        ty0 = __shfl(t2_reg2, friend_id0);
        sum0 += a13*((lane_id < 29)? tx0: ty0);
        tx0 = __shfl(t3_reg1, friend_id0);
        ty0 = __shfl(t3_reg2, friend_id0);
        sum0 += a22*((lane_id < 29)? tx0: ty0);

        friend_id0 = (lane_id+4 )&(warpSize-1);
        tx0 = __shfl(t1_reg1, friend_id0);
        ty0 = __shfl(t1_reg2, friend_id0);
        sum0 += a5*((lane_id < 28)? tx0: ty0);
        tx0 = __shfl(t2_reg1, friend_id0);
        ty0 = __shfl(t2_reg2, friend_id0);
        sum0 += a14*((lane_id < 28)? tx0: ty0);
        tx0 = __shfl(t3_reg1, friend_id0);
        ty0 = __shfl(t3_reg2, friend_id0);
        sum0 += a23*((lane_id < 28)? tx0: ty0);
       
        friend_id0 = (lane_id+4 )&(warpSize-1);
        tx0 = __shfl(t1_reg2, friend_id0);
        ty0 = __shfl(t1_reg3, friend_id0);
        sum0 += a6*((lane_id < 28)? tx0: ty0);
        tx0 = __shfl(t2_reg2, friend_id0);
        ty0 = __shfl(t2_reg3, friend_id0);
        sum0 += a15*((lane_id < 28)? tx0: ty0);
        tx0 = __shfl(t3_reg2, friend_id0);
        ty0 = __shfl(t3_reg3, friend_id0);
        sum0 += a24*((lane_id < 28)? tx0: ty0);


        friend_id0 = (lane_id+5 )&(warpSize-1);
        tx0 = __shfl(t1_reg2, friend_id0);
        ty0 = __shfl(t1_reg3, friend_id0);
        sum0 += a7*((lane_id < 27)? tx0: ty0);
        tx0 = __shfl(t2_reg2, friend_id0);
        ty0 = __shfl(t2_reg3, friend_id0);
        sum0 += a16*((lane_id < 27)? tx0: ty0);
        tx0 = __shfl(t3_reg2, friend_id0);
        ty0 = __shfl(t3_reg3, friend_id0);
        sum0 += a25*((lane_id < 27)? tx0: ty0);


        friend_id0 = (lane_id+6 )&(warpSize-1);
        tx0 = __shfl(t1_reg2, friend_id0);
        ty0 = __shfl(t1_reg3, friend_id0);
        sum0 += a8*((lane_id < 26)? tx0: ty0);
        tx0 = __shfl(t2_reg2, friend_id0);
        ty0 = __shfl(t2_reg3, friend_id0);
        sum0 += a17*((lane_id < 26)? tx0: ty0);
        tx0 = __shfl(t3_reg2, friend_id0);
        ty0 = __shfl(t3_reg3, friend_id0);
        sum0 += a26*((lane_id < 26)? tx0: ty0);

        OUT_3D(k,j  ,i) = sum0;
    }
}

__global__ void Stencil_Cuda_Sweep_Shfl2_1DWarp(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, 
        DATA_TYPE a8, DATA_TYPE a9, DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, 
        DATA_TYPE a13, DATA_TYPE a14, DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, 
        DATA_TYPE a18, DATA_TYPE a19, DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, 
        DATA_TYPE a23, DATA_TYPE a24, DATA_TYPE a25, DATA_TYPE a26 , 
        int z, int m, int n, int halo)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<1) + (lane_id>>5) + halo;

    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;
    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>5; // because the warp dimensions are 
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<1) + (lane_id>>5))>>0; // 1x1x32, warp_ids are division of 
    DATA_TYPE tx0, ty0, tz0;
    DATA_TYPE tx1, ty1, tz1;
    int friend_id0, friend_id1;
    int new_i, new_j;
    DATA_TYPE t3_reg0, t3_reg1, t3_reg2, t3_reg3, t3_reg4;
    DATA_TYPE t2_reg0, t2_reg1, t2_reg2, t2_reg3, t2_reg4;
    DATA_TYPE t1_reg0, t1_reg1, t1_reg2, t1_reg3, t1_reg4;

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;

    // t3 is current layer; t2 is previous layer
    new_i = (warp_id_x<<5) + lane_id%34;     // 10 is extended dimension of i
    new_j = (warp_id_y<<0) + lane_id/34;     
    t3_reg0 = IN_3D(k  , new_j, new_i);
    t2_reg0 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+32)%34;
    new_j = (warp_id_y<<0) + (lane_id+32)/34;
    t3_reg1 = IN_3D(k  , new_j, new_i);
    t2_reg1 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+64)%34;
    new_j = (warp_id_y<<0) + (lane_id+64)/34;
    t3_reg2 = IN_3D(k  , new_j, new_i);
    t2_reg2 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+96)%34;
    new_j = (warp_id_y<<0) + (lane_id+96)/34;
    t3_reg3 = IN_3D(k  , new_j, new_i);
    t2_reg3 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+128)%34;
    new_j = (warp_id_y<<0) + (lane_id+128)/34;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    t3_reg4 = IN_3D(k  , new_j, new_i);
    t2_reg4 = IN_3D(k-1, new_j, new_i);
    
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

        new_i = (warp_id_x<<5) + lane_id%34;  
        new_j = (warp_id_y<<0) + lane_id/34;     
        t3_reg0 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+32)%34;
        new_j = (warp_id_y<<0) + (lane_id+32)/34;
        t3_reg1 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+64)%34;
        new_j = (warp_id_y<<0) + (lane_id+64)/34;
        t3_reg2 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+96)%34;
        new_j = (warp_id_y<<0) + (lane_id+96)/34;
        t3_reg3 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+128)%34;
        new_j = (warp_id_y<<0) + (lane_id+128)/34;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo;
        t3_reg4 = IN_3D(k+1, new_j, new_i);
        

        friend_id0 = (lane_id+0 )&(warpSize-1);
        friend_id1 = (lane_id+2 )&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        sum0 += a0*tx0;
        sum1 += a0*((lane_id < 30)? tx1: ty1);
        tx0 = __shfl(t2_reg0, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        sum0 += a9*tx0;
        sum1 += a9*((lane_id < 30)? tx1: ty1);
        tx0 = __shfl(t3_reg0, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        sum0 += a18*tx0;
        sum1 += a18*((lane_id < 30)? tx1: ty1);

        
        friend_id0 = (lane_id+1 )&(warpSize-1);
        friend_id1 = (lane_id+3 )&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        sum0 += a1*((lane_id < 31)? tx0: ty0);
        sum1 += a1*((lane_id < 29)? tx1: ty1);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        sum0 += a10*((lane_id < 31)? tx0: ty0);
        sum1 += a10*((lane_id < 29)? tx1: ty1);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        sum0 += a19*((lane_id < 31)? tx0: ty0);
        sum1 += a19*((lane_id < 29)? tx1: ty1);

        friend_id0 = (lane_id+2 )&(warpSize-1);
        friend_id1 = (lane_id+4 )&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        sum0 += a2*((lane_id < 30)? tx0: ty0);
        sum1 += a2*((lane_id < 28)? tx1: ty1);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        sum0 += a11*((lane_id < 30)? tx0: ty0);
        sum1 += a11*((lane_id < 28)? tx1: ty1);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        sum0 += a20*((lane_id < 30)? tx0: ty0);
        sum1 += a20*((lane_id < 28)? tx1: ty1);

        friend_id0 = (lane_id+2 )&(warpSize-1);
        friend_id1 = (lane_id+4 )&(warpSize-1);
        tx0 = __shfl(t1_reg1, friend_id0);
        ty0 = __shfl(t1_reg2, friend_id0);
        tx1 = __shfl(t1_reg2, friend_id1);
        ty1 = __shfl(t1_reg3, friend_id1);
        sum0 += a3*((lane_id < 30)? tx0: ty0);
        sum1 += a3*((lane_id < 28)? tx1: ty1);
        tx0 = __shfl(t2_reg1, friend_id0);
        ty0 = __shfl(t2_reg2, friend_id0);
        tx1 = __shfl(t2_reg2, friend_id1);
        ty1 = __shfl(t2_reg3, friend_id1);
        sum0 += a12*((lane_id < 30)? tx0: ty0);
        sum1 += a12*((lane_id < 28)? tx1: ty1);
        tx0 = __shfl(t3_reg1, friend_id0);
        ty0 = __shfl(t3_reg2, friend_id0);
        tx1 = __shfl(t3_reg2, friend_id1);
        ty1 = __shfl(t3_reg3, friend_id1);
        sum0 += a21*((lane_id < 30)? tx0: ty0);
        sum1 += a21*((lane_id < 28)? tx1: ty1);
        
        friend_id0 = (lane_id+3 )&(warpSize-1);
        friend_id1 = (lane_id+5 )&(warpSize-1);
        tx0 = __shfl(t1_reg1, friend_id0);
        ty0 = __shfl(t1_reg2, friend_id0);
        tx1 = __shfl(t1_reg2, friend_id1);
        ty1 = __shfl(t1_reg3, friend_id1);
        sum0 += a4*((lane_id < 29)? tx0: ty0);
        sum1 += a4*((lane_id < 27)? tx1: ty1);
        tx0 = __shfl(t2_reg1, friend_id0);
        ty0 = __shfl(t2_reg2, friend_id0);
        tx1 = __shfl(t2_reg2, friend_id1);
        ty1 = __shfl(t2_reg3, friend_id1);
        sum0 += a13*((lane_id < 29)? tx0: ty0);
        sum1 += a13*((lane_id < 27)? tx1: ty1);
        tx0 = __shfl(t3_reg1, friend_id0);
        ty0 = __shfl(t3_reg2, friend_id0);
        tx1 = __shfl(t3_reg2, friend_id1);
        ty1 = __shfl(t3_reg3, friend_id1);
        sum0 += a22*((lane_id < 29)? tx0: ty0);
        sum1 += a22*((lane_id < 27)? tx1: ty1);

        friend_id0 = (lane_id+4 )&(warpSize-1);
        friend_id1 = (lane_id+6 )&(warpSize-1);
        tx0 = __shfl(t1_reg1, friend_id0);
        ty0 = __shfl(t1_reg2, friend_id0);
        tx1 = __shfl(t1_reg2, friend_id1);
        ty1 = __shfl(t1_reg3, friend_id1);
        sum0 += a5*((lane_id < 28)? tx0: ty0);
        sum1 += a5*((lane_id < 26)? tx1: ty1);
        tx0 = __shfl(t2_reg1, friend_id0);
        ty0 = __shfl(t2_reg2, friend_id0);
        tx1 = __shfl(t2_reg2, friend_id1);
        ty1 = __shfl(t2_reg3, friend_id1);
        sum0 += a14*((lane_id < 28)? tx0: ty0);
        sum1 += a14*((lane_id < 26)? tx1: ty1);
        tx0 = __shfl(t3_reg1, friend_id0);
        ty0 = __shfl(t3_reg2, friend_id0);
        tx1 = __shfl(t3_reg2, friend_id1);
        ty1 = __shfl(t3_reg3, friend_id1);
        sum0 += a23*((lane_id < 28)? tx0: ty0);
        sum1 += a23*((lane_id < 26)? tx1: ty1);
       
        friend_id0 = (lane_id+4 )&(warpSize-1);
        friend_id1 = (lane_id+6 )&(warpSize-1);
        tx0 = __shfl(t1_reg2, friend_id0);
        ty0 = __shfl(t1_reg3, friend_id0);
        tx1 = __shfl(t1_reg3, friend_id1);
        ty1 = __shfl(t1_reg4, friend_id1);
        sum0 += a6*((lane_id < 28)? tx0: ty0);
        sum1 += a6*((lane_id < 26)? tx1: ty1);
        tx0 = __shfl(t2_reg2, friend_id0);
        ty0 = __shfl(t2_reg3, friend_id0);
        tx1 = __shfl(t2_reg3, friend_id1);
        ty1 = __shfl(t2_reg4, friend_id1);
        sum0 += a15*((lane_id < 28)? tx0: ty0);
        sum1 += a15*((lane_id < 26)? tx1: ty1);
        tx0 = __shfl(t3_reg2, friend_id0);
        ty0 = __shfl(t3_reg3, friend_id0);
        tx1 = __shfl(t3_reg3, friend_id1);
        ty1 = __shfl(t3_reg4, friend_id1);
        sum0 += a24*((lane_id < 28)? tx0: ty0);
        sum1 += a24*((lane_id < 26)? tx1: ty1);


        friend_id0 = (lane_id+5 )&(warpSize-1);
        friend_id1 = (lane_id+7 )&(warpSize-1);
        tx0 = __shfl(t1_reg2, friend_id0);
        ty0 = __shfl(t1_reg3, friend_id0);
        tx1 = __shfl(t1_reg3, friend_id1);
        ty1 = __shfl(t1_reg4, friend_id1);
        sum0 += a7*((lane_id < 27)? tx0: ty0);
        sum1 += a7*((lane_id < 25)? tx1: ty1);
        tx0 = __shfl(t2_reg2, friend_id0);
        ty0 = __shfl(t2_reg3, friend_id0);
        tx1 = __shfl(t2_reg3, friend_id1);
        ty1 = __shfl(t2_reg4, friend_id1);
        sum0 += a16*((lane_id < 27)? tx0: ty0);
        sum1 += a16*((lane_id < 25)? tx1: ty1);
        tx0 = __shfl(t3_reg2, friend_id0);
        ty0 = __shfl(t3_reg3, friend_id0);
        tx1 = __shfl(t3_reg3, friend_id1);
        ty1 = __shfl(t3_reg4, friend_id1);
        sum0 += a25*((lane_id < 27)? tx0: ty0);
        sum1 += a25*((lane_id < 25)? tx1: ty1);


        friend_id0 = (lane_id+6 )&(warpSize-1);
        friend_id1 = (lane_id+8 )&(warpSize-1);
        tx0 = __shfl(t1_reg2, friend_id0);
        ty0 = __shfl(t1_reg3, friend_id0);
        tx1 = __shfl(t1_reg3, friend_id1);
        ty1 = __shfl(t1_reg4, friend_id1);
        sum0 += a8*((lane_id < 26)? tx0: ty0);
        sum1 += a8*((lane_id < 24)? tx1: ty1);
        tx0 = __shfl(t2_reg2, friend_id0);
        ty0 = __shfl(t2_reg3, friend_id0);
        tx1 = __shfl(t2_reg3, friend_id1);
        ty1 = __shfl(t2_reg4, friend_id1);
        sum0 += a17*((lane_id < 26)? tx0: ty0);
        sum1 += a17*((lane_id < 24)? tx1: ty1);
        tx0 = __shfl(t3_reg2, friend_id0);
        ty0 = __shfl(t3_reg3, friend_id0);
        tx1 = __shfl(t3_reg3, friend_id1);
        ty1 = __shfl(t3_reg4, friend_id1);
        sum0 += a26*((lane_id < 26)? tx0: ty0);
        sum1 += a26*((lane_id < 24)? tx1: ty1);

        OUT_3D(k,j  ,i) = sum0;
        OUT_3D(k,j+1,i) = sum1;
    }

}

__global__ void Stencil_Cuda_Sweep_Shfl4_1DWarp(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, 
        DATA_TYPE a8, DATA_TYPE a9, DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, 
        DATA_TYPE a13, DATA_TYPE a14, DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, 
        DATA_TYPE a18, DATA_TYPE a19, DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, 
        DATA_TYPE a23, DATA_TYPE a24, DATA_TYPE a25, DATA_TYPE a26 , 
        int z, int m, int n, int halo)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<2) + (lane_id>>5) + halo;

    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;
    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>5; // because the warp dimensions are 
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<2) + (lane_id>>5))>>0; // 1x1x32, warp_ids are division of 
    DATA_TYPE tx0, ty0, tz0;
    DATA_TYPE tx1, ty1, tz1;
    DATA_TYPE tx2, ty2, tz2;
    DATA_TYPE tx3, ty3, tz3;
    int friend_id0, friend_id1;
    int friend_id2, friend_id3;
    int new_i, new_j;
    DATA_TYPE t3_reg0, t3_reg1, t3_reg2, t3_reg3, t3_reg4, t3_reg5, t3_reg6;
    DATA_TYPE t2_reg0, t2_reg1, t2_reg2, t2_reg3, t2_reg4, t2_reg5, t2_reg6;
    DATA_TYPE t1_reg0, t1_reg1, t1_reg2, t1_reg3, t1_reg4, t1_reg5, t1_reg6;

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;

    // t3 is current layer; t2 is previous layer
    new_i = (warp_id_x<<5) + lane_id%34;     // 10 is extended dimension of i
    new_j = (warp_id_y<<0) + lane_id/34;     
    t3_reg0 = IN_3D(k  , new_j, new_i);
    t2_reg0 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+32)%34;
    new_j = (warp_id_y<<0) + (lane_id+32)/34;
    t3_reg1 = IN_3D(k  , new_j, new_i);
    t2_reg1 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+64)%34;
    new_j = (warp_id_y<<0) + (lane_id+64)/34;
    t3_reg2 = IN_3D(k  , new_j, new_i);
    t2_reg2 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+96)%34;
    new_j = (warp_id_y<<0) + (lane_id+96)/34;
    t3_reg3 = IN_3D(k  , new_j, new_i);
    t2_reg3 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+128)%34;
    new_j = (warp_id_y<<0) + (lane_id+128)/34;
    t3_reg4 = IN_3D(k  , new_j, new_i);
    t2_reg4 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+160)%34;
    new_j = (warp_id_y<<0) + (lane_id+160)/34;
    t3_reg5 = IN_3D(k  , new_j, new_i);
    t2_reg5 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+192)%34;
    new_j = (warp_id_y<<0) + (lane_id+192)/34;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    t3_reg6 = IN_3D(k  , new_j, new_i);
    t2_reg6 = IN_3D(k-1, new_j, new_i);

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

        new_i = (warp_id_x<<5) + lane_id%34;  
        new_j = (warp_id_y<<0) + lane_id/34;     
        t3_reg0 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+32)%34;
        new_j = (warp_id_y<<0) + (lane_id+32)/34;
        t3_reg1 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+64)%34;
        new_j = (warp_id_y<<0) + (lane_id+64)/34;
        t3_reg2 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+96)%34;
        new_j = (warp_id_y<<0) + (lane_id+96)/34;
        t3_reg3 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+128)%34;
        new_j = (warp_id_y<<0) + (lane_id+128)/34;
        t3_reg4 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+160)%34;
        new_j = (warp_id_y<<0) + (lane_id+160)/34;
        t3_reg5 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+192)%34;
        new_j = (warp_id_y<<0) + (lane_id+192)/34;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo;
        t3_reg6 = IN_3D(k+1, new_j, new_i);


        friend_id0 = (lane_id+0 )&(warpSize-1);
        friend_id1 = (lane_id+2 )&(warpSize-1);
        friend_id2 = (lane_id+4 )&(warpSize-1);
        friend_id3 = (lane_id+6 )&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        tx2 = __shfl(t1_reg2, friend_id2);
        ty2 = __shfl(t1_reg3, friend_id2);
        tx3 = __shfl(t1_reg3, friend_id3);
        ty3 = __shfl(t1_reg4, friend_id3);
        sum0 += a0*tx0;
        sum1 += a0*((lane_id < 30)? tx1: ty1);
        sum2 += a0*((lane_id < 28)? tx2: ty2);
        sum3 += a0*((lane_id < 26)? tx3: ty3);
        tx0 = __shfl(t2_reg0, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        tx2 = __shfl(t2_reg2, friend_id2);
        ty2 = __shfl(t2_reg3, friend_id2);
        tx3 = __shfl(t2_reg3, friend_id3);
        ty3 = __shfl(t2_reg4, friend_id3);
        sum0 += a9*tx0;
        sum1 += a9*((lane_id < 30)? tx1: ty1);
        sum2 += a9*((lane_id < 28)? tx2: ty2);
        sum3 += a9*((lane_id < 26)? tx3: ty3);
        tx0 = __shfl(t3_reg0, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        tx2 = __shfl(t3_reg2, friend_id2);
        ty2 = __shfl(t3_reg3, friend_id2);
        tx3 = __shfl(t3_reg3, friend_id3);
        ty3 = __shfl(t3_reg4, friend_id3);
        sum0 += a18*tx0;
        sum1 += a18*((lane_id < 30)? tx1: ty1);
        sum2 += a18*((lane_id < 28)? tx2: ty2);
        sum3 += a18*((lane_id < 26)? tx3: ty3);

        
        friend_id0 = (lane_id+1 )&(warpSize-1);
        friend_id1 = (lane_id+3 )&(warpSize-1);
        friend_id2 = (lane_id+5 )&(warpSize-1);
        friend_id3 = (lane_id+7 )&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        tx2 = __shfl(t1_reg2, friend_id2);
        ty2 = __shfl(t1_reg3, friend_id2);
        tx3 = __shfl(t1_reg3, friend_id3);
        ty3 = __shfl(t1_reg4, friend_id3);
        sum0 += a1*((lane_id < 31)? tx0: ty0);
        sum1 += a1*((lane_id < 29)? tx1: ty1);
        sum2 += a1*((lane_id < 27)? tx2: ty2);
        sum3 += a1*((lane_id < 25)? tx3: ty3);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        tx2 = __shfl(t2_reg2, friend_id2);
        ty2 = __shfl(t2_reg3, friend_id2);
        tx3 = __shfl(t2_reg3, friend_id3);
        ty3 = __shfl(t2_reg4, friend_id3);
        sum0 += a10*((lane_id < 31)? tx0: ty0);
        sum1 += a10*((lane_id < 29)? tx1: ty1);
        sum2 += a10*((lane_id < 27)? tx2: ty2);
        sum3 += a10*((lane_id < 25)? tx3: ty3);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        tx2 = __shfl(t3_reg2, friend_id2);
        ty2 = __shfl(t3_reg3, friend_id2);
        tx3 = __shfl(t3_reg3, friend_id3);
        ty3 = __shfl(t3_reg4, friend_id3);
        sum0 += a19*((lane_id < 31)? tx0: ty0);
        sum1 += a19*((lane_id < 29)? tx1: ty1);
        sum2 += a19*((lane_id < 27)? tx2: ty2);
        sum3 += a19*((lane_id < 25)? tx3: ty3);

        friend_id0 = (lane_id+2 )&(warpSize-1);
        friend_id1 = (lane_id+4 )&(warpSize-1);
        friend_id2 = (lane_id+6 )&(warpSize-1);
        friend_id3 = (lane_id+8 )&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        tx2 = __shfl(t1_reg2, friend_id2);
        ty2 = __shfl(t1_reg3, friend_id2);
        tx3 = __shfl(t1_reg3, friend_id3);
        ty3 = __shfl(t1_reg4, friend_id3);
        sum0 += a2*((lane_id < 30)? tx0: ty0);
        sum1 += a2*((lane_id < 28)? tx1: ty1);
        sum2 += a2*((lane_id < 26)? tx2: ty2);
        sum3 += a2*((lane_id < 24)? tx3: ty3);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        tx2 = __shfl(t2_reg2, friend_id2);
        ty2 = __shfl(t2_reg3, friend_id2);
        tx3 = __shfl(t2_reg3, friend_id3);
        ty3 = __shfl(t2_reg4, friend_id3);
        sum0 += a11*((lane_id < 30)? tx0: ty0);
        sum1 += a11*((lane_id < 28)? tx1: ty1);
        sum2 += a11*((lane_id < 26)? tx2: ty2);
        sum3 += a11*((lane_id < 24)? tx3: ty3);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        tx2 = __shfl(t3_reg2, friend_id2);
        ty2 = __shfl(t3_reg3, friend_id2);
        tx3 = __shfl(t3_reg3, friend_id3);
        ty3 = __shfl(t3_reg4, friend_id3);
        sum0 += a20*((lane_id < 30)? tx0: ty0);
        sum1 += a20*((lane_id < 28)? tx1: ty1);
        sum2 += a20*((lane_id < 26)? tx2: ty2);
        sum3 += a20*((lane_id < 24)? tx3: ty3);

        friend_id0 = (lane_id+2 )&(warpSize-1);
        friend_id1 = (lane_id+4 )&(warpSize-1);
        friend_id2 = (lane_id+6 )&(warpSize-1);
        friend_id3 = (lane_id+8 )&(warpSize-1);
        tx0 = __shfl(t1_reg1, friend_id0);
        ty0 = __shfl(t1_reg2, friend_id0);
        tx1 = __shfl(t1_reg2, friend_id1);
        ty1 = __shfl(t1_reg3, friend_id1);
        tx2 = __shfl(t1_reg3, friend_id2);
        ty2 = __shfl(t1_reg4, friend_id2);
        tx3 = __shfl(t1_reg4, friend_id3);
        ty3 = __shfl(t1_reg5, friend_id3);
        sum0 += a3*((lane_id < 30)? tx0: ty0);
        sum1 += a3*((lane_id < 28)? tx1: ty1);
        sum2 += a3*((lane_id < 26)? tx2: ty2);
        sum3 += a3*((lane_id < 24)? tx3: ty3);
        tx0 = __shfl(t2_reg1, friend_id0);
        ty0 = __shfl(t2_reg2, friend_id0);
        tx1 = __shfl(t2_reg2, friend_id1);
        ty1 = __shfl(t2_reg3, friend_id1);
        tx2 = __shfl(t2_reg3, friend_id2);
        ty2 = __shfl(t2_reg4, friend_id2);
        tx3 = __shfl(t2_reg4, friend_id3);
        ty3 = __shfl(t2_reg5, friend_id3);
        sum0 += a12*((lane_id < 30)? tx0: ty0);
        sum1 += a12*((lane_id < 28)? tx1: ty1);
        sum2 += a12*((lane_id < 26)? tx2: ty2);
        sum3 += a12*((lane_id < 24)? tx3: ty3);
        tx0 = __shfl(t3_reg1, friend_id0);
        ty0 = __shfl(t3_reg2, friend_id0);
        tx1 = __shfl(t3_reg2, friend_id1);
        ty1 = __shfl(t3_reg3, friend_id1);
        tx2 = __shfl(t3_reg3, friend_id2);
        ty2 = __shfl(t3_reg4, friend_id2);
        tx3 = __shfl(t3_reg4, friend_id3);
        ty3 = __shfl(t3_reg5, friend_id3);
        sum0 += a21*((lane_id < 30)? tx0: ty0);
        sum1 += a21*((lane_id < 28)? tx1: ty1);
        sum2 += a21*((lane_id < 26)? tx2: ty2);
        sum3 += a21*((lane_id < 24)? tx3: ty3);
        
        friend_id0 = (lane_id+3 )&(warpSize-1);
        friend_id1 = (lane_id+5 )&(warpSize-1);
        friend_id2 = (lane_id+7 )&(warpSize-1);
        friend_id3 = (lane_id+9 )&(warpSize-1);
        tx0 = __shfl(t1_reg1, friend_id0);
        ty0 = __shfl(t1_reg2, friend_id0);
        tx1 = __shfl(t1_reg2, friend_id1);
        ty1 = __shfl(t1_reg3, friend_id1);
        tx2 = __shfl(t1_reg3, friend_id2);
        ty2 = __shfl(t1_reg4, friend_id2);
        tx3 = __shfl(t1_reg4, friend_id3);
        ty3 = __shfl(t1_reg5, friend_id3);
        sum0 += a4*((lane_id < 29)? tx0: ty0);
        sum1 += a4*((lane_id < 27)? tx1: ty1);
        sum2 += a4*((lane_id < 25)? tx2: ty2);
        sum3 += a4*((lane_id < 23)? tx3: ty3);
        tx0 = __shfl(t2_reg1, friend_id0);
        ty0 = __shfl(t2_reg2, friend_id0);
        tx1 = __shfl(t2_reg2, friend_id1);
        ty1 = __shfl(t2_reg3, friend_id1);
        tx2 = __shfl(t2_reg3, friend_id2);
        ty2 = __shfl(t2_reg4, friend_id2);
        tx3 = __shfl(t2_reg4, friend_id3);
        ty3 = __shfl(t2_reg5, friend_id3);
        sum0 += a13*((lane_id < 29)? tx0: ty0);
        sum1 += a13*((lane_id < 27)? tx1: ty1);
        sum2 += a13*((lane_id < 25)? tx2: ty2);
        sum3 += a13*((lane_id < 23)? tx3: ty3);
        tx0 = __shfl(t3_reg1, friend_id0);
        ty0 = __shfl(t3_reg2, friend_id0);
        tx1 = __shfl(t3_reg2, friend_id1);
        ty1 = __shfl(t3_reg3, friend_id1);
        tx2 = __shfl(t3_reg3, friend_id2);
        ty2 = __shfl(t3_reg4, friend_id2);
        tx3 = __shfl(t3_reg4, friend_id3);
        ty3 = __shfl(t3_reg5, friend_id3);
        sum0 += a22*((lane_id < 29)? tx0: ty0);
        sum1 += a22*((lane_id < 27)? tx1: ty1);
        sum2 += a22*((lane_id < 25)? tx2: ty2);
        sum3 += a22*((lane_id < 23)? tx3: ty3);

        friend_id0 = (lane_id+4 )&(warpSize-1);
        friend_id1 = (lane_id+6 )&(warpSize-1);
        friend_id2 = (lane_id+8 )&(warpSize-1);
        friend_id3 = (lane_id+10)&(warpSize-1);
        tx0 = __shfl(t1_reg1, friend_id0);
        ty0 = __shfl(t1_reg2, friend_id0);
        tx1 = __shfl(t1_reg2, friend_id1);
        ty1 = __shfl(t1_reg3, friend_id1);
        tx2 = __shfl(t1_reg3, friend_id2);
        ty2 = __shfl(t1_reg4, friend_id2);
        tx3 = __shfl(t1_reg4, friend_id3);
        ty3 = __shfl(t1_reg5, friend_id3);
        sum0 += a5*((lane_id < 28)? tx0: ty0);
        sum1 += a5*((lane_id < 26)? tx1: ty1);
        sum2 += a5*((lane_id < 24)? tx2: ty2);
        sum3 += a5*((lane_id < 22)? tx3: ty3);
        tx0 = __shfl(t2_reg1, friend_id0);
        ty0 = __shfl(t2_reg2, friend_id0);
        tx1 = __shfl(t2_reg2, friend_id1);
        ty1 = __shfl(t2_reg3, friend_id1);
        tx2 = __shfl(t2_reg3, friend_id2);
        ty2 = __shfl(t2_reg4, friend_id2);
        tx3 = __shfl(t2_reg4, friend_id3);
        ty3 = __shfl(t2_reg5, friend_id3);
        sum0 += a14*((lane_id < 28)? tx0: ty0);
        sum1 += a14*((lane_id < 26)? tx1: ty1);
        sum2 += a14*((lane_id < 24)? tx2: ty2);
        sum3 += a14*((lane_id < 22)? tx3: ty3);
        tx0 = __shfl(t3_reg1, friend_id0);
        ty0 = __shfl(t3_reg2, friend_id0);
        tx1 = __shfl(t3_reg2, friend_id1);
        ty1 = __shfl(t3_reg3, friend_id1);
        tx2 = __shfl(t3_reg3, friend_id2);
        ty2 = __shfl(t3_reg4, friend_id2);
        tx3 = __shfl(t3_reg4, friend_id3);
        ty3 = __shfl(t3_reg5, friend_id3);
        sum0 += a23*((lane_id < 28)? tx0: ty0);
        sum1 += a23*((lane_id < 26)? tx1: ty1);
        sum2 += a23*((lane_id < 24)? tx2: ty2);
        sum3 += a23*((lane_id < 22)? tx3: ty3);
       
        friend_id0 = (lane_id+4 )&(warpSize-1);
        friend_id1 = (lane_id+6 )&(warpSize-1);
        friend_id2 = (lane_id+8 )&(warpSize-1);
        friend_id3 = (lane_id+10)&(warpSize-1);
        tx0 = __shfl(t1_reg2, friend_id0);
        ty0 = __shfl(t1_reg3, friend_id0);
        tx1 = __shfl(t1_reg3, friend_id1);
        ty1 = __shfl(t1_reg4, friend_id1);
        tx2 = __shfl(t1_reg4, friend_id2);
        ty2 = __shfl(t1_reg5, friend_id2);
        tx3 = __shfl(t1_reg5, friend_id3);
        ty3 = __shfl(t1_reg6, friend_id3);
        sum0 += a6*((lane_id < 28)? tx0: ty0);
        sum1 += a6*((lane_id < 26)? tx1: ty1);
        sum2 += a6*((lane_id < 24)? tx2: ty2);
        sum3 += a6*((lane_id < 22)? tx3: ty3);
        tx0 = __shfl(t2_reg2, friend_id0);
        ty0 = __shfl(t2_reg3, friend_id0);
        tx1 = __shfl(t2_reg3, friend_id1);
        ty1 = __shfl(t2_reg4, friend_id1);
        tx2 = __shfl(t2_reg4, friend_id2);
        ty2 = __shfl(t2_reg5, friend_id2);
        tx3 = __shfl(t2_reg5, friend_id3);
        ty3 = __shfl(t2_reg6, friend_id3);
        sum0 += a15*((lane_id < 28)? tx0: ty0);
        sum1 += a15*((lane_id < 26)? tx1: ty1);
        sum2 += a15*((lane_id < 24)? tx2: ty2);
        sum3 += a15*((lane_id < 22)? tx3: ty3);
        tx0 = __shfl(t3_reg2, friend_id0);
        ty0 = __shfl(t3_reg3, friend_id0);
        tx1 = __shfl(t3_reg3, friend_id1);
        ty1 = __shfl(t3_reg4, friend_id1);
        tx2 = __shfl(t3_reg4, friend_id2);
        ty2 = __shfl(t3_reg5, friend_id2);
        tx3 = __shfl(t3_reg5, friend_id3);
        ty3 = __shfl(t3_reg6, friend_id3);
        sum0 += a24*((lane_id < 28)? tx0: ty0);
        sum1 += a24*((lane_id < 26)? tx1: ty1);
        sum2 += a24*((lane_id < 24)? tx2: ty2);
        sum3 += a24*((lane_id < 22)? tx3: ty3);


        friend_id0 = (lane_id+5 )&(warpSize-1);
        friend_id1 = (lane_id+7 )&(warpSize-1);
        friend_id2 = (lane_id+9 )&(warpSize-1);
        friend_id3 = (lane_id+11)&(warpSize-1);
        tx0 = __shfl(t1_reg2, friend_id0);
        ty0 = __shfl(t1_reg3, friend_id0);
        tx1 = __shfl(t1_reg3, friend_id1);
        ty1 = __shfl(t1_reg4, friend_id1);
        tx2 = __shfl(t1_reg4, friend_id2);
        ty2 = __shfl(t1_reg5, friend_id2);
        tx3 = __shfl(t1_reg5, friend_id3);
        ty3 = __shfl(t1_reg6, friend_id3);
        sum0 += a7*((lane_id < 27)? tx0: ty0);
        sum1 += a7*((lane_id < 25)? tx1: ty1);
        sum2 += a7*((lane_id < 23)? tx2: ty2);
        sum3 += a7*((lane_id < 21)? tx3: ty3);
        tx0 = __shfl(t2_reg2, friend_id0);
        ty0 = __shfl(t2_reg3, friend_id0);
        tx1 = __shfl(t2_reg3, friend_id1);
        ty1 = __shfl(t2_reg4, friend_id1);
        tx2 = __shfl(t2_reg4, friend_id2);
        ty2 = __shfl(t2_reg5, friend_id2);
        tx3 = __shfl(t2_reg5, friend_id3);
        ty3 = __shfl(t2_reg6, friend_id3);
        sum0 += a16*((lane_id < 27)? tx0: ty0);
        sum1 += a16*((lane_id < 25)? tx1: ty1);
        sum2 += a16*((lane_id < 23)? tx2: ty2);
        sum3 += a16*((lane_id < 21)? tx3: ty3);
        tx0 = __shfl(t3_reg2, friend_id0);
        ty0 = __shfl(t3_reg3, friend_id0);
        tx1 = __shfl(t3_reg3, friend_id1);
        ty1 = __shfl(t3_reg4, friend_id1);
        tx2 = __shfl(t3_reg4, friend_id2);
        ty2 = __shfl(t3_reg5, friend_id2);
        tx3 = __shfl(t3_reg5, friend_id3);
        ty3 = __shfl(t3_reg6, friend_id3);
        sum0 += a25*((lane_id < 27)? tx0: ty0);
        sum1 += a25*((lane_id < 25)? tx1: ty1);
        sum2 += a25*((lane_id < 23)? tx2: ty2);
        sum3 += a25*((lane_id < 21)? tx3: ty3);


        friend_id0 = (lane_id+6 )&(warpSize-1);
        friend_id1 = (lane_id+8 )&(warpSize-1);
        friend_id2 = (lane_id+10)&(warpSize-1);
        friend_id3 = (lane_id+12)&(warpSize-1);
        tx0 = __shfl(t1_reg2, friend_id0);
        ty0 = __shfl(t1_reg3, friend_id0);
        tx1 = __shfl(t1_reg3, friend_id1);
        ty1 = __shfl(t1_reg4, friend_id1);
        tx2 = __shfl(t1_reg4, friend_id2);
        ty2 = __shfl(t1_reg5, friend_id2);
        tx3 = __shfl(t1_reg5, friend_id3);
        ty3 = __shfl(t1_reg6, friend_id3);
        sum0 += a8*((lane_id < 26)? tx0: ty0);
        sum1 += a8*((lane_id < 24)? tx1: ty1);
        sum2 += a8*((lane_id < 22)? tx2: ty2);
        sum3 += a8*((lane_id < 20)? tx3: ty3);
        tx0 = __shfl(t2_reg2, friend_id0);
        ty0 = __shfl(t2_reg3, friend_id0);
        tx1 = __shfl(t2_reg3, friend_id1);
        ty1 = __shfl(t2_reg4, friend_id1);
        tx2 = __shfl(t2_reg4, friend_id2);
        ty2 = __shfl(t2_reg5, friend_id2);
        tx3 = __shfl(t2_reg5, friend_id3);
        ty3 = __shfl(t2_reg6, friend_id3);
        sum0 += a17*((lane_id < 26)? tx0: ty0);
        sum1 += a17*((lane_id < 24)? tx1: ty1);
        sum2 += a17*((lane_id < 22)? tx2: ty2);
        sum3 += a17*((lane_id < 20)? tx3: ty3);
        tx0 = __shfl(t3_reg2, friend_id0);
        ty0 = __shfl(t3_reg3, friend_id0);
        tx1 = __shfl(t3_reg3, friend_id1);
        ty1 = __shfl(t3_reg4, friend_id1);
        tx2 = __shfl(t3_reg4, friend_id2);
        ty2 = __shfl(t3_reg5, friend_id2);
        tx3 = __shfl(t3_reg5, friend_id3);
        ty3 = __shfl(t3_reg6, friend_id3);
        sum0 += a26*((lane_id < 26)? tx0: ty0);
        sum1 += a26*((lane_id < 24)? tx1: ty1);
        sum2 += a26*((lane_id < 22)? tx2: ty2);
        sum3 += a26*((lane_id < 20)? tx3: ty3);

        OUT_3D(k,j  ,i) = sum0;
        OUT_3D(k,j+1,i) = sum1;
        OUT_3D(k,j+2,i) = sum2;
        OUT_3D(k,j+3,i) = sum3;
    }
}


__global__ void Stencil_Cuda_Sweep_Shfl4_1DWarpX(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, 
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE a7, 
        DATA_TYPE a8, DATA_TYPE a9, DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, 
        DATA_TYPE a13, DATA_TYPE a14, DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, 
        DATA_TYPE a18, DATA_TYPE a19, DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, 
        DATA_TYPE a23, DATA_TYPE a24, DATA_TYPE a25, DATA_TYPE a26 , 
        int z, int m, int n, int halo)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = (((threadIdx.x + blockIdx.x * blockDim.x)>>5)<<6) + (lane_id&31) + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<1) + (lane_id>>5) + halo;

    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;
    int warp_id_x = (i-halo)>>5; // because the warp dimensions are 
    int warp_id_y = (j-halo)>>0; // 1x1x32, warp_ids are division of 
    DATA_TYPE tx0, ty0, tz0;
    DATA_TYPE tx1, ty1, tz1;
    DATA_TYPE tx2, ty2, tz2;
    DATA_TYPE tx3, ty3, tz3;
    int friend_id0, friend_id1;
    int friend_id2, friend_id3;
    int new_i, new_j;
    DATA_TYPE t3_reg0, t3_reg1, t3_reg2, t3_reg3, t3_reg4, t3_reg5, t3_reg6, t3_reg7, t3_reg8;
    DATA_TYPE t2_reg0, t2_reg1, t2_reg2, t2_reg3, t2_reg4, t2_reg5, t2_reg6, t2_reg7, t2_reg8;
    DATA_TYPE t1_reg0, t1_reg1, t1_reg2, t1_reg3, t1_reg4, t1_reg5, t1_reg6, t1_reg7, t1_reg8;

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;

    // t3 is current layer; t2 is previous layer
    new_i = (warp_id_x<<5) + lane_id%66;     // 10 is extended dimension of i
    new_j = (warp_id_y<<0) + lane_id/66;     
    t3_reg0 = IN_3D(k  , new_j, new_i);
    t2_reg0 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+32)%66;
    new_j = (warp_id_y<<0) + (lane_id+32)/66;
    t3_reg1 = IN_3D(k  , new_j, new_i);
    t2_reg1 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+64)%66;
    new_j = (warp_id_y<<0) + (lane_id+64)/66;
    t3_reg2 = IN_3D(k  , new_j, new_i);
    t2_reg2 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+96)%66;
    new_j = (warp_id_y<<0) + (lane_id+96)/66;
    t3_reg3 = IN_3D(k  , new_j, new_i);
    t2_reg3 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+128)%66;
    new_j = (warp_id_y<<0) + (lane_id+128)/66;
    t3_reg4 = IN_3D(k  , new_j, new_i);
    t2_reg4 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+160)%66;
    new_j = (warp_id_y<<0) + (lane_id+160)/66;
    t3_reg5 = IN_3D(k  , new_j, new_i);
    t2_reg5 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+192)%66;
    new_j = (warp_id_y<<0) + (lane_id+192)/66;
    t3_reg6 = IN_3D(k  , new_j, new_i);
    t2_reg6 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+224)%66;
    new_j = (warp_id_y<<0) + (lane_id+224)/66;
    t3_reg7 = IN_3D(k  , new_j, new_i);
    t2_reg7 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<5) + (lane_id+256)%66;
    new_j = (warp_id_y<<0) + (lane_id+256)/66;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    t3_reg8 = IN_3D(k  , new_j, new_i);
    t2_reg8 = IN_3D(k-1, new_j, new_i);

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
        t1_reg7 = t2_reg7;
        t1_reg8 = t2_reg8;

        t2_reg0 = t3_reg0;
        t2_reg1 = t3_reg1;
        t2_reg2 = t3_reg2;
        t2_reg3 = t3_reg3;
        t2_reg4 = t3_reg4;
        t2_reg5 = t3_reg5;
        t2_reg6 = t3_reg6;
        t2_reg7 = t3_reg7;
        t2_reg8 = t3_reg8;

        new_i = (warp_id_x<<5) + lane_id%66;  
        new_j = (warp_id_y<<0) + lane_id/66;     
        t3_reg0 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+32)%66;
        new_j = (warp_id_y<<0) + (lane_id+32)/66;
        t3_reg1 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+64)%66;
        new_j = (warp_id_y<<0) + (lane_id+64)/66;
        t3_reg2 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+96)%66;
        new_j = (warp_id_y<<0) + (lane_id+96)/66;
        t3_reg3 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+128)%66;
        new_j = (warp_id_y<<0) + (lane_id+128)/66;
        t3_reg4 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+160)%66;
        new_j = (warp_id_y<<0) + (lane_id+160)/66;
        t3_reg5 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+192)%66;
        new_j = (warp_id_y<<0) + (lane_id+192)/66;
        t3_reg6 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+224)%66;
        new_j = (warp_id_y<<0) + (lane_id+224)/66;
        t3_reg7 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<5) + (lane_id+256)%66;
        new_j = (warp_id_y<<0) + (lane_id+256)/66;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo;
        t3_reg8 = IN_3D(k+1, new_j, new_i);


        friend_id0 = (lane_id+0 )&(warpSize-1);
        friend_id1 = (lane_id+0 )&(warpSize-1);
        friend_id2 = (lane_id+2 )&(warpSize-1);
        friend_id3 = (lane_id+2 )&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        tx2 = __shfl(t1_reg2, friend_id2);
        ty2 = __shfl(t1_reg3, friend_id2);
        tx3 = __shfl(t1_reg3, friend_id3);
        ty3 = __shfl(t1_reg4, friend_id3);
        sum0 += a0*tx0;
        sum1 += a0*tx1;
        sum2 += a0*((lane_id < 30)? tx2: ty2);
        sum3 += a0*((lane_id < 30)? tx3: ty3);
        tx0 = __shfl(t2_reg0, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        tx2 = __shfl(t2_reg2, friend_id2);
        ty2 = __shfl(t2_reg3, friend_id2);
        tx3 = __shfl(t2_reg3, friend_id3);
        ty3 = __shfl(t2_reg4, friend_id3);
        sum0 += a9*tx0;
        sum1 += a9*tx1;
        sum2 += a9*((lane_id < 30)? tx2: ty2);
        sum3 += a9*((lane_id < 30)? tx3: ty3);
        tx0 = __shfl(t3_reg0, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        tx2 = __shfl(t3_reg2, friend_id2);
        ty2 = __shfl(t3_reg3, friend_id2);
        tx3 = __shfl(t3_reg3, friend_id3);
        ty3 = __shfl(t3_reg4, friend_id3);
        sum0 += a18*tx0;
        sum1 += a18*tx1;
        sum2 += a18*((lane_id < 30)? tx2: ty2);
        sum3 += a18*((lane_id < 30)? tx3: ty3);

        
        friend_id0 = (lane_id+1 )&(warpSize-1);
        friend_id1 = (lane_id+1 )&(warpSize-1);
        friend_id2 = (lane_id+3 )&(warpSize-1);
        friend_id3 = (lane_id+3 )&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        tx2 = __shfl(t1_reg2, friend_id2);
        ty2 = __shfl(t1_reg3, friend_id2);
        tx3 = __shfl(t1_reg3, friend_id3);
        ty3 = __shfl(t1_reg4, friend_id3);
        sum0 += a1*((lane_id < 31)? tx0: ty0);
        sum1 += a1*((lane_id < 31)? tx1: ty1);
        sum2 += a1*((lane_id < 29)? tx2: ty2);
        sum3 += a1*((lane_id < 29)? tx3: ty3);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        tx2 = __shfl(t2_reg2, friend_id2);
        ty2 = __shfl(t2_reg3, friend_id2);
        tx3 = __shfl(t2_reg3, friend_id3);
        ty3 = __shfl(t2_reg4, friend_id3);
        sum0 += a10*((lane_id < 31)? tx0: ty0);
        sum1 += a10*((lane_id < 31)? tx1: ty1);
        sum2 += a10*((lane_id < 29)? tx2: ty2);
        sum3 += a10*((lane_id < 29)? tx3: ty3);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        tx2 = __shfl(t3_reg2, friend_id2);
        ty2 = __shfl(t3_reg3, friend_id2);
        tx3 = __shfl(t3_reg3, friend_id3);
        ty3 = __shfl(t3_reg4, friend_id3);
        sum0 += a19*((lane_id < 31)? tx0: ty0);
        sum1 += a19*((lane_id < 31)? tx1: ty1);
        sum2 += a19*((lane_id < 29)? tx2: ty2);
        sum3 += a19*((lane_id < 29)? tx3: ty3);

        friend_id0 = (lane_id+2 )&(warpSize-1);
        friend_id1 = (lane_id+2 )&(warpSize-1);
        friend_id2 = (lane_id+4 )&(warpSize-1);
        friend_id3 = (lane_id+4 )&(warpSize-1);
        tx0 = __shfl(t1_reg0, friend_id0);
        ty0 = __shfl(t1_reg1, friend_id0);
        tx1 = __shfl(t1_reg1, friend_id1);
        ty1 = __shfl(t1_reg2, friend_id1);
        tx2 = __shfl(t1_reg2, friend_id2);
        ty2 = __shfl(t1_reg3, friend_id2);
        tx3 = __shfl(t1_reg3, friend_id3);
        ty3 = __shfl(t1_reg4, friend_id3);
        sum0 += a2*((lane_id < 30)? tx0: ty0);
        sum1 += a2*((lane_id < 30)? tx1: ty1);
        sum2 += a2*((lane_id < 28)? tx2: ty2);
        sum3 += a2*((lane_id < 28)? tx3: ty3);
        tx0 = __shfl(t2_reg0, friend_id0);
        ty0 = __shfl(t2_reg1, friend_id0);
        tx1 = __shfl(t2_reg1, friend_id1);
        ty1 = __shfl(t2_reg2, friend_id1);
        tx2 = __shfl(t2_reg2, friend_id2);
        ty2 = __shfl(t2_reg3, friend_id2);
        tx3 = __shfl(t2_reg3, friend_id3);
        ty3 = __shfl(t2_reg4, friend_id3);
        sum0 += a11*((lane_id < 30)? tx0: ty0);
        sum1 += a11*((lane_id < 30)? tx1: ty1);
        sum2 += a11*((lane_id < 28)? tx2: ty2);
        sum3 += a11*((lane_id < 28)? tx3: ty3);
        tx0 = __shfl(t3_reg0, friend_id0);
        ty0 = __shfl(t3_reg1, friend_id0);
        tx1 = __shfl(t3_reg1, friend_id1);
        ty1 = __shfl(t3_reg2, friend_id1);
        tx2 = __shfl(t3_reg2, friend_id2);
        ty2 = __shfl(t3_reg3, friend_id2);
        tx3 = __shfl(t3_reg3, friend_id3);
        ty3 = __shfl(t3_reg4, friend_id3);
        sum0 += a20*((lane_id < 30)? tx0: ty0);
        sum1 += a20*((lane_id < 30)? tx1: ty1);
        sum2 += a20*((lane_id < 28)? tx2: ty2);
        sum3 += a20*((lane_id < 28)? tx3: ty3);

        friend_id0 = (lane_id+2 )&(warpSize-1);
        friend_id1 = (lane_id+2 )&(warpSize-1);
        friend_id2 = (lane_id+4 )&(warpSize-1);
        friend_id3 = (lane_id+4 )&(warpSize-1);
        tx0 = __shfl(t1_reg2, friend_id0);
        ty0 = __shfl(t1_reg3, friend_id0);
        tx1 = __shfl(t1_reg3, friend_id1);
        ty1 = __shfl(t1_reg4, friend_id1);
        tx2 = __shfl(t1_reg4, friend_id2);
        ty2 = __shfl(t1_reg5, friend_id2);
        tx3 = __shfl(t1_reg5, friend_id3);
        ty3 = __shfl(t1_reg6, friend_id3);
        sum0 += a3*((lane_id < 30)? tx0: ty0);
        sum1 += a3*((lane_id < 30)? tx1: ty1);
        sum2 += a3*((lane_id < 28)? tx2: ty2);
        sum3 += a3*((lane_id < 28)? tx3: ty3);
        tx0 = __shfl(t2_reg2, friend_id0);
        ty0 = __shfl(t2_reg3, friend_id0);
        tx1 = __shfl(t2_reg3, friend_id1);
        ty1 = __shfl(t2_reg4, friend_id1);
        tx2 = __shfl(t2_reg4, friend_id2);
        ty2 = __shfl(t2_reg5, friend_id2);
        tx3 = __shfl(t2_reg5, friend_id3);
        ty3 = __shfl(t2_reg6, friend_id3);
        sum0 += a12*((lane_id < 30)? tx0: ty0);
        sum1 += a12*((lane_id < 30)? tx1: ty1);
        sum2 += a12*((lane_id < 28)? tx2: ty2);
        sum3 += a12*((lane_id < 28)? tx3: ty3);
        tx0 = __shfl(t3_reg2, friend_id0);
        ty0 = __shfl(t3_reg3, friend_id0);
        tx1 = __shfl(t3_reg3, friend_id1);
        ty1 = __shfl(t3_reg4, friend_id1);
        tx2 = __shfl(t3_reg4, friend_id2);
        ty2 = __shfl(t3_reg5, friend_id2);
        tx3 = __shfl(t3_reg5, friend_id3);
        ty3 = __shfl(t3_reg6, friend_id3);
        sum0 += a21*((lane_id < 30)? tx0: ty0);
        sum1 += a21*((lane_id < 30)? tx1: ty1);
        sum2 += a21*((lane_id < 28)? tx2: ty2);
        sum3 += a21*((lane_id < 28)? tx3: ty3);
        
        friend_id0 = (lane_id+3 )&(warpSize-1);
        friend_id1 = (lane_id+3 )&(warpSize-1);
        friend_id2 = (lane_id+5 )&(warpSize-1);
        friend_id3 = (lane_id+5 )&(warpSize-1);
        tx0 = __shfl(t1_reg2, friend_id0);
        ty0 = __shfl(t1_reg3, friend_id0);
        tx1 = __shfl(t1_reg3, friend_id1);
        ty1 = __shfl(t1_reg4, friend_id1);
        tx2 = __shfl(t1_reg4, friend_id2);
        ty2 = __shfl(t1_reg5, friend_id2);
        tx3 = __shfl(t1_reg5, friend_id3);
        ty3 = __shfl(t1_reg6, friend_id3);
        sum0 += a4*((lane_id < 29)? tx0: ty0);
        sum1 += a4*((lane_id < 29)? tx1: ty1);
        sum2 += a4*((lane_id < 27)? tx2: ty2);
        sum3 += a4*((lane_id < 27)? tx3: ty3);
        tx0 = __shfl(t2_reg2, friend_id0);
        ty0 = __shfl(t2_reg3, friend_id0);
        tx1 = __shfl(t2_reg3, friend_id1);
        ty1 = __shfl(t2_reg4, friend_id1);
        tx2 = __shfl(t2_reg4, friend_id2);
        ty2 = __shfl(t2_reg5, friend_id2);
        tx3 = __shfl(t2_reg5, friend_id3);
        ty3 = __shfl(t2_reg6, friend_id3);
        sum0 += a13*((lane_id < 29)? tx0: ty0);
        sum1 += a13*((lane_id < 29)? tx1: ty1);
        sum2 += a13*((lane_id < 27)? tx2: ty2);
        sum3 += a13*((lane_id < 27)? tx3: ty3);
        tx0 = __shfl(t3_reg2, friend_id0);
        ty0 = __shfl(t3_reg3, friend_id0);
        tx1 = __shfl(t3_reg3, friend_id1);
        ty1 = __shfl(t3_reg4, friend_id1);
        tx2 = __shfl(t3_reg4, friend_id2);
        ty2 = __shfl(t3_reg5, friend_id2);
        tx3 = __shfl(t3_reg5, friend_id3);
        ty3 = __shfl(t3_reg6, friend_id3);
        sum0 += a22*((lane_id < 29)? tx0: ty0);
        sum1 += a22*((lane_id < 29)? tx1: ty1);
        sum2 += a22*((lane_id < 27)? tx2: ty2);
        sum3 += a22*((lane_id < 27)? tx3: ty3);

        friend_id0 = (lane_id+4 )&(warpSize-1);
        friend_id1 = (lane_id+4 )&(warpSize-1);
        friend_id2 = (lane_id+6 )&(warpSize-1);
        friend_id3 = (lane_id+6 )&(warpSize-1);
        tx0 = __shfl(t1_reg2, friend_id0);
        ty0 = __shfl(t1_reg3, friend_id0);
        tx1 = __shfl(t1_reg3, friend_id1);
        ty1 = __shfl(t1_reg4, friend_id1);
        tx2 = __shfl(t1_reg4, friend_id2);
        ty2 = __shfl(t1_reg5, friend_id2);
        tx3 = __shfl(t1_reg5, friend_id3);
        ty3 = __shfl(t1_reg6, friend_id3);
        sum0 += a5*((lane_id < 28)? tx0: ty0);
        sum1 += a5*((lane_id < 28)? tx1: ty1);
        sum2 += a5*((lane_id < 26)? tx2: ty2);
        sum3 += a5*((lane_id < 26)? tx3: ty3);
        tx0 = __shfl(t2_reg2, friend_id0);
        ty0 = __shfl(t2_reg3, friend_id0);
        tx1 = __shfl(t2_reg3, friend_id1);
        ty1 = __shfl(t2_reg4, friend_id1);
        tx2 = __shfl(t2_reg4, friend_id2);
        ty2 = __shfl(t2_reg5, friend_id2);
        tx3 = __shfl(t2_reg5, friend_id3);
        ty3 = __shfl(t2_reg6, friend_id3);
        sum0 += a14*((lane_id < 28)? tx0: ty0);
        sum1 += a14*((lane_id < 28)? tx1: ty1);
        sum2 += a14*((lane_id < 26)? tx2: ty2);
        sum3 += a14*((lane_id < 26)? tx3: ty3);
        tx0 = __shfl(t3_reg2, friend_id0);
        ty0 = __shfl(t3_reg3, friend_id0);
        tx1 = __shfl(t3_reg3, friend_id1);
        ty1 = __shfl(t3_reg4, friend_id1);
        tx2 = __shfl(t3_reg4, friend_id2);
        ty2 = __shfl(t3_reg5, friend_id2);
        tx3 = __shfl(t3_reg5, friend_id3);
        ty3 = __shfl(t3_reg6, friend_id3);
        sum0 += a23*((lane_id < 28)? tx0: ty0);
        sum1 += a23*((lane_id < 28)? tx1: ty1);
        sum2 += a23*((lane_id < 26)? tx2: ty2);
        sum3 += a23*((lane_id < 26)? tx3: ty3);
       
        friend_id0 = (lane_id+4 )&(warpSize-1);
        friend_id1 = (lane_id+4 )&(warpSize-1);
        friend_id2 = (lane_id+6 )&(warpSize-1);
        friend_id3 = (lane_id+6 )&(warpSize-1);
        tx0 = __shfl(t1_reg4, friend_id0);
        ty0 = __shfl(t1_reg5, friend_id0);
        tx1 = __shfl(t1_reg5, friend_id1);
        ty1 = __shfl(t1_reg6, friend_id1);
        tx2 = __shfl(t1_reg6, friend_id2);
        ty2 = __shfl(t1_reg7, friend_id2);
        tx3 = __shfl(t1_reg7, friend_id3);
        ty3 = __shfl(t1_reg8, friend_id3);
        sum0 += a6*((lane_id < 28)? tx0: ty0);
        sum1 += a6*((lane_id < 28)? tx1: ty1);
        sum2 += a6*((lane_id < 26)? tx2: ty2);
        sum3 += a6*((lane_id < 26)? tx3: ty3);
        tx0 = __shfl(t2_reg4, friend_id0);
        ty0 = __shfl(t2_reg5, friend_id0);
        tx1 = __shfl(t2_reg5, friend_id1);
        ty1 = __shfl(t2_reg6, friend_id1);
        tx2 = __shfl(t2_reg6, friend_id2);
        ty2 = __shfl(t2_reg7, friend_id2);
        tx3 = __shfl(t2_reg7, friend_id3);
        ty3 = __shfl(t2_reg8, friend_id3);
        sum0 += a15*((lane_id < 28)? tx0: ty0);
        sum1 += a15*((lane_id < 28)? tx1: ty1);
        sum2 += a15*((lane_id < 26)? tx2: ty2);
        sum3 += a15*((lane_id < 26)? tx3: ty3);
        tx0 = __shfl(t3_reg4, friend_id0);
        ty0 = __shfl(t3_reg5, friend_id0);
        tx1 = __shfl(t3_reg5, friend_id1);
        ty1 = __shfl(t3_reg6, friend_id1);
        tx2 = __shfl(t3_reg6, friend_id2);
        ty2 = __shfl(t3_reg7, friend_id2);
        tx3 = __shfl(t3_reg7, friend_id3);
        ty3 = __shfl(t3_reg8, friend_id3);
        sum0 += a24*((lane_id < 28)? tx0: ty0);
        sum1 += a24*((lane_id < 28)? tx1: ty1);
        sum2 += a24*((lane_id < 26)? tx2: ty2);
        sum3 += a24*((lane_id < 26)? tx3: ty3);


        friend_id0 = (lane_id+5 )&(warpSize-1);
        friend_id1 = (lane_id+5 )&(warpSize-1);
        friend_id2 = (lane_id+7 )&(warpSize-1);
        friend_id3 = (lane_id+7 )&(warpSize-1);
        tx0 = __shfl(t1_reg4, friend_id0);
        ty0 = __shfl(t1_reg5, friend_id0);
        tx1 = __shfl(t1_reg5, friend_id1);
        ty1 = __shfl(t1_reg6, friend_id1);
        tx2 = __shfl(t1_reg6, friend_id2);
        ty2 = __shfl(t1_reg7, friend_id2);
        tx3 = __shfl(t1_reg7, friend_id3);
        ty3 = __shfl(t1_reg8, friend_id3);
        sum0 += a7*((lane_id < 27)? tx0: ty0);
        sum1 += a7*((lane_id < 27)? tx1: ty1);
        sum2 += a7*((lane_id < 25)? tx2: ty2);
        sum3 += a7*((lane_id < 25)? tx3: ty3);
        tx0 = __shfl(t2_reg4, friend_id0);
        ty0 = __shfl(t2_reg5, friend_id0);
        tx1 = __shfl(t2_reg5, friend_id1);
        ty1 = __shfl(t2_reg6, friend_id1);
        tx2 = __shfl(t2_reg6, friend_id2);
        ty2 = __shfl(t2_reg7, friend_id2);
        tx3 = __shfl(t2_reg7, friend_id3);
        ty3 = __shfl(t2_reg8, friend_id3);
        sum0 += a16*((lane_id < 27)? tx0: ty0);
        sum1 += a16*((lane_id < 27)? tx1: ty1);
        sum2 += a16*((lane_id < 25)? tx2: ty2);
        sum3 += a16*((lane_id < 25)? tx3: ty3);
        tx0 = __shfl(t3_reg4, friend_id0);
        ty0 = __shfl(t3_reg5, friend_id0);
        tx1 = __shfl(t3_reg5, friend_id1);
        ty1 = __shfl(t3_reg6, friend_id1);
        tx2 = __shfl(t3_reg6, friend_id2);
        ty2 = __shfl(t3_reg7, friend_id2);
        tx3 = __shfl(t3_reg7, friend_id3);
        ty3 = __shfl(t3_reg8, friend_id3);
        sum0 += a25*((lane_id < 27)? tx0: ty0);
        sum1 += a25*((lane_id < 27)? tx1: ty1);
        sum2 += a25*((lane_id < 25)? tx2: ty2);
        sum3 += a25*((lane_id < 25)? tx3: ty3);


        friend_id0 = (lane_id+6 )&(warpSize-1);
        friend_id1 = (lane_id+6 )&(warpSize-1);
        friend_id2 = (lane_id+8 )&(warpSize-1);
        friend_id3 = (lane_id+8 )&(warpSize-1);
        tx0 = __shfl(t1_reg4, friend_id0);
        ty0 = __shfl(t1_reg5, friend_id0);
        tx1 = __shfl(t1_reg5, friend_id1);
        ty1 = __shfl(t1_reg6, friend_id1);
        tx2 = __shfl(t1_reg6, friend_id2);
        ty2 = __shfl(t1_reg7, friend_id2);
        tx3 = __shfl(t1_reg7, friend_id3);
        ty3 = __shfl(t1_reg8, friend_id3);
        sum0 += a8*((lane_id < 26)? tx0: ty0);
        sum1 += a8*((lane_id < 26)? tx1: ty1);
        sum2 += a8*((lane_id < 24)? tx2: ty2);
        sum3 += a8*((lane_id < 24)? tx3: ty3);
        tx0 = __shfl(t2_reg4, friend_id0);
        ty0 = __shfl(t2_reg5, friend_id0);
        tx1 = __shfl(t2_reg5, friend_id1);
        ty1 = __shfl(t2_reg6, friend_id1);
        tx2 = __shfl(t2_reg6, friend_id2);
        ty2 = __shfl(t2_reg7, friend_id2);
        tx3 = __shfl(t2_reg7, friend_id3);
        ty3 = __shfl(t2_reg8, friend_id3);
        sum0 += a17*((lane_id < 26)? tx0: ty0);
        sum1 += a17*((lane_id < 26)? tx1: ty1);
        sum2 += a17*((lane_id < 24)? tx2: ty2);
        sum3 += a17*((lane_id < 24)? tx3: ty3);
        tx0 = __shfl(t3_reg4, friend_id0);
        ty0 = __shfl(t3_reg5, friend_id0);
        tx1 = __shfl(t3_reg5, friend_id1);
        ty1 = __shfl(t3_reg6, friend_id1);
        tx2 = __shfl(t3_reg6, friend_id2);
        ty2 = __shfl(t3_reg7, friend_id2);
        tx3 = __shfl(t3_reg7, friend_id3);
        ty3 = __shfl(t3_reg8, friend_id3);
        sum0 += a26*((lane_id < 26)? tx0: ty0);
        sum1 += a26*((lane_id < 26)? tx1: ty1);
        sum2 += a26*((lane_id < 24)? tx2: ty2);
        sum3 += a26*((lane_id < 24)? tx3: ty3);

        OUT_3D(k,j  ,i   ) = sum0;
        OUT_3D(k,j  ,i+32) = sum1;
        OUT_3D(k,j+1,i   ) = sum2;
        OUT_3D(k,j+1,i+32) = sum3;
    }
}

int main(int argc, char **argv)
{
#ifdef __DEBUG
    int z = 4;
    int m = 8;
    int n = 128;
#else
    int z = 256; 
    int m = 256;
    int n = 256; 
#endif
    int halo = 1;
    int total = (z+2*halo)*(m+2*halo)*(n+2*halo);
    const int K = 27;
#ifdef __DEBUG
    DATA_TYPE args[K] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  
                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
#else
    DATA_TYPE args[K] = {0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 
                         0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037,  
                         0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037};
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
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
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
        Stencil_Cuda<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo); 
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
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
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
        Stencil_Cuda_Sweep<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sweep Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
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
        Stencil_Cuda_Sm_Branch<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo);
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
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
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
        Stencil_Cuda_Sm_Cyclic<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo);
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
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
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
        Stencil_Cuda_Sweep_Sm_Branch<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep_Sm_Branch: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sweep_Sm_Branch Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
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
        Stencil_Cuda_Sweep_Sm_Cyclic<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep_Sm_Cyclic: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sweep_Sm_Cyclic Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
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
        Stencil_Cuda_Shfl_2DWarp<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl_2DWarp: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Shfl_2DWarp Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
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
        Stencil_Cuda_Shfl2_2DWarp<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl2_2DWarp: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Shfl2_2DWarp Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
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
        Stencil_Cuda_Shfl4_2DWarp<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl4_2DWarp: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Shfl4_2DWarp Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    /*
    // Cuda 3D-Block with Shfl 8-Point (2D-Warp)
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dimGrid.x = (n)/8; 
    dimGrid.y = (m)/4; 
    dimGrid.z = (z)/(8*8);
    dimBlock.x = 8; 
    dimBlock.y = 4; 
    dimBlock.z = 8;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl8_2DWarp<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl8_2DWarp: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Shfl8_2DWarp Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));
    */

    /*
    // Cuda 2D Block Shfl version
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid8((n)/8, (m)/32, 4);
    dim3 dimBlock8(8, 32, 1);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sweep_Shfl_2DWarp<<<dimGrid8, dimBlock8>>>(
            in_d, out_d, args[0], args[1], args[2], args[3], args[4],
                args[5], args[6], args[7], args[8], args[9], args[10],
                args[11], args[12], args[13], args[14], args[15], args[16],
                args[17], args[18], args[19], args[20], args[21], args[22], 
                args[23], args[24], args[25], args[26], z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, z, m, n, "Output(Cuda_Sweep_Shfl):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep_Shfl_2DWarp: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sweep_Shfl_2DWarp Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda 2D Block Shfl2 version
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid9((n)/8, (m)/(32*2), 4);
    dim3 dimBlock9(8, 32, 1);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sweep_Shfl2_2DWarp<<<dimGrid9, dimBlock9>>>(
            in_d, out_d, args[0], args[1], args[2], args[3], args[4],
                args[5], args[6], args[7], args[8], args[9], args[10],
                args[11], args[12], args[13], args[14], args[15], args[16],
                args[17], args[18], args[19], args[20], args[21], args[22], 
                args[23], args[24], args[25], args[26], z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, z, m, n, "Output(Cuda_Sweep_Shfl2):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep_Shfl2_2DWarp: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sweep_Shfl2_2DWarp Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda 2D Block Shfl4 version
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid10((n)/8, (m)/(32*4), 4);
    dim3 dimBlock10(8, 32, 1);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sweep_Shfl4_2DWarp<<<dimGrid10, dimBlock10>>>(
            in_d, out_d, args[0], args[1], args[2], args[3], args[4],
                args[5], args[6], args[7], args[8], args[9], args[10],
                args[11], args[12], args[13], args[14], args[15], args[16],
                args[17], args[18], args[19], args[20], args[21], args[22], 
                args[23], args[24], args[25], args[26], z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, z, m, n, "Output(Cuda_Sweep_Shfl4):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep_Shfl4_2DWarp: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sweep_Shfl4_2DWarp Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));
    */

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
        Stencil_Cuda_Sweep_Shfl_1DWarp<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep_Shfl_1DWarp: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sweep_Shfl_1DWarp Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
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
        Stencil_Cuda_Sweep_Shfl2_1DWarp<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep_Shfl2_1DWarp: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sweep_Shfl2_1DWarp Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
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
        Stencil_Cuda_Sweep_Shfl4_1DWarp<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep_Shfl4_1DWarp: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sweep_Shfl4_1DWarp Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    /*
    // Cuda 2D Block Sweep Shfl4 1D-WarpX
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid18((n)/(64*2), (m)/(4*2), 4);
    dim3 dimBlock18(64, 4, 1);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sweep_Shfl4_1DWarpX<<<dimGrid18, dimBlock18>>>(
            in_d, out_d, args[0], args[1], args[2], args[3], args[4],
                args[5], args[6], args[7], args[8], args[9], args[10],
                args[11], args[12], args[13], args[14], args[15], args[16],
                args[17], args[18], args[19], args[20], args[21], args[22], 
                args[23], args[24], args[25], args[26], z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, z, m, n, halo, "Output(Cuda_Sweep_Shfl4_1DWarp):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep_Shfl4_1DWarp: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sweep_Shfl4_1DWarp Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));
    */

    cudaFree(in_d);
    cudaFree(out_d);

    delete[] in;
    delete[] out;
    delete[] out_ref;

}
