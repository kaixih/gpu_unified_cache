#include <iostream>
#include <metrics.h>
using namespace std;
#define IN_3D(_z,_y,_x)   in[(_z)*(m+2*halo)*(n+2*halo)+(_y)*(n+2*halo)+(_x)]
#define OUT_3D(_z,_y,_x) out[(_z)*(m+2*halo)*(n+2*halo)+(_y)*(n+2*halo)+(_x)]
#define ARG_3D(_l,_w,_x,_y)   args[(_l)]

#define SM_Z (8+2)
#define SM_M (4+2)
#define SM_N (8+2)
#define LOC_3D(_z,_y,_x) local[(_z)*(SM_M)*(SM_N)+(_y)*(SM_N)+(_x)]

#define SM_2D_M (4+2)
#define SM_2D_N (64+2)
#define LOC_2D(_y,_x) local[(_y)*(SM_2D_N)+(_x)]
// #define LOC_2D2(_y,_x) local[(_y)*(SM_2D_N2+2*halo)+(_x)]
#define LOC_L_2D(_z,_y,_x) local[(_z)*(SM_2D_M*SM_2D_N)+(_y)*(SM_2D_N)+(_x)]

// #define DATA_TYPE float
// #define DATA_TYPE double
#define warpSize 32 

// #define __DEBUG

#ifdef __DEBUG
#define ITER 1
#else
#define ITER 100
#endif


// #define TEMP
#define SPAC1

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

void Stencil_Seq(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, 
        int z, int m, int n, int halo)
{

    for(int k = halo; k < z+halo; k++)
    {
        for(int j = halo; j < m+halo; j++)
        {
            for(int i = halo; i < n+halo; i++)
            {
                OUT_3D(k,j,i) = ARG_3D(0 ,k,j,i) * IN_3D(k-1,j-1,i-1) +
                                ARG_3D(1 ,k,j,i) * IN_3D(k-1,j-1,i  ) +
                                ARG_3D(2 ,k,j,i) * IN_3D(k-1,j-1,i+1) +
                                ARG_3D(3 ,k,j,i) * IN_3D(k-1,j  ,i-1) +
                                ARG_3D(4 ,k,j,i) * IN_3D(k-1,j  ,i  ) +
                                ARG_3D(5 ,k,j,i) * IN_3D(k-1,j  ,i+1) +
                                ARG_3D(6 ,k,j,i) * IN_3D(k-1,j+1,i-1) + 
                                ARG_3D(7 ,k,j,i) * IN_3D(k-1,j+1,i  ) + 
                                ARG_3D(8 ,k,j,i) * IN_3D(k-1,j+1,i+1) + 
                                ARG_3D(9 ,k,j,i) * IN_3D(k  ,j-1,i-1) + 
                                ARG_3D(10,k,j,i) * IN_3D(k  ,j-1,i  ) + 
                                ARG_3D(11,k,j,i) * IN_3D(k  ,j-1,i+1) + 
                                ARG_3D(12,k,j,i) * IN_3D(k  ,j  ,i-1) + 
                                ARG_3D(13,k,j,i) * IN_3D(k  ,j  ,i  ) + 
                                ARG_3D(14,k,j,i) * IN_3D(k  ,j  ,i+1) + 
                                ARG_3D(15,k,j,i) * IN_3D(k  ,j+1,i-1) + 
                                ARG_3D(16,k,j,i) * IN_3D(k  ,j+1,i  ) + 
                                ARG_3D(17,k,j,i) * IN_3D(k  ,j+1,i+1) + 
                                ARG_3D(18,k,j,i) * IN_3D(k+1,j-1,i-1) + 
                                ARG_3D(19,k,j,i) * IN_3D(k+1,j-1,i  ) + 
                                ARG_3D(20,k,j,i) * IN_3D(k+1,j-1,i+1) + 
                                ARG_3D(21,k,j,i) * IN_3D(k+1,j  ,i-1) + 
                                ARG_3D(22,k,j,i) * IN_3D(k+1,j  ,i  ) + 
                                ARG_3D(23,k,j,i) * IN_3D(k+1,j  ,i+1) + 
                                ARG_3D(24,k,j,i) * IN_3D(k+1,j+1,i-1) + 
                                ARG_3D(25,k,j,i) * IN_3D(k+1,j+1,i  ) + 
                                ARG_3D(26,k,j,i) * IN_3D(k+1,j+1,i+1) ;
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

__global__ void Stencil_Cuda(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, 
        int z, int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = threadIdx.z + blockIdx.z * blockDim.z + halo;

    OUT_3D(k,j,i) = ARG_3D(0 ,k,j,i) * IN_3D(k-1,j-1,i-1) +
                    ARG_3D(1 ,k,j,i) * IN_3D(k-1,j-1,i  ) +
                    ARG_3D(2 ,k,j,i) * IN_3D(k-1,j-1,i+1) +
                    ARG_3D(3 ,k,j,i) * IN_3D(k-1,j  ,i-1) +
                    ARG_3D(4 ,k,j,i) * IN_3D(k-1,j  ,i  ) +
                    ARG_3D(5 ,k,j,i) * IN_3D(k-1,j  ,i+1) +
                    ARG_3D(6 ,k,j,i) * IN_3D(k-1,j+1,i-1) + 
                    ARG_3D(7 ,k,j,i) * IN_3D(k-1,j+1,i  ) + 
                    ARG_3D(8 ,k,j,i) * IN_3D(k-1,j+1,i+1) + 
                    ARG_3D(9 ,k,j,i) * IN_3D(k  ,j-1,i-1) + 
                    ARG_3D(10,k,j,i) * IN_3D(k  ,j-1,i  ) + 
                    ARG_3D(11,k,j,i) * IN_3D(k  ,j-1,i+1) + 
                    ARG_3D(12,k,j,i) * IN_3D(k  ,j  ,i-1) + 
                    ARG_3D(13,k,j,i) * IN_3D(k  ,j  ,i  ) + 
                    ARG_3D(14,k,j,i) * IN_3D(k  ,j  ,i+1) + 
                    ARG_3D(15,k,j,i) * IN_3D(k  ,j+1,i-1) + 
                    ARG_3D(16,k,j,i) * IN_3D(k  ,j+1,i  ) + 
                    ARG_3D(17,k,j,i) * IN_3D(k  ,j+1,i+1) + 
                    ARG_3D(18,k,j,i) * IN_3D(k+1,j-1,i-1) + 
                    ARG_3D(19,k,j,i) * IN_3D(k+1,j-1,i  ) + 
                    ARG_3D(20,k,j,i) * IN_3D(k+1,j-1,i+1) + 
                    ARG_3D(21,k,j,i) * IN_3D(k+1,j  ,i-1) + 
                    ARG_3D(22,k,j,i) * IN_3D(k+1,j  ,i  ) + 
                    ARG_3D(23,k,j,i) * IN_3D(k+1,j  ,i+1) + 
                    ARG_3D(24,k,j,i) * IN_3D(k+1,j+1,i-1) + 
                    ARG_3D(25,k,j,i) * IN_3D(k+1,j+1,i  ) + 
                    ARG_3D(26,k,j,i) * IN_3D(k+1,j+1,i+1) ;
}

__global__ void Stencil_Cuda_Sweep(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, 
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
        OUT_3D(k,j,i) = ARG_3D(0 ,k,j,i) * IN_3D(k-1,j-1,i-1) +
                        ARG_3D(1 ,k,j,i) * IN_3D(k-1,j-1,i  ) +
                        ARG_3D(2 ,k,j,i) * IN_3D(k-1,j-1,i+1) +
                        ARG_3D(3 ,k,j,i) * IN_3D(k-1,j  ,i-1) +
                        ARG_3D(4 ,k,j,i) * IN_3D(k-1,j  ,i  ) +
                        ARG_3D(5 ,k,j,i) * IN_3D(k-1,j  ,i+1) +
                        ARG_3D(6 ,k,j,i) * IN_3D(k-1,j+1,i-1) + 
                        ARG_3D(7 ,k,j,i) * IN_3D(k-1,j+1,i  ) + 
                        ARG_3D(8 ,k,j,i) * IN_3D(k-1,j+1,i+1) + 
                        ARG_3D(9 ,k,j,i) * IN_3D(k  ,j-1,i-1) + 
                        ARG_3D(10,k,j,i) * IN_3D(k  ,j-1,i  ) + 
                        ARG_3D(11,k,j,i) * IN_3D(k  ,j-1,i+1) + 
                        ARG_3D(12,k,j,i) * IN_3D(k  ,j  ,i-1) + 
                        ARG_3D(13,k,j,i) * IN_3D(k  ,j  ,i  ) + 
                        ARG_3D(14,k,j,i) * IN_3D(k  ,j  ,i+1) + 
                        ARG_3D(15,k,j,i) * IN_3D(k  ,j+1,i-1) + 
                        ARG_3D(16,k,j,i) * IN_3D(k  ,j+1,i  ) + 
                        ARG_3D(17,k,j,i) * IN_3D(k  ,j+1,i+1) + 
                        ARG_3D(18,k,j,i) * IN_3D(k+1,j-1,i-1) + 
                        ARG_3D(19,k,j,i) * IN_3D(k+1,j-1,i  ) + 
                        ARG_3D(20,k,j,i) * IN_3D(k+1,j-1,i+1) + 
                        ARG_3D(21,k,j,i) * IN_3D(k+1,j  ,i-1) + 
                        ARG_3D(22,k,j,i) * IN_3D(k+1,j  ,i  ) + 
                        ARG_3D(23,k,j,i) * IN_3D(k+1,j  ,i+1) + 
                        ARG_3D(24,k,j,i) * IN_3D(k+1,j+1,i-1) + 
                        ARG_3D(25,k,j,i) * IN_3D(k+1,j+1,i  ) + 
                        ARG_3D(26,k,j,i) * IN_3D(k+1,j+1,i+1) ;
    }
}

__global__ void Stencil_Cuda_Sweep_Sm(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, 
        int z, int m, int n, int halo)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;
    int li = threadIdx.x + 1;
    int lj = threadIdx.y + 1;

    extern __shared__ DATA_TYPE local[];

    int t1, t2, t3;
    t3 = 2; t2 = 1;
    LOC_L_2D(t3,lj,li) = IN_3D(k,j,i);
    LOC_L_2D(t2,lj,li) = IN_3D(k-1,j,i);
    if(li == halo)                                   
    {
        LOC_L_2D(t3,lj  ,li-1) = IN_3D(k  ,j  ,i-1); 
        LOC_L_2D(t2,lj  ,li-1) = IN_3D(k-1,j  ,i-1);
    }
    if(li == SM_2D_N-2)                         
    {
        LOC_L_2D(t3,lj  ,li+1) = IN_3D(k  ,j  ,i+1); 
        LOC_L_2D(t2,lj  ,li+1) = IN_3D(k-1,j  ,i+1);
    }
    if(lj == halo)                                   
    {
        LOC_L_2D(t3,lj-1,li  ) = IN_3D(k  ,j-1,i  ); 
        LOC_L_2D(t2,lj-1,li  ) = IN_3D(k-1,j-1,i  );
    }
    if(lj == SM_2D_M-2)                         
    {
        LOC_L_2D(t3,lj+1,li  ) = IN_3D(k  ,j+1,i  ); 
        LOC_L_2D(t2,lj+1,li  ) = IN_3D(k-1,j+1,i  );
    }
    if(li == halo && lj == halo)                     
    {
        LOC_L_2D(t3,lj-1,li-1) = IN_3D(k  ,j-1,i-1); 
        LOC_L_2D(t2,lj-1,li-1) = IN_3D(k-1,j-1,i-1);
    }
    if(li == SM_2D_N-2 && lj == halo)           
    {
        LOC_L_2D(t3,lj-1,li+1) = IN_3D(k  ,j-1,i+1); 
        LOC_L_2D(t2,lj-1,li+1) = IN_3D(k-1,j-1,i+1);
    }
    if(li == halo && lj == SM_2D_M-2)           
    { 
        LOC_L_2D(t3,lj+1,li-1) = IN_3D(k  ,j+1,i-1); 
        LOC_L_2D(t2,lj+1,li-1) = IN_3D(k-1,j+1,i-1);
    }
    if(li == SM_2D_N-2 && lj == SM_2D_M-2) 
    {
        LOC_L_2D(t3,lj+1,li+1) = IN_3D(k  ,j+1,i+1); 
        LOC_L_2D(t2,lj+1,li+1) = IN_3D(k-1,j+1,i+1);
    }

#pragma unroll // it seems the loop-unroll is useless to performance
    for(; k < k_end; ++k)
    {
        t1 = t2;
        t2 = t3;
        t3 = (t3+1)%3;
        LOC_L_2D(t3,lj,li) = IN_3D(k+1,j,i);
        if(li == halo)                         {LOC_L_2D(t3,lj  ,li-1) = IN_3D(k+1,j  ,i-1);}
        if(li == SM_2D_N-2)                    {LOC_L_2D(t3,lj  ,li+1) = IN_3D(k+1,j  ,i+1);}
        if(lj == halo)                         {LOC_L_2D(t3,lj-1,li  ) = IN_3D(k+1,j-1,i  );}
        if(lj == SM_2D_M-2)                    {LOC_L_2D(t3,lj+1,li  ) = IN_3D(k+1,j+1,i  );}
        if(li == halo && lj == halo)           {LOC_L_2D(t3,lj-1,li-1) = IN_3D(k+1,j-1,i-1);}
        if(li == SM_2D_N-2 && lj == halo)      {LOC_L_2D(t3,lj-1,li+1) = IN_3D(k+1,j-1,i+1);}
        if(li == halo && lj == SM_2D_M-2)      {LOC_L_2D(t3,lj+1,li-1) = IN_3D(k+1,j+1,i-1);}
        if(li == SM_2D_N-2 && lj == SM_2D_M-2) {LOC_L_2D(t3,lj+1,li+1) = IN_3D(k+1,j+1,i+1);}
        __syncthreads();

        OUT_3D(k,j,i) = ARG_3D(0 ,k,j,i) * LOC_L_2D(t1,lj-1,li-1) +
                        ARG_3D(1 ,k,j,i) * LOC_L_2D(t1,lj-1,li  ) +
                        ARG_3D(2 ,k,j,i) * LOC_L_2D(t1,lj-1,li+1) +
                        ARG_3D(3 ,k,j,i) * LOC_L_2D(t1,lj  ,li-1) +
                        ARG_3D(4 ,k,j,i) * LOC_L_2D(t1,lj  ,li  ) +
                        ARG_3D(5 ,k,j,i) * LOC_L_2D(t1,lj  ,li+1) +
                        ARG_3D(6 ,k,j,i) * LOC_L_2D(t1,lj+1,li-1) + 
                        ARG_3D(7 ,k,j,i) * LOC_L_2D(t1,lj+1,li  ) + 
                        ARG_3D(8 ,k,j,i) * LOC_L_2D(t1,lj+1,li+1) + 
                        ARG_3D(9 ,k,j,i) * LOC_L_2D(t2,lj-1,li-1) + 
                        ARG_3D(10,k,j,i) * LOC_L_2D(t2,lj-1,li  ) + 
                        ARG_3D(11,k,j,i) * LOC_L_2D(t2,lj-1,li+1) + 
                        ARG_3D(12,k,j,i) * LOC_L_2D(t2,lj  ,li-1) + 
                        ARG_3D(13,k,j,i) * LOC_L_2D(t2,lj  ,li  ) + 
                        ARG_3D(14,k,j,i) * LOC_L_2D(t2,lj  ,li+1) + 
                        ARG_3D(15,k,j,i) * LOC_L_2D(t2,lj+1,li-1) + 
                        ARG_3D(16,k,j,i) * LOC_L_2D(t2,lj+1,li  ) + 
                        ARG_3D(17,k,j,i) * LOC_L_2D(t2,lj+1,li+1) + 
                        ARG_3D(18,k,j,i) * LOC_L_2D(t3,lj-1,li-1) + 
                        ARG_3D(19,k,j,i) * LOC_L_2D(t3,lj-1,li  ) + 
                        ARG_3D(20,k,j,i) * LOC_L_2D(t3,lj-1,li+1) + 
                        ARG_3D(21,k,j,i) * LOC_L_2D(t3,lj  ,li-1) + 
                        ARG_3D(22,k,j,i) * LOC_L_2D(t3,lj  ,li  ) + 
                        ARG_3D(23,k,j,i) * LOC_L_2D(t3,lj  ,li+1) + 
                        ARG_3D(24,k,j,i) * LOC_L_2D(t3,lj+1,li-1) + 
                        ARG_3D(25,k,j,i) * LOC_L_2D(t3,lj+1,li  ) + 
                        ARG_3D(26,k,j,i) * LOC_L_2D(t3,lj+1,li+1) ;
        
        __syncthreads();
    }

}

__global__ void Stencil_Cuda_Sm(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, 
        int z, int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = threadIdx.z + blockIdx.z * blockDim.z + halo;

    int li = threadIdx.x + 1;
    int lj = threadIdx.y + 1;
    int lk = threadIdx.z + 1;

    __shared__ DATA_TYPE local[SM_Z*SM_M*SM_N];

    LOC_3D(lk,lj,li) = IN_3D(k,j,i);

    if(li == halo  ) LOC_3D(lk,lj,li-1) = IN_3D(k,j,i-1);
    if(li == SM_N-2) LOC_3D(lk,lj,li+1) = IN_3D(k,j,i+1);
    if(lj == halo  ) LOC_3D(lk,lj-1,li) = IN_3D(k,j-1,i);
    if(lj == SM_M-2) LOC_3D(lk,lj+1,li) = IN_3D(k,j+1,i);
    if(lk == halo  ) LOC_3D(lk-1,lj,li) = IN_3D(k-1,j,i);
    if(lk == SM_Z-2) LOC_3D(lk+1,lj,li) = IN_3D(k+1,j,i);
    if(li == halo   && lj == halo  ) LOC_3D(lk  ,lj-1,li-1) = IN_3D(k  ,j-1,i-1);
    if(li == halo   && lk == halo  ) LOC_3D(lk-1,lj  ,li-1) = IN_3D(k-1,j  ,i-1);
    if(lj == halo   && lk == halo  ) LOC_3D(lk-1,lj-1,li  ) = IN_3D(k-1,j-1,i  );
    if(li == SM_N-2 && lj == halo  ) LOC_3D(lk  ,lj-1,li+1) = IN_3D(k  ,j-1,i+1);
    if(li == SM_N-2 && lk == halo  ) LOC_3D(lk-1,lj  ,li+1) = IN_3D(k-1,j  ,i+1);
    if(lj == SM_M-2 && lk == halo  ) LOC_3D(lk-1,lj+1,li  ) = IN_3D(k-1,j+1,i  );
    if(li == halo   && lj == SM_M-2) LOC_3D(lk  ,lj+1,li-1) = IN_3D(k  ,j+1,i-1);
    if(li == halo   && lk == SM_Z-2) LOC_3D(lk+1,lj  ,li-1) = IN_3D(k+1,j  ,i-1);
    if(lj == halo   && lk == SM_Z-2) LOC_3D(lk+1,lj-1,li  ) = IN_3D(k+1,j-1,i  );
    if(li == SM_N-2 && lj == SM_M-2) LOC_3D(lk  ,lj+1,li+1) = IN_3D(k  ,j+1,i+1);
    if(li == SM_N-2 && lk == SM_Z-2) LOC_3D(lk+1,lj  ,li+1) = IN_3D(k+1,j  ,i+1);
    if(lj == SM_M-2 && lk == SM_Z-2) LOC_3D(lk+1,lj+1,li  ) = IN_3D(k+1,j+1,i  );
    if(li == halo   && lj == halo   && lk == halo  ) LOC_3D(lk-1,lj-1,li-1) = IN_3D(k-1,j-1,i-1);
    if(li == halo   && lj == halo   && lk == SM_Z-2) LOC_3D(lk+1,lj-1,li-1) = IN_3D(k+1,j-1,i-1);
    if(li == halo   && lj == SM_M-2 && lk == halo  ) LOC_3D(lk-1,lj+1,li-1) = IN_3D(k-1,j+1,i-1);
    if(li == halo   && lj == SM_M-2 && lk == SM_Z-2) LOC_3D(lk+1,lj+1,li-1) = IN_3D(k+1,j+1,i-1);
    if(li == SM_N-2 && lj == halo   && lk == halo  ) LOC_3D(lk-1,lj-1,li+1) = IN_3D(k-1,j-1,i+1);
    if(li == SM_N-2 && lj == halo   && lk == SM_Z-2) LOC_3D(lk+1,lj-1,li+1) = IN_3D(k+1,j-1,i+1);
    if(li == SM_N-2 && lj == SM_M-2 && lk == halo  ) LOC_3D(lk-1,lj+1,li+1) = IN_3D(k-1,j+1,i+1);
    if(li == SM_N-2 && lj == SM_M-2 && lk == SM_Z-2) LOC_3D(lk+1,lj+1,li+1) = IN_3D(k+1,j+1,i+1);

    __syncthreads();


    OUT_3D(k,j,i) = ARG_3D(0 ,k,j,i) * LOC_3D(lk-1,lj-1,li-1) +
                    ARG_3D(1 ,k,j,i) * LOC_3D(lk-1,lj-1,li  ) +
                    ARG_3D(2 ,k,j,i) * LOC_3D(lk-1,lj-1,li+1) +
                    ARG_3D(3 ,k,j,i) * LOC_3D(lk-1,lj  ,li-1) +
                    ARG_3D(4 ,k,j,i) * LOC_3D(lk-1,lj  ,li  ) +
                    ARG_3D(5 ,k,j,i) * LOC_3D(lk-1,lj  ,li+1) +
                    ARG_3D(6 ,k,j,i) * LOC_3D(lk-1,lj+1,li-1) + 
                    ARG_3D(7 ,k,j,i) * LOC_3D(lk-1,lj+1,li  ) + 
                    ARG_3D(8 ,k,j,i) * LOC_3D(lk-1,lj+1,li+1) + 
                    ARG_3D(9 ,k,j,i) * LOC_3D(lk  ,lj-1,li-1) + 
                    ARG_3D(10,k,j,i) * LOC_3D(lk  ,lj-1,li  ) + 
                    ARG_3D(11,k,j,i) * LOC_3D(lk  ,lj-1,li+1) + 
                    ARG_3D(12,k,j,i) * LOC_3D(lk  ,lj  ,li-1) + 
                    ARG_3D(13,k,j,i) * LOC_3D(lk  ,lj  ,li  ) + 
                    ARG_3D(14,k,j,i) * LOC_3D(lk  ,lj  ,li+1) + 
                    ARG_3D(15,k,j,i) * LOC_3D(lk  ,lj+1,li-1) + 
                    ARG_3D(16,k,j,i) * LOC_3D(lk  ,lj+1,li  ) + 
                    ARG_3D(17,k,j,i) * LOC_3D(lk  ,lj+1,li+1) + 
                    ARG_3D(18,k,j,i) * LOC_3D(lk+1,lj-1,li-1) + 
                    ARG_3D(19,k,j,i) * LOC_3D(lk+1,lj-1,li  ) + 
                    ARG_3D(20,k,j,i) * LOC_3D(lk+1,lj-1,li+1) + 
                    ARG_3D(21,k,j,i) * LOC_3D(lk+1,lj  ,li-1) + 
                    ARG_3D(22,k,j,i) * LOC_3D(lk+1,lj  ,li  ) + 
                    ARG_3D(23,k,j,i) * LOC_3D(lk+1,lj  ,li+1) + 
                    ARG_3D(24,k,j,i) * LOC_3D(lk+1,lj+1,li-1) + 
                    ARG_3D(25,k,j,i) * LOC_3D(lk+1,lj+1,li  ) + 
                    ARG_3D(26,k,j,i) * LOC_3D(lk+1,lj+1,li+1) ;
}

__global__ void Stencil_Cuda_Sm2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, 
        int z, int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = (((threadIdx.z + blockIdx.z * blockDim.z)>>3)<<4) + threadIdx.z + halo;

    int li = threadIdx.x + 1;
    int lj = threadIdx.y + 1;
    int lk = threadIdx.z + 1;

    __shared__ DATA_TYPE local[SM_Z*SM_M*SM_N];

    LOC_3D(lk,lj,li) = IN_3D(k,j,i);

    if(li == halo  ) LOC_3D(lk,lj,li-1) = IN_3D(k,j,i-1);
    if(li == SM_N-2) LOC_3D(lk,lj,li+1) = IN_3D(k,j,i+1);
    if(lj == halo  ) LOC_3D(lk,lj-1,li) = IN_3D(k,j-1,i);
    if(lj == SM_M-2) LOC_3D(lk,lj+1,li) = IN_3D(k,j+1,i);
    if(lk == halo  ) LOC_3D(lk-1,lj,li) = IN_3D(k-1,j,i);
    if(lk == SM_Z-2) LOC_3D(lk+1,lj,li) = IN_3D(k+1,j,i);
    if(li == halo   && lj == halo  ) LOC_3D(lk  ,lj-1,li-1) = IN_3D(k  ,j-1,i-1);
    if(li == halo   && lk == halo  ) LOC_3D(lk-1,lj  ,li-1) = IN_3D(k-1,j  ,i-1);
    if(lj == halo   && lk == halo  ) LOC_3D(lk-1,lj-1,li  ) = IN_3D(k-1,j-1,i  );
    if(li == SM_N-2 && lj == halo  ) LOC_3D(lk  ,lj-1,li+1) = IN_3D(k  ,j-1,i+1);
    if(li == SM_N-2 && lk == halo  ) LOC_3D(lk-1,lj  ,li+1) = IN_3D(k-1,j  ,i+1);
    if(lj == SM_M-2 && lk == halo  ) LOC_3D(lk-1,lj+1,li  ) = IN_3D(k-1,j+1,i  );
    if(li == halo   && lj == SM_M-2) LOC_3D(lk  ,lj+1,li-1) = IN_3D(k  ,j+1,i-1);
    if(li == halo   && lk == SM_Z-2) LOC_3D(lk+1,lj  ,li-1) = IN_3D(k+1,j  ,i-1);
    if(lj == halo   && lk == SM_Z-2) LOC_3D(lk+1,lj-1,li  ) = IN_3D(k+1,j-1,i  );
    if(li == SM_N-2 && lj == SM_M-2) LOC_3D(lk  ,lj+1,li+1) = IN_3D(k  ,j+1,i+1);
    if(li == SM_N-2 && lk == SM_Z-2) LOC_3D(lk+1,lj  ,li+1) = IN_3D(k+1,j  ,i+1);
    if(lj == SM_M-2 && lk == SM_Z-2) LOC_3D(lk+1,lj+1,li  ) = IN_3D(k+1,j+1,i  );
    if(li == halo   && lj == halo   && lk == halo  ) LOC_3D(lk-1,lj-1,li-1) = IN_3D(k-1,j-1,i-1);
    if(li == halo   && lj == halo   && lk == SM_Z-2) LOC_3D(lk+1,lj-1,li-1) = IN_3D(k+1,j-1,i-1);
    if(li == halo   && lj == SM_M-2 && lk == halo  ) LOC_3D(lk-1,lj+1,li-1) = IN_3D(k-1,j+1,i-1);
    if(li == halo   && lj == SM_M-2 && lk == SM_Z-2) LOC_3D(lk+1,lj+1,li-1) = IN_3D(k+1,j+1,i-1);
    if(li == SM_N-2 && lj == halo   && lk == halo  ) LOC_3D(lk-1,lj-1,li+1) = IN_3D(k-1,j-1,i+1);
    if(li == SM_N-2 && lj == halo   && lk == SM_Z-2) LOC_3D(lk+1,lj-1,li+1) = IN_3D(k+1,j-1,i+1);
    if(li == SM_N-2 && lj == SM_M-2 && lk == halo  ) LOC_3D(lk-1,lj+1,li+1) = IN_3D(k-1,j+1,i+1);
    if(li == SM_N-2 && lj == SM_M-2 && lk == SM_Z-2) LOC_3D(lk+1,lj+1,li+1) = IN_3D(k+1,j+1,i+1);

    __syncthreads();


    OUT_3D(k,j,i) = ARG_3D(0 ,k,j,i) * LOC_3D(lk-1,lj-1,li-1) +
                    ARG_3D(1 ,k,j,i) * LOC_3D(lk-1,lj-1,li  ) +
                    ARG_3D(2 ,k,j,i) * LOC_3D(lk-1,lj-1,li+1) +
                    ARG_3D(3 ,k,j,i) * LOC_3D(lk-1,lj  ,li-1) +
                    ARG_3D(4 ,k,j,i) * LOC_3D(lk-1,lj  ,li  ) +
                    ARG_3D(5 ,k,j,i) * LOC_3D(lk-1,lj  ,li+1) +
                    ARG_3D(6 ,k,j,i) * LOC_3D(lk-1,lj+1,li-1) + 
                    ARG_3D(7 ,k,j,i) * LOC_3D(lk-1,lj+1,li  ) + 
                    ARG_3D(8 ,k,j,i) * LOC_3D(lk-1,lj+1,li+1) + 
                    ARG_3D(9 ,k,j,i) * LOC_3D(lk  ,lj-1,li-1) + 
                    ARG_3D(10,k,j,i) * LOC_3D(lk  ,lj-1,li  ) + 
                    ARG_3D(11,k,j,i) * LOC_3D(lk  ,lj-1,li+1) + 
                    ARG_3D(12,k,j,i) * LOC_3D(lk  ,lj  ,li-1) + 
                    ARG_3D(13,k,j,i) * LOC_3D(lk  ,lj  ,li  ) + 
                    ARG_3D(14,k,j,i) * LOC_3D(lk  ,lj  ,li+1) + 
                    ARG_3D(15,k,j,i) * LOC_3D(lk  ,lj+1,li-1) + 
                    ARG_3D(16,k,j,i) * LOC_3D(lk  ,lj+1,li  ) + 
                    ARG_3D(17,k,j,i) * LOC_3D(lk  ,lj+1,li+1) + 
                    ARG_3D(18,k,j,i) * LOC_3D(lk+1,lj-1,li-1) + 
                    ARG_3D(19,k,j,i) * LOC_3D(lk+1,lj-1,li  ) + 
                    ARG_3D(20,k,j,i) * LOC_3D(lk+1,lj-1,li+1) + 
                    ARG_3D(21,k,j,i) * LOC_3D(lk+1,lj  ,li-1) + 
                    ARG_3D(22,k,j,i) * LOC_3D(lk+1,lj  ,li  ) + 
                    ARG_3D(23,k,j,i) * LOC_3D(lk+1,lj  ,li+1) + 
                    ARG_3D(24,k,j,i) * LOC_3D(lk+1,lj+1,li-1) + 
                    ARG_3D(25,k,j,i) * LOC_3D(lk+1,lj+1,li  ) + 
                    ARG_3D(26,k,j,i) * LOC_3D(lk+1,lj+1,li+1) ;

    __syncthreads();
    k+=8;
    LOC_3D(lk,lj,li) = IN_3D(k,j,i);

    if(li == halo  ) LOC_3D(lk,lj,li-1) = IN_3D(k,j,i-1);
    if(li == SM_N-2) LOC_3D(lk,lj,li+1) = IN_3D(k,j,i+1);
    if(lj == halo  ) LOC_3D(lk,lj-1,li) = IN_3D(k,j-1,i);
    if(lj == SM_M-2) LOC_3D(lk,lj+1,li) = IN_3D(k,j+1,i);
    if(lk == halo  ) LOC_3D(lk-1,lj,li) = IN_3D(k-1,j,i);
    if(lk == SM_Z-2) LOC_3D(lk+1,lj,li) = IN_3D(k+1,j,i);
    if(li == halo   && lj == halo  ) LOC_3D(lk  ,lj-1,li-1) = IN_3D(k  ,j-1,i-1);
    if(li == halo   && lk == halo  ) LOC_3D(lk-1,lj  ,li-1) = IN_3D(k-1,j  ,i-1);
    if(lj == halo   && lk == halo  ) LOC_3D(lk-1,lj-1,li  ) = IN_3D(k-1,j-1,i  );
    if(li == SM_N-2 && lj == halo  ) LOC_3D(lk  ,lj-1,li+1) = IN_3D(k  ,j-1,i+1);
    if(li == SM_N-2 && lk == halo  ) LOC_3D(lk-1,lj  ,li+1) = IN_3D(k-1,j  ,i+1);
    if(lj == SM_M-2 && lk == halo  ) LOC_3D(lk-1,lj+1,li  ) = IN_3D(k-1,j+1,i  );
    if(li == halo   && lj == SM_M-2) LOC_3D(lk  ,lj+1,li-1) = IN_3D(k  ,j+1,i-1);
    if(li == halo   && lk == SM_Z-2) LOC_3D(lk+1,lj  ,li-1) = IN_3D(k+1,j  ,i-1);
    if(lj == halo   && lk == SM_Z-2) LOC_3D(lk+1,lj-1,li  ) = IN_3D(k+1,j-1,i  );
    if(li == SM_N-2 && lj == SM_M-2) LOC_3D(lk  ,lj+1,li+1) = IN_3D(k  ,j+1,i+1);
    if(li == SM_N-2 && lk == SM_Z-2) LOC_3D(lk+1,lj  ,li+1) = IN_3D(k+1,j  ,i+1);
    if(lj == SM_M-2 && lk == SM_Z-2) LOC_3D(lk+1,lj+1,li  ) = IN_3D(k+1,j+1,i  );
    if(li == halo   && lj == halo   && lk == halo  ) LOC_3D(lk-1,lj-1,li-1) = IN_3D(k-1,j-1,i-1);
    if(li == halo   && lj == halo   && lk == SM_Z-2) LOC_3D(lk+1,lj-1,li-1) = IN_3D(k+1,j-1,i-1);
    if(li == halo   && lj == SM_M-2 && lk == halo  ) LOC_3D(lk-1,lj+1,li-1) = IN_3D(k-1,j+1,i-1);
    if(li == halo   && lj == SM_M-2 && lk == SM_Z-2) LOC_3D(lk+1,lj+1,li-1) = IN_3D(k+1,j+1,i-1);
    if(li == SM_N-2 && lj == halo   && lk == halo  ) LOC_3D(lk-1,lj-1,li+1) = IN_3D(k-1,j-1,i+1);
    if(li == SM_N-2 && lj == halo   && lk == SM_Z-2) LOC_3D(lk+1,lj-1,li+1) = IN_3D(k+1,j-1,i+1);
    if(li == SM_N-2 && lj == SM_M-2 && lk == halo  ) LOC_3D(lk-1,lj+1,li+1) = IN_3D(k-1,j+1,i+1);
    if(li == SM_N-2 && lj == SM_M-2 && lk == SM_Z-2) LOC_3D(lk+1,lj+1,li+1) = IN_3D(k+1,j+1,i+1);

    __syncthreads();


    OUT_3D(k,j,i) = ARG_3D(0 ,k,j,i) * LOC_3D(lk-1,lj-1,li-1) +
                    ARG_3D(1 ,k,j,i) * LOC_3D(lk-1,lj-1,li  ) +
                    ARG_3D(2 ,k,j,i) * LOC_3D(lk-1,lj-1,li+1) +
                    ARG_3D(3 ,k,j,i) * LOC_3D(lk-1,lj  ,li-1) +
                    ARG_3D(4 ,k,j,i) * LOC_3D(lk-1,lj  ,li  ) +
                    ARG_3D(5 ,k,j,i) * LOC_3D(lk-1,lj  ,li+1) +
                    ARG_3D(6 ,k,j,i) * LOC_3D(lk-1,lj+1,li-1) + 
                    ARG_3D(7 ,k,j,i) * LOC_3D(lk-1,lj+1,li  ) + 
                    ARG_3D(8 ,k,j,i) * LOC_3D(lk-1,lj+1,li+1) + 
                    ARG_3D(9 ,k,j,i) * LOC_3D(lk  ,lj-1,li-1) + 
                    ARG_3D(10,k,j,i) * LOC_3D(lk  ,lj-1,li  ) + 
                    ARG_3D(11,k,j,i) * LOC_3D(lk  ,lj-1,li+1) + 
                    ARG_3D(12,k,j,i) * LOC_3D(lk  ,lj  ,li-1) + 
                    ARG_3D(13,k,j,i) * LOC_3D(lk  ,lj  ,li  ) + 
                    ARG_3D(14,k,j,i) * LOC_3D(lk  ,lj  ,li+1) + 
                    ARG_3D(15,k,j,i) * LOC_3D(lk  ,lj+1,li-1) + 
                    ARG_3D(16,k,j,i) * LOC_3D(lk  ,lj+1,li  ) + 
                    ARG_3D(17,k,j,i) * LOC_3D(lk  ,lj+1,li+1) + 
                    ARG_3D(18,k,j,i) * LOC_3D(lk+1,lj-1,li-1) + 
                    ARG_3D(19,k,j,i) * LOC_3D(lk+1,lj-1,li  ) + 
                    ARG_3D(20,k,j,i) * LOC_3D(lk+1,lj-1,li+1) + 
                    ARG_3D(21,k,j,i) * LOC_3D(lk+1,lj  ,li-1) + 
                    ARG_3D(22,k,j,i) * LOC_3D(lk+1,lj  ,li  ) + 
                    ARG_3D(23,k,j,i) * LOC_3D(lk+1,lj  ,li+1) + 
                    ARG_3D(24,k,j,i) * LOC_3D(lk+1,lj+1,li-1) + 
                    ARG_3D(25,k,j,i) * LOC_3D(lk+1,lj+1,li  ) + 
                    ARG_3D(26,k,j,i) * LOC_3D(lk+1,lj+1,li+1) ;
}

__global__ void Stencil_Cuda_SmX(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, 
        int z, int m, int n, int halo) 
{
    __shared__ DATA_TYPE local[SM_Z*SM_M*SM_N];
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = threadIdx.z + blockIdx.z * blockDim.z + halo;

    int li = threadIdx.x + 1;
    int lj = threadIdx.y + 1;
    int lk = threadIdx.z + 1;

    int lane_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;

    int blk_id_x = blockIdx.x; //(threadIdx.x + blockIdx.x * blockDim.x)>>3; 
    int blk_id_y = blockIdx.y; //(threadIdx.y + blockIdx.y * blockDim.y)>>2;
    int blk_id_z = blockIdx.z; //(threadIdx.z + blockIdx.z * blockDim.z)>>3; 

    int new_i = (blk_id_x<<3) + lane_id%10 ;     
    int new_j = (blk_id_y<<2) + (lane_id/10)%6 ; 
    int new_k = (blk_id_z<<3) + lane_id/60 ;     
    int new_li = lane_id%10;
    int new_lj = (lane_id/10)%6;
    int new_lk = lane_id/60;
    LOC_3D(new_lk,new_lj,new_li) = IN_3D(new_k,new_j,new_i);
    new_i = (blk_id_x<<3) + (lane_id+256)%10 ;
    new_j = (blk_id_y<<2) + ((lane_id+256)/10)%6 ;
    new_k = (blk_id_z<<3) + (lane_id+256)/60 ;
    new_li = (lane_id+256)%10;
    new_lj = ((lane_id+256)/10)%6;
    new_lk = (lane_id+256)/60; 
    LOC_3D(new_lk,new_lj,new_li) = IN_3D(new_k,new_j,new_i);
    new_i = (blk_id_x<<3) + (lane_id+512)%10 ;
    new_j = (blk_id_y<<2) + ((lane_id+512)/10)%6 ;
    new_k = (blk_id_z<<3) + (lane_id+512)/60 ;
    new_li = (lane_id+512)%10;
    new_lj = ((lane_id+512)/10)%6;
    new_lk = (lane_id+512)/60; 
    if(new_li < 10 &&  new_lj < 6 && new_lk < 10 )
        LOC_3D(new_lk,new_lj,new_li) = IN_3D(new_k,new_j,new_i);

    __syncthreads();

    OUT_3D(k,j,i) = ARG_3D(0 ,k,j,i) * LOC_3D(lk-1,lj-1,li-1) +
                    ARG_3D(1 ,k,j,i) * LOC_3D(lk-1,lj-1,li  ) +
                    ARG_3D(2 ,k,j,i) * LOC_3D(lk-1,lj-1,li+1) +
                    ARG_3D(3 ,k,j,i) * LOC_3D(lk-1,lj  ,li-1) +
                    ARG_3D(4 ,k,j,i) * LOC_3D(lk-1,lj  ,li  ) +
                    ARG_3D(5 ,k,j,i) * LOC_3D(lk-1,lj  ,li+1) +
                    ARG_3D(6 ,k,j,i) * LOC_3D(lk-1,lj+1,li-1) + 
                    ARG_3D(7 ,k,j,i) * LOC_3D(lk-1,lj+1,li  ) + 
                    ARG_3D(8 ,k,j,i) * LOC_3D(lk-1,lj+1,li+1) + 
                    ARG_3D(9 ,k,j,i) * LOC_3D(lk  ,lj-1,li-1) + 
                    ARG_3D(10,k,j,i) * LOC_3D(lk  ,lj-1,li  ) + 
                    ARG_3D(11,k,j,i) * LOC_3D(lk  ,lj-1,li+1) + 
                    ARG_3D(12,k,j,i) * LOC_3D(lk  ,lj  ,li-1) + 
                    ARG_3D(13,k,j,i) * LOC_3D(lk  ,lj  ,li  ) + 
                    ARG_3D(14,k,j,i) * LOC_3D(lk  ,lj  ,li+1) + 
                    ARG_3D(15,k,j,i) * LOC_3D(lk  ,lj+1,li-1) + 
                    ARG_3D(16,k,j,i) * LOC_3D(lk  ,lj+1,li  ) + 
                    ARG_3D(17,k,j,i) * LOC_3D(lk  ,lj+1,li+1) + 
                    ARG_3D(18,k,j,i) * LOC_3D(lk+1,lj-1,li-1) + 
                    ARG_3D(19,k,j,i) * LOC_3D(lk+1,lj-1,li  ) + 
                    ARG_3D(20,k,j,i) * LOC_3D(lk+1,lj-1,li+1) + 
                    ARG_3D(21,k,j,i) * LOC_3D(lk+1,lj  ,li-1) + 
                    ARG_3D(22,k,j,i) * LOC_3D(lk+1,lj  ,li  ) + 
                    ARG_3D(23,k,j,i) * LOC_3D(lk+1,lj  ,li+1) + 
                    ARG_3D(24,k,j,i) * LOC_3D(lk+1,lj+1,li-1) + 
                    ARG_3D(25,k,j,i) * LOC_3D(lk+1,lj+1,li  ) + 
                    ARG_3D(26,k,j,i) * LOC_3D(lk+1,lj+1,li+1) ;

}


__global__ void Stencil_Cuda_Shfl(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, 
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
    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5;
    threadInput0 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+32)/60;
    threadInput1 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10;
    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+64)/60;
    threadInput2 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10;
    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+96)/60;
    threadInput3 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10;
    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+128)/60;
    threadInput4 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10;
    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+160)/60;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    new_k = (new_k < z+2*halo)? new_k: z+2*halo-1;
    threadInput5 = IN_3D(new_k, new_j, new_i);

    DATA_TYPE sum = 0.0;
    int friend_id;
    DATA_TYPE tx, ty, tz;

    friend_id = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput0, friend_id);
    ty = __shfl(threadInput1, friend_id);
    sum += ARG_3D(0,k,j,i)*((lane_id < 26)? tx: ty);

    friend_id = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput0, friend_id);
    ty = __shfl(threadInput1, friend_id);
    sum += ARG_3D(1,k,j,i)*((lane_id < 25)? tx: ty);

    friend_id = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput0, friend_id);
    ty = __shfl(threadInput1, friend_id);
    sum += ARG_3D(2,k,j,i)*((lane_id < 24)? tx: ty);

    friend_id = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput0, friend_id);
    ty = __shfl(threadInput1, friend_id);
    sum += ARG_3D(3,k,j,i)*((lane_id < 18)? tx: ty);

    friend_id = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput0, friend_id);
    ty = __shfl(threadInput1, friend_id);
    sum += ARG_3D(4,k,j,i)*((lane_id < 17)? tx: ty);

    friend_id = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput0, friend_id);
    ty = __shfl(threadInput1, friend_id);
    sum += ARG_3D(5,k,j,i)*((lane_id < 16)? tx: ty);

    friend_id = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput0, friend_id);
    ty = __shfl(threadInput1, friend_id);
    sum += ARG_3D(6,k,j,i)*((lane_id < 10)? tx: ty);

    friend_id = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput0, friend_id);
    ty = __shfl(threadInput1, friend_id);
    sum += ARG_3D(7,k,j,i)*((lane_id < 9 )? tx: ty);

    friend_id = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput0, friend_id);
    ty = __shfl(threadInput1, friend_id);
    sum += ARG_3D(8,k,j,i)*((lane_id < 8 )? tx: ty);

    friend_id = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput1, friend_id);
    ty = __shfl(threadInput2, friend_id);
    tz = __shfl(threadInput3, friend_id);
    sum += ARG_3D(9,k,j,i)*((lane_id < 4 )? tx: ((lane_id < 30)? ty: tz));

    friend_id = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput1, friend_id);
    ty = __shfl(threadInput2, friend_id);
    tz = __shfl(threadInput3, friend_id);
    sum += ARG_3D(10,k,j,i)*((lane_id < 3 )? tx: ((lane_id < 29)? ty: tz));

    friend_id = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput1, friend_id);
    ty = __shfl(threadInput2, friend_id);
    tz = __shfl(threadInput3, friend_id);
    sum += ARG_3D(11,k,j,i)*((lane_id < 2 )? tx: ((lane_id < 28)? ty: tz));

    friend_id = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput2, friend_id);
    ty = __shfl(threadInput3, friend_id);
    sum += ARG_3D(12,k,j,i)*((lane_id < 22)? tx: ty);

    friend_id = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput2, friend_id);
    ty = __shfl(threadInput3, friend_id);
    sum += ARG_3D(13,k,j,i)*((lane_id < 21)? tx: ty);

    friend_id = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput2, friend_id);
    ty = __shfl(threadInput3, friend_id);
    sum += ARG_3D(14,k,j,i)*((lane_id < 20)? tx: ty);

    friend_id = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput2, friend_id);
    ty = __shfl(threadInput3, friend_id);
    sum += ARG_3D(15,k,j,i)*((lane_id < 14)? tx: ty);

    friend_id = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput2, friend_id);
    ty = __shfl(threadInput3, friend_id);
    sum += ARG_3D(16,k,j,i)*((lane_id < 13)? tx: ty);

    friend_id = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput2, friend_id);
    ty = __shfl(threadInput3, friend_id);
    sum += ARG_3D(17,k,j,i)*((lane_id < 12)? tx: ty);

    friend_id = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput3, friend_id);
    ty = __shfl(threadInput4, friend_id);
    sum += ARG_3D(18,k,j,i)*((lane_id < 8 )? tx: ty);

    friend_id = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput3, friend_id);
    ty = __shfl(threadInput4, friend_id);
    sum += ARG_3D(19,k,j,i)*((lane_id < 7 )? tx: ty);

    friend_id = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput3, friend_id);
    ty = __shfl(threadInput4, friend_id);
    sum += ARG_3D(20,k,j,i)*((lane_id < 6 )? tx: ty);

    friend_id = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput4, friend_id);
    ty = __shfl(threadInput5, friend_id);
    sum += ARG_3D(21,k,j,i)*((lane_id < 24)? tx: ty);

    friend_id = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput4, friend_id);
    ty = __shfl(threadInput5, friend_id);
    sum += ARG_3D(22,k,j,i)*((lane_id < 24)? tx: ty);

    friend_id = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput4, friend_id);
    ty = __shfl(threadInput5, friend_id);
    sum += ARG_3D(23,k,j,i)*((lane_id < 24)? tx: ty);

    friend_id = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput4, friend_id);
    ty = __shfl(threadInput5, friend_id);
    sum += ARG_3D(24,k,j,i)*((lane_id < 16)? tx: ty);

    friend_id = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput4, friend_id);
    ty = __shfl(threadInput5, friend_id);
    sum += ARG_3D(25,k,j,i)*((lane_id < 16)? tx: ty);

    friend_id = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    tx = __shfl(threadInput4, friend_id);
    ty = __shfl(threadInput5, friend_id);
    sum += ARG_3D(26,k,j,i)*((lane_id < 16)? tx: ty);

    
    OUT_3D(k,j,i) = sum;
}

__global__ void Stencil_Cuda_Shfl2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, 
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
    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5,
              threadInput6, threadInput7;
    threadInput0 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+32)/60;
    threadInput1 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10;
    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+64)/60;
    threadInput2 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10;
    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+96)/60;
    threadInput3 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10;
    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+128)/60;
    threadInput4 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10;
    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+160)/60;
    threadInput5 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+192)%10;
    new_j = (warp_id_y<<2) + ((lane_id+192)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+192)/60;
    threadInput6 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+224)%10;
    new_j = (warp_id_y<<2) + ((lane_id+224)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+224)/60;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    new_k = (new_k < z+2*halo)? new_k: z+2*halo-1;
    threadInput7 = IN_3D(new_k, new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    int friend_id0, friend_id1;
    DATA_TYPE tx0, ty0, tz0, tx1, ty1, tz1;

    friend_id0 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tz1 = __shfl(threadInput3, friend_id1);
    sum0 += ARG_3D(0,k  ,j,i)*((lane_id < 26)? tx0: ty0);
    sum1 += ARG_3D(0,k+1,j,i)*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));

    friend_id0 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tz1 = __shfl(threadInput3, friend_id1);
    sum0 += ARG_3D(1,k  ,j,i)*((lane_id < 25)? tx0: ty0);
    sum1 += ARG_3D(1,k+1,j,i)*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));

    friend_id0 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tz1 = __shfl(threadInput3, friend_id1);
    sum0 += ARG_3D(2,k  ,j,i)*((lane_id < 24)? tx0: ty0);
    sum1 += ARG_3D(2,k+1,j,i)*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));

    friend_id0 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput2, friend_id1);
    ty1 = __shfl(threadInput3, friend_id1);
    sum0 += ARG_3D(3,k  ,j,i)*((lane_id < 18)? tx0: ty0);
    sum1 += ARG_3D(3,k+1,j,i)*((lane_id < 22)? tx1: ty1);

    friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput2, friend_id1);
    ty1 = __shfl(threadInput3, friend_id1);
    sum0 += ARG_3D(4,k  ,j,i)*((lane_id < 17)? tx0: ty0);
    sum1 += ARG_3D(4,k+1,j,i)*((lane_id < 21)? tx1: ty1);

    friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput2, friend_id1);
    ty1 = __shfl(threadInput3, friend_id1);
    sum0 += ARG_3D(5,k  ,j,i)*((lane_id < 16)? tx0: ty0);
    sum1 += ARG_3D(5,k+1,j,i)*((lane_id < 20)? tx1: ty1);

    friend_id0 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput2, friend_id1);
    ty1 = __shfl(threadInput3, friend_id1);
    sum0 += ARG_3D(6,k  ,j,i)*((lane_id < 10)? tx0: ty0);
    sum1 += ARG_3D(6,k+1,j,i)*((lane_id < 14)? tx1: ty1);

    friend_id0 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput2, friend_id1);
    ty1 = __shfl(threadInput3, friend_id1);
    sum0 += ARG_3D(7,k  ,j,i)*((lane_id < 9 )? tx0: ty0);
    sum1 += ARG_3D(7,k+1,j,i)*((lane_id < 13)? tx1: ty1);

    friend_id0 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput2, friend_id1);
    ty1 = __shfl(threadInput3, friend_id1);
    sum0 += ARG_3D(8,k  ,j,i)*((lane_id < 8 )? tx0: ty0);
    sum1 += ARG_3D(8,k+1,j,i)*((lane_id < 12)? tx1: ty1);

    friend_id0 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput1, friend_id0);
    ty0 = __shfl(threadInput2, friend_id0);
    tz0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput3, friend_id1);
    ty1 = __shfl(threadInput4, friend_id1);
    sum0 += ARG_3D(9,k  ,j,i)*((lane_id < 4 )? tx0: ((lane_id < 30)? ty0: tz0));
    sum1 += ARG_3D(9,k+1,j,i)*((lane_id < 8)? tx1: ty1);

    friend_id0 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput1, friend_id0);
    ty0 = __shfl(threadInput2, friend_id0);
    tz0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput3, friend_id1);
    ty1 = __shfl(threadInput4, friend_id1);
    sum0 += ARG_3D(10,k  ,j,i)*((lane_id < 3 )? tx0: ((lane_id < 29)? ty0: tz0));
    sum1 += ARG_3D(10,k+1,j,i)*((lane_id < 7)? tx1: ty1);

    friend_id0 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput1, friend_id0);
    ty0 = __shfl(threadInput2, friend_id0);
    tz0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput3, friend_id1);
    ty1 = __shfl(threadInput4, friend_id1);
    sum0 += ARG_3D(11,k  ,j,i)*((lane_id < 2 )? tx0: ((lane_id < 28)? ty0: tz0));
    sum1 += ARG_3D(11,k+1,j,i)*((lane_id < 6)? tx1: ty1);

    friend_id0 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput2, friend_id0);
    ty0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput4, friend_id1);
    ty1 = __shfl(threadInput5, friend_id1);
    sum0 += ARG_3D(12,k  ,j,i)*((lane_id < 22)? tx0: ty0);
    sum1 += ARG_3D(12,k+1,j,i)*((lane_id < 24)? tx1: ty1);

    friend_id0 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput2, friend_id0);
    ty0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput4, friend_id1);
    ty1 = __shfl(threadInput5, friend_id1);
    sum0 += ARG_3D(13,k  ,j,i)*((lane_id < 21)? tx0: ty0);
    sum1 += ARG_3D(13,k+1,j,i)*((lane_id < 24)? tx1: ty1);

    friend_id0 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput2, friend_id0);
    ty0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput4, friend_id1);
    ty1 = __shfl(threadInput5, friend_id1);
    sum0 += ARG_3D(14,k  ,j,i)*((lane_id < 20)? tx0: ty0);
    sum1 += ARG_3D(14,k+1,j,i)*((lane_id < 24)? tx1: ty1);

    friend_id0 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput2, friend_id0);
    ty0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput4, friend_id1);
    ty1 = __shfl(threadInput5, friend_id1);
    sum0 += ARG_3D(15,k  ,j,i)*((lane_id < 14)? tx0: ty0);
    sum1 += ARG_3D(15,k+1,j,i)*((lane_id < 16)? tx1: ty1);

    friend_id0 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput2, friend_id0);
    ty0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput4, friend_id1);
    ty1 = __shfl(threadInput5, friend_id1);
    sum0 += ARG_3D(16,k  ,j,i)*((lane_id < 13)? tx0: ty0);
    sum1 += ARG_3D(16,k+1,j,i)*((lane_id < 16)? tx1: ty1);

    friend_id0 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput2, friend_id0);
    ty0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput4, friend_id1);
    ty1 = __shfl(threadInput5, friend_id1);
    sum0 += ARG_3D(17,k  ,j,i)*((lane_id < 12)? tx0: ty0);
    sum1 += ARG_3D(17,k+1,j,i)*((lane_id < 16)? tx1: ty1);

    friend_id0 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput3, friend_id0);
    ty0 = __shfl(threadInput4, friend_id0);
    tx1 = __shfl(threadInput5, friend_id1);
    ty1 = __shfl(threadInput6, friend_id1);
    sum0 += ARG_3D(18,k  ,j,i)*((lane_id < 8 )? tx0: ty0);
    sum1 += ARG_3D(18,k+1,j,i)*((lane_id < 10)? tx1: ty1);

    friend_id0 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput3, friend_id0);
    ty0 = __shfl(threadInput4, friend_id0);
    tx1 = __shfl(threadInput5, friend_id1);
    ty1 = __shfl(threadInput6, friend_id1);
    sum0 += ARG_3D(19,k  ,j,i)*((lane_id < 7 )? tx0: ty0);
    sum1 += ARG_3D(19,k+1,j,i)*((lane_id < 9 )? tx1: ty1);

    friend_id0 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput3, friend_id0);
    ty0 = __shfl(threadInput4, friend_id0);
    tx1 = __shfl(threadInput5, friend_id1);
    ty1 = __shfl(threadInput6, friend_id1);
    sum0 += ARG_3D(20,k  ,j,i)*((lane_id < 6 )? tx0: ty0);
    sum1 += ARG_3D(20,k+1,j,i)*((lane_id < 8 )? tx1: ty1);

    friend_id0 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput4, friend_id0);
    ty0 = __shfl(threadInput5, friend_id0);
    tx1 = __shfl(threadInput5, friend_id1);
    ty1 = __shfl(threadInput6, friend_id1);
    tz1 = __shfl(threadInput7, friend_id1);
    sum0 += ARG_3D(21,k  ,j,i)*((lane_id < 24)? tx0: ty0);
    sum1 += ARG_3D(21,k+1,j,i)*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));

    friend_id0 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput4, friend_id0);
    ty0 = __shfl(threadInput5, friend_id0);
    tx1 = __shfl(threadInput5, friend_id1);
    ty1 = __shfl(threadInput6, friend_id1);
    tz1 = __shfl(threadInput7, friend_id1);
    sum0 += ARG_3D(22,k  ,j,i)*((lane_id < 24)? tx0: ty0);
    sum1 += ARG_3D(22,k+1,j,i)*((lane_id < 1 )? tx1: ((lane_id < 27)? ty1: tz1));

    friend_id0 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput4, friend_id0);
    ty0 = __shfl(threadInput5, friend_id0);
    tx1 = __shfl(threadInput6, friend_id1);
    ty1 = __shfl(threadInput7, friend_id1);
    sum0 += ARG_3D(23,k  ,j,i)*((lane_id < 24)? tx0: ty0);
    sum1 += ARG_3D(23,k+1,j,i)*((lane_id < 26)? tx1: ty1);

    friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput4, friend_id0);
    ty0 = __shfl(threadInput5, friend_id0);
    tx1 = __shfl(threadInput6, friend_id1);
    ty1 = __shfl(threadInput7, friend_id1);
    sum0 += ARG_3D(24,k  ,j,i)*((lane_id < 16)? tx0: ty0);
    sum1 += ARG_3D(24,k+1,j,i)*((lane_id < 20)? tx1: ty1);

    friend_id0 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput4, friend_id0);
    ty0 = __shfl(threadInput5, friend_id0);
    tx1 = __shfl(threadInput6, friend_id1);
    ty1 = __shfl(threadInput7, friend_id1);
    sum0 += ARG_3D(25,k  ,j,i)*((lane_id < 16)? tx0: ty0);
    sum1 += ARG_3D(25,k+1,j,i)*((lane_id < 19)? tx1: ty1);

    friend_id0 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput4, friend_id0);
    ty0 = __shfl(threadInput5, friend_id0);
    tx1 = __shfl(threadInput6, friend_id1);
    ty1 = __shfl(threadInput7, friend_id1);
    sum0 += ARG_3D(26,k  ,j,i)*((lane_id < 16)? tx0: ty0);
    sum1 += ARG_3D(26,k+1,j,i)*((lane_id < 18)? tx1: ty1);


    OUT_3D(k  ,j,i) = sum0;
    OUT_3D(k+1,j,i) = sum1;

}

__global__ void Stencil_Cuda_Shfl4(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args,
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
    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5,
              threadInput6, threadInput7, threadInput8, threadInput9, threadInput10, threadInput11;
    threadInput0 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+32)/60;
    threadInput1 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10;
    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+64)/60;
    threadInput2 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10;
    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+96)/60;
    threadInput3 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10;
    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+128)/60;
    threadInput4 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10;
    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+160)/60;
    threadInput5 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+192)%10;
    new_j = (warp_id_y<<2) + ((lane_id+192)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+192)/60;
    threadInput6 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+224)%10;
    new_j = (warp_id_y<<2) + ((lane_id+224)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+224)/60;
    threadInput7 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+256)%10;
    new_j = (warp_id_y<<2) + ((lane_id+256)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+256)/60;
    threadInput8 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+288)%10;
    new_j = (warp_id_y<<2) + ((lane_id+288)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+288)/60;
    threadInput9 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+320)%10;
    new_j = (warp_id_y<<2) + ((lane_id+320)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+320)/60;
    threadInput10 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+352)%10;
    new_j = (warp_id_y<<2) + ((lane_id+352)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+352)/60;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    new_k = (new_k < z+2*halo)? new_k: z+2*halo-1;
    threadInput11 = IN_3D(new_k, new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    int friend_id0, friend_id1, friend_id2, friend_id3;
    DATA_TYPE tx0, ty0, tz0, tx1, ty1, tz1, tx2, ty2, tz2, tx3, ty3, tz3;

    friend_id0 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tz1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput3, friend_id2);
    ty2 = __shfl(threadInput4, friend_id2);
    tx3 = __shfl(threadInput5, friend_id3);
    ty3 = __shfl(threadInput6, friend_id3);
    sum0 += ARG_3D(0,k  ,j,i)*((lane_id < 26)? tx0: ty0);
    sum1 += ARG_3D(0,k+1,j,i)*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
    sum2 += ARG_3D(0,k+2,j,i)*((lane_id < 8 )? tx2: ty2);
    sum3 += ARG_3D(0,k+3,j,i)*((lane_id < 10)? tx3: ty3);

    friend_id0 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tz1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput3, friend_id2);
    ty2 = __shfl(threadInput4, friend_id2);
    tx3 = __shfl(threadInput5, friend_id3);
    ty3 = __shfl(threadInput6, friend_id3);
    sum0 += ARG_3D(1,k  ,j,i)*((lane_id < 25)? tx0: ty0);
    sum1 += ARG_3D(1,k+1,j,i)*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
    sum2 += ARG_3D(1,k+2,j,i)*((lane_id < 7 )? tx2: ty2);
    sum3 += ARG_3D(1,k+3,j,i)*((lane_id < 9 )? tx3: ty3);

    friend_id0 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tz1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput3, friend_id2);
    ty2 = __shfl(threadInput4, friend_id2);
    tx3 = __shfl(threadInput5, friend_id3);
    ty3 = __shfl(threadInput6, friend_id3);
    sum0 += ARG_3D(2,k  ,j,i)*((lane_id < 24)? tx0: ty0);
    sum1 += ARG_3D(2,k+1,j,i)*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
    sum2 += ARG_3D(2,k+2,j,i)*((lane_id < 6 )? tx2: ty2);
    sum3 += ARG_3D(2,k+3,j,i)*((lane_id < 8 )? tx3: ty3);

    friend_id0 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput2, friend_id1);
    ty1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput4, friend_id2);
    ty2 = __shfl(threadInput5, friend_id2);
    tx3 = __shfl(threadInput5, friend_id3);
    ty3 = __shfl(threadInput6, friend_id3);
    tz3 = __shfl(threadInput7, friend_id3);
    sum0 += ARG_3D(3,k  ,j,i)*((lane_id < 18)? tx0: ty0);
    sum1 += ARG_3D(3,k+1,j,i)*((lane_id < 22)? tx1: ty1);
    sum2 += ARG_3D(3,k+2,j,i)*((lane_id < 24)? tx2: ty2);
    sum3 += ARG_3D(3,k+3,j,i)*((lane_id < 2 )? tx3: ((lane_id < 28)? ty3: tz3));

    friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput2, friend_id1);
    ty1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput4, friend_id2);
    ty2 = __shfl(threadInput5, friend_id2);
    tx3 = __shfl(threadInput5, friend_id3);
    ty3 = __shfl(threadInput6, friend_id3);
    tz3 = __shfl(threadInput7, friend_id3);
    sum0 += ARG_3D(4,k  ,j,i)*((lane_id < 17)? tx0: ty0);
    sum1 += ARG_3D(4,k+1,j,i)*((lane_id < 21)? tx1: ty1);
    sum2 += ARG_3D(4,k+2,j,i)*((lane_id < 24)? tx2: ty2);
    sum3 += ARG_3D(4,k+3,j,i)*((lane_id < 1 )? tx3: ((lane_id < 27)? ty3: tz3));

    friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput2, friend_id1);
    ty1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput4, friend_id2);
    ty2 = __shfl(threadInput5, friend_id2);
    tx3 = __shfl(threadInput6, friend_id3);
    ty3 = __shfl(threadInput7, friend_id3);
    sum0 += ARG_3D(5,k  ,j,i)*((lane_id < 16)? tx0: ty0);
    sum1 += ARG_3D(5,k+1,j,i)*((lane_id < 20)? tx1: ty1);
    sum2 += ARG_3D(5,k+2,j,i)*((lane_id < 24)? tx2: ty2);
    sum3 += ARG_3D(5,k+3,j,i)*((lane_id < 26)? tx3: ty3);

    friend_id0 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput2, friend_id1);
    ty1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput4, friend_id2);
    ty2 = __shfl(threadInput5, friend_id2);
    tx3 = __shfl(threadInput6, friend_id3);
    ty3 = __shfl(threadInput7, friend_id3);
    sum0 += ARG_3D(6,k  ,j,i)*((lane_id < 10)? tx0: ty0);
    sum1 += ARG_3D(6,k+1,j,i)*((lane_id < 14)? tx1: ty1);
    sum2 += ARG_3D(6,k+2,j,i)*((lane_id < 16)? tx2: ty2);
    sum3 += ARG_3D(6,k+3,j,i)*((lane_id < 20)? tx3: ty3);

    friend_id0 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput2, friend_id1);
    ty1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput4, friend_id2);
    ty2 = __shfl(threadInput5, friend_id2);
    tx3 = __shfl(threadInput6, friend_id3);
    ty3 = __shfl(threadInput7, friend_id3);
    sum0 += ARG_3D(7,k  ,j,i)*((lane_id < 9 )? tx0: ty0);
    sum1 += ARG_3D(7,k+1,j,i)*((lane_id < 13)? tx1: ty1);
    sum2 += ARG_3D(7,k+2,j,i)*((lane_id < 16)? tx2: ty2);
    sum3 += ARG_3D(7,k+3,j,i)*((lane_id < 19)? tx3: ty3);

    friend_id0 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput2, friend_id1);
    ty1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput4, friend_id2);
    ty2 = __shfl(threadInput5, friend_id2);
    tx3 = __shfl(threadInput6, friend_id3);
    ty3 = __shfl(threadInput7, friend_id3);
    sum0 += ARG_3D(8,k  ,j,i)*((lane_id < 8 )? tx0: ty0);
    sum1 += ARG_3D(8,k+1,j,i)*((lane_id < 12)? tx1: ty1);
    sum2 += ARG_3D(8,k+2,j,i)*((lane_id < 16)? tx2: ty2);
    sum3 += ARG_3D(8,k+3,j,i)*((lane_id < 18)? tx3: ty3);

    friend_id0 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput1, friend_id0);
    ty0 = __shfl(threadInput2, friend_id0);
    tz0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput3, friend_id1);
    ty1 = __shfl(threadInput4, friend_id1);
    tx2 = __shfl(threadInput5, friend_id2);
    ty2 = __shfl(threadInput6, friend_id2);
    tx3 = __shfl(threadInput7, friend_id3);
    ty3 = __shfl(threadInput8, friend_id3);
    sum0 += ARG_3D(9,k  ,j,i)*((lane_id < 4 )? tx0: ((lane_id < 30)? ty0: tz0));
    sum1 += ARG_3D(9,k+1,j,i)*((lane_id < 8)? tx1: ty1);
    sum2 += ARG_3D(9,k+2,j,i)*((lane_id < 10)? tx2: ty2);
    sum3 += ARG_3D(9,k+3,j,i)*((lane_id < 14)? tx3: ty3);

    friend_id0 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput1, friend_id0);
    ty0 = __shfl(threadInput2, friend_id0);
    tz0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput3, friend_id1);
    ty1 = __shfl(threadInput4, friend_id1);
    tx2 = __shfl(threadInput5, friend_id2);
    ty2 = __shfl(threadInput6, friend_id2);
    tx3 = __shfl(threadInput7, friend_id3);
    ty3 = __shfl(threadInput8, friend_id3);
    sum0 += ARG_3D(10,k  ,j,i)*((lane_id < 3 )? tx0: ((lane_id < 29)? ty0: tz0));
    sum1 += ARG_3D(10,k+1,j,i)*((lane_id < 7 )? tx1: ty1);
    sum2 += ARG_3D(10,k+2,j,i)*((lane_id < 9 )? tx2: ty2);
    sum3 += ARG_3D(10,k+3,j,i)*((lane_id < 13)? tx3: ty3);

    friend_id0 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput1, friend_id0);
    ty0 = __shfl(threadInput2, friend_id0);
    tz0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput3, friend_id1);
    ty1 = __shfl(threadInput4, friend_id1);
    tx2 = __shfl(threadInput5, friend_id2);
    ty2 = __shfl(threadInput6, friend_id2);
    tx3 = __shfl(threadInput7, friend_id3);
    ty3 = __shfl(threadInput8, friend_id3);
    sum0 += ARG_3D(11,k  ,j,i)*((lane_id < 2 )? tx0: ((lane_id < 28)? ty0: tz0));
    sum1 += ARG_3D(11,k+1,j,i)*((lane_id < 6 )? tx1: ty1);
    sum2 += ARG_3D(11,k+2,j,i)*((lane_id < 8 )? tx2: ty2);
    sum3 += ARG_3D(11,k+3,j,i)*((lane_id < 12)? tx3: ty3);

    friend_id0 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput2, friend_id0);
    ty0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput4, friend_id1);
    ty1 = __shfl(threadInput5, friend_id1);
    tx2 = __shfl(threadInput5, friend_id2);
    ty2 = __shfl(threadInput6, friend_id2);
    tz2 = __shfl(threadInput7, friend_id2);
    tx3 = __shfl(threadInput7, friend_id3);
    ty3 = __shfl(threadInput8, friend_id3);
    sum0 += ARG_3D(12,k  ,j,i)*((lane_id < 22)? tx0: ty0);
    sum1 += ARG_3D(12,k+1,j,i)*((lane_id < 24)? tx1: ty1);
    sum2 += ARG_3D(12,k+2,j,i)*((lane_id < 2 )? tx2: ((lane_id < 28)? ty2: tz2));
    sum3 += ARG_3D(12,k+3,j,i)*((lane_id < 6 )? tx3: ty3);

    friend_id0 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput2, friend_id0);
    ty0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput4, friend_id1);
    ty1 = __shfl(threadInput5, friend_id1);
    tx2 = __shfl(threadInput5, friend_id2);
    ty2 = __shfl(threadInput6, friend_id2);
    tz2 = __shfl(threadInput7, friend_id2);
    tx3 = __shfl(threadInput7, friend_id3);
    ty3 = __shfl(threadInput8, friend_id3);
    tz3 = __shfl(threadInput9, friend_id3);
    sum0 += ARG_3D(13,k  ,j,i)*((lane_id < 21)? tx0: ty0);
    sum1 += ARG_3D(13,k+1,j,i)*((lane_id < 24)? tx1: ty1);
    sum2 += ARG_3D(13,k+2,j,i)*((lane_id < 1 )? tx2: ((lane_id < 27)? ty2: tz2));
    sum3 += ARG_3D(13,k+3,j,i)*((lane_id < 5 )? tx3: ((lane_id < 31)? ty3: tz3));

    friend_id0 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput2, friend_id0);
    ty0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput4, friend_id1);
    ty1 = __shfl(threadInput5, friend_id1);
    tx2 = __shfl(threadInput6, friend_id2);
    ty2 = __shfl(threadInput7, friend_id2);
    tx3 = __shfl(threadInput7, friend_id3);
    ty3 = __shfl(threadInput8, friend_id3);
    tz3 = __shfl(threadInput9, friend_id3);
    sum0 += ARG_3D(14,k  ,j,i)*((lane_id < 20)? tx0: ty0);
    sum1 += ARG_3D(14,k+1,j,i)*((lane_id < 24)? tx1: ty1);
    sum2 += ARG_3D(14,k+2,j,i)*((lane_id < 26)? tx2: ty2);
    sum3 += ARG_3D(14,k+3,j,i)*((lane_id < 4 )? tx3: ((lane_id < 30)? ty3: tz3));

    friend_id0 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput2, friend_id0);
    ty0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput4, friend_id1);
    ty1 = __shfl(threadInput5, friend_id1);
    tx2 = __shfl(threadInput6, friend_id2);
    ty2 = __shfl(threadInput7, friend_id2);
    tx3 = __shfl(threadInput8, friend_id3);
    ty3 = __shfl(threadInput9, friend_id3);
    sum0 += ARG_3D(15,k  ,j,i)*((lane_id < 14)? tx0: ty0);
    sum1 += ARG_3D(15,k+1,j,i)*((lane_id < 16)? tx1: ty1);
    sum2 += ARG_3D(15,k+2,j,i)*((lane_id < 20)? tx2: ty2);
    sum3 += ARG_3D(15,k+3,j,i)*((lane_id < 24)? tx3: ty3);

    friend_id0 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput2, friend_id0);
    ty0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput4, friend_id1);
    ty1 = __shfl(threadInput5, friend_id1);
    tx2 = __shfl(threadInput6, friend_id2);
    ty2 = __shfl(threadInput7, friend_id2);
    tx3 = __shfl(threadInput8, friend_id3);
    ty3 = __shfl(threadInput9, friend_id3);
    sum0 += ARG_3D(16,k  ,j,i)*((lane_id < 13)? tx0: ty0);
    sum1 += ARG_3D(16,k+1,j,i)*((lane_id < 16)? tx1: ty1);
    sum2 += ARG_3D(16,k+2,j,i)*((lane_id < 19)? tx2: ty2);
    sum3 += ARG_3D(16,k+3,j,i)*((lane_id < 23)? tx3: ty3);

    friend_id0 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput2, friend_id0);
    ty0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput4, friend_id1);
    ty1 = __shfl(threadInput5, friend_id1);
    tx2 = __shfl(threadInput6, friend_id2);
    ty2 = __shfl(threadInput7, friend_id2);
    tx3 = __shfl(threadInput8, friend_id3);
    ty3 = __shfl(threadInput9, friend_id3);
    sum0 += ARG_3D(17,k  ,j,i)*((lane_id < 12)? tx0: ty0);
    sum1 += ARG_3D(17,k+1,j,i)*((lane_id < 16)? tx1: ty1);
    sum2 += ARG_3D(17,k+2,j,i)*((lane_id < 18)? tx2: ty2);
    sum3 += ARG_3D(17,k+3,j,i)*((lane_id < 22)? tx3: ty3);

    friend_id0 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput3, friend_id0);
    ty0 = __shfl(threadInput4, friend_id0);
    tx1 = __shfl(threadInput5, friend_id1);
    ty1 = __shfl(threadInput6, friend_id1);
    tx2 = __shfl(threadInput7, friend_id2);
    ty2 = __shfl(threadInput8, friend_id2);
    tx3 = __shfl(threadInput9 , friend_id3);
    ty3 = __shfl(threadInput10, friend_id3);
    sum0 += ARG_3D(18,k  ,j,i)*((lane_id < 8 )? tx0: ty0);
    sum1 += ARG_3D(18,k+1,j,i)*((lane_id < 10)? tx1: ty1);
    sum2 += ARG_3D(18,k+2,j,i)*((lane_id < 14)? tx2: ty2);
    sum3 += ARG_3D(18,k+3,j,i)*((lane_id < 16)? tx3: ty3);

    friend_id0 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput3, friend_id0);
    ty0 = __shfl(threadInput4, friend_id0);
    tx1 = __shfl(threadInput5, friend_id1);
    ty1 = __shfl(threadInput6, friend_id1);
    tx2 = __shfl(threadInput7, friend_id2);
    ty2 = __shfl(threadInput8, friend_id2);
    tx3 = __shfl(threadInput9 , friend_id3);
    ty3 = __shfl(threadInput10, friend_id3);
    sum0 += ARG_3D(19,k  ,j,i)*((lane_id < 7 )? tx0: ty0);
    sum1 += ARG_3D(19,k+1,j,i)*((lane_id < 9 )? tx1: ty1);
    sum2 += ARG_3D(19,k+2,j,i)*((lane_id < 13)? tx2: ty2);
    sum3 += ARG_3D(19,k+3,j,i)*((lane_id < 16)? tx3: ty3);

    friend_id0 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput3, friend_id0);
    ty0 = __shfl(threadInput4, friend_id0);
    tx1 = __shfl(threadInput5, friend_id1);
    ty1 = __shfl(threadInput6, friend_id1);
    tx2 = __shfl(threadInput7, friend_id2);
    ty2 = __shfl(threadInput8, friend_id2);
    tx3 = __shfl(threadInput9 , friend_id3);
    ty3 = __shfl(threadInput10, friend_id3);
    sum0 += ARG_3D(20,k  ,j,i)*((lane_id < 6 )? tx0: ty0);
    sum1 += ARG_3D(20,k+1,j,i)*((lane_id < 8 )? tx1: ty1);
    sum2 += ARG_3D(20,k+2,j,i)*((lane_id < 12)? tx2: ty2);
    sum3 += ARG_3D(20,k+3,j,i)*((lane_id < 16)? tx3: ty3);

    friend_id0 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput4, friend_id0);
    ty0 = __shfl(threadInput5, friend_id0);
    tx1 = __shfl(threadInput5, friend_id1);
    ty1 = __shfl(threadInput6, friend_id1);
    tz1 = __shfl(threadInput7, friend_id1);
    tx2 = __shfl(threadInput7, friend_id2);
    ty2 = __shfl(threadInput8, friend_id2);
    tx3 = __shfl(threadInput9 , friend_id3);
    ty3 = __shfl(threadInput10, friend_id3);
    sum0 += ARG_3D(21,k  ,j,i)*((lane_id < 24)? tx0: ty0);
    sum1 += ARG_3D(21,k+1,j,i)*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
    sum2 += ARG_3D(21,k+2,j,i)*((lane_id < 6 )? tx2: ty2);
    sum3 += ARG_3D(21,k+3,j,i)*((lane_id < 8 )? tx3: ty3);

    friend_id0 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+23+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput4, friend_id0);
    ty0 = __shfl(threadInput5, friend_id0);
    tx1 = __shfl(threadInput5, friend_id1);
    ty1 = __shfl(threadInput6, friend_id1);
    tz1 = __shfl(threadInput7, friend_id1);
    tx2 = __shfl(threadInput7, friend_id2);
    ty2 = __shfl(threadInput8, friend_id2);
    tz2 = __shfl(threadInput9, friend_id2);
    tx3 = __shfl(threadInput9 , friend_id3);
    ty3 = __shfl(threadInput10, friend_id3);
    sum0 += ARG_3D(22,k  ,j,i)*((lane_id < 24)? tx0: ty0);
    sum1 += ARG_3D(22,k+1,j,i)*((lane_id < 1 )? tx1: ((lane_id < 27)? ty1: tz1));
    sum2 += ARG_3D(22,k+2,j,i)*((lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2));
    sum3 += ARG_3D(22,k+3,j,i)*((lane_id < 8 )? tx3: ty3);

    friend_id0 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput4, friend_id0);
    ty0 = __shfl(threadInput5, friend_id0);
    tx1 = __shfl(threadInput6, friend_id1);
    ty1 = __shfl(threadInput7, friend_id1);
    tx2 = __shfl(threadInput7, friend_id2);
    ty2 = __shfl(threadInput8, friend_id2);
    tz2 = __shfl(threadInput9, friend_id2);
    tx3 = __shfl(threadInput9 , friend_id3);
    ty3 = __shfl(threadInput10, friend_id3);
    sum0 += ARG_3D(23,k  ,j,i)*((lane_id < 24)? tx0: ty0);
    sum1 += ARG_3D(23,k+1,j,i)*((lane_id < 26)? tx1: ty1);
    sum2 += ARG_3D(23,k+2,j,i)*((lane_id < 4 )? tx2: ((lane_id < 30)? ty2: tz2));
    sum3 += ARG_3D(23,k+3,j,i)*((lane_id < 8 )? tx3: ty3);

    friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput4, friend_id0);
    ty0 = __shfl(threadInput5, friend_id0);
    tx1 = __shfl(threadInput6, friend_id1);
    ty1 = __shfl(threadInput7, friend_id1);
    tx2 = __shfl(threadInput8, friend_id2);
    ty2 = __shfl(threadInput9, friend_id2);
    tx3 = __shfl(threadInput10, friend_id3);
    ty3 = __shfl(threadInput11, friend_id3);
    sum0 += ARG_3D(24,k  ,j,i)*((lane_id < 16)? tx0: ty0);
    sum1 += ARG_3D(24,k+1,j,i)*((lane_id < 20)? tx1: ty1);
    sum2 += ARG_3D(24,k+2,j,i)*((lane_id < 24)? tx2: ty2);
    sum3 += ARG_3D(24,k+3,j,i)*((lane_id < 26)? tx3: ty3);

    friend_id0 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput4, friend_id0);
    ty0 = __shfl(threadInput5, friend_id0);
    tx1 = __shfl(threadInput6, friend_id1);
    ty1 = __shfl(threadInput7, friend_id1);
    tx2 = __shfl(threadInput8, friend_id2);
    ty2 = __shfl(threadInput9, friend_id2);
    tx3 = __shfl(threadInput10, friend_id3);
    ty3 = __shfl(threadInput11, friend_id3);
    sum0 += ARG_3D(25,k  ,j,i)*((lane_id < 16)? tx0: ty0);
    sum1 += ARG_3D(25,k+1,j,i)*((lane_id < 19)? tx1: ty1);
    sum2 += ARG_3D(25,k+2,j,i)*((lane_id < 23)? tx2: ty2);
    sum3 += ARG_3D(25,k+3,j,i)*((lane_id < 25)? tx3: ty3);

    friend_id0 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput4, friend_id0);
    ty0 = __shfl(threadInput5, friend_id0);
    tx1 = __shfl(threadInput6, friend_id1);
    ty1 = __shfl(threadInput7, friend_id1);
    tx2 = __shfl(threadInput8, friend_id2);
    ty2 = __shfl(threadInput9, friend_id2);
    tx3 = __shfl(threadInput10, friend_id3);
    ty3 = __shfl(threadInput11, friend_id3);
    sum0 += ARG_3D(26,k  ,j,i)*((lane_id < 16)? tx0: ty0);
    sum1 += ARG_3D(26,k+1,j,i)*((lane_id < 18)? tx1: ty1);
    sum2 += ARG_3D(26,k+2,j,i)*((lane_id < 22)? tx2: ty2);
    sum3 += ARG_3D(26,k+3,j,i)*((lane_id < 24)? tx3: ty3);


    OUT_3D(k  ,j,i) = sum0;
    OUT_3D(k+1,j,i) = sum1;
    OUT_3D(k+2,j,i) = sum2;
    OUT_3D(k+3,j,i) = sum3;
}

__global__ void Stencil_Cuda_Shfl8(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, 
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
    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5,
              threadInput6, threadInput7, threadInput8, threadInput9, threadInput10, threadInput11,
              threadInput12, threadInput13, threadInput14, threadInput15, threadInput16, threadInput17,
              threadInput18;
    threadInput0 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+32)/60;
    threadInput1 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10;
    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+64)/60;
    threadInput2 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10;
    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+96)/60;
    threadInput3 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10;
    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+128)/60;
    threadInput4 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10;
    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+160)/60;
    threadInput5 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+192)%10;
    new_j = (warp_id_y<<2) + ((lane_id+192)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+192)/60;
    threadInput6 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+224)%10;
    new_j = (warp_id_y<<2) + ((lane_id+224)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+224)/60;
    threadInput7 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+256)%10;
    new_j = (warp_id_y<<2) + ((lane_id+256)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+256)/60;
    threadInput8 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+288)%10;
    new_j = (warp_id_y<<2) + ((lane_id+288)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+288)/60;
    threadInput9 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+320)%10;
    new_j = (warp_id_y<<2) + ((lane_id+320)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+320)/60;
    threadInput10 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+352)%10;
    new_j = (warp_id_y<<2) + ((lane_id+352)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+352)/60;
    threadInput11 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+384)%10;
    new_j = (warp_id_y<<2) + ((lane_id+384)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+384)/60;
    threadInput12 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+416)%10;
    new_j = (warp_id_y<<2) + ((lane_id+416)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+416)/60;
    threadInput13 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+448)%10;
    new_j = (warp_id_y<<2) + ((lane_id+448)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+448)/60;
    threadInput14 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+480)%10;
    new_j = (warp_id_y<<2) + ((lane_id+480)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+480)/60;
    threadInput15 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+512)%10;
    new_j = (warp_id_y<<2) + ((lane_id+512)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+512)/60;
    threadInput16 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+544)%10;
    new_j = (warp_id_y<<2) + ((lane_id+544)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+544)/60;
    threadInput17 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+576)%10;
    new_j = (warp_id_y<<2) + ((lane_id+576)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+576)/60;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    new_k = (new_k < z+2*halo)? new_k: z+2*halo-1;
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

    friend_id0 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tz1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput3, friend_id2);
    ty2 = __shfl(threadInput4, friend_id2);
    tx3 = __shfl(threadInput5, friend_id3);
    ty3 = __shfl(threadInput6, friend_id3);
    rx0 = __shfl(threadInput7, friend_id4);
    ry0 = __shfl(threadInput8, friend_id4);
    rx1 = __shfl(threadInput9 , friend_id5);
    ry1 = __shfl(threadInput10, friend_id5);
    rx2 = __shfl(threadInput11, friend_id6);
    ry2 = __shfl(threadInput12, friend_id6);
    rx3 = __shfl(threadInput13, friend_id7);
    ry3 = __shfl(threadInput14, friend_id7);
    sum0 += ARG_3D(0,k  ,j,i)*((lane_id < 26)? tx0: ty0);
    sum1 += ARG_3D(0,k+1,j,i)*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
    sum2 += ARG_3D(0,k+2,j,i)*((lane_id < 8 )? tx2: ty2);
    sum3 += ARG_3D(0,k+3,j,i)*((lane_id < 10)? tx3: ty3);
    sum4 += ARG_3D(0,k+4,j,i)*((lane_id < 14)? rx0: ry0);
    sum5 += ARG_3D(0,k+5,j,i)*((lane_id < 16)? rx1: ry1);
    sum6 += ARG_3D(0,k+6,j,i)*((lane_id < 20)? rx2: ry2);
    sum7 += ARG_3D(0,k+7,j,i)*((lane_id < 24)? rx3: ry3);

    friend_id0 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tz1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput3, friend_id2);
    ty2 = __shfl(threadInput4, friend_id2);
    tx3 = __shfl(threadInput5, friend_id3);
    ty3 = __shfl(threadInput6, friend_id3);
    rx0 = __shfl(threadInput7, friend_id4);
    ry0 = __shfl(threadInput8, friend_id4);
    rx1 = __shfl(threadInput9 , friend_id5);
    ry1 = __shfl(threadInput10, friend_id5);
    rx2 = __shfl(threadInput11, friend_id6);
    ry2 = __shfl(threadInput12, friend_id6);
    rx3 = __shfl(threadInput13, friend_id7);
    ry3 = __shfl(threadInput14, friend_id7);
    sum0 += ARG_3D(1,k  ,j,i)*((lane_id < 25)? tx0: ty0);
    sum1 += ARG_3D(1,k+1,j,i)*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
    sum2 += ARG_3D(1,k+2,j,i)*((lane_id < 7 )? tx2: ty2);
    sum3 += ARG_3D(1,k+3,j,i)*((lane_id < 9 )? tx3: ty3);
    sum4 += ARG_3D(1,k+4,j,i)*((lane_id < 13)? rx0: ry0);
    sum5 += ARG_3D(1,k+5,j,i)*((lane_id < 16)? rx1: ry1);
    sum6 += ARG_3D(1,k+6,j,i)*((lane_id < 19)? rx2: ry2);
    sum7 += ARG_3D(1,k+7,j,i)*((lane_id < 23)? rx3: ry3);

    friend_id0 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput1, friend_id1);
    ty1 = __shfl(threadInput2, friend_id1);
    tz1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput3, friend_id2);
    ty2 = __shfl(threadInput4, friend_id2);
    tx3 = __shfl(threadInput5, friend_id3);
    ty3 = __shfl(threadInput6, friend_id3);
    rx0 = __shfl(threadInput7, friend_id4);
    ry0 = __shfl(threadInput8, friend_id4);
    rx1 = __shfl(threadInput9 , friend_id5);
    ry1 = __shfl(threadInput10, friend_id5);
    rx2 = __shfl(threadInput11, friend_id6);
    ry2 = __shfl(threadInput12, friend_id6);
    rx3 = __shfl(threadInput13, friend_id7);
    ry3 = __shfl(threadInput14, friend_id7);
    sum0 += ARG_3D(2,k  ,j,i)*((lane_id < 24)? tx0: ty0);
    sum1 += ARG_3D(2,k+1,j,i)*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
    sum2 += ARG_3D(2,k+2,j,i)*((lane_id < 6 )? tx2: ty2);
    sum3 += ARG_3D(2,k+3,j,i)*((lane_id < 8 )? tx3: ty3);
    sum4 += ARG_3D(2,k+4,j,i)*((lane_id < 12)? rx0: ry0);
    sum5 += ARG_3D(2,k+5,j,i)*((lane_id < 16)? rx1: ry1);
    sum6 += ARG_3D(2,k+6,j,i)*((lane_id < 18)? rx2: ry2);
    sum7 += ARG_3D(2,k+7,j,i)*((lane_id < 22)? rx3: ry3);

    friend_id0 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput2, friend_id1);
    ty1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput4, friend_id2);
    ty2 = __shfl(threadInput5, friend_id2);
    tx3 = __shfl(threadInput5, friend_id3);
    ty3 = __shfl(threadInput6, friend_id3);
    tz3 = __shfl(threadInput7, friend_id3);
    rx0 = __shfl(threadInput7, friend_id4);
    ry0 = __shfl(threadInput8, friend_id4);
    rx1 = __shfl(threadInput9 , friend_id5);
    ry1 = __shfl(threadInput10, friend_id5);
    rx2 = __shfl(threadInput11, friend_id6);
    ry2 = __shfl(threadInput12, friend_id6);
    rx3 = __shfl(threadInput13, friend_id7);
    ry3 = __shfl(threadInput14, friend_id7);
    sum0 += ARG_3D(3,k  ,j,i)*((lane_id < 18)? tx0: ty0);
    sum1 += ARG_3D(3,k+1,j,i)*((lane_id < 22)? tx1: ty1);
    sum2 += ARG_3D(3,k+2,j,i)*((lane_id < 24)? tx2: ty2);
    sum3 += ARG_3D(3,k+3,j,i)*((lane_id < 2 )? tx3: ((lane_id < 28)? ty3: tz3));
    sum4 += ARG_3D(3,k+4,j,i)*((lane_id < 6 )? rx0: ry0);
    sum5 += ARG_3D(3,k+5,j,i)*((lane_id < 8 )? rx1: ry1);
    sum6 += ARG_3D(3,k+6,j,i)*((lane_id < 12)? rx2: ry2);
    sum7 += ARG_3D(3,k+7,j,i)*((lane_id < 16)? rx3: ry3);

    friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+23+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+15+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput2, friend_id1);
    ty1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput4, friend_id2);
    ty2 = __shfl(threadInput5, friend_id2);
    tx3 = __shfl(threadInput5, friend_id3);
    ty3 = __shfl(threadInput6, friend_id3);
    tz3 = __shfl(threadInput7, friend_id3);
    rx0 = __shfl(threadInput7, friend_id4);
    ry0 = __shfl(threadInput8, friend_id4);
    rz0 = __shfl(threadInput9, friend_id4);
    rx1 = __shfl(threadInput9 , friend_id5);
    ry1 = __shfl(threadInput10, friend_id5);
    rx2 = __shfl(threadInput11, friend_id6);
    ry2 = __shfl(threadInput12, friend_id6);
    rx3 = __shfl(threadInput13, friend_id7);
    ry3 = __shfl(threadInput14, friend_id7);
    sum0 += ARG_3D(4,k  ,j,i)*((lane_id < 17)? tx0: ty0);
    sum1 += ARG_3D(4,k+1,j,i)*((lane_id < 21)? tx1: ty1);
    sum2 += ARG_3D(4,k+2,j,i)*((lane_id < 24)? tx2: ty2);
    sum3 += ARG_3D(4,k+3,j,i)*((lane_id < 1 )? tx3: ((lane_id < 27)? ty3: tz3));
    sum4 += ARG_3D(4,k+4,j,i)*((lane_id < 5 )? rx0: ((lane_id < 31)? ry0: rz0));
    sum5 += ARG_3D(4,k+5,j,i)*((lane_id < 8 )? rx1: ry1);
    sum6 += ARG_3D(4,k+6,j,i)*((lane_id < 11)? rx2: ry2);
    sum7 += ARG_3D(4,k+7,j,i)*((lane_id < 15)? rx3: ry3);

    friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput2, friend_id1);
    ty1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput4, friend_id2);
    ty2 = __shfl(threadInput5, friend_id2);
    tx3 = __shfl(threadInput6, friend_id3);
    ty3 = __shfl(threadInput7, friend_id3);
    rx0 = __shfl(threadInput7, friend_id4);
    ry0 = __shfl(threadInput8, friend_id4);
    rz0 = __shfl(threadInput9, friend_id4);
    rx1 = __shfl(threadInput9 , friend_id5);
    ry1 = __shfl(threadInput10, friend_id5);
    rx2 = __shfl(threadInput11, friend_id6);
    ry2 = __shfl(threadInput12, friend_id6);
    rx3 = __shfl(threadInput13, friend_id7);
    ry3 = __shfl(threadInput14, friend_id7);
    sum0 += ARG_3D(5,k  ,j,i)*((lane_id < 16)? tx0: ty0);
    sum1 += ARG_3D(5,k+1,j,i)*((lane_id < 20)? tx1: ty1);
    sum2 += ARG_3D(5,k+2,j,i)*((lane_id < 24)? tx2: ty2);
    sum3 += ARG_3D(5,k+3,j,i)*((lane_id < 26)? tx3: ty3);
    sum4 += ARG_3D(5,k+4,j,i)*((lane_id < 4 )? rx0: ((lane_id < 30)? ry0: rz0));
    sum5 += ARG_3D(5,k+5,j,i)*((lane_id < 8 )? rx1: ry1);
    sum6 += ARG_3D(5,k+6,j,i)*((lane_id < 10)? rx2: ry2);
    sum7 += ARG_3D(5,k+7,j,i)*((lane_id < 14)? rx3: ry3);

    friend_id0 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput2, friend_id1);
    ty1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput4, friend_id2);
    ty2 = __shfl(threadInput5, friend_id2);
    tx3 = __shfl(threadInput6, friend_id3);
    ty3 = __shfl(threadInput7, friend_id3);
    rx0 = __shfl(threadInput8, friend_id4);
    ry0 = __shfl(threadInput9, friend_id4);
    rx1 = __shfl(threadInput10, friend_id5);
    ry1 = __shfl(threadInput11, friend_id5);
    rx2 = __shfl(threadInput11, friend_id6);
    ry2 = __shfl(threadInput12, friend_id6);
    rz2 = __shfl(threadInput13, friend_id6);
    rx3 = __shfl(threadInput13, friend_id7);
    ry3 = __shfl(threadInput14, friend_id7);
    sum0 += ARG_3D(6,k  ,j,i)*((lane_id < 10)? tx0: ty0);
    sum1 += ARG_3D(6,k+1,j,i)*((lane_id < 14)? tx1: ty1);
    sum2 += ARG_3D(6,k+2,j,i)*((lane_id < 16)? tx2: ty2);
    sum3 += ARG_3D(6,k+3,j,i)*((lane_id < 20)? tx3: ty3);
    sum4 += ARG_3D(6,k+4,j,i)*((lane_id < 24)? rx0: ry0);
    sum5 += ARG_3D(6,k+5,j,i)*((lane_id < 26)? rx1: ry1);
    sum6 += ARG_3D(6,k+6,j,i)*((lane_id < 4 )? rx2: ((lane_id < 30)? ry2: rz2));
    sum7 += ARG_3D(6,k+7,j,i)*((lane_id < 8 )? rx3: ry3);

    friend_id0 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput2, friend_id1);
    ty1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput4, friend_id2);
    ty2 = __shfl(threadInput5, friend_id2);
    tx3 = __shfl(threadInput6, friend_id3);
    ty3 = __shfl(threadInput7, friend_id3);
    rx0 = __shfl(threadInput8, friend_id4);
    ry0 = __shfl(threadInput9, friend_id4);
    rx1 = __shfl(threadInput10, friend_id5);
    ry1 = __shfl(threadInput11, friend_id5);
    rx2 = __shfl(threadInput11, friend_id6);
    ry2 = __shfl(threadInput12, friend_id6);
    rz2 = __shfl(threadInput13, friend_id6);
    rx3 = __shfl(threadInput13, friend_id7);
    ry3 = __shfl(threadInput14, friend_id7);
    sum0 += ARG_3D(7,k  ,j,i)*((lane_id < 9 )? tx0: ty0);
    sum1 += ARG_3D(7,k+1,j,i)*((lane_id < 13)? tx1: ty1);
    sum2 += ARG_3D(7,k+2,j,i)*((lane_id < 16)? tx2: ty2);
    sum3 += ARG_3D(7,k+3,j,i)*((lane_id < 19)? tx3: ty3);
    sum4 += ARG_3D(7,k+4,j,i)*((lane_id < 23)? rx0: ry0);
    sum5 += ARG_3D(7,k+5,j,i)*((lane_id < 25)? rx1: ry1);
    sum6 += ARG_3D(7,k+6,j,i)*((lane_id < 3 )? rx2: ((lane_id < 29)? ry2: rz2));
    sum7 += ARG_3D(7,k+7,j,i)*((lane_id < 7 )? rx3: ry3);

    friend_id0 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput0, friend_id0);
    ty0 = __shfl(threadInput1, friend_id0);
    tx1 = __shfl(threadInput2, friend_id1);
    ty1 = __shfl(threadInput3, friend_id1);
    tx2 = __shfl(threadInput4, friend_id2);
    ty2 = __shfl(threadInput5, friend_id2);
    tx3 = __shfl(threadInput6, friend_id3);
    ty3 = __shfl(threadInput7, friend_id3);
    rx0 = __shfl(threadInput8, friend_id4);
    ry0 = __shfl(threadInput9, friend_id4);
    rx1 = __shfl(threadInput10, friend_id5);
    ry1 = __shfl(threadInput11, friend_id5);
    rx2 = __shfl(threadInput11, friend_id6);
    ry2 = __shfl(threadInput12, friend_id6);
    rz2 = __shfl(threadInput13, friend_id6);
    rx3 = __shfl(threadInput13, friend_id7);
    ry3 = __shfl(threadInput14, friend_id7);
    sum0 += ARG_3D(8,k  ,j,i)*((lane_id < 8 )? tx0: ty0);
    sum1 += ARG_3D(8,k+1,j,i)*((lane_id < 12)? tx1: ty1);
    sum2 += ARG_3D(8,k+2,j,i)*((lane_id < 16)? tx2: ty2);
    sum3 += ARG_3D(8,k+3,j,i)*((lane_id < 18)? tx3: ty3);
    sum4 += ARG_3D(8,k+4,j,i)*((lane_id < 22)? rx0: ry0);
    sum5 += ARG_3D(8,k+5,j,i)*((lane_id < 24)? rx1: ry1);
    sum6 += ARG_3D(8,k+6,j,i)*((lane_id < 2 )? rx2: ((lane_id < 28)? ry2: rz2));
    sum7 += ARG_3D(8,k+7,j,i)*((lane_id < 6 )? rx3: ry3);

    friend_id0 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput1, friend_id0);
    ty0 = __shfl(threadInput2, friend_id0);
    tz0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput3, friend_id1);
    ty1 = __shfl(threadInput4, friend_id1);
    tx2 = __shfl(threadInput5, friend_id2);
    ty2 = __shfl(threadInput6, friend_id2);
    tx3 = __shfl(threadInput7, friend_id3);
    ty3 = __shfl(threadInput8, friend_id3);
    rx0 = __shfl(threadInput9 , friend_id4);
    ry0 = __shfl(threadInput10, friend_id4);
    rx1 = __shfl(threadInput11, friend_id5);
    ry1 = __shfl(threadInput12, friend_id5);
    rx2 = __shfl(threadInput13, friend_id6);
    ry2 = __shfl(threadInput14, friend_id6);
    rx3 = __shfl(threadInput15, friend_id7);
    ry3 = __shfl(threadInput16, friend_id7);
    sum0 += ARG_3D(9,k  ,j,i)*((lane_id < 4 )? tx0: ((lane_id < 30)? ty0: tz0));
    sum1 += ARG_3D(9,k+1,j,i)*((lane_id < 8)? tx1: ty1);
    sum2 += ARG_3D(9,k+2,j,i)*((lane_id < 10)? tx2: ty2);
    sum3 += ARG_3D(9,k+3,j,i)*((lane_id < 14)? tx3: ty3);
    sum4 += ARG_3D(9,k+4,j,i)*((lane_id < 16)? rx0: ry0);
    sum5 += ARG_3D(9,k+5,j,i)*((lane_id < 20)? rx1: ry1);
    sum6 += ARG_3D(9,k+6,j,i)*((lane_id < 24)? rx2: ry2);
    sum7 += ARG_3D(9,k+7,j,i)*((lane_id < 26)? rx3: ry3);

    friend_id0 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput1, friend_id0);
    ty0 = __shfl(threadInput2, friend_id0);
    tz0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput3, friend_id1);
    ty1 = __shfl(threadInput4, friend_id1);
    tx2 = __shfl(threadInput5, friend_id2);
    ty2 = __shfl(threadInput6, friend_id2);
    tx3 = __shfl(threadInput7, friend_id3);
    ty3 = __shfl(threadInput8, friend_id3);
    rx0 = __shfl(threadInput9 , friend_id4);
    ry0 = __shfl(threadInput10, friend_id4);
    rx1 = __shfl(threadInput11, friend_id5);
    ry1 = __shfl(threadInput12, friend_id5);
    rx2 = __shfl(threadInput13, friend_id6);
    ry2 = __shfl(threadInput14, friend_id6);
    rx3 = __shfl(threadInput15, friend_id7);
    ry3 = __shfl(threadInput16, friend_id7);
    sum0 += ARG_3D(10,k  ,j,i)*((lane_id < 3 )? tx0: ((lane_id < 29)? ty0: tz0));
    sum1 += ARG_3D(10,k+1,j,i)*((lane_id < 7 )? tx1: ty1);
    sum2 += ARG_3D(10,k+2,j,i)*((lane_id < 9 )? tx2: ty2);
    sum3 += ARG_3D(10,k+3,j,i)*((lane_id < 13)? tx3: ty3);
    sum4 += ARG_3D(10,k+4,j,i)*((lane_id < 16)? rx0: ry0);
    sum5 += ARG_3D(10,k+5,j,i)*((lane_id < 19)? rx1: ry1);
    sum6 += ARG_3D(10,k+6,j,i)*((lane_id < 23)? rx2: ry2);
    sum7 += ARG_3D(10,k+7,j,i)*((lane_id < 25)? rx3: ry3);

    friend_id0 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput1, friend_id0);
    ty0 = __shfl(threadInput2, friend_id0);
    tz0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput3, friend_id1);
    ty1 = __shfl(threadInput4, friend_id1);
    tx2 = __shfl(threadInput5, friend_id2);
    ty2 = __shfl(threadInput6, friend_id2);
    tx3 = __shfl(threadInput7, friend_id3);
    ty3 = __shfl(threadInput8, friend_id3);
    rx0 = __shfl(threadInput9 , friend_id4);
    ry0 = __shfl(threadInput10, friend_id4);
    rx1 = __shfl(threadInput11, friend_id5);
    ry1 = __shfl(threadInput12, friend_id5);
    rx2 = __shfl(threadInput13, friend_id6);
    ry2 = __shfl(threadInput14, friend_id6);
    rx3 = __shfl(threadInput15, friend_id7);
    ry3 = __shfl(threadInput16, friend_id7);
    sum0 += ARG_3D(11,k  ,j,i)*((lane_id < 2 )? tx0: ((lane_id < 28)? ty0: tz0));
    sum1 += ARG_3D(11,k+1,j,i)*((lane_id < 6 )? tx1: ty1);
    sum2 += ARG_3D(11,k+2,j,i)*((lane_id < 8 )? tx2: ty2);
    sum3 += ARG_3D(11,k+3,j,i)*((lane_id < 12)? tx3: ty3);
    sum4 += ARG_3D(11,k+4,j,i)*((lane_id < 16)? rx0: ry0);
    sum5 += ARG_3D(11,k+5,j,i)*((lane_id < 18)? rx1: ry1);
    sum6 += ARG_3D(11,k+6,j,i)*((lane_id < 22)? rx2: ry2);
    sum7 += ARG_3D(11,k+7,j,i)*((lane_id < 24)? rx3: ry3);

    friend_id0 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput2, friend_id0);
    ty0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput4, friend_id1);
    ty1 = __shfl(threadInput5, friend_id1);
    tx2 = __shfl(threadInput5, friend_id2);
    ty2 = __shfl(threadInput6, friend_id2);
    tz2 = __shfl(threadInput7, friend_id2);
    tx3 = __shfl(threadInput7, friend_id3);
    ty3 = __shfl(threadInput8, friend_id3);
    rx0 = __shfl(threadInput9 , friend_id4);
    ry0 = __shfl(threadInput10, friend_id4);
    rx1 = __shfl(threadInput11, friend_id5);
    ry1 = __shfl(threadInput12, friend_id5);
    rx2 = __shfl(threadInput13, friend_id6);
    ry2 = __shfl(threadInput14, friend_id6);
    rx3 = __shfl(threadInput15, friend_id7);
    ry3 = __shfl(threadInput16, friend_id7);
    sum0 += ARG_3D(12,k  ,j,i)*((lane_id < 22)? tx0: ty0);
    sum1 += ARG_3D(12,k+1,j,i)*((lane_id < 24)? tx1: ty1);
    sum2 += ARG_3D(12,k+2,j,i)*((lane_id < 2 )? tx2: ((lane_id < 28)? ty2: tz2));
    sum3 += ARG_3D(12,k+3,j,i)*((lane_id < 6 )? tx3: ty3);
    sum4 += ARG_3D(12,k+4,j,i)*((lane_id < 8 )? rx0: ry0);
    sum5 += ARG_3D(12,k+5,j,i)*((lane_id < 12)? rx1: ry1);
    sum6 += ARG_3D(12,k+6,j,i)*((lane_id < 16)? rx2: ry2);
    sum7 += ARG_3D(12,k+7,j,i)*((lane_id < 18)? rx3: ry3);

    friend_id0 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+23+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+15+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput2, friend_id0);
    ty0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput4, friend_id1);
    ty1 = __shfl(threadInput5, friend_id1);
    tx2 = __shfl(threadInput5, friend_id2);
    ty2 = __shfl(threadInput6, friend_id2);
    tz2 = __shfl(threadInput7, friend_id2);
    tx3 = __shfl(threadInput7, friend_id3);
    ty3 = __shfl(threadInput8, friend_id3);
    tz3 = __shfl(threadInput9, friend_id3);
    rx0 = __shfl(threadInput9 , friend_id4);
    ry0 = __shfl(threadInput10, friend_id4);
    rx1 = __shfl(threadInput11, friend_id5);
    ry1 = __shfl(threadInput12, friend_id5);
    rx2 = __shfl(threadInput13, friend_id6);
    ry2 = __shfl(threadInput14, friend_id6);
    rx3 = __shfl(threadInput15, friend_id7);
    ry3 = __shfl(threadInput16, friend_id7);
    sum0 += ARG_3D(13,k  ,j,i)*((lane_id < 21)? tx0: ty0);
    sum1 += ARG_3D(13,k+1,j,i)*((lane_id < 24)? tx1: ty1);
    sum2 += ARG_3D(13,k+2,j,i)*((lane_id < 1 )? tx2: ((lane_id < 27)? ty2: tz2));
    sum3 += ARG_3D(13,k+3,j,i)*((lane_id < 5 )? tx3: ((lane_id < 31)? ty3: tz3));
    sum4 += ARG_3D(13,k+4,j,i)*((lane_id < 8 )? rx0: ry0);
    sum5 += ARG_3D(13,k+5,j,i)*((lane_id < 11)? rx1: ry1);
    sum6 += ARG_3D(13,k+6,j,i)*((lane_id < 15)? rx2: ry2);
    sum7 += ARG_3D(13,k+7,j,i)*((lane_id < 17)? rx3: ry3);

    friend_id0 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput2, friend_id0);
    ty0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput4, friend_id1);
    ty1 = __shfl(threadInput5, friend_id1);
    tx2 = __shfl(threadInput6, friend_id2);
    ty2 = __shfl(threadInput7, friend_id2);
    tx3 = __shfl(threadInput7, friend_id3);
    ty3 = __shfl(threadInput8, friend_id3);
    tz3 = __shfl(threadInput9, friend_id3);
    rx0 = __shfl(threadInput9 , friend_id4);
    ry0 = __shfl(threadInput10, friend_id4);
    rx1 = __shfl(threadInput11, friend_id5);
    ry1 = __shfl(threadInput12, friend_id5);
    rx2 = __shfl(threadInput13, friend_id6);
    ry2 = __shfl(threadInput14, friend_id6);
    rx3 = __shfl(threadInput15, friend_id7);
    ry3 = __shfl(threadInput16, friend_id7);
    sum0 += ARG_3D(14,k  ,j,i)*((lane_id < 20)? tx0: ty0);
    sum1 += ARG_3D(14,k+1,j,i)*((lane_id < 24)? tx1: ty1);
    sum2 += ARG_3D(14,k+2,j,i)*((lane_id < 26)? tx2: ty2);
    sum3 += ARG_3D(14,k+3,j,i)*((lane_id < 4 )? tx3: ((lane_id < 30)? ty3: tz3));
    sum4 += ARG_3D(14,k+4,j,i)*((lane_id < 8 )? rx0: ry0);
    sum5 += ARG_3D(14,k+5,j,i)*((lane_id < 10)? rx1: ry1);
    sum6 += ARG_3D(14,k+6,j,i)*((lane_id < 14)? rx2: ry2);
    sum7 += ARG_3D(14,k+7,j,i)*((lane_id < 16)? rx3: ry3);

    friend_id0 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+0+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput2, friend_id0);
    ty0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput4, friend_id1);
    ty1 = __shfl(threadInput5, friend_id1);
    tx2 = __shfl(threadInput6, friend_id2);
    ty2 = __shfl(threadInput7, friend_id2);
    tx3 = __shfl(threadInput8, friend_id3);
    ty3 = __shfl(threadInput9, friend_id3);
    rx0 = __shfl(threadInput10, friend_id4);
    ry0 = __shfl(threadInput11, friend_id4);
    rx1 = __shfl(threadInput11, friend_id5);
    ry1 = __shfl(threadInput12, friend_id5);
    rz1 = __shfl(threadInput13, friend_id5);
    rx2 = __shfl(threadInput13, friend_id6);
    ry2 = __shfl(threadInput14, friend_id6);
    rx3 = __shfl(threadInput15, friend_id7);
    ry3 = __shfl(threadInput16, friend_id7);
    sum0 += ARG_3D(15,k  ,j,i)*((lane_id < 14)? tx0: ty0);
    sum1 += ARG_3D(15,k+1,j,i)*((lane_id < 16)? tx1: ty1);
    sum2 += ARG_3D(15,k+2,j,i)*((lane_id < 20)? tx2: ty2);
    sum3 += ARG_3D(15,k+3,j,i)*((lane_id < 24)? tx3: ty3);
    sum4 += ARG_3D(15,k+4,j,i)*((lane_id < 26)? rx0: ry0);
    sum5 += ARG_3D(15,k+5,j,i)*((lane_id < 4 )? rx1: ((lane_id < 30)? ry1: rz1));
    sum6 += ARG_3D(15,k+6,j,i)*((lane_id < 8 )? rx2: ry2);
    sum7 += ARG_3D(15,k+7,j,i)*((lane_id < 10)? rx3: ry3);

    friend_id0 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+1+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput2, friend_id0);
    ty0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput4, friend_id1);
    ty1 = __shfl(threadInput5, friend_id1);
    tx2 = __shfl(threadInput6, friend_id2);
    ty2 = __shfl(threadInput7, friend_id2);
    tx3 = __shfl(threadInput8, friend_id3);
    ty3 = __shfl(threadInput9, friend_id3);
    rx0 = __shfl(threadInput10, friend_id4);
    ry0 = __shfl(threadInput11, friend_id4);
    rx1 = __shfl(threadInput11, friend_id5);
    ry1 = __shfl(threadInput12, friend_id5);
    rz1 = __shfl(threadInput13, friend_id5);
    rx2 = __shfl(threadInput13, friend_id6);
    ry2 = __shfl(threadInput14, friend_id6);
    rx3 = __shfl(threadInput15, friend_id7);
    ry3 = __shfl(threadInput16, friend_id7);
    sum0 += ARG_3D(16,k  ,j,i)*((lane_id < 13)? tx0: ty0);
    sum1 += ARG_3D(16,k+1,j,i)*((lane_id < 16)? tx1: ty1);
    sum2 += ARG_3D(16,k+2,j,i)*((lane_id < 19)? tx2: ty2);
    sum3 += ARG_3D(16,k+3,j,i)*((lane_id < 23)? tx3: ty3);
    sum4 += ARG_3D(16,k+4,j,i)*((lane_id < 25)? rx0: ry0);
    sum5 += ARG_3D(16,k+5,j,i)*((lane_id < 3 )? rx1: ((lane_id < 29)? ry1: rz1));
    sum6 += ARG_3D(16,k+6,j,i)*((lane_id < 7 )? rx2: ry2);
    sum7 += ARG_3D(16,k+7,j,i)*((lane_id < 9 )? rx3: ry3);

    friend_id0 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+2+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput2, friend_id0);
    ty0 = __shfl(threadInput3, friend_id0);
    tx1 = __shfl(threadInput4, friend_id1);
    ty1 = __shfl(threadInput5, friend_id1);
    tx2 = __shfl(threadInput6, friend_id2);
    ty2 = __shfl(threadInput7, friend_id2);
    tx3 = __shfl(threadInput8, friend_id3);
    ty3 = __shfl(threadInput9, friend_id3);
    rx0 = __shfl(threadInput10, friend_id4);
    ry0 = __shfl(threadInput11, friend_id4);
    rx1 = __shfl(threadInput11, friend_id5);
    ry1 = __shfl(threadInput12, friend_id5);
    rz1 = __shfl(threadInput13, friend_id5);
    rx2 = __shfl(threadInput13, friend_id6);
    ry2 = __shfl(threadInput14, friend_id6);
    rx3 = __shfl(threadInput15, friend_id7);
    ry3 = __shfl(threadInput16, friend_id7);
    sum0 += ARG_3D(17,k  ,j,i)*((lane_id < 12)? tx0: ty0);
    sum1 += ARG_3D(17,k+1,j,i)*((lane_id < 16)? tx1: ty1);
    sum2 += ARG_3D(17,k+2,j,i)*((lane_id < 18)? tx2: ty2);
    sum3 += ARG_3D(17,k+3,j,i)*((lane_id < 22)? tx3: ty3);
    sum4 += ARG_3D(17,k+4,j,i)*((lane_id < 24)? rx0: ry0);
    sum5 += ARG_3D(17,k+5,j,i)*((lane_id < 2 )? rx1: ((lane_id < 28)? ry1: rz1));
    sum6 += ARG_3D(17,k+6,j,i)*((lane_id < 6 )? rx2: ry2);
    sum7 += ARG_3D(17,k+7,j,i)*((lane_id < 8 )? rx3: ry3);

    friend_id0 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput3, friend_id0);
    ty0 = __shfl(threadInput4, friend_id0);
    tx1 = __shfl(threadInput5, friend_id1);
    ty1 = __shfl(threadInput6, friend_id1);
    tx2 = __shfl(threadInput7, friend_id2);
    ty2 = __shfl(threadInput8, friend_id2);
    tx3 = __shfl(threadInput9 , friend_id3);
    ty3 = __shfl(threadInput10, friend_id3);
    rx0 = __shfl(threadInput11, friend_id4);
    ry0 = __shfl(threadInput12, friend_id4);
    rx1 = __shfl(threadInput13, friend_id5);
    ry1 = __shfl(threadInput14, friend_id5);
    rx2 = __shfl(threadInput15, friend_id6);
    ry2 = __shfl(threadInput16, friend_id6);
    rx3 = __shfl(threadInput16, friend_id7);
    ry3 = __shfl(threadInput17, friend_id7);
    rz3 = __shfl(threadInput18, friend_id7);
    sum0 += ARG_3D(18,k  ,j,i)*((lane_id < 8 )? tx0: ty0);
    sum1 += ARG_3D(18,k+1,j,i)*((lane_id < 10)? tx1: ty1);
    sum2 += ARG_3D(18,k+2,j,i)*((lane_id < 14)? tx2: ty2);
    sum3 += ARG_3D(18,k+3,j,i)*((lane_id < 16)? tx3: ty3);
    sum4 += ARG_3D(18,k+4,j,i)*((lane_id < 20)? rx0: ry0);
    sum5 += ARG_3D(18,k+5,j,i)*((lane_id < 24)? rx1: ry1);
    sum6 += ARG_3D(18,k+6,j,i)*((lane_id < 26)? rx2: ry2);
    sum7 += ARG_3D(18,k+7,j,i)*((lane_id < 4 )? rx3: ((lane_id < 30)? ry3: rz3));

    friend_id0 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput3, friend_id0);
    ty0 = __shfl(threadInput4, friend_id0);
    tx1 = __shfl(threadInput5, friend_id1);
    ty1 = __shfl(threadInput6, friend_id1);
    tx2 = __shfl(threadInput7, friend_id2);
    ty2 = __shfl(threadInput8, friend_id2);
    tx3 = __shfl(threadInput9 , friend_id3);
    ty3 = __shfl(threadInput10, friend_id3);
    rx0 = __shfl(threadInput11, friend_id4);
    ry0 = __shfl(threadInput12, friend_id4);
    rx1 = __shfl(threadInput13, friend_id5);
    ry1 = __shfl(threadInput14, friend_id5);
    rx2 = __shfl(threadInput15, friend_id6);
    ry2 = __shfl(threadInput16, friend_id6);
    rx3 = __shfl(threadInput16, friend_id7);
    ry3 = __shfl(threadInput17, friend_id7);
    rz3 = __shfl(threadInput18, friend_id7);
    sum0 += ARG_3D(19,k  ,j,i)*((lane_id < 7 )? tx0: ty0);
    sum1 += ARG_3D(19,k+1,j,i)*((lane_id < 9 )? tx1: ty1);
    sum2 += ARG_3D(19,k+2,j,i)*((lane_id < 13)? tx2: ty2);
    sum3 += ARG_3D(19,k+3,j,i)*((lane_id < 16)? tx3: ty3);
    sum4 += ARG_3D(19,k+4,j,i)*((lane_id < 19)? rx0: ry0);
    sum5 += ARG_3D(19,k+5,j,i)*((lane_id < 23)? rx1: ry1);
    sum6 += ARG_3D(19,k+6,j,i)*((lane_id < 25)? rx2: ry2);
    sum7 += ARG_3D(19,k+7,j,i)*((lane_id < 3 )? rx3: ((lane_id < 29)? ry3: rz3));

    friend_id0 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput3, friend_id0);
    ty0 = __shfl(threadInput4, friend_id0);
    tx1 = __shfl(threadInput5, friend_id1);
    ty1 = __shfl(threadInput6, friend_id1);
    tx2 = __shfl(threadInput7, friend_id2);
    ty2 = __shfl(threadInput8, friend_id2);
    tx3 = __shfl(threadInput9 , friend_id3);
    ty3 = __shfl(threadInput10, friend_id3);
    rx0 = __shfl(threadInput11, friend_id4);
    ry0 = __shfl(threadInput12, friend_id4);
    rx1 = __shfl(threadInput13, friend_id5);
    ry1 = __shfl(threadInput14, friend_id5);
    rx2 = __shfl(threadInput15, friend_id6);
    ry2 = __shfl(threadInput16, friend_id6);
    rx3 = __shfl(threadInput16, friend_id7);
    ry3 = __shfl(threadInput17, friend_id7);
    rz3 = __shfl(threadInput18, friend_id7);
    sum0 += ARG_3D(20,k  ,j,i)*((lane_id < 6 )? tx0: ty0);
    sum1 += ARG_3D(20,k+1,j,i)*((lane_id < 8 )? tx1: ty1);
    sum2 += ARG_3D(20,k+2,j,i)*((lane_id < 12)? tx2: ty2);
    sum3 += ARG_3D(20,k+3,j,i)*((lane_id < 16)? tx3: ty3);
    sum4 += ARG_3D(20,k+4,j,i)*((lane_id < 18)? rx0: ry0);
    sum5 += ARG_3D(20,k+5,j,i)*((lane_id < 22)? rx1: ry1);
    sum6 += ARG_3D(20,k+6,j,i)*((lane_id < 24)? rx2: ry2);
    sum7 += ARG_3D(20,k+7,j,i)*((lane_id < 2 )? rx3: ((lane_id < 28)? ry3: rz3));

    friend_id0 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput4, friend_id0);
    ty0 = __shfl(threadInput5, friend_id0);
    tx1 = __shfl(threadInput5, friend_id1);
    ty1 = __shfl(threadInput6, friend_id1);
    tz1 = __shfl(threadInput7, friend_id1);
    tx2 = __shfl(threadInput7, friend_id2);
    ty2 = __shfl(threadInput8, friend_id2);
    tx3 = __shfl(threadInput9 , friend_id3);
    ty3 = __shfl(threadInput10, friend_id3);
    rx0 = __shfl(threadInput11, friend_id4);
    ry0 = __shfl(threadInput12, friend_id4);
    rx1 = __shfl(threadInput13, friend_id5);
    ry1 = __shfl(threadInput14, friend_id5);
    rx2 = __shfl(threadInput15, friend_id6);
    ry2 = __shfl(threadInput16, friend_id6);
    rx3 = __shfl(threadInput17, friend_id7);
    ry3 = __shfl(threadInput18, friend_id7);
    sum0 += ARG_3D(21,k  ,j,i)*((lane_id < 24)? tx0: ty0);
    sum1 += ARG_3D(21,k+1,j,i)*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
    sum2 += ARG_3D(21,k+2,j,i)*((lane_id < 6 )? tx2: ty2);
    sum3 += ARG_3D(21,k+3,j,i)*((lane_id < 8 )? tx3: ty3);
    sum4 += ARG_3D(21,k+4,j,i)*((lane_id < 12)? rx0: ry0);
    sum5 += ARG_3D(21,k+5,j,i)*((lane_id < 16)? rx1: ry1);
    sum6 += ARG_3D(21,k+6,j,i)*((lane_id < 18)? rx2: ry2);
    sum7 += ARG_3D(21,k+7,j,i)*((lane_id < 22)? rx3: ry3);

    friend_id0 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+23+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+15+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput4, friend_id0);
    ty0 = __shfl(threadInput5, friend_id0);
    tx1 = __shfl(threadInput5, friend_id1);
    ty1 = __shfl(threadInput6, friend_id1);
    tz1 = __shfl(threadInput7, friend_id1);
    tx2 = __shfl(threadInput7, friend_id2);
    ty2 = __shfl(threadInput8, friend_id2);
    tz2 = __shfl(threadInput9, friend_id2);
    tx3 = __shfl(threadInput9 , friend_id3);
    ty3 = __shfl(threadInput10, friend_id3);
    rx0 = __shfl(threadInput11, friend_id4);
    ry0 = __shfl(threadInput12, friend_id4);
    rx1 = __shfl(threadInput13, friend_id5);
    ry1 = __shfl(threadInput14, friend_id5);
    rx2 = __shfl(threadInput15, friend_id6);
    ry2 = __shfl(threadInput16, friend_id6);
    rx3 = __shfl(threadInput17, friend_id7);
    ry3 = __shfl(threadInput18, friend_id7);
    sum0 += ARG_3D(22,k  ,j,i)*((lane_id < 24)? tx0: ty0);
    sum1 += ARG_3D(22,k+1,j,i)*((lane_id < 1 )? tx1: ((lane_id < 27)? ty1: tz1));
    sum2 += ARG_3D(22,k+2,j,i)*((lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2));
    sum3 += ARG_3D(22,k+3,j,i)*((lane_id < 8 )? tx3: ty3);
    sum4 += ARG_3D(22,k+4,j,i)*((lane_id < 11)? rx0: ry0);
    sum5 += ARG_3D(22,k+5,j,i)*((lane_id < 15)? rx1: ry1);
    sum6 += ARG_3D(22,k+6,j,i)*((lane_id < 17)? rx2: ry2);
    sum7 += ARG_3D(22,k+7,j,i)*((lane_id < 21)? rx3: ry3);

    friend_id0 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput4, friend_id0);
    ty0 = __shfl(threadInput5, friend_id0);
    tx1 = __shfl(threadInput6, friend_id1);
    ty1 = __shfl(threadInput7, friend_id1);
    tx2 = __shfl(threadInput7, friend_id2);
    ty2 = __shfl(threadInput8, friend_id2);
    tz2 = __shfl(threadInput9, friend_id2);
    tx3 = __shfl(threadInput9 , friend_id3);
    ty3 = __shfl(threadInput10, friend_id3);
    rx0 = __shfl(threadInput11, friend_id4);
    ry0 = __shfl(threadInput12, friend_id4);
    rx1 = __shfl(threadInput13, friend_id5);
    ry1 = __shfl(threadInput14, friend_id5);
    rx2 = __shfl(threadInput15, friend_id6);
    ry2 = __shfl(threadInput16, friend_id6);
    rx3 = __shfl(threadInput17, friend_id7);
    ry3 = __shfl(threadInput18, friend_id7);
    sum0 += ARG_3D(23,k  ,j,i)*((lane_id < 24)? tx0: ty0);
    sum1 += ARG_3D(23,k+1,j,i)*((lane_id < 26)? tx1: ty1);
    sum2 += ARG_3D(23,k+2,j,i)*((lane_id < 4 )? tx2: ((lane_id < 30)? ty2: tz2));
    sum3 += ARG_3D(23,k+3,j,i)*((lane_id < 8 )? tx3: ty3);
    sum4 += ARG_3D(23,k+4,j,i)*((lane_id < 10)? rx0: ry0);
    sum5 += ARG_3D(23,k+5,j,i)*((lane_id < 14)? rx1: ry1);
    sum6 += ARG_3D(23,k+6,j,i)*((lane_id < 16)? rx2: ry2);
    sum7 += ARG_3D(23,k+7,j,i)*((lane_id < 20)? rx3: ry3);

    friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput4, friend_id0);
    ty0 = __shfl(threadInput5, friend_id0);
    tx1 = __shfl(threadInput6, friend_id1);
    ty1 = __shfl(threadInput7, friend_id1);
    tx2 = __shfl(threadInput8, friend_id2);
    ty2 = __shfl(threadInput9, friend_id2);
    tx3 = __shfl(threadInput10, friend_id3);
    ty3 = __shfl(threadInput11, friend_id3);
    rx0 = __shfl(threadInput11, friend_id4);
    ry0 = __shfl(threadInput12, friend_id4);
    rz0 = __shfl(threadInput13, friend_id4);
    rx1 = __shfl(threadInput13, friend_id5);
    ry1 = __shfl(threadInput14, friend_id5);
    rx2 = __shfl(threadInput15, friend_id6);
    ry2 = __shfl(threadInput16, friend_id6);
    rx3 = __shfl(threadInput17, friend_id7);
    ry3 = __shfl(threadInput18, friend_id7);
    sum0 += ARG_3D(24,k  ,j,i)*((lane_id < 16)? tx0: ty0);
    sum1 += ARG_3D(24,k+1,j,i)*((lane_id < 20)? tx1: ty1);
    sum2 += ARG_3D(24,k+2,j,i)*((lane_id < 24)? tx2: ty2);
    sum3 += ARG_3D(24,k+3,j,i)*((lane_id < 26)? tx3: ty3);
    sum4 += ARG_3D(24,k+4,j,i)*((lane_id < 4 )? rx0: ((lane_id < 30)? ry0: rz0));
    sum5 += ARG_3D(24,k+5,j,i)*((lane_id < 8 )? rx1: ry1);
    sum6 += ARG_3D(24,k+6,j,i)*((lane_id < 10)? rx2: ry2);
    sum7 += ARG_3D(24,k+7,j,i)*((lane_id < 14)? rx3: ry3);

    friend_id0 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput4, friend_id0);
    ty0 = __shfl(threadInput5, friend_id0);
    tx1 = __shfl(threadInput6, friend_id1);
    ty1 = __shfl(threadInput7, friend_id1);
    tx2 = __shfl(threadInput8, friend_id2);
    ty2 = __shfl(threadInput9, friend_id2);
    tx3 = __shfl(threadInput10, friend_id3);
    ty3 = __shfl(threadInput11, friend_id3);
    rx0 = __shfl(threadInput11, friend_id4);
    ry0 = __shfl(threadInput12, friend_id4);
    rz0 = __shfl(threadInput13, friend_id4);
    rx1 = __shfl(threadInput13, friend_id5);
    ry1 = __shfl(threadInput14, friend_id5);
    rx2 = __shfl(threadInput15, friend_id6);
    ry2 = __shfl(threadInput16, friend_id6);
    rx3 = __shfl(threadInput17, friend_id7);
    ry3 = __shfl(threadInput18, friend_id7);
    sum0 += ARG_3D(25,k  ,j,i)*((lane_id < 16)? tx0: ty0);
    sum1 += ARG_3D(25,k+1,j,i)*((lane_id < 19)? tx1: ty1);
    sum2 += ARG_3D(25,k+2,j,i)*((lane_id < 23)? tx2: ty2);
    sum3 += ARG_3D(25,k+3,j,i)*((lane_id < 25)? tx3: ty3);
    sum4 += ARG_3D(25,k+4,j,i)*((lane_id < 3 )? rx0: ((lane_id < 29)? ry0: rz0));
    sum5 += ARG_3D(25,k+5,j,i)*((lane_id < 7 )? rx1: ry1);
    sum6 += ARG_3D(25,k+6,j,i)*((lane_id < 9 )? rx2: ry2);
    sum7 += ARG_3D(25,k+7,j,i)*((lane_id < 13)? rx3: ry3);

    friend_id0 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = __shfl(threadInput4, friend_id0);
    ty0 = __shfl(threadInput5, friend_id0);
    tx1 = __shfl(threadInput6, friend_id1);
    ty1 = __shfl(threadInput7, friend_id1);
    tx2 = __shfl(threadInput8, friend_id2);
    ty2 = __shfl(threadInput9, friend_id2);
    tx3 = __shfl(threadInput10, friend_id3);
    ty3 = __shfl(threadInput11, friend_id3);
    rx0 = __shfl(threadInput11, friend_id4);
    ry0 = __shfl(threadInput12, friend_id4);
    rz0 = __shfl(threadInput13, friend_id4);
    rx1 = __shfl(threadInput13, friend_id5);
    ry1 = __shfl(threadInput14, friend_id5);
    rx2 = __shfl(threadInput15, friend_id6);
    ry2 = __shfl(threadInput16, friend_id6);
    rx3 = __shfl(threadInput17, friend_id7);
    ry3 = __shfl(threadInput18, friend_id7);
    sum0 += ARG_3D(26,k  ,j,i)*((lane_id < 16)? tx0: ty0);
    sum1 += ARG_3D(26,k+1,j,i)*((lane_id < 18)? tx1: ty1);
    sum2 += ARG_3D(26,k+2,j,i)*((lane_id < 22)? tx2: ty2);
    sum3 += ARG_3D(26,k+3,j,i)*((lane_id < 24)? tx3: ty3);
    sum4 += ARG_3D(26,k+4,j,i)*((lane_id < 2 )? rx0: ((lane_id < 28)? ry0: rz0));
    sum5 += ARG_3D(26,k+5,j,i)*((lane_id < 6 )? rx1: ry1);
    sum6 += ARG_3D(26,k+6,j,i)*((lane_id < 8 )? rx2: ry2);
    sum7 += ARG_3D(26,k+7,j,i)*((lane_id < 12)? rx3: ry3);


    OUT_3D(k  ,j,i) = sum0;
    OUT_3D(k+1,j,i) = sum1;
    OUT_3D(k+2,j,i) = sum2;
    OUT_3D(k+3,j,i) = sum3;
    OUT_3D(k+4,j,i) = sum4;
    OUT_3D(k+5,j,i) = sum5;
    OUT_3D(k+6,j,i) = sum6;
    OUT_3D(k+7,j,i) = sum7;
}

__global__ void Stencil_Cuda_Sweep_Shfl(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, 
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
    DATA_TYPE t3_threadInput0, t3_threadInput1;
    DATA_TYPE t2_threadInput0, t2_threadInput1;
    DATA_TYPE t1_threadInput0, t1_threadInput1;

#define SM_2D_M2 32 
#define SM_2D_N2 8 

    DATA_TYPE sum = 0.0;

    // t3 is current layer; t2 is previous layer
    new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
    new_j = (warp_id_y<<2) + lane_id/10;     
    t3_threadInput0 = IN_3D(k  , new_j, new_i);
    t2_threadInput0 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + (lane_id+32)/10;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    t3_threadInput1 = IN_3D(k  , new_j, new_i);
    t2_threadInput1 = IN_3D(k-1, new_j, new_i);

#pragma unroll // it seems the loop-unroll is useless to performance
    for(; k < k_end; ++k)
    {
        sum = 0.0;
        // move the current storage down 
        t1_threadInput0 = t2_threadInput0;
        t1_threadInput1 = t2_threadInput1;
        t2_threadInput0 = t3_threadInput0;
        t2_threadInput1 = t3_threadInput1;

        new_i = (warp_id_x<<3) + lane_id%10;  
        new_j = (warp_id_y<<2) + lane_id/10;     
        t3_threadInput0 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+32)%10;
        new_j = (warp_id_y<<2) + (lane_id+32)/10;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
        t3_threadInput1 = IN_3D(k+1, new_j, new_i);

        friend_id = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
        tx = __shfl(t1_threadInput0, friend_id);
        ty = __shfl(t1_threadInput1, friend_id);
        sum += ARG_3D(0,k,j,i)*((lane_id < 26)? tx: ty);
        tx = __shfl(t2_threadInput0, friend_id);
        ty = __shfl(t2_threadInput1, friend_id);
        sum += ARG_3D(9,k,j,i)*((lane_id < 26)? tx: ty);
        tx = __shfl(t3_threadInput0, friend_id);
        ty = __shfl(t3_threadInput1, friend_id);
        sum += ARG_3D(18,k,j,i)*((lane_id < 26)? tx: ty);

        friend_id = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
        tx = __shfl(t1_threadInput0, friend_id);
        ty = __shfl(t1_threadInput1, friend_id);
        sum += ARG_3D(1,k,j,i)*((lane_id < 25)? tx: ty);
        tx = __shfl(t2_threadInput0, friend_id);
        ty = __shfl(t2_threadInput1, friend_id);
        sum += ARG_3D(10,k,j,i)*((lane_id < 25)? tx: ty);
        tx = __shfl(t3_threadInput0, friend_id);
        ty = __shfl(t3_threadInput1, friend_id);
        sum += ARG_3D(19,k,j,i)*((lane_id < 25)? tx: ty);

        friend_id = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
        tx = __shfl(t1_threadInput0, friend_id);
        ty = __shfl(t1_threadInput1, friend_id);
        sum += ARG_3D(2,k,j,i)*((lane_id < 24)? tx: ty);
        tx = __shfl(t2_threadInput0, friend_id);
        ty = __shfl(t2_threadInput1, friend_id);
        sum += ARG_3D(11,k,j,i)*((lane_id < 24)? tx: ty);
        tx = __shfl(t3_threadInput0, friend_id);
        ty = __shfl(t3_threadInput1, friend_id);
        sum += ARG_3D(20,k,j,i)*((lane_id < 24)? tx: ty);

        friend_id = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
        tx = __shfl(t1_threadInput0, friend_id);
        ty = __shfl(t1_threadInput1, friend_id);
        sum += ARG_3D(3,k,j,i)*((lane_id < 18)? tx: ty);
        tx = __shfl(t2_threadInput0, friend_id);
        ty = __shfl(t2_threadInput1, friend_id);
        sum += ARG_3D(12,k,j,i)*((lane_id < 18)? tx: ty);
        tx = __shfl(t3_threadInput0, friend_id);
        ty = __shfl(t3_threadInput1, friend_id);
        sum += ARG_3D(21,k,j,i)*((lane_id < 18)? tx: ty);

        friend_id = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
        tx = __shfl(t1_threadInput0, friend_id);
        ty = __shfl(t1_threadInput1, friend_id);
        sum += ARG_3D(4,k,j,i)*((lane_id < 17)? tx: ty);
        tx = __shfl(t2_threadInput0, friend_id);
        ty = __shfl(t2_threadInput1, friend_id);
        sum += ARG_3D(13,k,j,i)*((lane_id < 17)? tx: ty);
        tx = __shfl(t3_threadInput0, friend_id);
        ty = __shfl(t3_threadInput1, friend_id);
        sum += ARG_3D(22,k,j,i)*((lane_id < 17)? tx: ty);

        friend_id = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
        tx = __shfl(t1_threadInput0, friend_id);
        ty = __shfl(t1_threadInput1, friend_id);
        sum += ARG_3D(5,k,j,i)*((lane_id < 16)? tx: ty);
        tx = __shfl(t2_threadInput0, friend_id);
        ty = __shfl(t2_threadInput1, friend_id);
        sum += ARG_3D(14,k,j,i)*((lane_id < 16)? tx: ty);
        tx = __shfl(t3_threadInput0, friend_id);
        ty = __shfl(t3_threadInput1, friend_id);
        sum += ARG_3D(23,k,j,i)*((lane_id < 16)? tx: ty);

        friend_id = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
        tx = __shfl(t1_threadInput0, friend_id);
        ty = __shfl(t1_threadInput1, friend_id);
        sum += ARG_3D(6,k,j,i)*((lane_id < 10)? tx: ty);
        tx = __shfl(t2_threadInput0, friend_id);
        ty = __shfl(t2_threadInput1, friend_id);
        sum += ARG_3D(15,k,j,i)*((lane_id < 10)? tx: ty);
        tx = __shfl(t3_threadInput0, friend_id);
        ty = __shfl(t3_threadInput1, friend_id);
        sum += ARG_3D(24,k,j,i)*((lane_id < 10)? tx: ty);

        friend_id = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
        tx = __shfl(t1_threadInput0, friend_id);
        ty = __shfl(t1_threadInput1, friend_id);
        sum += ARG_3D(7,k,j,i)*((lane_id < 9 )? tx: ty);
        tx = __shfl(t2_threadInput0, friend_id);
        ty = __shfl(t2_threadInput1, friend_id);
        sum += ARG_3D(16,k,j,i)*((lane_id < 9 )? tx: ty);
        tx = __shfl(t3_threadInput0, friend_id);
        ty = __shfl(t3_threadInput1, friend_id);
        sum += ARG_3D(25,k,j,i)*((lane_id < 9 )? tx: ty);
        
        friend_id = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
        tx = __shfl(t1_threadInput0, friend_id);
        ty = __shfl(t1_threadInput1, friend_id);
        sum += ARG_3D(8,k,j,i)*((lane_id < 8 )? tx: ty);
        tx = __shfl(t2_threadInput0, friend_id);
        ty = __shfl(t2_threadInput1, friend_id);
        sum += ARG_3D(17,k,j,i)*((lane_id < 8 )? tx: ty);
        tx = __shfl(t3_threadInput0, friend_id);
        ty = __shfl(t3_threadInput1, friend_id);
        sum += ARG_3D(26,k,j,i)*((lane_id < 8 )? tx: ty);

        OUT_3D(k,j,i) = sum;
    }
}

__global__ void Stencil_Cuda_Sweep_Shfl2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, 
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
    DATA_TYPE t3_threadInput0, t3_threadInput1, t3_threadInput2, t3_threadInput3;
    DATA_TYPE t2_threadInput0, t2_threadInput1, t2_threadInput2, t2_threadInput3;
    DATA_TYPE t1_threadInput0, t1_threadInput1, t1_threadInput2, t1_threadInput3;

#define SM_2D_M2 32 
#define SM_2D_N2 8 

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;

    // t3 is current layer; t2 is previous layer
    new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
    new_j = (warp_id_y<<2) + lane_id/10;     
    t3_threadInput0 = IN_3D(k  , new_j, new_i);
    t2_threadInput0 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + (lane_id+32)/10;
    t3_threadInput1 = IN_3D(k  , new_j, new_i);
    t2_threadInput1 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10;
    new_j = (warp_id_y<<2) + (lane_id+64)/10;
    t3_threadInput2 = IN_3D(k  , new_j, new_i);
    t2_threadInput2 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10;
    new_j = (warp_id_y<<2) + (lane_id+96)/10;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    t3_threadInput3 = IN_3D(k  , new_j, new_i);
    t2_threadInput3 = IN_3D(k-1, new_j, new_i);

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

        new_i = (warp_id_x<<3) + lane_id%10;  
        new_j = (warp_id_y<<2) + lane_id/10;     
        t3_threadInput0 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+32)%10;
        new_j = (warp_id_y<<2) + (lane_id+32)/10;
        t3_threadInput1 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+64)%10;
        new_j = (warp_id_y<<2) + (lane_id+64)/10;
        t3_threadInput2 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+96)%10;
        new_j = (warp_id_y<<2) + (lane_id+96)/10;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
        t3_threadInput3 = IN_3D(k+1, new_j, new_i);

        friend_id0 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        sum0 += ARG_3D(0,k,j  ,i)*((lane_id < 26)? tx0: ty0);
        sum1 += ARG_3D(0,k,j+4,i)*((lane_id < 20)? tx1: ty1);
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        sum0 += ARG_3D(9,k,j  ,i)*((lane_id < 26)? tx0: ty0);
        sum1 += ARG_3D(9,k,j+4,i)*((lane_id < 20)? tx1: ty1);
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        sum0 += ARG_3D(18,k,j  ,i)*((lane_id < 26)? tx0: ty0);
        sum1 += ARG_3D(18,k,j+4,i)*((lane_id < 20)? tx1: ty1);

        friend_id0 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        sum0 += ARG_3D(1,k,j  ,i)*((lane_id < 25)? tx0: ty0);
        sum1 += ARG_3D(1,k,j+4,i)*((lane_id < 19)? tx1: ty1);
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        sum0 += ARG_3D(10,k,j  ,i)*((lane_id < 25)? tx0: ty0);
        sum1 += ARG_3D(10,k,j+4,i)*((lane_id < 19)? tx1: ty1);
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        sum0 += ARG_3D(19,k,j  ,i)*((lane_id < 25)? tx0: ty0);
        sum1 += ARG_3D(19,k,j+4,i)*((lane_id < 19)? tx1: ty1);

        friend_id0 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        sum0 += ARG_3D(2,k,j  ,i)*((lane_id < 24)? tx0: ty0);
        sum1 += ARG_3D(2,k,j+4,i)*((lane_id < 18)? tx1: ty1);
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        sum0 += ARG_3D(11,k,j  ,i)*((lane_id < 24)? tx0: ty0);
        sum1 += ARG_3D(11,k,j+4,i)*((lane_id < 18)? tx1: ty1);
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        sum0 += ARG_3D(20,k,j  ,i)*((lane_id < 24)? tx0: ty0);
        sum1 += ARG_3D(20,k,j+4,i)*((lane_id < 18)? tx1: ty1);

        friend_id0 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        sum0 += ARG_3D(3,k,j  ,i)*((lane_id < 18)? tx0: ty0);
        sum1 += ARG_3D(3,k,j+4,i)*((lane_id < 12)? tx1: ty1);
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        sum0 += ARG_3D(12,k,j  ,i)*((lane_id < 18)? tx0: ty0);
        sum1 += ARG_3D(12,k,j+4,i)*((lane_id < 12)? tx1: ty1);
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        sum0 += ARG_3D(21,k,j  ,i)*((lane_id < 18)? tx0: ty0);
        sum1 += ARG_3D(21,k,j+4,i)*((lane_id < 12)? tx1: ty1);

        friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        sum0 += ARG_3D(4,k,j  ,i)*((lane_id < 17)? tx0: ty0);
        sum1 += ARG_3D(4,k,j+4,i)*((lane_id < 11)? tx1: ty1);
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        sum0 += ARG_3D(13,k,j  ,i)*((lane_id < 17)? tx0: ty0);
        sum1 += ARG_3D(13,k,j+4,i)*((lane_id < 11)? tx1: ty1);
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        sum0 += ARG_3D(22,k,j  ,i)*((lane_id < 17)? tx0: ty0);
        sum1 += ARG_3D(22,k,j+4,i)*((lane_id < 11)? tx1: ty1);

        friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        sum0 += ARG_3D(5,k,j  ,i)*((lane_id < 16)? tx0: ty0);
        sum1 += ARG_3D(5,k,j+4,i)*((lane_id < 10)? tx1: ty1);
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        sum0 += ARG_3D(14,k,j  ,i)*((lane_id < 16)? tx0: ty0);
        sum1 += ARG_3D(14,k,j+4,i)*((lane_id < 10)? tx1: ty1);
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        sum0 += ARG_3D(23,k,j  ,i)*((lane_id < 16)? tx0: ty0);
        sum1 += ARG_3D(23,k,j+4,i)*((lane_id < 10)? tx1: ty1);

        friend_id0 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        tz1 = __shfl(t1_threadInput3, friend_id1);
        sum0 += ARG_3D(6,k,j  ,i)*((lane_id < 10)? tx0: ty0);
        sum1 += ARG_3D(6,k,j+4,i)*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        tz1 = __shfl(t2_threadInput3, friend_id1);
        sum0 += ARG_3D(15,k,j  ,i)*((lane_id < 10)? tx0: ty0);
        sum1 += ARG_3D(15,k,j+4,i)*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        tz1 = __shfl(t3_threadInput3, friend_id1);
        sum0 += ARG_3D(24,k,j  ,i)*((lane_id < 10)? tx0: ty0);
        sum1 += ARG_3D(24,k,j+4,i)*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));

        friend_id0 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        tz1 = __shfl(t1_threadInput3, friend_id1);
        sum0 += ARG_3D(7,k,j  ,i)*((lane_id < 9 )? tx0: ty0);
        sum1 += ARG_3D(7,k,j+4,i)*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        tz1 = __shfl(t2_threadInput3, friend_id1);
        sum0 += ARG_3D(16,k,j  ,i)*((lane_id < 9 )? tx0: ty0);
        sum1 += ARG_3D(16,k,j+4,i)*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        tz1 = __shfl(t3_threadInput3, friend_id1);
        sum0 += ARG_3D(25,k,j  ,i)*((lane_id < 9 )? tx0: ty0);
        sum1 += ARG_3D(25,k,j+4,i)*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
        
        friend_id0 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        tz1 = __shfl(t1_threadInput3, friend_id1);
        sum0 += ARG_3D(8,k,j  ,i)*((lane_id < 8 )? tx0: ty0);
        sum1 += ARG_3D(8,k,j+4,i)*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        tz1 = __shfl(t2_threadInput3, friend_id1);
        sum0 += ARG_3D(17,k,j  ,i)*((lane_id < 8 )? tx0: ty0);
        sum1 += ARG_3D(17,k,j+4,i)*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        tz1 = __shfl(t3_threadInput3, friend_id1);
        sum0 += ARG_3D(26,k,j  ,i)*((lane_id < 8 )? tx0: ty0);
        sum1 += ARG_3D(26,k,j+4,i)*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
        

        OUT_3D(k,j  ,i) = sum0;
        OUT_3D(k,j+4,i) = sum1;
    }
}

__global__ void Stencil_Cuda_Sweep_Shfl4(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, 
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
    new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
    new_j = (warp_id_y<<2) + lane_id/10;     
    t3_threadInput0 = IN_3D(k  , new_j, new_i);
    t2_threadInput0 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + (lane_id+32)/10;
    t3_threadInput1 = IN_3D(k  , new_j, new_i);
    t2_threadInput1 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10;
    new_j = (warp_id_y<<2) + (lane_id+64)/10;
    t3_threadInput2 = IN_3D(k  , new_j, new_i);
    t2_threadInput2 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10;
    new_j = (warp_id_y<<2) + (lane_id+96)/10;
    t3_threadInput3 = IN_3D(k  , new_j, new_i);
    t2_threadInput3 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10;
    new_j = (warp_id_y<<2) + (lane_id+128)/10;
    t3_threadInput4 = IN_3D(k  , new_j, new_i);
    t2_threadInput4 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10;
    new_j = (warp_id_y<<2) + (lane_id+160)/10;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    t3_threadInput5 = IN_3D(k  , new_j, new_i);
    t2_threadInput5 = IN_3D(k-1, new_j, new_i);

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

        new_i = (warp_id_x<<3) + lane_id%10;  
        new_j = (warp_id_y<<2) + lane_id/10;     
        t3_threadInput0 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+32)%10;
        new_j = (warp_id_y<<2) + (lane_id+32)/10;
        t3_threadInput1 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+64)%10;
        new_j = (warp_id_y<<2) + (lane_id+64)/10;
        t3_threadInput2 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+96)%10;
        new_j = (warp_id_y<<2) + (lane_id+96)/10;
        t3_threadInput3 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+128)%10;
        new_j = (warp_id_y<<2) + (lane_id+128)/10;
        t3_threadInput4 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+160)%10;
        new_j = (warp_id_y<<2) + (lane_id+160)/10;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo;
        t3_threadInput5 = IN_3D(k+1, new_j, new_i);


        friend_id0 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        tx2 = __shfl(t1_threadInput2, friend_id2);
        ty2 = __shfl(t1_threadInput3, friend_id2);
        tx3 = __shfl(t1_threadInput3, friend_id3);
        ty3 = __shfl(t1_threadInput4, friend_id3);
        sum0 += ARG_3D(0,k,j   ,i)*((lane_id < 26)? tx0: ty0);
        sum1 += ARG_3D(0,k,j+4 ,i)*((lane_id < 20)? tx1: ty1);
        sum2 += ARG_3D(0,k,j+8 ,i)*((lane_id < 14)? tx2: ty2);
        sum3 += ARG_3D(0,k,j+12,i)*((lane_id < 8 )? tx3: ty3);
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        tx2 = __shfl(t2_threadInput2, friend_id2);
        ty2 = __shfl(t2_threadInput3, friend_id2);
        tx3 = __shfl(t2_threadInput3, friend_id3);
        ty3 = __shfl(t2_threadInput4, friend_id3);
        sum0 += ARG_3D(9,k,j   ,i)*((lane_id < 26)? tx0: ty0);
        sum1 += ARG_3D(9,k,j+4 ,i)*((lane_id < 20)? tx1: ty1);
        sum2 += ARG_3D(9,k,j+8 ,i)*((lane_id < 14)? tx2: ty2);
        sum3 += ARG_3D(9,k,j+12,i)*((lane_id < 8 )? tx3: ty3);
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        tx2 = __shfl(t3_threadInput2, friend_id2);
        ty2 = __shfl(t3_threadInput3, friend_id2);
        tx3 = __shfl(t3_threadInput3, friend_id3);
        ty3 = __shfl(t3_threadInput4, friend_id3);
        sum0 += ARG_3D(18,k,j   ,i)*((lane_id < 26)? tx0: ty0);
        sum1 += ARG_3D(18,k,j+4 ,i)*((lane_id < 20)? tx1: ty1);
        sum2 += ARG_3D(18,k,j+8 ,i)*((lane_id < 14)? tx2: ty2);
        sum3 += ARG_3D(18,k,j+12,i)*((lane_id < 8 )? tx3: ty3);

        friend_id0 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        tx2 = __shfl(t1_threadInput2, friend_id2);
        ty2 = __shfl(t1_threadInput3, friend_id2);
        tx3 = __shfl(t1_threadInput3, friend_id3);
        ty3 = __shfl(t1_threadInput4, friend_id3);
        sum0 += ARG_3D(1,k,j   ,i)*((lane_id < 25)? tx0: ty0);
        sum1 += ARG_3D(1,k,j+4 ,i)*((lane_id < 19)? tx1: ty1);
        sum2 += ARG_3D(1,k,j+8 ,i)*((lane_id < 13)? tx2: ty2);
        sum3 += ARG_3D(1,k,j+12,i)*((lane_id < 7 )? tx3: ty3);
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        tx2 = __shfl(t2_threadInput2, friend_id2);
        ty2 = __shfl(t2_threadInput3, friend_id2);
        tx3 = __shfl(t2_threadInput3, friend_id3);
        ty3 = __shfl(t2_threadInput4, friend_id3);
        sum0 += ARG_3D(10,k,j   ,i)*((lane_id < 25)? tx0: ty0);
        sum1 += ARG_3D(10,k,j+4 ,i)*((lane_id < 19)? tx1: ty1);
        sum2 += ARG_3D(10,k,j+8 ,i)*((lane_id < 13)? tx2: ty2);
        sum3 += ARG_3D(10,k,j+12,i)*((lane_id < 7 )? tx3: ty3);
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        tx2 = __shfl(t3_threadInput2, friend_id2);
        ty2 = __shfl(t3_threadInput3, friend_id2);
        tx3 = __shfl(t3_threadInput3, friend_id3);
        ty3 = __shfl(t3_threadInput4, friend_id3);
        sum0 += ARG_3D(19,k,j   ,i)*((lane_id < 25)? tx0: ty0);
        sum1 += ARG_3D(19,k,j+4 ,i)*((lane_id < 19)? tx1: ty1);
        sum2 += ARG_3D(19,k,j+8 ,i)*((lane_id < 13)? tx2: ty2);
        sum3 += ARG_3D(19,k,j+12,i)*((lane_id < 7 )? tx3: ty3);

        friend_id0 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        tx2 = __shfl(t1_threadInput2, friend_id2);
        ty2 = __shfl(t1_threadInput3, friend_id2);
        tx3 = __shfl(t1_threadInput3, friend_id3);
        ty3 = __shfl(t1_threadInput4, friend_id3);
        sum0 += ARG_3D(2,k,j   ,i)*((lane_id < 24)? tx0: ty0);
        sum1 += ARG_3D(2,k,j+4 ,i)*((lane_id < 18)? tx1: ty1);
        sum2 += ARG_3D(2,k,j+8 ,i)*((lane_id < 12)? tx2: ty2);
        sum3 += ARG_3D(2,k,j+12,i)*((lane_id < 6 )? tx3: ty3);
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        tx2 = __shfl(t2_threadInput2, friend_id2);
        ty2 = __shfl(t2_threadInput3, friend_id2);
        tx3 = __shfl(t2_threadInput3, friend_id3);
        ty3 = __shfl(t2_threadInput4, friend_id3);
        sum0 += ARG_3D(11,k,j   ,i)*((lane_id < 24)? tx0: ty0);
        sum1 += ARG_3D(11,k,j+4 ,i)*((lane_id < 18)? tx1: ty1);
        sum2 += ARG_3D(11,k,j+8 ,i)*((lane_id < 12)? tx2: ty2);
        sum3 += ARG_3D(11,k,j+12,i)*((lane_id < 6 )? tx3: ty3);
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        tx2 = __shfl(t3_threadInput2, friend_id2);
        ty2 = __shfl(t3_threadInput3, friend_id2);
        tx3 = __shfl(t3_threadInput3, friend_id3);
        ty3 = __shfl(t3_threadInput4, friend_id3);
        sum0 += ARG_3D(20,k,j   ,i)*((lane_id < 24)? tx0: ty0);
        sum1 += ARG_3D(20,k,j+4 ,i)*((lane_id < 18)? tx1: ty1);
        sum2 += ARG_3D(20,k,j+8 ,i)*((lane_id < 12)? tx2: ty2);
        sum3 += ARG_3D(20,k,j+12,i)*((lane_id < 6 )? tx3: ty3);

        friend_id0 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        tx2 = __shfl(t1_threadInput2, friend_id2);
        ty2 = __shfl(t1_threadInput3, friend_id2);
        ty3 = __shfl(t1_threadInput4, friend_id3);
        tz3 = __shfl(t1_threadInput5, friend_id3);
        sum0 += ARG_3D(3,k,j   ,i)*((lane_id < 18)? tx0: ty0);
        sum1 += ARG_3D(3,k,j+4 ,i)*((lane_id < 12)? tx1: ty1);
        sum2 += ARG_3D(3,k,j+8 ,i)*((lane_id < 6 )? tx2: ty2);
        sum3 += ARG_3D(3,k,j+12,i)*((lane_id < 24)? ty3: tz3);
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        tx2 = __shfl(t2_threadInput2, friend_id2);
        ty2 = __shfl(t2_threadInput3, friend_id2);
        ty3 = __shfl(t2_threadInput4, friend_id3);
        tz3 = __shfl(t2_threadInput5, friend_id3);
        sum0 += ARG_3D(12,k,j   ,i)*((lane_id < 18)? tx0: ty0);
        sum1 += ARG_3D(12,k,j+4 ,i)*((lane_id < 12)? tx1: ty1);
        sum2 += ARG_3D(12,k,j+8 ,i)*((lane_id < 6 )? tx2: ty2);
        sum3 += ARG_3D(12,k,j+12,i)*((lane_id < 24)? ty3: tz3);
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        tx2 = __shfl(t3_threadInput2, friend_id2);
        ty2 = __shfl(t3_threadInput3, friend_id2);
        ty3 = __shfl(t3_threadInput4, friend_id3);
        tz3 = __shfl(t3_threadInput5, friend_id3);
        sum0 += ARG_3D(21,k,j   ,i)*((lane_id < 18)? tx0: ty0);
        sum1 += ARG_3D(21,k,j+4 ,i)*((lane_id < 12)? tx1: ty1);
        sum2 += ARG_3D(21,k,j+8 ,i)*((lane_id < 6 )? tx2: ty2);
        sum3 += ARG_3D(21,k,j+12,i)*((lane_id < 24)? ty3: tz3);

        friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        tx2 = __shfl(t1_threadInput2, friend_id2);
        ty2 = __shfl(t1_threadInput3, friend_id2);
        tz2 = __shfl(t1_threadInput4, friend_id2);
        ty3 = __shfl(t1_threadInput4, friend_id3);
        tz3 = __shfl(t1_threadInput5, friend_id3);
        sum0 += ARG_3D(4,k,j   ,i)*((lane_id < 17)? tx0: ty0);
        sum1 += ARG_3D(4,k,j+4 ,i)*((lane_id < 11)? tx1: ty1);
        sum2 += ARG_3D(4,k,j+8 ,i)*((lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2));
        sum3 += ARG_3D(4,k,j+12,i)*((lane_id < 24)? ty3: tz3);
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        tx2 = __shfl(t2_threadInput2, friend_id2);
        ty2 = __shfl(t2_threadInput3, friend_id2);
        tz2 = __shfl(t2_threadInput4, friend_id2);
        ty3 = __shfl(t2_threadInput4, friend_id3);
        tz3 = __shfl(t2_threadInput5, friend_id3);
        sum0 += ARG_3D(13,k,j   ,i)*((lane_id < 17)? tx0: ty0);
        sum1 += ARG_3D(13,k,j+4 ,i)*((lane_id < 11)? tx1: ty1);
        sum2 += ARG_3D(13,k,j+8 ,i)*((lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2));
        sum3 += ARG_3D(13,k,j+12,i)*((lane_id < 24)? ty3: tz3);
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        tx2 = __shfl(t3_threadInput2, friend_id2);
        ty2 = __shfl(t3_threadInput3, friend_id2);
        tz2 = __shfl(t3_threadInput4, friend_id2);
        ty3 = __shfl(t3_threadInput4, friend_id3);
        tz3 = __shfl(t3_threadInput5, friend_id3);
        sum0 += ARG_3D(22,k,j   ,i)*((lane_id < 17)? tx0: ty0);
        sum1 += ARG_3D(22,k,j+4 ,i)*((lane_id < 11)? tx1: ty1);
        sum2 += ARG_3D(22,k,j+8 ,i)*((lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2));
        sum3 += ARG_3D(22,k,j+12,i)*((lane_id < 24)? ty3: tz3);

        friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        tx2 = __shfl(t1_threadInput2, friend_id2);
        ty2 = __shfl(t1_threadInput3, friend_id2);
        tz2 = __shfl(t1_threadInput4, friend_id2);
        ty3 = __shfl(t1_threadInput4, friend_id3);
        tz3 = __shfl(t1_threadInput5, friend_id3);
        sum0 += ARG_3D(5,k,j   ,i)*((lane_id < 16)? tx0: ty0);
        sum1 += ARG_3D(5,k,j+4 ,i)*((lane_id < 10)? tx1: ty1);
        sum2 += ARG_3D(5,k,j+8 ,i)*((lane_id < 4 )? tx2: ((lane_id < 30)? ty2: tz2));
        sum3 += ARG_3D(5,k,j+12,i)*((lane_id < 24)? ty3: tz3);
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        tx2 = __shfl(t2_threadInput2, friend_id2);
        ty2 = __shfl(t2_threadInput3, friend_id2);
        tz2 = __shfl(t2_threadInput4, friend_id2);
        ty3 = __shfl(t2_threadInput4, friend_id3);
        tz3 = __shfl(t2_threadInput5, friend_id3);
        sum0 += ARG_3D(14,k,j   ,i)*((lane_id < 16)? tx0: ty0);
        sum1 += ARG_3D(14,k,j+4 ,i)*((lane_id < 10)? tx1: ty1);
        sum2 += ARG_3D(14,k,j+8 ,i)*((lane_id < 4 )? tx2: ((lane_id < 30)? ty2: tz2));
        sum3 += ARG_3D(14,k,j+12,i)*((lane_id < 24)? ty3: tz3);
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        tx2 = __shfl(t3_threadInput2, friend_id2);
        ty2 = __shfl(t3_threadInput3, friend_id2);
        tz2 = __shfl(t3_threadInput4, friend_id2);
        ty3 = __shfl(t3_threadInput4, friend_id3);
        tz3 = __shfl(t3_threadInput5, friend_id3);
        sum0 += ARG_3D(23,k,j   ,i)*((lane_id < 16)? tx0: ty0);
        sum1 += ARG_3D(23,k,j+4 ,i)*((lane_id < 10)? tx1: ty1);
        sum2 += ARG_3D(23,k,j+8 ,i)*((lane_id < 4 )? tx2: ((lane_id < 30)? ty2: tz2));
        sum3 += ARG_3D(23,k,j+12,i)*((lane_id < 24)? ty3: tz3);

        friend_id0 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        tz1 = __shfl(t1_threadInput3, friend_id1);
        ty2 = __shfl(t1_threadInput3, friend_id2);
        tz2 = __shfl(t1_threadInput4, friend_id2);
        ty3 = __shfl(t1_threadInput4, friend_id3);
        tz3 = __shfl(t1_threadInput5, friend_id3);
        sum0 += ARG_3D(6,k,j   ,i)*((lane_id < 10)? tx0: ty0);
        sum1 += ARG_3D(6,k,j+4 ,i)*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
        sum2 += ARG_3D(6,k,j+8 ,i)*((lane_id < 24)? ty2: tz2);
        sum3 += ARG_3D(6,k,j+12,i)*((lane_id < 16)? ty3: tz3);
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        tz1 = __shfl(t2_threadInput3, friend_id1);
        ty2 = __shfl(t2_threadInput3, friend_id2);
        tz2 = __shfl(t2_threadInput4, friend_id2);
        ty3 = __shfl(t2_threadInput4, friend_id3);
        tz3 = __shfl(t2_threadInput5, friend_id3);
        sum0 += ARG_3D(15,k,j   ,i)*((lane_id < 10)? tx0: ty0);
        sum1 += ARG_3D(15,k,j+4 ,i)*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
        sum2 += ARG_3D(15,k,j+8 ,i)*((lane_id < 24)? ty2: tz2);
        sum3 += ARG_3D(15,k,j+12,i)*((lane_id < 16)? ty3: tz3);
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        tz1 = __shfl(t3_threadInput3, friend_id1);
        ty2 = __shfl(t3_threadInput3, friend_id2);
        tz2 = __shfl(t3_threadInput4, friend_id2);
        ty3 = __shfl(t3_threadInput4, friend_id3);
        tz3 = __shfl(t3_threadInput5, friend_id3);
        sum0 += ARG_3D(24,k,j   ,i)*((lane_id < 10)? tx0: ty0);
        sum1 += ARG_3D(24,k,j+4 ,i)*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
        sum2 += ARG_3D(24,k,j+8 ,i)*((lane_id < 24)? ty2: tz2);
        sum3 += ARG_3D(24,k,j+12,i)*((lane_id < 16)? ty3: tz3);

        friend_id0 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        tz1 = __shfl(t1_threadInput3, friend_id1);
        ty2 = __shfl(t1_threadInput3, friend_id2);
        tz2 = __shfl(t1_threadInput4, friend_id2);
        ty3 = __shfl(t1_threadInput4, friend_id3);
        tz3 = __shfl(t1_threadInput5, friend_id3);
        sum0 += ARG_3D(7,k,j   ,i)*((lane_id < 9 )? tx0: ty0);
        sum1 += ARG_3D(7,k,j+4 ,i)*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
        sum2 += ARG_3D(7,k,j+8 ,i)*((lane_id < 23)? ty2: tz2);
        sum3 += ARG_3D(7,k,j+12,i)*((lane_id < 16)? ty3: tz3);
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        tz1 = __shfl(t2_threadInput3, friend_id1);
        ty2 = __shfl(t2_threadInput3, friend_id2);
        tz2 = __shfl(t2_threadInput4, friend_id2);
        ty3 = __shfl(t2_threadInput4, friend_id3);
        tz3 = __shfl(t2_threadInput5, friend_id3);
        sum0 += ARG_3D(16,k,j   ,i)*((lane_id < 9 )? tx0: ty0);
        sum1 += ARG_3D(16,k,j+4 ,i)*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
        sum2 += ARG_3D(16,k,j+8 ,i)*((lane_id < 23)? ty2: tz2);
        sum3 += ARG_3D(16,k,j+12,i)*((lane_id < 16)? ty3: tz3);
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        tz1 = __shfl(t3_threadInput3, friend_id1);
        ty2 = __shfl(t3_threadInput3, friend_id2);
        tz2 = __shfl(t3_threadInput4, friend_id2);
        ty3 = __shfl(t3_threadInput4, friend_id3);
        tz3 = __shfl(t3_threadInput5, friend_id3);
        sum0 += ARG_3D(25,k,j   ,i)*((lane_id < 9 )? tx0: ty0);
        sum1 += ARG_3D(25,k,j+4 ,i)*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
        sum2 += ARG_3D(25,k,j+8 ,i)*((lane_id < 23)? ty2: tz2);
        sum3 += ARG_3D(25,k,j+12,i)*((lane_id < 16)? ty3: tz3);
        
        friend_id0 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        tz1 = __shfl(t1_threadInput3, friend_id1);
        ty2 = __shfl(t1_threadInput3, friend_id2);
        tz2 = __shfl(t1_threadInput4, friend_id2);
        ty3 = __shfl(t1_threadInput4, friend_id3);
        tz3 = __shfl(t1_threadInput5, friend_id3);
        sum0 += ARG_3D(8,k,j   ,i)*((lane_id < 8 )? tx0: ty0);
        sum1 += ARG_3D(8,k,j+4 ,i)*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
        sum2 += ARG_3D(8,k,j+8 ,i)*((lane_id < 22)? ty2: tz2);
        sum3 += ARG_3D(8,k,j+12,i)*((lane_id < 16)? ty3: tz3);
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        tz1 = __shfl(t2_threadInput3, friend_id1);
        ty2 = __shfl(t2_threadInput3, friend_id2);
        tz2 = __shfl(t2_threadInput4, friend_id2);
        ty3 = __shfl(t2_threadInput4, friend_id3);
        tz3 = __shfl(t2_threadInput5, friend_id3);
        sum0 += ARG_3D(17,k,j   ,i)*((lane_id < 8 )? tx0: ty0);
        sum1 += ARG_3D(17,k,j+4 ,i)*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
        sum2 += ARG_3D(17,k,j+8 ,i)*((lane_id < 22)? ty2: tz2);
        sum3 += ARG_3D(17,k,j+12,i)*((lane_id < 16)? ty3: tz3);
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        tz1 = __shfl(t3_threadInput3, friend_id1);
        ty2 = __shfl(t3_threadInput3, friend_id2);
        tz2 = __shfl(t3_threadInput4, friend_id2);
        ty3 = __shfl(t3_threadInput4, friend_id3);
        tz3 = __shfl(t3_threadInput5, friend_id3);
        sum0 += ARG_3D(26,k,j   ,i)*((lane_id < 8 )? tx0: ty0);
        sum1 += ARG_3D(26,k,j+4 ,i)*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
        sum2 += ARG_3D(26,k,j+8 ,i)*((lane_id < 22)? ty2: tz2);
        sum3 += ARG_3D(26,k,j+12,i)*((lane_id < 16)? ty3: tz3);

        
        OUT_3D(k,j   ,i) = sum0;
        OUT_3D(k,j+4 ,i) = sum1;
        OUT_3D(k,j+8 ,i) = sum2;
        OUT_3D(k,j+12,i) = sum3;
    }
}

__global__ void Stencil_Cuda_Sweep_Shfl8(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, 
        int z, int m, int n, int halo)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<5) + (lane_id>>3) + halo;

    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;
    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<5) + (lane_id>>3))>>2; // 1x4x8, warp_ids are division of 
    DATA_TYPE tx0, ty0;
    DATA_TYPE tx1, ty1, tz1;
    DATA_TYPE tx2, ty2, tz2;
    DATA_TYPE tx3, ty3, tz3;
    DATA_TYPE rx0, ry0, rx1, ry1, rz1, rx2, ry2, rz2, rx3, ry3, rz3;
    int friend_id0, friend_id1;
    int friend_id2, friend_id3;
    int new_i, new_j;
    DATA_TYPE t3_threadInput0, t3_threadInput1, t3_threadInput2, t3_threadInput3, t3_threadInput4, t3_threadInput5;
    DATA_TYPE t3_threadInput6, t3_threadInput7, t3_threadInput8, t3_threadInput9, t3_threadInput10;
    DATA_TYPE t2_threadInput0, t2_threadInput1, t2_threadInput2, t2_threadInput3, t2_threadInput4, t2_threadInput5;
    DATA_TYPE t2_threadInput6, t2_threadInput7, t2_threadInput8, t2_threadInput9, t2_threadInput10;
    DATA_TYPE t1_threadInput0, t1_threadInput1, t1_threadInput2, t1_threadInput3, t1_threadInput4, t1_threadInput5;
    DATA_TYPE t1_threadInput6, t1_threadInput7, t1_threadInput8, t1_threadInput9, t1_threadInput10;

#define SM_2D_M2 32 
#define SM_2D_N2 8 

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    DATA_TYPE sum4 = 0.0;
    DATA_TYPE sum5 = 0.0;
    DATA_TYPE sum6 = 0.0;
    DATA_TYPE sum7 = 0.0;

    // t3 is current layer; t2 is previous layer
    new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
    new_j = (warp_id_y<<2) + lane_id/10;     
    t3_threadInput0 = IN_3D(k  , new_j, new_i);
    t2_threadInput0 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + (lane_id+32)/10;
    t3_threadInput1 = IN_3D(k  , new_j, new_i);
    t2_threadInput1 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10;
    new_j = (warp_id_y<<2) + (lane_id+64)/10;
    t3_threadInput2 = IN_3D(k  , new_j, new_i);
    t2_threadInput2 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10;
    new_j = (warp_id_y<<2) + (lane_id+96)/10;
    t3_threadInput3 = IN_3D(k  , new_j, new_i);
    t2_threadInput3 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10;
    new_j = (warp_id_y<<2) + (lane_id+128)/10;
    t3_threadInput4 = IN_3D(k  , new_j, new_i);
    t2_threadInput4 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10;
    new_j = (warp_id_y<<2) + (lane_id+160)/10;
    t3_threadInput5 = IN_3D(k  , new_j, new_i);
    t2_threadInput5 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+192)%10;
    new_j = (warp_id_y<<2) + (lane_id+192)/10;
    t3_threadInput6 = IN_3D(k  , new_j, new_i);
    t2_threadInput6 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+224)%10;
    new_j = (warp_id_y<<2) + (lane_id+224)/10;
    t3_threadInput7 = IN_3D(k  , new_j, new_i);
    t2_threadInput7 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+256)%10;
    new_j = (warp_id_y<<2) + (lane_id+256)/10;
    t3_threadInput8 = IN_3D(k  , new_j, new_i);
    t2_threadInput8 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+288)%10;
    new_j = (warp_id_y<<2) + (lane_id+288)/10;
    t3_threadInput9 = IN_3D(k  , new_j, new_i);
    t2_threadInput9 = IN_3D(k-1, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+320)%10;
    new_j = (warp_id_y<<2) + (lane_id+320)/10;
    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
    t3_threadInput10 = IN_3D(k  , new_j, new_i);
    t2_threadInput10 = IN_3D(k-1, new_j, new_i);

#pragma unroll // it seems the loop-unroll is useless to performance
    for(; k < k_end; ++k)
    {
        sum0 = 0.0;
        sum1 = 0.0;
        sum2 = 0.0;
        sum3 = 0.0;
        sum4 = 0.0;
        sum5 = 0.0;
        sum6 = 0.0;
        sum7 = 0.0;
        // move the current storage down 
        t1_threadInput0  = t2_threadInput0 ;
        t1_threadInput1  = t2_threadInput1 ;
        t1_threadInput2  = t2_threadInput2 ;
        t1_threadInput3  = t2_threadInput3 ;
        t1_threadInput4  = t2_threadInput4 ;
        t1_threadInput5  = t2_threadInput5 ;
        t1_threadInput6  = t2_threadInput6 ;
        t1_threadInput7  = t2_threadInput7 ;
        t1_threadInput8  = t2_threadInput8 ;
        t1_threadInput9  = t2_threadInput9 ;
        t1_threadInput10 = t2_threadInput10;

        t2_threadInput0  = t3_threadInput0 ;
        t2_threadInput1  = t3_threadInput1 ;
        t2_threadInput2  = t3_threadInput2 ;
        t2_threadInput3  = t3_threadInput3 ;
        t2_threadInput4  = t3_threadInput4 ;
        t2_threadInput5  = t3_threadInput5 ;
        t2_threadInput6  = t3_threadInput6 ;
        t2_threadInput7  = t3_threadInput7 ;
        t2_threadInput8  = t3_threadInput8 ;
        t2_threadInput9  = t3_threadInput9 ;
        t2_threadInput10 = t3_threadInput10;

        new_i = (warp_id_x<<3) + lane_id%10;  
        new_j = (warp_id_y<<2) + lane_id/10;     
        t3_threadInput0 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+32)%10;
        new_j = (warp_id_y<<2) + (lane_id+32)/10;
        t3_threadInput1 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+64)%10;
        new_j = (warp_id_y<<2) + (lane_id+64)/10;
        t3_threadInput2 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+96)%10;
        new_j = (warp_id_y<<2) + (lane_id+96)/10;
        t3_threadInput3 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+128)%10;
        new_j = (warp_id_y<<2) + (lane_id+128)/10;
        t3_threadInput4 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+160)%10;
        new_j = (warp_id_y<<2) + (lane_id+160)/10;
        t3_threadInput5 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+192)%10;
        new_j = (warp_id_y<<2) + (lane_id+192)/10;
        t3_threadInput6 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+224)%10;
        new_j = (warp_id_y<<2) + (lane_id+224)/10;
        t3_threadInput7 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+256)%10;
        new_j = (warp_id_y<<2) + (lane_id+256)/10;
        t3_threadInput8 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+288)%10;
        new_j = (warp_id_y<<2) + (lane_id+288)/10;
        t3_threadInput9 = IN_3D(k+1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+320)%10;
        new_j = (warp_id_y<<2) + (lane_id+320)/10;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
        t3_threadInput10 = IN_3D(k+1, new_j, new_i);


        friend_id0 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        tx2 = __shfl(t1_threadInput2, friend_id2);
        ty2 = __shfl(t1_threadInput3, friend_id2);
        tx3 = __shfl(t1_threadInput3, friend_id3);
        ty3 = __shfl(t1_threadInput4, friend_id3);
        rx0 = __shfl(t1_threadInput5, friend_id0);
        ry0 = __shfl(t1_threadInput6, friend_id0);
        rx1 = __shfl(t1_threadInput6, friend_id1);
        ry1 = __shfl(t1_threadInput7, friend_id1);
        rx2 = __shfl(t1_threadInput7, friend_id2);
        ry2 = __shfl(t1_threadInput8, friend_id2);
        rx3 = __shfl(t1_threadInput8, friend_id3);
        ry3 = __shfl(t1_threadInput9, friend_id3);
        sum0 += ARG_3D(0,k,j,i)*((lane_id < 26)? tx0: ty0);
        sum1 += ARG_3D(0,k,j,i)*((lane_id < 20)? tx1: ty1);
        sum2 += ARG_3D(0,k,j,i)*((lane_id < 14)? tx2: ty2);
        sum3 += ARG_3D(0,k,j,i)*((lane_id < 8 )? tx3: ty3);
        sum4 += ARG_3D(0,k,j,i)*((lane_id < 26)? rx0: ry0);
        sum5 += ARG_3D(0,k,j,i)*((lane_id < 20)? rx1: ry1);
        sum6 += ARG_3D(0,k,j,i)*((lane_id < 14)? rx2: ry2);
        sum7 += ARG_3D(0,k,j,i)*((lane_id < 8 )? rx3: ry3);
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        tx2 = __shfl(t2_threadInput2, friend_id2);
        ty2 = __shfl(t2_threadInput3, friend_id2);
        tx3 = __shfl(t2_threadInput3, friend_id3);
        ty3 = __shfl(t2_threadInput4, friend_id3);
        rx0 = __shfl(t2_threadInput5, friend_id0);
        ry0 = __shfl(t2_threadInput6, friend_id0);
        rx1 = __shfl(t2_threadInput6, friend_id1);
        ry1 = __shfl(t2_threadInput7, friend_id1);
        rx2 = __shfl(t2_threadInput7, friend_id2);
        ry2 = __shfl(t2_threadInput8, friend_id2);
        rx3 = __shfl(t2_threadInput8, friend_id3);
        ry3 = __shfl(t2_threadInput9, friend_id3);
        sum0 += ARG_3D(9,k,j,i)*((lane_id < 26)? tx0: ty0);
        sum1 += ARG_3D(9,k,j,i)*((lane_id < 20)? tx1: ty1);
        sum2 += ARG_3D(9,k,j,i)*((lane_id < 14)? tx2: ty2);
        sum3 += ARG_3D(9,k,j,i)*((lane_id < 8 )? tx3: ty3);
        sum4 += ARG_3D(9,k,j,i)*((lane_id < 26)? rx0: ry0);
        sum5 += ARG_3D(9,k,j,i)*((lane_id < 20)? rx1: ry1);
        sum6 += ARG_3D(9,k,j,i)*((lane_id < 14)? rx2: ry2);
        sum7 += ARG_3D(9,k,j,i)*((lane_id < 8 )? rx3: ry3);
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        tx2 = __shfl(t3_threadInput2, friend_id2);
        ty2 = __shfl(t3_threadInput3, friend_id2);
        tx3 = __shfl(t3_threadInput3, friend_id3);
        ty3 = __shfl(t3_threadInput4, friend_id3);
        rx0 = __shfl(t3_threadInput5, friend_id0);
        ry0 = __shfl(t3_threadInput6, friend_id0);
        rx1 = __shfl(t3_threadInput6, friend_id1);
        ry1 = __shfl(t3_threadInput7, friend_id1);
        rx2 = __shfl(t3_threadInput7, friend_id2);
        ry2 = __shfl(t3_threadInput8, friend_id2);
        rx3 = __shfl(t3_threadInput8, friend_id3);
        ry3 = __shfl(t3_threadInput9, friend_id3);
        sum0 += ARG_3D(18,k,j,i)*((lane_id < 26)? tx0: ty0);
        sum1 += ARG_3D(18,k,j,i)*((lane_id < 20)? tx1: ty1);
        sum2 += ARG_3D(18,k,j,i)*((lane_id < 14)? tx2: ty2);
        sum3 += ARG_3D(18,k,j,i)*((lane_id < 8 )? tx3: ty3);
        sum4 += ARG_3D(18,k,j,i)*((lane_id < 26)? rx0: ry0);
        sum5 += ARG_3D(18,k,j,i)*((lane_id < 20)? rx1: ry1);
        sum6 += ARG_3D(18,k,j,i)*((lane_id < 14)? rx2: ry2);
        sum7 += ARG_3D(18,k,j,i)*((lane_id < 8 )? rx3: ry3);

        friend_id0 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        tx2 = __shfl(t1_threadInput2, friend_id2);
        ty2 = __shfl(t1_threadInput3, friend_id2);
        tx3 = __shfl(t1_threadInput3, friend_id3);
        ty3 = __shfl(t1_threadInput4, friend_id3);
        rx0 = __shfl(t1_threadInput5, friend_id0);
        ry0 = __shfl(t1_threadInput6, friend_id0);
        rx1 = __shfl(t1_threadInput6, friend_id1);
        ry1 = __shfl(t1_threadInput7, friend_id1);
        rx2 = __shfl(t1_threadInput7, friend_id2);
        ry2 = __shfl(t1_threadInput8, friend_id2);
        rx3 = __shfl(t1_threadInput8, friend_id3);
        ry3 = __shfl(t1_threadInput9, friend_id3);
        sum0 += ARG_3D(1,k,j,i)*((lane_id < 25)? tx0: ty0);
        sum1 += ARG_3D(1,k,j,i)*((lane_id < 19)? tx1: ty1);
        sum2 += ARG_3D(1,k,j,i)*((lane_id < 13)? tx2: ty2);
        sum3 += ARG_3D(1,k,j,i)*((lane_id < 7 )? tx3: ty3);
        sum4 += ARG_3D(1,k,j,i)*((lane_id < 25)? rx0: ry0);
        sum5 += ARG_3D(1,k,j,i)*((lane_id < 19)? rx1: ry1);
        sum6 += ARG_3D(1,k,j,i)*((lane_id < 13)? rx2: ry2);
        sum7 += ARG_3D(1,k,j,i)*((lane_id < 7 )? rx3: ry3);
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        tx2 = __shfl(t2_threadInput2, friend_id2);
        ty2 = __shfl(t2_threadInput3, friend_id2);
        tx3 = __shfl(t2_threadInput3, friend_id3);
        ty3 = __shfl(t2_threadInput4, friend_id3);
        rx0 = __shfl(t2_threadInput5, friend_id0);
        ry0 = __shfl(t2_threadInput6, friend_id0);
        rx1 = __shfl(t2_threadInput6, friend_id1);
        ry1 = __shfl(t2_threadInput7, friend_id1);
        rx2 = __shfl(t2_threadInput7, friend_id2);
        ry2 = __shfl(t2_threadInput8, friend_id2);
        rx3 = __shfl(t2_threadInput8, friend_id3);
        ry3 = __shfl(t2_threadInput9, friend_id3);
        sum0 += ARG_3D(10,k,j,i)*((lane_id < 25)? tx0: ty0);
        sum1 += ARG_3D(10,k,j,i)*((lane_id < 19)? tx1: ty1);
        sum2 += ARG_3D(10,k,j,i)*((lane_id < 13)? tx2: ty2);
        sum3 += ARG_3D(10,k,j,i)*((lane_id < 7 )? tx3: ty3);
        sum4 += ARG_3D(10,k,j,i)*((lane_id < 25)? rx0: ry0);
        sum5 += ARG_3D(10,k,j,i)*((lane_id < 19)? rx1: ry1);
        sum6 += ARG_3D(10,k,j,i)*((lane_id < 13)? rx2: ry2);
        sum7 += ARG_3D(10,k,j,i)*((lane_id < 7 )? rx3: ry3);
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        tx2 = __shfl(t3_threadInput2, friend_id2);
        ty2 = __shfl(t3_threadInput3, friend_id2);
        tx3 = __shfl(t3_threadInput3, friend_id3);
        ty3 = __shfl(t3_threadInput4, friend_id3);
        rx0 = __shfl(t3_threadInput5, friend_id0);
        ry0 = __shfl(t3_threadInput6, friend_id0);
        rx1 = __shfl(t3_threadInput6, friend_id1);
        ry1 = __shfl(t3_threadInput7, friend_id1);
        rx2 = __shfl(t3_threadInput7, friend_id2);
        ry2 = __shfl(t3_threadInput8, friend_id2);
        rx3 = __shfl(t3_threadInput8, friend_id3);
        ry3 = __shfl(t3_threadInput9, friend_id3);
        sum0 += ARG_3D(19,k,j,i)*((lane_id < 25)? tx0: ty0);
        sum1 += ARG_3D(19,k,j,i)*((lane_id < 19)? tx1: ty1);
        sum2 += ARG_3D(19,k,j,i)*((lane_id < 13)? tx2: ty2);
        sum3 += ARG_3D(19,k,j,i)*((lane_id < 7 )? tx3: ty3);
        sum4 += ARG_3D(19,k,j,i)*((lane_id < 25)? rx0: ry0);
        sum5 += ARG_3D(19,k,j,i)*((lane_id < 19)? rx1: ry1);
        sum6 += ARG_3D(19,k,j,i)*((lane_id < 13)? rx2: ry2);
        sum7 += ARG_3D(19,k,j,i)*((lane_id < 7 )? rx3: ry3);

        friend_id0 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0, friend_id0);
        ty0 = __shfl(t1_threadInput1, friend_id0);
        tx1 = __shfl(t1_threadInput1, friend_id1);
        ty1 = __shfl(t1_threadInput2, friend_id1);
        tx2 = __shfl(t1_threadInput2, friend_id2);
        ty2 = __shfl(t1_threadInput3, friend_id2);
        tx3 = __shfl(t1_threadInput3, friend_id3);
        ty3 = __shfl(t1_threadInput4, friend_id3);
        rx0 = __shfl(t1_threadInput5, friend_id0);
        ry0 = __shfl(t1_threadInput6, friend_id0);
        rx1 = __shfl(t1_threadInput6, friend_id1);
        ry1 = __shfl(t1_threadInput7, friend_id1);
        rx2 = __shfl(t1_threadInput7, friend_id2);
        ry2 = __shfl(t1_threadInput8, friend_id2);
        rx3 = __shfl(t1_threadInput8, friend_id3);
        ry3 = __shfl(t1_threadInput9, friend_id3);
        sum0 += ARG_3D(2,k,j,i)*((lane_id < 24)? tx0: ty0);
        sum1 += ARG_3D(2,k,j,i)*((lane_id < 18)? tx1: ty1);
        sum2 += ARG_3D(2,k,j,i)*((lane_id < 12)? tx2: ty2);
        sum3 += ARG_3D(2,k,j,i)*((lane_id < 6 )? tx3: ty3);
        sum4 += ARG_3D(2,k,j,i)*((lane_id < 24)? rx0: ry0);
        sum5 += ARG_3D(2,k,j,i)*((lane_id < 18)? rx1: ry1);
        sum6 += ARG_3D(2,k,j,i)*((lane_id < 12)? rx2: ry2);
        sum7 += ARG_3D(2,k,j,i)*((lane_id < 6 )? rx3: ry3);
        tx0 = __shfl(t2_threadInput0, friend_id0);
        ty0 = __shfl(t2_threadInput1, friend_id0);
        tx1 = __shfl(t2_threadInput1, friend_id1);
        ty1 = __shfl(t2_threadInput2, friend_id1);
        tx2 = __shfl(t2_threadInput2, friend_id2);
        ty2 = __shfl(t2_threadInput3, friend_id2);
        tx3 = __shfl(t2_threadInput3, friend_id3);
        ty3 = __shfl(t2_threadInput4, friend_id3);
        rx0 = __shfl(t2_threadInput5, friend_id0);
        ry0 = __shfl(t2_threadInput6, friend_id0);
        rx1 = __shfl(t2_threadInput6, friend_id1);
        ry1 = __shfl(t2_threadInput7, friend_id1);
        rx2 = __shfl(t2_threadInput7, friend_id2);
        ry2 = __shfl(t2_threadInput8, friend_id2);
        rx3 = __shfl(t2_threadInput8, friend_id3);
        ry3 = __shfl(t2_threadInput9, friend_id3);
        sum0 += ARG_3D(11,k,j,i)*((lane_id < 24)? tx0: ty0);
        sum1 += ARG_3D(11,k,j,i)*((lane_id < 18)? tx1: ty1);
        sum2 += ARG_3D(11,k,j,i)*((lane_id < 12)? tx2: ty2);
        sum3 += ARG_3D(11,k,j,i)*((lane_id < 6 )? tx3: ty3);
        sum4 += ARG_3D(11,k,j,i)*((lane_id < 24)? rx0: ry0);
        sum5 += ARG_3D(11,k,j,i)*((lane_id < 18)? rx1: ry1);
        sum6 += ARG_3D(11,k,j,i)*((lane_id < 12)? rx2: ry2);
        sum7 += ARG_3D(11,k,j,i)*((lane_id < 6 )? rx3: ry3);
        tx0 = __shfl(t3_threadInput0, friend_id0);
        ty0 = __shfl(t3_threadInput1, friend_id0);
        tx1 = __shfl(t3_threadInput1, friend_id1);
        ty1 = __shfl(t3_threadInput2, friend_id1);
        tx2 = __shfl(t3_threadInput2, friend_id2);
        ty2 = __shfl(t3_threadInput3, friend_id2);
        tx3 = __shfl(t3_threadInput3, friend_id3);
        ty3 = __shfl(t3_threadInput4, friend_id3);
        rx0 = __shfl(t3_threadInput5, friend_id0);
        ry0 = __shfl(t3_threadInput6, friend_id0);
        rx1 = __shfl(t3_threadInput6, friend_id1);
        ry1 = __shfl(t3_threadInput7, friend_id1);
        rx2 = __shfl(t3_threadInput7, friend_id2);
        ry2 = __shfl(t3_threadInput8, friend_id2);
        rx3 = __shfl(t3_threadInput8, friend_id3);
        ry3 = __shfl(t3_threadInput9, friend_id3);
        sum0 += ARG_3D(20,k,j,i)*((lane_id < 24)? tx0: ty0);
        sum1 += ARG_3D(20,k,j,i)*((lane_id < 18)? tx1: ty1);
        sum2 += ARG_3D(20,k,j,i)*((lane_id < 12)? tx2: ty2);
        sum3 += ARG_3D(20,k,j,i)*((lane_id < 6 )? tx3: ty3);
        sum4 += ARG_3D(20,k,j,i)*((lane_id < 24)? rx0: ry0);
        sum5 += ARG_3D(20,k,j,i)*((lane_id < 18)? rx1: ry1);
        sum6 += ARG_3D(20,k,j,i)*((lane_id < 12)? rx2: ry2);
        sum7 += ARG_3D(20,k,j,i)*((lane_id < 6 )? rx3: ry3);

        friend_id0 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0 , friend_id0);
        ty0 = __shfl(t1_threadInput1 , friend_id0);
        tx1 = __shfl(t1_threadInput1 , friend_id1);
        ty1 = __shfl(t1_threadInput2 , friend_id1);
        tx2 = __shfl(t1_threadInput2 , friend_id2);
        ty2 = __shfl(t1_threadInput3 , friend_id2);
        ty3 = __shfl(t1_threadInput4 , friend_id3);
        tz3 = __shfl(t1_threadInput5 , friend_id3);
        rx0 = __shfl(t1_threadInput5 , friend_id0);
        ry0 = __shfl(t1_threadInput6 , friend_id0);
        rx1 = __shfl(t1_threadInput6 , friend_id1);
        ry1 = __shfl(t1_threadInput7 , friend_id1);
        rx2 = __shfl(t1_threadInput7 , friend_id2);
        ry2 = __shfl(t1_threadInput8 , friend_id2);
        ry3 = __shfl(t1_threadInput9 , friend_id3);
        rz3 = __shfl(t1_threadInput10, friend_id3);
        sum0 += ARG_3D(3,k,j,i)*((lane_id < 18)? tx0: ty0);
        sum1 += ARG_3D(3,k,j,i)*((lane_id < 12)? tx1: ty1);
        sum2 += ARG_3D(3,k,j,i)*((lane_id < 6 )? tx2: ty2);
        sum3 += ARG_3D(3,k,j,i)*((lane_id < 24)? ty3: tz3);
        sum4 += ARG_3D(3,k,j,i)*((lane_id < 18)? rx0: ry0);
        sum5 += ARG_3D(3,k,j,i)*((lane_id < 12)? rx1: ry1);
        sum6 += ARG_3D(3,k,j,i)*((lane_id < 6 )? rx2: ry2);
        sum7 += ARG_3D(3,k,j,i)*((lane_id < 24)? ry3: rz3);
        tx0 = __shfl(t2_threadInput0 , friend_id0);
        ty0 = __shfl(t2_threadInput1 , friend_id0);
        tx1 = __shfl(t2_threadInput1 , friend_id1);
        ty1 = __shfl(t2_threadInput2 , friend_id1);
        tx2 = __shfl(t2_threadInput2 , friend_id2);
        ty2 = __shfl(t2_threadInput3 , friend_id2);
        ty3 = __shfl(t2_threadInput4 , friend_id3);
        tz3 = __shfl(t2_threadInput5 , friend_id3);
        rx0 = __shfl(t2_threadInput5 , friend_id0);
        ry0 = __shfl(t2_threadInput6 , friend_id0);
        rx1 = __shfl(t2_threadInput6 , friend_id1);
        ry1 = __shfl(t2_threadInput7 , friend_id1);
        rx2 = __shfl(t2_threadInput7 , friend_id2);
        ry2 = __shfl(t2_threadInput8 , friend_id2);
        ry3 = __shfl(t2_threadInput9 , friend_id3);
        rz3 = __shfl(t2_threadInput10, friend_id3);
        sum0 += ARG_3D(12,k,j,i)*((lane_id < 18)? tx0: ty0);
        sum1 += ARG_3D(12,k,j,i)*((lane_id < 12)? tx1: ty1);
        sum2 += ARG_3D(12,k,j,i)*((lane_id < 6 )? tx2: ty2);
        sum3 += ARG_3D(12,k,j,i)*((lane_id < 24)? ty3: tz3);
        sum4 += ARG_3D(12,k,j,i)*((lane_id < 18)? rx0: ry0);
        sum5 += ARG_3D(12,k,j,i)*((lane_id < 12)? rx1: ry1);
        sum6 += ARG_3D(12,k,j,i)*((lane_id < 6 )? rx2: ry2);
        sum7 += ARG_3D(12,k,j,i)*((lane_id < 24)? ry3: rz3);
        tx0 = __shfl(t3_threadInput0 , friend_id0);
        ty0 = __shfl(t3_threadInput1 , friend_id0);
        tx1 = __shfl(t3_threadInput1 , friend_id1);
        ty1 = __shfl(t3_threadInput2 , friend_id1);
        tx2 = __shfl(t3_threadInput2 , friend_id2);
        ty2 = __shfl(t3_threadInput3 , friend_id2);
        ty3 = __shfl(t3_threadInput4 , friend_id3);
        tz3 = __shfl(t3_threadInput5 , friend_id3);
        rx0 = __shfl(t3_threadInput5 , friend_id0);
        ry0 = __shfl(t3_threadInput6 , friend_id0);
        rx1 = __shfl(t3_threadInput6 , friend_id1);
        ry1 = __shfl(t3_threadInput7 , friend_id1);
        rx2 = __shfl(t3_threadInput7 , friend_id2);
        ry2 = __shfl(t3_threadInput8 , friend_id2);
        ry3 = __shfl(t3_threadInput9 , friend_id3);
        rz3 = __shfl(t3_threadInput10, friend_id3);
        sum0 += ARG_3D(21,k,j,i)*((lane_id < 18)? tx0: ty0);
        sum1 += ARG_3D(21,k,j,i)*((lane_id < 12)? tx1: ty1);
        sum2 += ARG_3D(21,k,j,i)*((lane_id < 6 )? tx2: ty2);
        sum3 += ARG_3D(21,k,j,i)*((lane_id < 24)? ty3: tz3);
        sum4 += ARG_3D(21,k,j,i)*((lane_id < 18)? rx0: ry0);
        sum5 += ARG_3D(21,k,j,i)*((lane_id < 12)? rx1: ry1);
        sum6 += ARG_3D(21,k,j,i)*((lane_id < 6 )? rx2: ry2);
        sum7 += ARG_3D(21,k,j,i)*((lane_id < 24)? ry3: rz3);

        friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0 , friend_id0);
        ty0 = __shfl(t1_threadInput1 , friend_id0);
        tx1 = __shfl(t1_threadInput1 , friend_id1);
        ty1 = __shfl(t1_threadInput2 , friend_id1);
        tx2 = __shfl(t1_threadInput2 , friend_id2);
        ty2 = __shfl(t1_threadInput3 , friend_id2);
        tz2 = __shfl(t1_threadInput4 , friend_id2);
        ty3 = __shfl(t1_threadInput4 , friend_id3);
        tz3 = __shfl(t1_threadInput5 , friend_id3);
        rx0 = __shfl(t1_threadInput5 , friend_id0);
        ry0 = __shfl(t1_threadInput6 , friend_id0);
        rx1 = __shfl(t1_threadInput6 , friend_id1);
        ry1 = __shfl(t1_threadInput7 , friend_id1);
        rx2 = __shfl(t1_threadInput7 , friend_id2);
        ry2 = __shfl(t1_threadInput8 , friend_id2);
        rz2 = __shfl(t1_threadInput9 , friend_id2);
        ry3 = __shfl(t1_threadInput9 , friend_id3);
        rz3 = __shfl(t1_threadInput10, friend_id3);
        sum0 += ARG_3D(4,k,j,i)*((lane_id < 17)? tx0: ty0);
        sum1 += ARG_3D(4,k,j,i)*((lane_id < 11)? tx1: ty1);
        sum2 += ARG_3D(4,k,j,i)*((lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2));
        sum3 += ARG_3D(4,k,j,i)*((lane_id < 24)? ty3: tz3);
        sum4 += ARG_3D(4,k,j,i)*((lane_id < 17)? rx0: ry0);
        sum5 += ARG_3D(4,k,j,i)*((lane_id < 11)? rx1: ry1);
        sum6 += ARG_3D(4,k,j,i)*((lane_id < 5 )? rx2: ((lane_id < 31)? ry2: rz2));
        sum7 += ARG_3D(4,k,j,i)*((lane_id < 24)? ry3: rz3);
        tx0 = __shfl(t2_threadInput0 , friend_id0);
        ty0 = __shfl(t2_threadInput1 , friend_id0);
        tx1 = __shfl(t2_threadInput1 , friend_id1);
        ty1 = __shfl(t2_threadInput2 , friend_id1);
        tx2 = __shfl(t2_threadInput2 , friend_id2);
        ty2 = __shfl(t2_threadInput3 , friend_id2);
        tz2 = __shfl(t2_threadInput4 , friend_id2);
        ty3 = __shfl(t2_threadInput4 , friend_id3);
        tz3 = __shfl(t2_threadInput5 , friend_id3);
        rx0 = __shfl(t2_threadInput5 , friend_id0);
        ry0 = __shfl(t2_threadInput6 , friend_id0);
        rx1 = __shfl(t2_threadInput6 , friend_id1);
        ry1 = __shfl(t2_threadInput7 , friend_id1);
        rx2 = __shfl(t2_threadInput7 , friend_id2);
        ry2 = __shfl(t2_threadInput8 , friend_id2);
        rz2 = __shfl(t2_threadInput9 , friend_id2);
        ry3 = __shfl(t2_threadInput9 , friend_id3);
        rz3 = __shfl(t2_threadInput10, friend_id3);
        sum0 += ARG_3D(13,k,j,i)*((lane_id < 17)? tx0: ty0);
        sum1 += ARG_3D(13,k,j,i)*((lane_id < 11)? tx1: ty1);
        sum2 += ARG_3D(13,k,j,i)*((lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2));
        sum3 += ARG_3D(13,k,j,i)*((lane_id < 24)? ty3: tz3);
        sum4 += ARG_3D(13,k,j,i)*((lane_id < 17)? rx0: ry0);
        sum5 += ARG_3D(13,k,j,i)*((lane_id < 11)? rx1: ry1);
        sum6 += ARG_3D(13,k,j,i)*((lane_id < 5 )? rx2: ((lane_id < 31)? ry2: rz2));
        sum7 += ARG_3D(13,k,j,i)*((lane_id < 24)? ry3: rz3);
        tx0 = __shfl(t3_threadInput0 , friend_id0);
        ty0 = __shfl(t3_threadInput1 , friend_id0);
        tx1 = __shfl(t3_threadInput1 , friend_id1);
        ty1 = __shfl(t3_threadInput2 , friend_id1);
        tx2 = __shfl(t3_threadInput2 , friend_id2);
        ty2 = __shfl(t3_threadInput3 , friend_id2);
        tz2 = __shfl(t3_threadInput4 , friend_id2);
        ty3 = __shfl(t3_threadInput4 , friend_id3);
        tz3 = __shfl(t3_threadInput5 , friend_id3);
        rx0 = __shfl(t3_threadInput5 , friend_id0);
        ry0 = __shfl(t3_threadInput6 , friend_id0);
        rx1 = __shfl(t3_threadInput6 , friend_id1);
        ry1 = __shfl(t3_threadInput7 , friend_id1);
        rx2 = __shfl(t3_threadInput7 , friend_id2);
        ry2 = __shfl(t3_threadInput8 , friend_id2);
        rz2 = __shfl(t3_threadInput9 , friend_id2);
        ry3 = __shfl(t3_threadInput9 , friend_id3);
        rz3 = __shfl(t3_threadInput10, friend_id3);
        sum0 += ARG_3D(22,k,j,i)*((lane_id < 17)? tx0: ty0);
        sum1 += ARG_3D(22,k,j,i)*((lane_id < 11)? tx1: ty1);
        sum2 += ARG_3D(22,k,j,i)*((lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2));
        sum3 += ARG_3D(22,k,j,i)*((lane_id < 24)? ty3: tz3);
        sum4 += ARG_3D(22,k,j,i)*((lane_id < 17)? rx0: ry0);
        sum5 += ARG_3D(22,k,j,i)*((lane_id < 11)? rx1: ry1);
        sum6 += ARG_3D(22,k,j,i)*((lane_id < 5 )? rx2: ((lane_id < 31)? ry2: rz2));
        sum7 += ARG_3D(22,k,j,i)*((lane_id < 24)? ry3: rz3);

        friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0 , friend_id0);
        ty0 = __shfl(t1_threadInput1 , friend_id0);
        tx1 = __shfl(t1_threadInput1 , friend_id1);
        ty1 = __shfl(t1_threadInput2 , friend_id1);
        tx2 = __shfl(t1_threadInput2 , friend_id2);
        ty2 = __shfl(t1_threadInput3 , friend_id2);
        tz2 = __shfl(t1_threadInput4 , friend_id2);
        ty3 = __shfl(t1_threadInput4 , friend_id3);
        tz3 = __shfl(t1_threadInput5 , friend_id3);
        rx0 = __shfl(t1_threadInput5 , friend_id0);
        ry0 = __shfl(t1_threadInput6 , friend_id0);
        rx1 = __shfl(t1_threadInput6 , friend_id1);
        ry1 = __shfl(t1_threadInput7 , friend_id1);
        rx2 = __shfl(t1_threadInput7 , friend_id2);
        ry2 = __shfl(t1_threadInput8 , friend_id2);
        rz2 = __shfl(t1_threadInput9 , friend_id2);
        ry3 = __shfl(t1_threadInput9 , friend_id3);
        rz3 = __shfl(t1_threadInput10, friend_id3);
        sum0 += ARG_3D(5,k,j,i)*((lane_id < 16)? tx0: ty0);
        sum1 += ARG_3D(5,k,j,i)*((lane_id < 10)? tx1: ty1);
        sum2 += ARG_3D(5,k,j,i)*((lane_id < 4 )? tx2: ((lane_id < 30)? ty2: tz2));
        sum3 += ARG_3D(5,k,j,i)*((lane_id < 24)? ty3: tz3);
        sum4 += ARG_3D(5,k,j,i)*((lane_id < 16)? rx0: ry0);
        sum5 += ARG_3D(5,k,j,i)*((lane_id < 10)? rx1: ry1);
        sum6 += ARG_3D(5,k,j,i)*((lane_id < 4 )? rx2: ((lane_id < 30)? ry2: rz2));
        sum7 += ARG_3D(5,k,j,i)*((lane_id < 24)? ry3: rz3);
        tx0 = __shfl(t2_threadInput0 , friend_id0);
        ty0 = __shfl(t2_threadInput1 , friend_id0);
        tx1 = __shfl(t2_threadInput1 , friend_id1);
        ty1 = __shfl(t2_threadInput2 , friend_id1);
        tx2 = __shfl(t2_threadInput2 , friend_id2);
        ty2 = __shfl(t2_threadInput3 , friend_id2);
        tz2 = __shfl(t2_threadInput4 , friend_id2);
        ty3 = __shfl(t2_threadInput4 , friend_id3);
        tz3 = __shfl(t2_threadInput5 , friend_id3);
        rx0 = __shfl(t2_threadInput5 , friend_id0);
        ry0 = __shfl(t2_threadInput6 , friend_id0);
        rx1 = __shfl(t2_threadInput6 , friend_id1);
        ry1 = __shfl(t2_threadInput7 , friend_id1);
        rx2 = __shfl(t2_threadInput7 , friend_id2);
        ry2 = __shfl(t2_threadInput8 , friend_id2);
        rz2 = __shfl(t2_threadInput9 , friend_id2);
        ry3 = __shfl(t2_threadInput9 , friend_id3);
        rz3 = __shfl(t2_threadInput10, friend_id3);
        sum0 += ARG_3D(14,k,j,i)*((lane_id < 16)? tx0: ty0);
        sum1 += ARG_3D(14,k,j,i)*((lane_id < 10)? tx1: ty1);
        sum2 += ARG_3D(14,k,j,i)*((lane_id < 4 )? tx2: ((lane_id < 30)? ty2: tz2));
        sum3 += ARG_3D(14,k,j,i)*((lane_id < 24)? ty3: tz3);
        sum4 += ARG_3D(14,k,j,i)*((lane_id < 16)? rx0: ry0);
        sum5 += ARG_3D(14,k,j,i)*((lane_id < 10)? rx1: ry1);
        sum6 += ARG_3D(14,k,j,i)*((lane_id < 4 )? rx2: ((lane_id < 30)? ry2: rz2));
        sum7 += ARG_3D(14,k,j,i)*((lane_id < 24)? ry3: rz3);
        tx0 = __shfl(t3_threadInput0 , friend_id0);
        ty0 = __shfl(t3_threadInput1 , friend_id0);
        tx1 = __shfl(t3_threadInput1 , friend_id1);
        ty1 = __shfl(t3_threadInput2 , friend_id1);
        tx2 = __shfl(t3_threadInput2 , friend_id2);
        ty2 = __shfl(t3_threadInput3 , friend_id2);
        tz2 = __shfl(t3_threadInput4 , friend_id2);
        ty3 = __shfl(t3_threadInput4 , friend_id3);
        tz3 = __shfl(t3_threadInput5 , friend_id3);
        rx0 = __shfl(t3_threadInput5 , friend_id0);
        ry0 = __shfl(t3_threadInput6 , friend_id0);
        rx1 = __shfl(t3_threadInput6 , friend_id1);
        ry1 = __shfl(t3_threadInput7 , friend_id1);
        rx2 = __shfl(t3_threadInput7 , friend_id2);
        ry2 = __shfl(t3_threadInput8 , friend_id2);
        rz2 = __shfl(t3_threadInput9 , friend_id2);
        ry3 = __shfl(t3_threadInput9 , friend_id3);
        rz3 = __shfl(t3_threadInput10, friend_id3);
        sum0 += ARG_3D(23,k,j,i)*((lane_id < 16)? tx0: ty0);
        sum1 += ARG_3D(23,k,j,i)*((lane_id < 10)? tx1: ty1);
        sum2 += ARG_3D(23,k,j,i)*((lane_id < 4 )? tx2: ((lane_id < 30)? ty2: tz2));
        sum3 += ARG_3D(23,k,j,i)*((lane_id < 24)? ty3: tz3);
        sum4 += ARG_3D(23,k,j,i)*((lane_id < 16)? rx0: ry0);
        sum5 += ARG_3D(23,k,j,i)*((lane_id < 10)? rx1: ry1);
        sum6 += ARG_3D(23,k,j,i)*((lane_id < 4 )? rx2: ((lane_id < 30)? ry2: rz2));
        sum7 += ARG_3D(23,k,j,i)*((lane_id < 24)? ry3: rz3);

        friend_id0 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0 , friend_id0);
        ty0 = __shfl(t1_threadInput1 , friend_id0);
        tx1 = __shfl(t1_threadInput1 , friend_id1);
        ty1 = __shfl(t1_threadInput2 , friend_id1);
        tz1 = __shfl(t1_threadInput3 , friend_id1);
        ty2 = __shfl(t1_threadInput3 , friend_id2);
        tz2 = __shfl(t1_threadInput4 , friend_id2);
        ty3 = __shfl(t1_threadInput4 , friend_id3);
        tz3 = __shfl(t1_threadInput5 , friend_id3);
        rx0 = __shfl(t1_threadInput5 , friend_id0);
        ry0 = __shfl(t1_threadInput6 , friend_id0);
        rx1 = __shfl(t1_threadInput6 , friend_id1);
        ry1 = __shfl(t1_threadInput7 , friend_id1);
        rz1 = __shfl(t1_threadInput8 , friend_id1);
        ry2 = __shfl(t1_threadInput8 , friend_id2);
        rz2 = __shfl(t1_threadInput9 , friend_id2);
        ry3 = __shfl(t1_threadInput9 , friend_id3);
        rz3 = __shfl(t1_threadInput10, friend_id3);
        sum0 += ARG_3D(6,k,j,i)*((lane_id < 10)? tx0: ty0);
        sum1 += ARG_3D(6,k,j,i)*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
        sum2 += ARG_3D(6,k,j,i)*((lane_id < 24)? ty2: tz2);
        sum3 += ARG_3D(6,k,j,i)*((lane_id < 16)? ty3: tz3);
        sum4 += ARG_3D(6,k,j,i)*((lane_id < 10)? rx0: ry0);
        sum5 += ARG_3D(6,k,j,i)*((lane_id < 4 )? rx1: ((lane_id < 30)? ry1: rz1));
        sum6 += ARG_3D(6,k,j,i)*((lane_id < 24)? ry2: rz2);
        sum7 += ARG_3D(6,k,j,i)*((lane_id < 16)? ry3: rz3);
        tx0 = __shfl(t2_threadInput0 , friend_id0);
        ty0 = __shfl(t2_threadInput1 , friend_id0);
        tx1 = __shfl(t2_threadInput1 , friend_id1);
        ty1 = __shfl(t2_threadInput2 , friend_id1);
        tz1 = __shfl(t2_threadInput3 , friend_id1);
        ty2 = __shfl(t2_threadInput3 , friend_id2);
        tz2 = __shfl(t2_threadInput4 , friend_id2);
        ty3 = __shfl(t2_threadInput4 , friend_id3);
        tz3 = __shfl(t2_threadInput5 , friend_id3);
        rx0 = __shfl(t2_threadInput5 , friend_id0);
        ry0 = __shfl(t2_threadInput6 , friend_id0);
        rx1 = __shfl(t2_threadInput6 , friend_id1);
        ry1 = __shfl(t2_threadInput7 , friend_id1);
        rz1 = __shfl(t2_threadInput8 , friend_id1);
        ry2 = __shfl(t2_threadInput8 , friend_id2);
        rz2 = __shfl(t2_threadInput9 , friend_id2);
        ry3 = __shfl(t2_threadInput9 , friend_id3);
        rz3 = __shfl(t2_threadInput10, friend_id3);
        sum0 += ARG_3D(15,k,j,i)*((lane_id < 10)? tx0: ty0);
        sum1 += ARG_3D(15,k,j,i)*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
        sum2 += ARG_3D(15,k,j,i)*((lane_id < 24)? ty2: tz2);
        sum3 += ARG_3D(15,k,j,i)*((lane_id < 16)? ty3: tz3);
        sum4 += ARG_3D(15,k,j,i)*((lane_id < 10)? rx0: ry0);
        sum5 += ARG_3D(15,k,j,i)*((lane_id < 4 )? rx1: ((lane_id < 30)? ry1: rz1));
        sum6 += ARG_3D(15,k,j,i)*((lane_id < 24)? ry2: rz2);
        sum7 += ARG_3D(15,k,j,i)*((lane_id < 16)? ry3: rz3);
        tx0 = __shfl(t3_threadInput0 , friend_id0);
        ty0 = __shfl(t3_threadInput1 , friend_id0);
        tx1 = __shfl(t3_threadInput1 , friend_id1);
        ty1 = __shfl(t3_threadInput2 , friend_id1);
        tz1 = __shfl(t3_threadInput3 , friend_id1);
        ty2 = __shfl(t3_threadInput3 , friend_id2);
        tz2 = __shfl(t3_threadInput4 , friend_id2);
        ty3 = __shfl(t3_threadInput4 , friend_id3);
        tz3 = __shfl(t3_threadInput5 , friend_id3);
        rx0 = __shfl(t3_threadInput5 , friend_id0);
        ry0 = __shfl(t3_threadInput6 , friend_id0);
        rx1 = __shfl(t3_threadInput6 , friend_id1);
        ry1 = __shfl(t3_threadInput7 , friend_id1);
        rz1 = __shfl(t3_threadInput8 , friend_id1);
        ry2 = __shfl(t3_threadInput8 , friend_id2);
        rz2 = __shfl(t3_threadInput9 , friend_id2);
        ry3 = __shfl(t3_threadInput9 , friend_id3);
        rz3 = __shfl(t3_threadInput10, friend_id3);
        sum0 += ARG_3D(24,k,j,i)*((lane_id < 10)? tx0: ty0);
        sum1 += ARG_3D(24,k,j,i)*((lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1));
        sum2 += ARG_3D(24,k,j,i)*((lane_id < 24)? ty2: tz2);
        sum3 += ARG_3D(24,k,j,i)*((lane_id < 16)? ty3: tz3);
        sum4 += ARG_3D(24,k,j,i)*((lane_id < 10)? rx0: ry0);
        sum5 += ARG_3D(24,k,j,i)*((lane_id < 4 )? rx1: ((lane_id < 30)? ry1: rz1));
        sum6 += ARG_3D(24,k,j,i)*((lane_id < 24)? ry2: rz2);
        sum7 += ARG_3D(24,k,j,i)*((lane_id < 16)? ry3: rz3);

        friend_id0 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0 , friend_id0);
        ty0 = __shfl(t1_threadInput1 , friend_id0);
        tx1 = __shfl(t1_threadInput1 , friend_id1);
        ty1 = __shfl(t1_threadInput2 , friend_id1);
        tz1 = __shfl(t1_threadInput3 , friend_id1);
        ty2 = __shfl(t1_threadInput3 , friend_id2);
        tz2 = __shfl(t1_threadInput4 , friend_id2);
        ty3 = __shfl(t1_threadInput4 , friend_id3);
        tz3 = __shfl(t1_threadInput5 , friend_id3);
        rx0 = __shfl(t1_threadInput5 , friend_id0);
        ry0 = __shfl(t1_threadInput6 , friend_id0);
        rx1 = __shfl(t1_threadInput6 , friend_id1);
        ry1 = __shfl(t1_threadInput7 , friend_id1);
        rz1 = __shfl(t1_threadInput8 , friend_id1);
        ry2 = __shfl(t1_threadInput8 , friend_id2);
        rz2 = __shfl(t1_threadInput9 , friend_id2);
        ry3 = __shfl(t1_threadInput9 , friend_id3);
        rz3 = __shfl(t1_threadInput10, friend_id3);
        sum0 += ARG_3D(7,k,j,i)*((lane_id < 9 )? tx0: ty0);
        sum1 += ARG_3D(7,k,j,i)*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
        sum2 += ARG_3D(7,k,j,i)*((lane_id < 23)? ty2: tz2);
        sum3 += ARG_3D(7,k,j,i)*((lane_id < 16)? ty3: tz3);
        sum4 += ARG_3D(7,k,j,i)*((lane_id < 9 )? rx0: ry0);
        sum5 += ARG_3D(7,k,j,i)*((lane_id < 3 )? rx1: ((lane_id < 29)? ry1: rz1));
        sum6 += ARG_3D(7,k,j,i)*((lane_id < 23)? ry2: rz2);
        sum7 += ARG_3D(7,k,j,i)*((lane_id < 16)? ry3: rz3);
        tx0 = __shfl(t2_threadInput0 , friend_id0);
        ty0 = __shfl(t2_threadInput1 , friend_id0);
        tx1 = __shfl(t2_threadInput1 , friend_id1);
        ty1 = __shfl(t2_threadInput2 , friend_id1);
        tz1 = __shfl(t2_threadInput3 , friend_id1);
        ty2 = __shfl(t2_threadInput3 , friend_id2);
        tz2 = __shfl(t2_threadInput4 , friend_id2);
        ty3 = __shfl(t2_threadInput4 , friend_id3);
        tz3 = __shfl(t2_threadInput5 , friend_id3);
        rx0 = __shfl(t2_threadInput5 , friend_id0);
        ry0 = __shfl(t2_threadInput6 , friend_id0);
        rx1 = __shfl(t2_threadInput6 , friend_id1);
        ry1 = __shfl(t2_threadInput7 , friend_id1);
        rz1 = __shfl(t2_threadInput8 , friend_id1);
        ry2 = __shfl(t2_threadInput8 , friend_id2);
        rz2 = __shfl(t2_threadInput9 , friend_id2);
        ry3 = __shfl(t2_threadInput9 , friend_id3);
        rz3 = __shfl(t2_threadInput10, friend_id3);
        sum0 += ARG_3D(16,k,j,i)*((lane_id < 9 )? tx0: ty0);
        sum1 += ARG_3D(16,k,j,i)*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
        sum2 += ARG_3D(16,k,j,i)*((lane_id < 23)? ty2: tz2);
        sum3 += ARG_3D(16,k,j,i)*((lane_id < 16)? ty3: tz3);
        sum4 += ARG_3D(16,k,j,i)*((lane_id < 9 )? rx0: ry0);
        sum5 += ARG_3D(16,k,j,i)*((lane_id < 3 )? rx1: ((lane_id < 29)? ry1: rz1));
        sum6 += ARG_3D(16,k,j,i)*((lane_id < 23)? ry2: rz2);
        sum7 += ARG_3D(16,k,j,i)*((lane_id < 16)? ry3: rz3);
        tx0 = __shfl(t3_threadInput0 , friend_id0);
        ty0 = __shfl(t3_threadInput1 , friend_id0);
        tx1 = __shfl(t3_threadInput1 , friend_id1);
        ty1 = __shfl(t3_threadInput2 , friend_id1);
        tz1 = __shfl(t3_threadInput3 , friend_id1);
        ty2 = __shfl(t3_threadInput3 , friend_id2);
        tz2 = __shfl(t3_threadInput4 , friend_id2);
        ty3 = __shfl(t3_threadInput4 , friend_id3);
        tz3 = __shfl(t3_threadInput5 , friend_id3);
        rx0 = __shfl(t3_threadInput5 , friend_id0);
        ry0 = __shfl(t3_threadInput6 , friend_id0);
        rx1 = __shfl(t3_threadInput6 , friend_id1);
        ry1 = __shfl(t3_threadInput7 , friend_id1);
        rz1 = __shfl(t3_threadInput8 , friend_id1);
        ry2 = __shfl(t3_threadInput8 , friend_id2);
        rz2 = __shfl(t3_threadInput9 , friend_id2);
        ry3 = __shfl(t3_threadInput9 , friend_id3);
        rz3 = __shfl(t3_threadInput10, friend_id3);
        sum0 += ARG_3D(25,k,j,i)*((lane_id < 9 )? tx0: ty0);
        sum1 += ARG_3D(25,k,j,i)*((lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1));
        sum2 += ARG_3D(25,k,j,i)*((lane_id < 23)? ty2: tz2);
        sum3 += ARG_3D(25,k,j,i)*((lane_id < 16)? ty3: tz3);
        sum4 += ARG_3D(25,k,j,i)*((lane_id < 9 )? rx0: ry0);
        sum5 += ARG_3D(25,k,j,i)*((lane_id < 3 )? rx1: ((lane_id < 29)? ry1: rz1));
        sum6 += ARG_3D(25,k,j,i)*((lane_id < 23)? ry2: rz2);
        sum7 += ARG_3D(25,k,j,i)*((lane_id < 16)? ry3: rz3);
        
        friend_id0 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(t1_threadInput0 , friend_id0);
        ty0 = __shfl(t1_threadInput1 , friend_id0);
        tx1 = __shfl(t1_threadInput1 , friend_id1);
        ty1 = __shfl(t1_threadInput2 , friend_id1);
        tz1 = __shfl(t1_threadInput3 , friend_id1);
        ty2 = __shfl(t1_threadInput3 , friend_id2);
        tz2 = __shfl(t1_threadInput4 , friend_id2);
        ty3 = __shfl(t1_threadInput4 , friend_id3);
        tz3 = __shfl(t1_threadInput5 , friend_id3);
        rx0 = __shfl(t1_threadInput5 , friend_id0);
        ry0 = __shfl(t1_threadInput6 , friend_id0);
        rx1 = __shfl(t1_threadInput6 , friend_id1);
        ry1 = __shfl(t1_threadInput7 , friend_id1);
        rz1 = __shfl(t1_threadInput8 , friend_id1);
        ry2 = __shfl(t1_threadInput8 , friend_id2);
        rz2 = __shfl(t1_threadInput9 , friend_id2);
        ry3 = __shfl(t1_threadInput9 , friend_id3);
        rz3 = __shfl(t1_threadInput10, friend_id3);
        sum0 += ARG_3D(8,k,j,i)*((lane_id < 8 )? tx0: ty0);
        sum1 += ARG_3D(8,k,j,i)*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
        sum2 += ARG_3D(8,k,j,i)*((lane_id < 22)? ty2: tz2);
        sum3 += ARG_3D(8,k,j,i)*((lane_id < 16)? ty3: tz3);
        sum4 += ARG_3D(8,k,j,i)*((lane_id < 8 )? rx0: ry0);
        sum5 += ARG_3D(8,k,j,i)*((lane_id < 2 )? rx1: ((lane_id < 28)? ry1: rz1));
        sum6 += ARG_3D(8,k,j,i)*((lane_id < 22)? ry2: rz2);
        sum7 += ARG_3D(8,k,j,i)*((lane_id < 16)? ry3: rz3);
        tx0 = __shfl(t2_threadInput0 , friend_id0);
        ty0 = __shfl(t2_threadInput1 , friend_id0);
        tx1 = __shfl(t2_threadInput1 , friend_id1);
        ty1 = __shfl(t2_threadInput2 , friend_id1);
        tz1 = __shfl(t2_threadInput3 , friend_id1);
        ty2 = __shfl(t2_threadInput3 , friend_id2);
        tz2 = __shfl(t2_threadInput4 , friend_id2);
        ty3 = __shfl(t2_threadInput4 , friend_id3);
        tz3 = __shfl(t2_threadInput5 , friend_id3);
        rx0 = __shfl(t2_threadInput5 , friend_id0);
        ry0 = __shfl(t2_threadInput6 , friend_id0);
        rx1 = __shfl(t2_threadInput6 , friend_id1);
        ry1 = __shfl(t2_threadInput7 , friend_id1);
        rz1 = __shfl(t2_threadInput8 , friend_id1);
        ry2 = __shfl(t2_threadInput8 , friend_id2);
        rz2 = __shfl(t2_threadInput9 , friend_id2);
        ry3 = __shfl(t2_threadInput9 , friend_id3);
        rz3 = __shfl(t2_threadInput10, friend_id3);
        sum0 += ARG_3D(17,k,j,i)*((lane_id < 8 )? tx0: ty0);
        sum1 += ARG_3D(17,k,j,i)*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
        sum2 += ARG_3D(17,k,j,i)*((lane_id < 22)? ty2: tz2);
        sum3 += ARG_3D(17,k,j,i)*((lane_id < 16)? ty3: tz3);
        sum4 += ARG_3D(17,k,j,i)*((lane_id < 8 )? rx0: ry0);
        sum5 += ARG_3D(17,k,j,i)*((lane_id < 2 )? rx1: ((lane_id < 28)? ry1: rz1));
        sum6 += ARG_3D(17,k,j,i)*((lane_id < 22)? ry2: rz2);
        sum7 += ARG_3D(17,k,j,i)*((lane_id < 16)? ry3: rz3);
        tx0 = __shfl(t3_threadInput0 , friend_id0);
        ty0 = __shfl(t3_threadInput1 , friend_id0);
        tx1 = __shfl(t3_threadInput1 , friend_id1);
        ty1 = __shfl(t3_threadInput2 , friend_id1);
        tz1 = __shfl(t3_threadInput3 , friend_id1);
        ty2 = __shfl(t3_threadInput3 , friend_id2);
        tz2 = __shfl(t3_threadInput4 , friend_id2);
        ty3 = __shfl(t3_threadInput4 , friend_id3);
        tz3 = __shfl(t3_threadInput5 , friend_id3);
        rx0 = __shfl(t3_threadInput5 , friend_id0);
        ry0 = __shfl(t3_threadInput6 , friend_id0);
        rx1 = __shfl(t3_threadInput6 , friend_id1);
        ry1 = __shfl(t3_threadInput7 , friend_id1);
        rz1 = __shfl(t3_threadInput8 , friend_id1);
        ry2 = __shfl(t3_threadInput8 , friend_id2);
        rz2 = __shfl(t3_threadInput9 , friend_id2);
        ry3 = __shfl(t3_threadInput9 , friend_id3);
        rz3 = __shfl(t3_threadInput10, friend_id3);
        sum0 += ARG_3D(26,k,j,i)*((lane_id < 8 )? tx0: ty0);
        sum1 += ARG_3D(26,k,j,i)*((lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1));
        sum2 += ARG_3D(26,k,j,i)*((lane_id < 22)? ty2: tz2);
        sum3 += ARG_3D(26,k,j,i)*((lane_id < 16)? ty3: tz3);
        sum4 += ARG_3D(26,k,j,i)*((lane_id < 8 )? rx0: ry0);
        sum5 += ARG_3D(26,k,j,i)*((lane_id < 2 )? rx1: ((lane_id < 28)? ry1: rz1));
        sum6 += ARG_3D(26,k,j,i)*((lane_id < 22)? ry2: rz2);
        sum7 += ARG_3D(26,k,j,i)*((lane_id < 16)? ry3: rz3);

        
        OUT_3D(k,j   ,i) = sum0;
        OUT_3D(k,j+4 ,i) = sum1;
        OUT_3D(k,j+8 ,i) = sum2;
        OUT_3D(k,j+12,i) = sum3;
        OUT_3D(k,j+16,i) = sum4;
        OUT_3D(k,j+20,i) = sum5;
        OUT_3D(k,j+24,i) = sum6;
        OUT_3D(k,j+28,i) = sum7;
}
}int main(int argc, char **argv)
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
    Init_Input_3D(in, z, m, n, halo, seed);

    // Show_Me(in, z, m, n, halo, "Input:");
    for(int i = 0; i < ITER; i++)
    {
        Stencil_Seq(in, out_ref, args, z, m, n, halo);
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
    DATA_TYPE *args_d;
    DATA_TYPE *out_d;
    DATA_TYPE *out = new DATA_TYPE[total];
    cudaMalloc((void**)&in_d, total*sizeof(DATA_TYPE));
    cudaMalloc((void**)&out_d, total*sizeof(DATA_TYPE));
    cudaMalloc((void**)&args_d, (K)*sizeof(DATA_TYPE));
    cudaMemcpy(args_d, args, (K)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    // Cuda version
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed);
    Clear_Output_3D(out, z, m, n, halo);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid((n)/8, (m)/4, (z)/8);
    dim3 dimBlock(8, 4, 8);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda<<<dimGrid, dimBlock>>>(in_d, out_d, args_d, z, m, n, halo); 
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, z, m, n, halo, "Output(Cuda):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(1, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda Sweep version
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid1((n)/64, (m)/4, 4);
    dim3 dimBlock1(64, 4, 1);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sweep<<<dimGrid1, dimBlock1>>>(in_d, out_d, args_d, z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, z, m, n, "Output(Sweep):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sweep Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(1, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));


    // Cuda 3D Block SM version
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid2((n)/8, (m)/4, (z)/8);
    dim3 dimBlock2(8, 4, 8);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sm<<<dimGrid2, dimBlock2>>>(
            in_d, out_d, args_d, z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, z, m, n, "Output(Cuda_Sm):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sm: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sm Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(1, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));


    // Cuda 2D Block SM version
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid3((n)/64, (m)/4, 4);
    dim3 dimBlock3(64, 4, 1);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sweep_Sm<<<dimGrid3, dimBlock3, ((SM_2D_M)*(SM_2D_N)*3*sizeof(DATA_TYPE))>>>(
            in_d, out_d, args_d, z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, z, m, n, "Output(Cuda_Sweep_Sm):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep_Sm: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sweep_Sm Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(1, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));


    // Cuda 3D Block Shfl version
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid4((n)/8, (m)/4, (z)/8);
    dim3 dimBlock4(8, 4, 8);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl<<<dimGrid4, dimBlock4>>>(
            in_d, out_d, args_d, z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, z, m, n, "Output(Cuda_Shfl):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Shfl Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(1, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));


    // Cuda 3D Block Shfl2 version
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid5((n)/8, (m)/4, (z)/(8*2));
    dim3 dimBlock5(8, 4, 8);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl2<<<dimGrid5, dimBlock5>>>(
            in_d, out_d, args_d, z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, z, m, n, "Output(Cuda_Shfl2):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl2: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Shfl2 Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(1, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));


    // Cuda 3D Block Shfl4 version
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid6((n)/8, (m)/4, (z)/(8*4));
    dim3 dimBlock6(8, 4, 8);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl4<<<dimGrid6, dimBlock6>>>(
            in_d, out_d, args_d, z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, z, m, n, "Output(Cuda_Shfl4):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl4: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Shfl4 Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(1, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));


    // Cuda 3D Block Shfl8 version
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid7((n)/8, (m)/4, (z)/(8*8));
    dim3 dimBlock7(8, 4, 8);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl8<<<dimGrid7, dimBlock7>>>(
            in_d, out_d, args_d, z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, z, m, n, halo, "Output(Cuda_Shfl8):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl8: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Shfl8 Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(1, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));


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
        Stencil_Cuda_Sweep_Shfl<<<dimGrid8, dimBlock8>>>(
            in_d, out_d, args_d, z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, z, m, n, "Output(Cuda_Sweep_Shfl):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep_Shfl: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sweep_Shfl Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(1, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

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
        Stencil_Cuda_Sweep_Shfl2<<<dimGrid9, dimBlock9>>>(
            in_d, out_d, args_d, z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, z, m, n, "Output(Cuda_Sweep_Shfl2):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep_Shfl2: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sweep_Shfl2 Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(1, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

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
        Stencil_Cuda_Sweep_Shfl4<<<dimGrid10, dimBlock10>>>(
            in_d, out_d, args_d, z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, z, m, n, "Output(Cuda_Sweep_Shfl4):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep_Shfl4: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sweep_Shfl4 Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(1, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda 2D Block Shfl8 version
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed); // reset input
    Clear_Output_3D(out, z, m, n, halo); // flush output
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid11((n)/8, (m)/(32*8), 4);
    dim3 dimBlock11(8, 32, 1);
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sweep_Shfl8<<<dimGrid11, dimBlock11>>>(
            in_d, out_d, args_d, z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    // Show_Me(out, z, m, n, "Output(Cuda_Sweep_Shfl8):");
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep_Shfl8: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sweep_Shfl8 Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(1, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));


    cudaFree(in_d);
    cudaFree(out_d);

    delete[] in;
    delete[] out;
    delete[] out_ref;

}
