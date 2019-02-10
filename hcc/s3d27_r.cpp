#include <iostream>
#include <cmath>
#include <hc.hpp>
#include "sten_metrics.h"

using namespace hc;

#define warpSize 64

// #define __DEBUG

// #ifdef __DEBUG
// #define ITER 1
// #else
// #define ITER 100
// #endif

typedef union
{
    int split[2];
    double d_val;
}split_db;

double __shfl(double x, int lane) restrict(amp)
{
    int lo, hi;
    split_db my_union;
    my_union.d_val = x;
    lo = my_union.split[0];
    hi = my_union.split[1];

    lo = __shfl(lo, lane);
    hi = __shfl(hi, lane);

    my_union.split[0] = lo;
    my_union.split[1] = hi;

    return my_union.d_val;
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

void Stencil_Seq(DATA_TYPE *in, DATA_TYPE *out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , DATA_TYPE a7 , DATA_TYPE a8 , DATA_TYPE a9 , 
        DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, DATA_TYPE a13, DATA_TYPE a14, 
        DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, DATA_TYPE a18, DATA_TYPE a19, 
        DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, DATA_TYPE a23, DATA_TYPE a24, 
        DATA_TYPE a25, DATA_TYPE a26, 
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

void Stencil_Hcc_L1_3Blk(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , DATA_TYPE a7 , DATA_TYPE a8 , DATA_TYPE a9 , 
        DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, DATA_TYPE a13, DATA_TYPE a14, 
        DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, DATA_TYPE a18, DATA_TYPE a19, 
        DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, DATA_TYPE a23, DATA_TYPE a24, 
        DATA_TYPE a25, DATA_TYPE a26, 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(z, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 4, 8, 8);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        int i = tidx.global[2] + halo;
        int j = tidx.global[1] + halo;
        int k = tidx.global[0] + halo;

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
    });
    fut.wait();
}

void Stencil_Hcc_Sweep(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , DATA_TYPE a7 , DATA_TYPE a8 , DATA_TYPE a9 , 
        DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, DATA_TYPE a13, DATA_TYPE a14, 
        DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, DATA_TYPE a18, DATA_TYPE a19, 
        DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, DATA_TYPE a23, DATA_TYPE a24, 
        DATA_TYPE a25, DATA_TYPE a26, 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(4, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 1, 8, 32);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        int i = tidx.global[2] + halo;
        int j = tidx.global[1] + halo;

        const int block_z = z / 4;
        int k = block_z * tidx.tile[0] + halo;
        const int k_end = k + block_z;

#pragma unroll 
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
    });
    fut.wait();
}

void Stencil_Hcc_L1_25Blk(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , DATA_TYPE a7 , DATA_TYPE a8 , DATA_TYPE a9 , 
        DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, DATA_TYPE a13, DATA_TYPE a14, 
        DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, DATA_TYPE a18, DATA_TYPE a19, 
        DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, DATA_TYPE a23, DATA_TYPE a24, 
        DATA_TYPE a25, DATA_TYPE a26, 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(4, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 1, 4, 64);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        int i = tidx.global[2] + halo;
        int j = tidx.global[1] + halo;

        const int block_z = z / 4;
        int k = block_z * tidx.tile[0] + halo;
        const int k_end = k + block_z;

#pragma unroll 
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
    });
    fut.wait();
}

void Stencil_Hcc_Lds_3BlkBrc(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , DATA_TYPE a7 , DATA_TYPE a8 , DATA_TYPE a9 , 
        DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, DATA_TYPE a13, DATA_TYPE a14, 
        DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, DATA_TYPE a18, DATA_TYPE a19, 
        DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, DATA_TYPE a23, DATA_TYPE a24, 
        DATA_TYPE a25, DATA_TYPE a26, 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(z, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 4, 8, 8);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        int i = tidx.global[2] + halo;
        int j = tidx.global[1] + halo;
        int k = tidx.global[0] + halo;

        int li = tidx.local[2] + 1;
        int lj = tidx.local[1] + 1;
        int lk = tidx.local[0] + 1;

        tile_static DATA_TYPE local[4+2][8+2][8+2];
        LOC_3D(lk,lj,li) = IN_3D(k,j,i);

        if(li == halo) LOC_3D(lk,lj,li-1) = IN_3D(k,j,i-1);
        if(li == 8   ) LOC_3D(lk,lj,li+1) = IN_3D(k,j,i+1);
        if(lj == halo) LOC_3D(lk,lj-1,li) = IN_3D(k,j-1,i);
        if(lj == 8   ) LOC_3D(lk,lj+1,li) = IN_3D(k,j+1,i);
        if(lk == halo) LOC_3D(lk-1,lj,li) = IN_3D(k-1,j,i);
        if(lk == 4   ) LOC_3D(lk+1,lj,li) = IN_3D(k+1,j,i);
        if(li == halo && lj == halo) LOC_3D(lk  ,lj-1,li-1) = IN_3D(k  ,j-1,i-1);
        if(li == halo && lk == halo) LOC_3D(lk-1,lj  ,li-1) = IN_3D(k-1,j  ,i-1);
        if(lj == halo && lk == halo) LOC_3D(lk-1,lj-1,li  ) = IN_3D(k-1,j-1,i  );
        if(li == 8    && lj == halo) LOC_3D(lk  ,lj-1,li+1) = IN_3D(k  ,j-1,i+1);
        if(li == 8    && lk == halo) LOC_3D(lk-1,lj  ,li+1) = IN_3D(k-1,j  ,i+1);
        if(lj == 8    && lk == halo) LOC_3D(lk-1,lj+1,li  ) = IN_3D(k-1,j+1,i  );
        if(li == halo && lj == 8   ) LOC_3D(lk  ,lj+1,li-1) = IN_3D(k  ,j+1,i-1);
        if(li == halo && lk == 4   ) LOC_3D(lk+1,lj  ,li-1) = IN_3D(k+1,j  ,i-1);
        if(lj == halo && lk == 4   ) LOC_3D(lk+1,lj-1,li  ) = IN_3D(k+1,j-1,i  );
        if(li == 8    && lj == 8   ) LOC_3D(lk  ,lj+1,li+1) = IN_3D(k  ,j+1,i+1);
        if(li == 8    && lk == 4   ) LOC_3D(lk+1,lj  ,li+1) = IN_3D(k+1,j  ,i+1);
        if(lj == 8    && lk == 4   ) LOC_3D(lk+1,lj+1,li  ) = IN_3D(k+1,j+1,i  );
        if(li == halo && lj == halo && lk == halo) LOC_3D(lk-1,lj-1,li-1) = IN_3D(k-1,j-1,i-1);
        if(li == halo && lj == halo && lk == 4   ) LOC_3D(lk+1,lj-1,li-1) = IN_3D(k+1,j-1,i-1);
        if(li == halo && lj == 8    && lk == halo) LOC_3D(lk-1,lj+1,li-1) = IN_3D(k-1,j+1,i-1);
        if(li == halo && lj == 8    && lk == 4   ) LOC_3D(lk+1,lj+1,li-1) = IN_3D(k+1,j+1,i-1);
        if(li == 8    && lj == halo && lk == halo) LOC_3D(lk-1,lj-1,li+1) = IN_3D(k-1,j-1,i+1);
        if(li == 8    && lj == halo && lk == 4   ) LOC_3D(lk+1,lj-1,li+1) = IN_3D(k+1,j-1,i+1);
        if(li == 8    && lj == 8    && lk == halo) LOC_3D(lk-1,lj+1,li+1) = IN_3D(k-1,j+1,i+1);
        if(li == 8    && lj == 8    && lk == 4   ) LOC_3D(lk+1,lj+1,li+1) = IN_3D(k+1,j+1,i+1);
        tidx.barrier.wait();

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
    });
    fut.wait();
}

void Stencil_Hcc_Sm_Branch(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , DATA_TYPE a7 , DATA_TYPE a8 , DATA_TYPE a9 , 
        DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, DATA_TYPE a13, DATA_TYPE a14, 
        DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, DATA_TYPE a18, DATA_TYPE a19, 
        DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, DATA_TYPE a23, DATA_TYPE a24, 
        DATA_TYPE a25, DATA_TYPE a26, 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(z, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 8, 4, 8);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        int i = tidx.global[2] + halo;
        int j = tidx.global[1] + halo;
        int k = tidx.global[0] + halo;

        int li = tidx.local[2] + 1;
        int lj = tidx.local[1] + 1;
        int lk = tidx.local[0] + 1;

        tile_static DATA_TYPE local[8+2][4+2][8+2];
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
        tidx.barrier.wait();

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
    });
    fut.wait();
}

void Stencil_Hcc_Lds_3BlkCyc(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , DATA_TYPE a7 , DATA_TYPE a8 , DATA_TYPE a9 , 
        DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, DATA_TYPE a13, DATA_TYPE a14, 
        DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, DATA_TYPE a18, DATA_TYPE a19, 
        DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, DATA_TYPE a23, DATA_TYPE a24, 
        DATA_TYPE a25, DATA_TYPE a26, 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(z, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 4, 8, 8);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        int i = tidx.global[2] + halo;
        int j = tidx.global[1] + halo;
        int k = tidx.global[0] + halo;

        int li = tidx.local[2] + 1;
        int lj = tidx.local[1] + 1;
        int lk = tidx.local[0] + 1;

        tile_static DATA_TYPE local[4+2][8+2][8+2];

        int lane_id = tidx.local[2] + tidx.local[1] * tidx.tile_dim[2] +
            tidx.local[0] * tidx.tile_dim[2] * tidx.tile_dim[1];

        int blk_id_x = tidx.tile[2];
        int blk_id_y = tidx.tile[1];
        int blk_id_z = tidx.tile[0];

        int new_i, new_j, new_k, new_li, new_lj, new_lk;
        new_i  = (blk_id_x<<3) + lane_id%10 ;     
        new_j  = (blk_id_y<<3) + (lane_id/10)%10 ; 
        new_k  = (blk_id_z<<2) + lane_id/100 ;     
        new_li = lane_id%10;
        new_lj = (lane_id/10)%10;
        new_lk = lane_id/100;
        LOC_3D(new_lk,new_lj,new_li) = IN_3D(new_k,new_j,new_i);
        new_i  = (blk_id_x<<3) + (lane_id+256)%10 ;
        new_j  = (blk_id_y<<3) + ((lane_id+256)/10)%10 ;
        new_k  = (blk_id_z<<2) + (lane_id+256)/100 ;
        new_li = (lane_id+256)%10;
        new_lj = ((lane_id+256)/10)%10 ;
        new_lk = (lane_id+256)/100; 
        LOC_3D(new_lk,new_lj,new_li) = IN_3D(new_k,new_j,new_i);
        new_i  = (blk_id_x<<3) + (lane_id+512)%10 ;
        new_j  = (blk_id_y<<3) + ((lane_id+512)/10)%10 ;
        new_k  = (blk_id_z<<2) + (lane_id+512)/100 ;
        new_li = (lane_id+512)%10;
        new_lj = ((lane_id+512)/10)%10;
        new_lk = (lane_id+512)/100; 
        if(new_li < 10 &&  new_lj < 10 && new_lk < 6 )
            LOC_3D(new_lk,new_lj,new_li) = IN_3D(new_k,new_j,new_i);

        tidx.barrier.wait();

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
    });
    fut.wait();
}
void Stencil_Hcc_Sm_Cyclic(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , DATA_TYPE a7 , DATA_TYPE a8 , DATA_TYPE a9 , 
        DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, DATA_TYPE a13, DATA_TYPE a14, 
        DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, DATA_TYPE a18, DATA_TYPE a19, 
        DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, DATA_TYPE a23, DATA_TYPE a24, 
        DATA_TYPE a25, DATA_TYPE a26, 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(z, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 8, 4, 8);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        int i = tidx.global[2] + halo;
        int j = tidx.global[1] + halo;
        int k = tidx.global[0] + halo;

        int li = tidx.local[2] + 1;
        int lj = tidx.local[1] + 1;
        int lk = tidx.local[0] + 1;

        tile_static DATA_TYPE local[8+2][4+2][8+2];

        int lane_id = tidx.local[2] + tidx.local[1] * tidx.tile_dim[2] +
            tidx.local[0] * tidx.tile_dim[2] * tidx.tile_dim[1];

        int blk_id_x = tidx.tile[2];
        int blk_id_y = tidx.tile[1];
        int blk_id_z = tidx.tile[0];

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

        tidx.barrier.wait();

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
    });
    fut.wait();
}

void Stencil_Hcc_Sweep_Sm_Branch(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , DATA_TYPE a7 , DATA_TYPE a8 , DATA_TYPE a9 , 
        DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, DATA_TYPE a13, DATA_TYPE a14, 
        DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, DATA_TYPE a18, DATA_TYPE a19, 
        DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, DATA_TYPE a23, DATA_TYPE a24, 
        DATA_TYPE a25, DATA_TYPE a26, 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(4, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 1, 8, 32);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        int i = tidx.global[2] + halo;
        int j = tidx.global[1] + halo;

        const int block_z = z / 4;
        int k = block_z * tidx.tile[0] + halo;
        const int k_end = k + block_z;

        int li = tidx.local[2] + 1;
        int lj = tidx.local[1] + 1;

        tile_static DATA_TYPE local[3][8+2][32+2];

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
            tidx.barrier.wait();

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
        }
    });
    fut.wait();
}

void Stencil_Hcc_Lds_25BlkBrc(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , DATA_TYPE a7 , DATA_TYPE a8 , DATA_TYPE a9 , 
        DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, DATA_TYPE a13, DATA_TYPE a14, 
        DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, DATA_TYPE a18, DATA_TYPE a19, 
        DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, DATA_TYPE a23, DATA_TYPE a24, 
        DATA_TYPE a25, DATA_TYPE a26, 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(4, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 1, 4, 64);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        int i = tidx.global[2] + halo;
        int j = tidx.global[1] + halo;

        const int block_z = z / 4;
        int k = block_z * tidx.tile[0] + halo;
        const int k_end = k + block_z;

        int li = tidx.local[2] + 1;
        int lj = tidx.local[1] + 1;

        tile_static DATA_TYPE local[3][4+2][64+2];

        int t1, t2, t3;
        t3 = 2; t2 = 1;

        LOC_3D(t3,lj,li) = IN_3D(k  ,j,i);
        LOC_3D(t2,lj,li) = IN_3D(k-1,j,i);
        if(li == halo)                                   
        {
            LOC_3D(t3,lj  ,li-1) = IN_3D(k  ,j  ,i-1); 
            LOC_3D(t2,lj  ,li-1) = IN_3D(k-1,j  ,i-1);
        }
        if(li == 64  )                         
        {
            LOC_3D(t3,lj  ,li+1) = IN_3D(k  ,j  ,i+1); 
            LOC_3D(t2,lj  ,li+1) = IN_3D(k-1,j  ,i+1);
        }
        if(lj == halo)                                   
        {
            LOC_3D(t3,lj-1,li  ) = IN_3D(k  ,j-1,i  ); 
            LOC_3D(t2,lj-1,li  ) = IN_3D(k-1,j-1,i  );
        }
        if(lj == 4   )                         
        {
            LOC_3D(t3,lj+1,li  ) = IN_3D(k  ,j+1,i  ); 
            LOC_3D(t2,lj+1,li  ) = IN_3D(k-1,j+1,i  );
        }
        if(li == halo && lj == halo)                     
        {
            LOC_3D(t3,lj-1,li-1) = IN_3D(k  ,j-1,i-1); 
            LOC_3D(t2,lj-1,li-1) = IN_3D(k-1,j-1,i-1);
        }
        if(li == 64   && lj == halo)           
        {
            LOC_3D(t3,lj-1,li+1) = IN_3D(k  ,j-1,i+1); 
            LOC_3D(t2,lj-1,li+1) = IN_3D(k-1,j-1,i+1);
        }
        if(li == halo && lj == 4   )           
        { 
            LOC_3D(t3,lj+1,li-1) = IN_3D(k  ,j+1,i-1); 
            LOC_3D(t2,lj+1,li-1) = IN_3D(k-1,j+1,i-1);
        }
        if(li == 64   && lj == 4   ) 
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
            if(li == 64  )               {LOC_3D(t3,lj  ,li+1) = IN_3D(k+1,j  ,i+1);}
            if(lj == halo)               {LOC_3D(t3,lj-1,li  ) = IN_3D(k+1,j-1,i  );}
            if(lj == 4   )               {LOC_3D(t3,lj+1,li  ) = IN_3D(k+1,j+1,i  );}
            if(li == halo && lj == halo) {LOC_3D(t3,lj-1,li-1) = IN_3D(k+1,j-1,i-1);}
            if(li == 64   && lj == halo) {LOC_3D(t3,lj-1,li+1) = IN_3D(k+1,j-1,i+1);}
            if(li == halo && lj == 4   ) {LOC_3D(t3,lj+1,li-1) = IN_3D(k+1,j+1,i-1);}
            if(li == 64   && lj == 4   ) {LOC_3D(t3,lj+1,li+1) = IN_3D(k+1,j+1,i+1);}
            tidx.barrier.wait();

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
        }
    });
    fut.wait();
}

void Stencil_Hcc_Sweep_Sm_Cyclic(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , DATA_TYPE a7 , DATA_TYPE a8 , DATA_TYPE a9 , 
        DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, DATA_TYPE a13, DATA_TYPE a14, 
        DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, DATA_TYPE a18, DATA_TYPE a19, 
        DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, DATA_TYPE a23, DATA_TYPE a24, 
        DATA_TYPE a25, DATA_TYPE a26, 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(4, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 1, 8, 32);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        int i = tidx.global[2] + halo;
        int j = tidx.global[1] + halo;

        const int block_z = z / 4;
        int k = block_z * tidx.tile[0] + halo;
        const int k_end = k + block_z;

        int li = tidx.local[2] + 1;
        int lj = tidx.local[1] + 1;

        tile_static DATA_TYPE local[3][8+2][32+2];

        int t1, t2, t3;
        t3 = 2; t2 = 1;

        unsigned int lane_id = tidx.local[2] + tidx.local[1] * tidx.tile_dim[2];
        int blk_id_x = tidx.tile[2];
        int blk_id_y = tidx.tile[1];
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
            tidx.barrier.wait();

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
        }
    });
    fut.wait();
}

void Stencil_Hcc_Lds_25BlkCyc(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , DATA_TYPE a7 , DATA_TYPE a8 , DATA_TYPE a9 , 
        DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, DATA_TYPE a13, DATA_TYPE a14, 
        DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, DATA_TYPE a18, DATA_TYPE a19, 
        DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, DATA_TYPE a23, DATA_TYPE a24, 
        DATA_TYPE a25, DATA_TYPE a26, 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(4, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 1, 4, 64);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        int i = tidx.global[2] + halo;
        int j = tidx.global[1] + halo;

        const int block_z = z / 4;
        int k = block_z * tidx.tile[0] + halo;
        const int k_end = k + block_z;

        int li = tidx.local[2] + 1;
        int lj = tidx.local[1] + 1;

        tile_static DATA_TYPE local[3][4+2][64+2];

        int t1, t2, t3;
        t3 = 2; t2 = 1;

        unsigned int lane_id = tidx.local[2] + tidx.local[1] * tidx.tile_dim[2];
        int blk_id_x = tidx.tile[2];
        int blk_id_y = tidx.tile[1];
        int new_i, new_j, new_li, new_lj;
        new_i  = (blk_id_x<<6) + lane_id%66;
        new_j  = (blk_id_y<<2) + lane_id/66;
        new_li = lane_id%66;
        new_lj = lane_id/66;
        LOC_3D(t3,new_lj,new_li) = IN_3D(k  ,new_j,new_i);
        LOC_3D(t2,new_lj,new_li) = IN_3D(k-1,new_j,new_i);
        new_i  = (blk_id_x<<6) + (lane_id+256)%66;
        new_j  = (blk_id_y<<2) + (lane_id+256)/66;
        new_li = (lane_id+256)%66;
        new_lj = (lane_id+256)/66;
        new_i  = (new_i < n+2*halo)? new_i: n+2*halo-1;
        new_j  = (new_j < m+2*halo)? new_j: m+2*halo-1;
        if(new_li < 66 &&  new_lj < 6)
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

            new_i  = (blk_id_x<<6) + lane_id%66;
            new_j  = (blk_id_y<<2) + lane_id/66;
            new_li = lane_id%66;
            new_lj = lane_id/66;
            LOC_3D(t3,new_lj,new_li) = IN_3D(k+1,new_j,new_i);
            new_i  = (blk_id_x<<6) + (lane_id+256)%66;
            new_j  = (blk_id_y<<2) + (lane_id+256)/66;
            new_li = (lane_id+256)%66;
            new_lj = (lane_id+256)/66;
            new_i  = (new_i < n+2*halo)? new_i: n+2*halo-1;
            new_j  = (new_j < m+2*halo)? new_j: m+2*halo-1;
            if(new_li < 66 &&  new_lj < 6)
            {
                LOC_3D(t3,new_lj,new_li) = IN_3D(k+1,new_j,new_i);
            }
            tidx.barrier.wait();

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
        }
    });
    fut.wait();
}

void Stencil_Hcc_Reg1_3Blk2Wf(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , DATA_TYPE a7 , DATA_TYPE a8 , DATA_TYPE a9 , 
        DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, DATA_TYPE a13, DATA_TYPE a14, 
        DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, DATA_TYPE a18, DATA_TYPE a19, 
        DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, DATA_TYPE a23, DATA_TYPE a24, 
        DATA_TYPE a25, DATA_TYPE a26, 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(z, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 4, 8, 8);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        int i = tidx.global[2] + halo;
        int j = tidx.global[1] + halo;
        int k = tidx.global[0] + halo;
        const int lane_id = __lane_id();
        int lane_id_it = lane_id;

        int warp_id0 = (tidx.global[2])>>3;
        int warp_id1 = (tidx.global[1])>>3;
        int warp_id2 = (tidx.global[0])>>0;
        int new_id0 ;
        int new_id1 ;
        int new_id2 ;
        DATA_TYPE reg0, reg1, reg2, reg3, reg4;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg0 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg1 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg2 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg3 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
        new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
        new_id2 = (new_id2 < z+2)? new_id2 : z+1 ;
        reg4 = IN_3D(new_id2, new_id1, new_id0) ;

        DATA_TYPE sum0 = 0.0;
        int friend_id0;
        DATA_TYPE tx0, ty0, tz0;

        // process (0, 0, 0)
        friend_id0 = (lane_id+ 0+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a0 *((lane_id < 52 )? tx0: ty0);
        // process (1, 0, 0)
        friend_id0 = (lane_id+ 1+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a1 *((lane_id < 51 )? tx0: ty0);
        // process (2, 0, 0)
        friend_id0 = (lane_id+ 2+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a2 *((lane_id < 50 )? tx0: ty0);
        // process (0, 1, 0)
        friend_id0 = (lane_id+10+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a3 *((lane_id < 44 )? tx0: ty0);
        // process (1, 1, 0)
        friend_id0 = (lane_id+11+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a4 *((lane_id < 43 )? tx0: ty0);
        // process (2, 1, 0)
        friend_id0 = (lane_id+12+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a5 *((lane_id < 42 )? tx0: ty0);
        // process (0, 2, 0)
        friend_id0 = (lane_id+20+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a6 *((lane_id < 36 )? tx0: ty0);
        // process (1, 2, 0)
        friend_id0 = (lane_id+21+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a7 *((lane_id < 35 )? tx0: ty0);
        // process (2, 2, 0)
        friend_id0 = (lane_id+22+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a8 *((lane_id < 34 )? tx0: ty0);
        // process (0, 0, 1)
        friend_id0 = (lane_id+36+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a9 *((lane_id < 24 )? tx0: ty0);
        // process (1, 0, 1)
        friend_id0 = (lane_id+37+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a10*((lane_id < 23 )? tx0: ty0);
        // process (2, 0, 1)
        friend_id0 = (lane_id+38+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a11*((lane_id < 22 )? tx0: ty0);
        // process (0, 1, 1)
        friend_id0 = (lane_id+46+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a12*((lane_id < 16 )? tx0: ty0);
        // process (1, 1, 1)
        friend_id0 = (lane_id+47+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a13*((lane_id < 15 )? tx0: ty0);
        // process (2, 1, 1)
        friend_id0 = (lane_id+48+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a14*((lane_id < 14 )? tx0: ty0);
        // process (0, 2, 1)
        friend_id0 = (lane_id+56+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tz0 = __shfl(reg3, friend_id0);
        sum0 += a15*((lane_id < 8 )? tx0: ((lane_id < 58)? ty0: tz0));
        // process (1, 2, 1)
        friend_id0 = (lane_id+57+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tz0 = __shfl(reg3, friend_id0);
        sum0 += a16*((lane_id < 7 )? tx0: ((lane_id < 57)? ty0: tz0));
        // process (2, 2, 1)
        friend_id0 = (lane_id+58+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tz0 = __shfl(reg3, friend_id0);
        sum0 += a17*((lane_id < 6 )? tx0: ((lane_id < 56)? ty0: tz0));
        // process (0, 0, 2)
        friend_id0 = (lane_id+ 8+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a18*((lane_id < 46 )? tx0: ty0);
        // process (1, 0, 2)
        friend_id0 = (lane_id+ 9+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a19*((lane_id < 45 )? tx0: ty0);
        // process (2, 0, 2)
        friend_id0 = (lane_id+10+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a20*((lane_id < 44 )? tx0: ty0);
        // process (0, 1, 2)
        friend_id0 = (lane_id+18+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a21*((lane_id < 38 )? tx0: ty0);
        // process (1, 1, 2)
        friend_id0 = (lane_id+19+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a22*((lane_id < 37 )? tx0: ty0);
        // process (2, 1, 2)
        friend_id0 = (lane_id+20+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a23*((lane_id < 36 )? tx0: ty0);
        // process (0, 2, 2)
        friend_id0 = (lane_id+28+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a24*((lane_id < 30 )? tx0: ty0);
        // process (1, 2, 2)
        friend_id0 = (lane_id+29+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a25*((lane_id < 29 )? tx0: ty0);
        // process (2, 2, 2)
        friend_id0 = (lane_id+30+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a26*((lane_id < 28 )? tx0: ty0);

        OUT_3D(k,j,i) = sum0;

    });
    fut.wait();
}

void Stencil_Hcc_Reg2_3Blk2Wf(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , DATA_TYPE a7 , DATA_TYPE a8 , DATA_TYPE a9 , 
        DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, DATA_TYPE a13, DATA_TYPE a14, 
        DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, DATA_TYPE a18, DATA_TYPE a19, 
        DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, DATA_TYPE a23, DATA_TYPE a24, 
        DATA_TYPE a25, DATA_TYPE a26, 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(z/2, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 4, 8, 8);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        int i = tidx.global[2] + halo;
        int j = tidx.global[1] + halo;
        int k = (((tidx.global[0])>>0)<<1) + halo;
        const int lane_id = __lane_id();
        int lane_id_it = lane_id;

        int warp_id0 = (tidx.global[2])>>3;
        int warp_id1 = (tidx.global[1])>>3;
        int warp_id2 = (((tidx.global[0])>>0)<<1)>>0;
        int new_id0 ;
        int new_id1 ;
        int new_id2 ;
        DATA_TYPE reg0, reg1, reg2, reg3, reg4, reg5, reg6;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg0 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg1 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg2 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg3 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg4 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg5 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
        new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
        new_id2 = (new_id2 < z+2)? new_id2 : z+1 ;
        reg6 = IN_3D(new_id2, new_id1, new_id0) ;

        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;
        int friend_id0;
        int friend_id1;
        DATA_TYPE tx0, ty0, tz0;
        DATA_TYPE tx1, ty1, tz1;

        // process (0, 0, 0)
        friend_id0 = (lane_id+ 0+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a0 *((lane_id < 52 )? tx0: ty0);
        friend_id1 = (lane_id+36+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a0 *((lane_id < 24 )? tx1: ty1);
        // process (1, 0, 0)
        friend_id0 = (lane_id+ 1+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a1 *((lane_id < 51 )? tx0: ty0);
        friend_id1 = (lane_id+37+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a1 *((lane_id < 23 )? tx1: ty1);
        // process (2, 0, 0)
        friend_id0 = (lane_id+ 2+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a2 *((lane_id < 50 )? tx0: ty0);
        friend_id1 = (lane_id+38+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a2 *((lane_id < 22 )? tx1: ty1);
        // process (0, 1, 0)
        friend_id0 = (lane_id+10+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a3 *((lane_id < 44 )? tx0: ty0);
        friend_id1 = (lane_id+46+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a3 *((lane_id < 16 )? tx1: ty1);
        // process (1, 1, 0)
        friend_id0 = (lane_id+11+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a4 *((lane_id < 43 )? tx0: ty0);
        friend_id1 = (lane_id+47+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a4 *((lane_id < 15 )? tx1: ty1);
        // process (2, 1, 0)
        friend_id0 = (lane_id+12+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a5 *((lane_id < 42 )? tx0: ty0);
        friend_id1 = (lane_id+48+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a5 *((lane_id < 14 )? tx1: ty1);
        // process (0, 2, 0)
        friend_id0 = (lane_id+20+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a6 *((lane_id < 36 )? tx0: ty0);
        friend_id1 = (lane_id+56+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        tz1 = __shfl(reg3, friend_id1);
        sum1 += a6 *((lane_id < 8 )? tx1: ((lane_id < 58)? ty1: tz1));
        // process (1, 2, 0)
        friend_id0 = (lane_id+21+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a7 *((lane_id < 35 )? tx0: ty0);
        friend_id1 = (lane_id+57+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        tz1 = __shfl(reg3, friend_id1);
        sum1 += a7 *((lane_id < 7 )? tx1: ((lane_id < 57)? ty1: tz1));
        // process (2, 2, 0)
        friend_id0 = (lane_id+22+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a8 *((lane_id < 34 )? tx0: ty0);
        friend_id1 = (lane_id+58+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        tz1 = __shfl(reg3, friend_id1);
        sum1 += a8 *((lane_id < 6 )? tx1: ((lane_id < 56)? ty1: tz1));
        // process (0, 0, 1)
        friend_id0 = (lane_id+36+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a9 *((lane_id < 24 )? tx0: ty0);
        friend_id1 = (lane_id+ 8+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a9 *((lane_id < 46 )? tx1: ty1);
        // process (1, 0, 1)
        friend_id0 = (lane_id+37+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a10*((lane_id < 23 )? tx0: ty0);
        friend_id1 = (lane_id+ 9+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a10*((lane_id < 45 )? tx1: ty1);
        // process (2, 0, 1)
        friend_id0 = (lane_id+38+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a11*((lane_id < 22 )? tx0: ty0);
        friend_id1 = (lane_id+10+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a11*((lane_id < 44 )? tx1: ty1);
        // process (0, 1, 1)
        friend_id0 = (lane_id+46+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a12*((lane_id < 16 )? tx0: ty0);
        friend_id1 = (lane_id+18+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a12*((lane_id < 38 )? tx1: ty1);
        // process (1, 1, 1)
        friend_id0 = (lane_id+47+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a13*((lane_id < 15 )? tx0: ty0);
        friend_id1 = (lane_id+19+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a13*((lane_id < 37 )? tx1: ty1);
        // process (2, 1, 1)
        friend_id0 = (lane_id+48+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a14*((lane_id < 14 )? tx0: ty0);
        friend_id1 = (lane_id+20+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a14*((lane_id < 36 )? tx1: ty1);
        // process (0, 2, 1)
        friend_id0 = (lane_id+56+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tz0 = __shfl(reg3, friend_id0);
        sum0 += a15*((lane_id < 8 )? tx0: ((lane_id < 58)? ty0: tz0));
        friend_id1 = (lane_id+28+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a15*((lane_id < 30 )? tx1: ty1);
        // process (1, 2, 1)
        friend_id0 = (lane_id+57+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tz0 = __shfl(reg3, friend_id0);
        sum0 += a16*((lane_id < 7 )? tx0: ((lane_id < 57)? ty0: tz0));
        friend_id1 = (lane_id+29+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a16*((lane_id < 29 )? tx1: ty1);
        // process (2, 2, 1)
        friend_id0 = (lane_id+58+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tz0 = __shfl(reg3, friend_id0);
        sum0 += a17*((lane_id < 6 )? tx0: ((lane_id < 56)? ty0: tz0));
        friend_id1 = (lane_id+30+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a17*((lane_id < 28 )? tx1: ty1);
        // process (0, 0, 2)
        friend_id0 = (lane_id+ 8+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a18*((lane_id < 46 )? tx0: ty0);
        friend_id1 = (lane_id+44+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg4, friend_id1);
        ty1 = __shfl(reg5, friend_id1);
        sum1 += a18*((lane_id < 16 )? tx1: ty1);
        // process (1, 0, 2)
        friend_id0 = (lane_id+ 9+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a19*((lane_id < 45 )? tx0: ty0);
        friend_id1 = (lane_id+45+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg4, friend_id1);
        ty1 = __shfl(reg5, friend_id1);
        sum1 += a19*((lane_id < 16 )? tx1: ty1);
        // process (2, 0, 2)
        friend_id0 = (lane_id+10+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a20*((lane_id < 44 )? tx0: ty0);
        friend_id1 = (lane_id+46+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg4, friend_id1);
        ty1 = __shfl(reg5, friend_id1);
        sum1 += a20*((lane_id < 16 )? tx1: ty1);
        // process (0, 1, 2)
        friend_id0 = (lane_id+18+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a21*((lane_id < 38 )? tx0: ty0);
        friend_id1 = (lane_id+54+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg4, friend_id1);
        ty1 = __shfl(reg5, friend_id1);
        tz1 = __shfl(reg6, friend_id1);
        sum1 += a21*((lane_id < 8 )? tx1: ((lane_id < 60)? ty1: tz1));
        // process (1, 1, 2)
        friend_id0 = (lane_id+19+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a22*((lane_id < 37 )? tx0: ty0);
        friend_id1 = (lane_id+55+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg4, friend_id1);
        ty1 = __shfl(reg5, friend_id1);
        tz1 = __shfl(reg6, friend_id1);
        sum1 += a22*((lane_id < 8 )? tx1: ((lane_id < 59)? ty1: tz1));
        // process (2, 1, 2)
        friend_id0 = (lane_id+20+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a23*((lane_id < 36 )? tx0: ty0);
        friend_id1 = (lane_id+56+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg4, friend_id1);
        ty1 = __shfl(reg5, friend_id1);
        tz1 = __shfl(reg6, friend_id1);
        sum1 += a23*((lane_id < 8 )? tx1: ((lane_id < 58)? ty1: tz1));
        // process (0, 2, 2)
        friend_id0 = (lane_id+28+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a24*((lane_id < 30 )? tx0: ty0);
        friend_id1 = (lane_id+ 0+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg5, friend_id1);
        ty1 = __shfl(reg6, friend_id1);
        sum1 += a24*((lane_id < 52 )? tx1: ty1);
        // process (1, 2, 2)
        friend_id0 = (lane_id+29+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a25*((lane_id < 29 )? tx0: ty0);
        friend_id1 = (lane_id+ 1+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg5, friend_id1);
        ty1 = __shfl(reg6, friend_id1);
        sum1 += a25*((lane_id < 51 )? tx1: ty1);
        // process (2, 2, 2)
        friend_id0 = (lane_id+30+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a26*((lane_id < 28 )? tx0: ty0);
        friend_id1 = (lane_id+ 2+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg5, friend_id1);
        ty1 = __shfl(reg6, friend_id1);
        sum1 += a26*((lane_id < 50 )? tx1: ty1);

        OUT_3D(k  ,j,i) = sum0;
        OUT_3D(k+1,j,i) = sum1;

    });
    fut.wait();
}

void Stencil_Hcc_Reg4_3Blk2Wf(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , DATA_TYPE a7 , DATA_TYPE a8 , DATA_TYPE a9 , 
        DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, DATA_TYPE a13, DATA_TYPE a14, 
        DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, DATA_TYPE a18, DATA_TYPE a19, 
        DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, DATA_TYPE a23, DATA_TYPE a24, 
        DATA_TYPE a25, DATA_TYPE a26, 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(z/4, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 4, 8, 8);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        int i = tidx.global[2] + halo;
        int j = tidx.global[1] + halo;
        int k = (((tidx.global[0])>>0)<<2) + halo;
        const int lane_id = __lane_id();
        int lane_id_it = lane_id;

        int warp_id0 = (tidx.global[2])>>3;
        int warp_id1 = (tidx.global[1])>>3;
        int warp_id2 = (((tidx.global[0])>>0)<<2)>>0;
        DATA_TYPE reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9;
        int new_id0 ;
        int new_id1 ;
        int new_id2 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg0 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg1 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg2 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg3 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg4 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg5 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg6 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg7 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        reg8 = IN_3D(new_id2, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10%10 ;
        new_id2 = (warp_id2<<0) + lane_id_it/100 ;
        new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
        new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
        new_id2 = (new_id2 < z+2)? new_id2 : z+1 ;
        reg9 = IN_3D(new_id2, new_id1, new_id0) ;

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

        // process (0, 0, 0)
        friend_id0 = (lane_id+ 0+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a0 *((lane_id < 52 )? tx0: ty0);
        friend_id1 = (lane_id+36+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a0 *((lane_id < 24 )? tx1: ty1);
        friend_id2 = (lane_id+ 8+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg3, friend_id2);
        ty2 = __shfl(reg4, friend_id2);
        sum2 += a0 *((lane_id < 46 )? tx2: ty2);
        friend_id3 = (lane_id+44+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg4, friend_id3);
        ty3 = __shfl(reg5, friend_id3);
        sum3 += a0 *((lane_id < 16 )? tx3: ty3);
        // process (1, 0, 0)
        friend_id0 = (lane_id+ 1+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a1 *((lane_id < 51 )? tx0: ty0);
        friend_id1 = (lane_id+37+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a1 *((lane_id < 23 )? tx1: ty1);
        friend_id2 = (lane_id+ 9+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg3, friend_id2);
        ty2 = __shfl(reg4, friend_id2);
        sum2 += a1 *((lane_id < 45 )? tx2: ty2);
        friend_id3 = (lane_id+45+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg4, friend_id3);
        ty3 = __shfl(reg5, friend_id3);
        sum3 += a1 *((lane_id < 16 )? tx3: ty3);
        // process (2, 0, 0)
        friend_id0 = (lane_id+ 2+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a2 *((lane_id < 50 )? tx0: ty0);
        friend_id1 = (lane_id+38+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a2 *((lane_id < 22 )? tx1: ty1);
        friend_id2 = (lane_id+10+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg3, friend_id2);
        ty2 = __shfl(reg4, friend_id2);
        sum2 += a2 *((lane_id < 44 )? tx2: ty2);
        friend_id3 = (lane_id+46+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg4, friend_id3);
        ty3 = __shfl(reg5, friend_id3);
        sum3 += a2 *((lane_id < 16 )? tx3: ty3);
        // process (0, 1, 0)
        friend_id0 = (lane_id+10+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a3 *((lane_id < 44 )? tx0: ty0);
        friend_id1 = (lane_id+46+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a3 *((lane_id < 16 )? tx1: ty1);
        friend_id2 = (lane_id+18+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg3, friend_id2);
        ty2 = __shfl(reg4, friend_id2);
        sum2 += a3 *((lane_id < 38 )? tx2: ty2);
        friend_id3 = (lane_id+54+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg4, friend_id3);
        ty3 = __shfl(reg5, friend_id3);
        tz3 = __shfl(reg6, friend_id3);
        sum3 += a3 *((lane_id < 8 )? tx3: ((lane_id < 60)? ty3: tz3));
        // process (1, 1, 0)
        friend_id0 = (lane_id+11+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a4 *((lane_id < 43 )? tx0: ty0);
        friend_id1 = (lane_id+47+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a4 *((lane_id < 15 )? tx1: ty1);
        friend_id2 = (lane_id+19+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg3, friend_id2);
        ty2 = __shfl(reg4, friend_id2);
        sum2 += a4 *((lane_id < 37 )? tx2: ty2);
        friend_id3 = (lane_id+55+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg4, friend_id3);
        ty3 = __shfl(reg5, friend_id3);
        tz3 = __shfl(reg6, friend_id3);
        sum3 += a4 *((lane_id < 8 )? tx3: ((lane_id < 59)? ty3: tz3));
        // process (2, 1, 0)
        friend_id0 = (lane_id+12+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a5 *((lane_id < 42 )? tx0: ty0);
        friend_id1 = (lane_id+48+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a5 *((lane_id < 14 )? tx1: ty1);
        friend_id2 = (lane_id+20+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg3, friend_id2);
        ty2 = __shfl(reg4, friend_id2);
        sum2 += a5 *((lane_id < 36 )? tx2: ty2);
        friend_id3 = (lane_id+56+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg4, friend_id3);
        ty3 = __shfl(reg5, friend_id3);
        tz3 = __shfl(reg6, friend_id3);
        sum3 += a5 *((lane_id < 8 )? tx3: ((lane_id < 58)? ty3: tz3));
        // process (0, 2, 0)
        friend_id0 = (lane_id+20+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a6 *((lane_id < 36 )? tx0: ty0);
        friend_id1 = (lane_id+56+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        tz1 = __shfl(reg3, friend_id1);
        sum1 += a6 *((lane_id < 8 )? tx1: ((lane_id < 58)? ty1: tz1));
        friend_id2 = (lane_id+28+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg3, friend_id2);
        ty2 = __shfl(reg4, friend_id2);
        sum2 += a6 *((lane_id < 30 )? tx2: ty2);
        friend_id3 = (lane_id+ 0+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg5, friend_id3);
        ty3 = __shfl(reg6, friend_id3);
        sum3 += a6 *((lane_id < 52 )? tx3: ty3);
        // process (1, 2, 0)
        friend_id0 = (lane_id+21+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a7 *((lane_id < 35 )? tx0: ty0);
        friend_id1 = (lane_id+57+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        tz1 = __shfl(reg3, friend_id1);
        sum1 += a7 *((lane_id < 7 )? tx1: ((lane_id < 57)? ty1: tz1));
        friend_id2 = (lane_id+29+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg3, friend_id2);
        ty2 = __shfl(reg4, friend_id2);
        sum2 += a7 *((lane_id < 29 )? tx2: ty2);
        friend_id3 = (lane_id+ 1+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg5, friend_id3);
        ty3 = __shfl(reg6, friend_id3);
        sum3 += a7 *((lane_id < 51 )? tx3: ty3);
        // process (2, 2, 0)
        friend_id0 = (lane_id+22+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a8 *((lane_id < 34 )? tx0: ty0);
        friend_id1 = (lane_id+58+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        tz1 = __shfl(reg3, friend_id1);
        sum1 += a8 *((lane_id < 6 )? tx1: ((lane_id < 56)? ty1: tz1));
        friend_id2 = (lane_id+30+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg3, friend_id2);
        ty2 = __shfl(reg4, friend_id2);
        sum2 += a8 *((lane_id < 28 )? tx2: ty2);
        friend_id3 = (lane_id+ 2+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg5, friend_id3);
        ty3 = __shfl(reg6, friend_id3);
        sum3 += a8 *((lane_id < 50 )? tx3: ty3);
        // process (0, 0, 1)
        friend_id0 = (lane_id+36+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a9 *((lane_id < 24 )? tx0: ty0);
        friend_id1 = (lane_id+ 8+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a9 *((lane_id < 46 )? tx1: ty1);
        friend_id2 = (lane_id+44+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg4, friend_id2);
        ty2 = __shfl(reg5, friend_id2);
        sum2 += a9 *((lane_id < 16 )? tx2: ty2);
        friend_id3 = (lane_id+16+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg6, friend_id3);
        ty3 = __shfl(reg7, friend_id3);
        sum3 += a9 *((lane_id < 40 )? tx3: ty3);
        // process (1, 0, 1)
        friend_id0 = (lane_id+37+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a10*((lane_id < 23 )? tx0: ty0);
        friend_id1 = (lane_id+ 9+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a10*((lane_id < 45 )? tx1: ty1);
        friend_id2 = (lane_id+45+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg4, friend_id2);
        ty2 = __shfl(reg5, friend_id2);
        sum2 += a10*((lane_id < 16 )? tx2: ty2);
        friend_id3 = (lane_id+17+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg6, friend_id3);
        ty3 = __shfl(reg7, friend_id3);
        sum3 += a10*((lane_id < 39 )? tx3: ty3);
        // process (2, 0, 1)
        friend_id0 = (lane_id+38+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a11*((lane_id < 22 )? tx0: ty0);
        friend_id1 = (lane_id+10+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a11*((lane_id < 44 )? tx1: ty1);
        friend_id2 = (lane_id+46+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg4, friend_id2);
        ty2 = __shfl(reg5, friend_id2);
        sum2 += a11*((lane_id < 16 )? tx2: ty2);
        friend_id3 = (lane_id+18+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg6, friend_id3);
        ty3 = __shfl(reg7, friend_id3);
        sum3 += a11*((lane_id < 38 )? tx3: ty3);
        // process (0, 1, 1)
        friend_id0 = (lane_id+46+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a12*((lane_id < 16 )? tx0: ty0);
        friend_id1 = (lane_id+18+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a12*((lane_id < 38 )? tx1: ty1);
        friend_id2 = (lane_id+54+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg4, friend_id2);
        ty2 = __shfl(reg5, friend_id2);
        tz2 = __shfl(reg6, friend_id2);
        sum2 += a12*((lane_id < 8 )? tx2: ((lane_id < 60)? ty2: tz2));
        friend_id3 = (lane_id+26+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg6, friend_id3);
        ty3 = __shfl(reg7, friend_id3);
        sum3 += a12*((lane_id < 32 )? tx3: ty3);
        // process (1, 1, 1)
        friend_id0 = (lane_id+47+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a13*((lane_id < 15 )? tx0: ty0);
        friend_id1 = (lane_id+19+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a13*((lane_id < 37 )? tx1: ty1);
        friend_id2 = (lane_id+55+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg4, friend_id2);
        ty2 = __shfl(reg5, friend_id2);
        tz2 = __shfl(reg6, friend_id2);
        sum2 += a13*((lane_id < 8 )? tx2: ((lane_id < 59)? ty2: tz2));
        friend_id3 = (lane_id+27+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg6, friend_id3);
        ty3 = __shfl(reg7, friend_id3);
        sum3 += a13*((lane_id < 31 )? tx3: ty3);
        // process (2, 1, 1)
        friend_id0 = (lane_id+48+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a14*((lane_id < 14 )? tx0: ty0);
        friend_id1 = (lane_id+20+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a14*((lane_id < 36 )? tx1: ty1);
        friend_id2 = (lane_id+56+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg4, friend_id2);
        ty2 = __shfl(reg5, friend_id2);
        tz2 = __shfl(reg6, friend_id2);
        sum2 += a14*((lane_id < 8 )? tx2: ((lane_id < 58)? ty2: tz2));
        friend_id3 = (lane_id+28+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg6, friend_id3);
        ty3 = __shfl(reg7, friend_id3);
        sum3 += a14*((lane_id < 30 )? tx3: ty3);
        // process (0, 2, 1)
        friend_id0 = (lane_id+56+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tz0 = __shfl(reg3, friend_id0);
        sum0 += a15*((lane_id < 8 )? tx0: ((lane_id < 58)? ty0: tz0));
        friend_id1 = (lane_id+28+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a15*((lane_id < 30 )? tx1: ty1);
        friend_id2 = (lane_id+ 0+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg5, friend_id2);
        ty2 = __shfl(reg6, friend_id2);
        sum2 += a15*((lane_id < 52 )? tx2: ty2);
        friend_id3 = (lane_id+36+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg6, friend_id3);
        ty3 = __shfl(reg7, friend_id3);
        sum3 += a15*((lane_id < 24 )? tx3: ty3);
        // process (1, 2, 1)
        friend_id0 = (lane_id+57+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tz0 = __shfl(reg3, friend_id0);
        sum0 += a16*((lane_id < 7 )? tx0: ((lane_id < 57)? ty0: tz0));
        friend_id1 = (lane_id+29+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a16*((lane_id < 29 )? tx1: ty1);
        friend_id2 = (lane_id+ 1+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg5, friend_id2);
        ty2 = __shfl(reg6, friend_id2);
        sum2 += a16*((lane_id < 51 )? tx2: ty2);
        friend_id3 = (lane_id+37+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg6, friend_id3);
        ty3 = __shfl(reg7, friend_id3);
        sum3 += a16*((lane_id < 23 )? tx3: ty3);
        // process (2, 2, 1)
        friend_id0 = (lane_id+58+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tz0 = __shfl(reg3, friend_id0);
        sum0 += a17*((lane_id < 6 )? tx0: ((lane_id < 56)? ty0: tz0));
        friend_id1 = (lane_id+30+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a17*((lane_id < 28 )? tx1: ty1);
        friend_id2 = (lane_id+ 2+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg5, friend_id2);
        ty2 = __shfl(reg6, friend_id2);
        sum2 += a17*((lane_id < 50 )? tx2: ty2);
        friend_id3 = (lane_id+38+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg6, friend_id3);
        ty3 = __shfl(reg7, friend_id3);
        sum3 += a17*((lane_id < 22 )? tx3: ty3);
        // process (0, 0, 2)
        friend_id0 = (lane_id+ 8+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a18*((lane_id < 46 )? tx0: ty0);
        friend_id1 = (lane_id+44+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg4, friend_id1);
        ty1 = __shfl(reg5, friend_id1);
        sum1 += a18*((lane_id < 16 )? tx1: ty1);
        friend_id2 = (lane_id+16+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg6, friend_id2);
        ty2 = __shfl(reg7, friend_id2);
        sum2 += a18*((lane_id < 40 )? tx2: ty2);
        friend_id3 = (lane_id+52+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg7, friend_id3);
        ty3 = __shfl(reg8, friend_id3);
        tz3 = __shfl(reg9, friend_id3);
        sum3 += a18*((lane_id < 10 )? tx3: ((lane_id < 62)? ty3: tz3));
        // process (1, 0, 2)
        friend_id0 = (lane_id+ 9+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a19*((lane_id < 45 )? tx0: ty0);
        friend_id1 = (lane_id+45+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg4, friend_id1);
        ty1 = __shfl(reg5, friend_id1);
        sum1 += a19*((lane_id < 16 )? tx1: ty1);
        friend_id2 = (lane_id+17+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg6, friend_id2);
        ty2 = __shfl(reg7, friend_id2);
        sum2 += a19*((lane_id < 39 )? tx2: ty2);
        friend_id3 = (lane_id+53+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg7, friend_id3);
        ty3 = __shfl(reg8, friend_id3);
        tz3 = __shfl(reg9, friend_id3);
        sum3 += a19*((lane_id < 9 )? tx3: ((lane_id < 61)? ty3: tz3));
        // process (2, 0, 2)
        friend_id0 = (lane_id+10+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a20*((lane_id < 44 )? tx0: ty0);
        friend_id1 = (lane_id+46+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg4, friend_id1);
        ty1 = __shfl(reg5, friend_id1);
        sum1 += a20*((lane_id < 16 )? tx1: ty1);
        friend_id2 = (lane_id+18+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg6, friend_id2);
        ty2 = __shfl(reg7, friend_id2);
        sum2 += a20*((lane_id < 38 )? tx2: ty2);
        friend_id3 = (lane_id+54+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg7, friend_id3);
        ty3 = __shfl(reg8, friend_id3);
        tz3 = __shfl(reg9, friend_id3);
        sum3 += a20*((lane_id < 8 )? tx3: ((lane_id < 60)? ty3: tz3));
        // process (0, 1, 2)
        friend_id0 = (lane_id+18+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a21*((lane_id < 38 )? tx0: ty0);
        friend_id1 = (lane_id+54+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg4, friend_id1);
        ty1 = __shfl(reg5, friend_id1);
        tz1 = __shfl(reg6, friend_id1);
        sum1 += a21*((lane_id < 8 )? tx1: ((lane_id < 60)? ty1: tz1));
        friend_id2 = (lane_id+26+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg6, friend_id2);
        ty2 = __shfl(reg7, friend_id2);
        sum2 += a21*((lane_id < 32 )? tx2: ty2);
        friend_id3 = (lane_id+62+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg7, friend_id3);
        ty3 = __shfl(reg8, friend_id3);
        tz3 = __shfl(reg9, friend_id3);
        sum3 += a21*((lane_id < 2 )? tx3: ((lane_id < 54)? ty3: tz3));
        // process (1, 1, 2)
        friend_id0 = (lane_id+19+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a22*((lane_id < 37 )? tx0: ty0);
        friend_id1 = (lane_id+55+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg4, friend_id1);
        ty1 = __shfl(reg5, friend_id1);
        tz1 = __shfl(reg6, friend_id1);
        sum1 += a22*((lane_id < 8 )? tx1: ((lane_id < 59)? ty1: tz1));
        friend_id2 = (lane_id+27+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg6, friend_id2);
        ty2 = __shfl(reg7, friend_id2);
        sum2 += a22*((lane_id < 31 )? tx2: ty2);
        friend_id3 = (lane_id+63+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg7, friend_id3);
        ty3 = __shfl(reg8, friend_id3);
        tz3 = __shfl(reg9, friend_id3);
        sum3 += a22*((lane_id < 1 )? tx3: ((lane_id < 53)? ty3: tz3));
        // process (2, 1, 2)
        friend_id0 = (lane_id+20+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a23*((lane_id < 36 )? tx0: ty0);
        friend_id1 = (lane_id+56+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg4, friend_id1);
        ty1 = __shfl(reg5, friend_id1);
        tz1 = __shfl(reg6, friend_id1);
        sum1 += a23*((lane_id < 8 )? tx1: ((lane_id < 58)? ty1: tz1));
        friend_id2 = (lane_id+28+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg6, friend_id2);
        ty2 = __shfl(reg7, friend_id2);
        sum2 += a23*((lane_id < 30 )? tx2: ty2);
        friend_id3 = (lane_id+ 0+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg8, friend_id3);
        ty3 = __shfl(reg9, friend_id3);
        sum3 += a23*((lane_id < 52 )? tx3: ty3);
        // process (0, 2, 2)
        friend_id0 = (lane_id+28+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a24*((lane_id < 30 )? tx0: ty0);
        friend_id1 = (lane_id+ 0+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg5, friend_id1);
        ty1 = __shfl(reg6, friend_id1);
        sum1 += a24*((lane_id < 52 )? tx1: ty1);
        friend_id2 = (lane_id+36+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg6, friend_id2);
        ty2 = __shfl(reg7, friend_id2);
        sum2 += a24*((lane_id < 24 )? tx2: ty2);
        friend_id3 = (lane_id+ 8+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg8, friend_id3);
        ty3 = __shfl(reg9, friend_id3);
        sum3 += a24*((lane_id < 46 )? tx3: ty3);
        // process (1, 2, 2)
        friend_id0 = (lane_id+29+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a25*((lane_id < 29 )? tx0: ty0);
        friend_id1 = (lane_id+ 1+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg5, friend_id1);
        ty1 = __shfl(reg6, friend_id1);
        sum1 += a25*((lane_id < 51 )? tx1: ty1);
        friend_id2 = (lane_id+37+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg6, friend_id2);
        ty2 = __shfl(reg7, friend_id2);
        sum2 += a25*((lane_id < 23 )? tx2: ty2);
        friend_id3 = (lane_id+ 9+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg8, friend_id3);
        ty3 = __shfl(reg9, friend_id3);
        sum3 += a25*((lane_id < 45 )? tx3: ty3);
        // process (2, 2, 2)
        friend_id0 = (lane_id+30+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a26*((lane_id < 28 )? tx0: ty0);
        friend_id1 = (lane_id+ 2+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg5, friend_id1);
        ty1 = __shfl(reg6, friend_id1);
        sum1 += a26*((lane_id < 50 )? tx1: ty1);
        friend_id2 = (lane_id+38+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg6, friend_id2);
        ty2 = __shfl(reg7, friend_id2);
        sum2 += a26*((lane_id < 22 )? tx2: ty2);
        friend_id3 = (lane_id+10+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg8, friend_id3);
        ty3 = __shfl(reg9, friend_id3);
        sum3 += a26*((lane_id < 44 )? tx3: ty3);

        OUT_3D(k  ,j,i) = sum0;
        OUT_3D(k+1,j,i) = sum1;
        OUT_3D(k+2,j,i) = sum2;
        OUT_3D(k+3,j,i) = sum3;

    });
    fut.wait();
}

void Stencil_Hcc_Reg1_25Blk1Wf(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , DATA_TYPE a7 , DATA_TYPE a8 , DATA_TYPE a9 , 
        DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, DATA_TYPE a13, DATA_TYPE a14, 
        DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, DATA_TYPE a18, DATA_TYPE a19, 
        DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, DATA_TYPE a23, DATA_TYPE a24, 
        DATA_TYPE a25, DATA_TYPE a26, 
       int z, int m, int n, int halo)
{
    extent<3> comp_domain(4, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 1, 4, 64);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        int i = tidx.global[2] + halo;
        int j = (((tidx.global[1])>>0)<<0) + halo;

        const int block_z = z / 4;
        int k = block_z * tidx.tile[0] + halo;
        const int k_end = k + block_z;

        int warp_id0 = (tidx.global[2])>>6;             // because the warp dimensions are 
        int warp_id1 = ((((tidx.global[1])>>0)<<0))>>0; // 1x1x64, warp_ids are division of these numbers

        const int lane_id = __lane_id();
        int lane_id_it = lane_id;

        DATA_TYPE t3_reg0, t3_reg1, t3_reg2, t3_reg3;
        DATA_TYPE t2_reg0, t2_reg1, t2_reg2, t2_reg3;
        DATA_TYPE t1_reg0, t1_reg1, t1_reg2, t1_reg3;
        int new_id0 ;
        int new_id1 ;

        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        t3_reg0 = IN_3D(k  , new_id1, new_id0) ;
        t2_reg0 = IN_3D(k-1, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        t3_reg1 = IN_3D(k  , new_id1, new_id0) ;
        t2_reg1 = IN_3D(k-1, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        t3_reg2 = IN_3D(k  , new_id1, new_id0) ;
        t2_reg2 = IN_3D(k-1, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
        new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
        t3_reg3 = IN_3D(k  , new_id1, new_id0) ;
        t2_reg3 = IN_3D(k-1, new_id1, new_id0) ;

        DATA_TYPE sum0 = 0.0;
        int friend_id0;
        DATA_TYPE tx0, ty0, tz0;

#pragma unroll 
        for(; k < k_end; ++k)
        {
            sum0 = 0.0;
            t1_reg0 = t2_reg0;
            t1_reg1 = t2_reg1;
            t1_reg2 = t2_reg2;
            t1_reg3 = t2_reg3;

            t2_reg0 = t3_reg0;
            t2_reg1 = t3_reg1;
            t2_reg2 = t3_reg2;
            t2_reg3 = t3_reg3;

            lane_id_it = lane_id;

            new_id0 = (warp_id0<<6) + lane_id_it%66 ;
            new_id1 = (warp_id1<<0) + lane_id_it/66 ;
            t3_reg0 = IN_3D(k+1, new_id1, new_id0) ;
            lane_id_it += 64 ;
            new_id0 = (warp_id0<<6) + lane_id_it%66 ;
            new_id1 = (warp_id1<<0) + lane_id_it/66 ;
            t3_reg1 = IN_3D(k+1, new_id1, new_id0) ;
            lane_id_it += 64 ;
            new_id0 = (warp_id0<<6) + lane_id_it%66 ;
            new_id1 = (warp_id1<<0) + lane_id_it/66 ;
            t3_reg2 = IN_3D(k+1, new_id1, new_id0) ;
            lane_id_it += 64 ;
            new_id0 = (warp_id0<<6) + lane_id_it%66 ;
            new_id1 = (warp_id1<<0) + lane_id_it/66 ;
            new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
            new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
            t3_reg3 = IN_3D(k+1, new_id1, new_id0) ;

            // process (0, 0, 0)
            friend_id0 = (lane_id+ 0)&63 ;
            tx0 = __shfl(t1_reg0, friend_id0);
            sum0 += a0 *(tx0);
            tx0 = __shfl(t2_reg0, friend_id0);
            sum0 += a9 *(tx0);
            tx0 = __shfl(t3_reg0, friend_id0);
            sum0 += a18*(tx0);
            // process (1, 0, 0)
            friend_id0 = (lane_id+ 1)&63 ;
            tx0 = __shfl(t1_reg0, friend_id0);
            ty0 = __shfl(t1_reg1, friend_id0);
            sum0 += a1 *((lane_id < 63 )? tx0: ty0);
            tx0 = __shfl(t2_reg0, friend_id0);
            ty0 = __shfl(t2_reg1, friend_id0);
            sum0 += a10*((lane_id < 63 )? tx0: ty0);
            tx0 = __shfl(t3_reg0, friend_id0);
            ty0 = __shfl(t3_reg1, friend_id0);
            sum0 += a19*((lane_id < 63 )? tx0: ty0);
            // process (2, 0, 0)
            friend_id0 = (lane_id+ 2)&63 ;
            tx0 = __shfl(t1_reg0, friend_id0);
            ty0 = __shfl(t1_reg1, friend_id0);
            sum0 += a2 *((lane_id < 62 )? tx0: ty0);
            tx0 = __shfl(t2_reg0, friend_id0);
            ty0 = __shfl(t2_reg1, friend_id0);
            sum0 += a11*((lane_id < 62 )? tx0: ty0);
            tx0 = __shfl(t3_reg0, friend_id0);
            ty0 = __shfl(t3_reg1, friend_id0);
            sum0 += a20*((lane_id < 62 )? tx0: ty0);
            // process (0, 1, 0)
            friend_id0 = (lane_id+ 2)&63 ;
            tx0 = __shfl(t1_reg1, friend_id0);
            ty0 = __shfl(t1_reg2, friend_id0);
            sum0 += a3 *((lane_id < 62 )? tx0: ty0);
            tx0 = __shfl(t2_reg1, friend_id0);
            ty0 = __shfl(t2_reg2, friend_id0);
            sum0 += a12*((lane_id < 62 )? tx0: ty0);
            tx0 = __shfl(t3_reg1, friend_id0);
            ty0 = __shfl(t3_reg2, friend_id0);
            sum0 += a21*((lane_id < 62 )? tx0: ty0);
            // process (1, 1, 0)
            friend_id0 = (lane_id+ 3)&63 ;
            tx0 = __shfl(t1_reg1, friend_id0);
            ty0 = __shfl(t1_reg2, friend_id0);
            sum0 += a4 *((lane_id < 61 )? tx0: ty0);
            tx0 = __shfl(t2_reg1, friend_id0);
            ty0 = __shfl(t2_reg2, friend_id0);
            sum0 += a13*((lane_id < 61 )? tx0: ty0);
            tx0 = __shfl(t3_reg1, friend_id0);
            ty0 = __shfl(t3_reg2, friend_id0);
            sum0 += a22*((lane_id < 61 )? tx0: ty0);
            // process (2, 1, 0)
            friend_id0 = (lane_id+ 4)&63 ;
            tx0 = __shfl(t1_reg1, friend_id0);
            ty0 = __shfl(t1_reg2, friend_id0);
            sum0 += a5 *((lane_id < 60 )? tx0: ty0);
            tx0 = __shfl(t2_reg1, friend_id0);
            ty0 = __shfl(t2_reg2, friend_id0);
            sum0 += a14*((lane_id < 60 )? tx0: ty0);
            tx0 = __shfl(t3_reg1, friend_id0);
            ty0 = __shfl(t3_reg2, friend_id0);
            sum0 += a23*((lane_id < 60 )? tx0: ty0);
            // process (0, 2, 0)
            friend_id0 = (lane_id+ 4)&63 ;
            tx0 = __shfl(t1_reg2, friend_id0);
            ty0 = __shfl(t1_reg3, friend_id0);
            sum0 += a6 *((lane_id < 60 )? tx0: ty0);
            tx0 = __shfl(t2_reg2, friend_id0);
            ty0 = __shfl(t2_reg3, friend_id0);
            sum0 += a15*((lane_id < 60 )? tx0: ty0);
            tx0 = __shfl(t3_reg2, friend_id0);
            ty0 = __shfl(t3_reg3, friend_id0);
            sum0 += a24*((lane_id < 60 )? tx0: ty0);
            // process (1, 2, 0)
            friend_id0 = (lane_id+ 5)&63 ;
            tx0 = __shfl(t1_reg2, friend_id0);
            ty0 = __shfl(t1_reg3, friend_id0);
            sum0 += a7 *((lane_id < 59 )? tx0: ty0);
            tx0 = __shfl(t2_reg2, friend_id0);
            ty0 = __shfl(t2_reg3, friend_id0);
            sum0 += a16*((lane_id < 59 )? tx0: ty0);
            tx0 = __shfl(t3_reg2, friend_id0);
            ty0 = __shfl(t3_reg3, friend_id0);
            sum0 += a25*((lane_id < 59 )? tx0: ty0);
            // process (2, 2, 0)
            friend_id0 = (lane_id+ 6)&63 ;
            tx0 = __shfl(t1_reg2, friend_id0);
            ty0 = __shfl(t1_reg3, friend_id0);
            sum0 += a8 *((lane_id < 58 )? tx0: ty0);
            tx0 = __shfl(t2_reg2, friend_id0);
            ty0 = __shfl(t2_reg3, friend_id0);
            sum0 += a17*((lane_id < 58 )? tx0: ty0);
            tx0 = __shfl(t3_reg2, friend_id0);
            ty0 = __shfl(t3_reg3, friend_id0);
            sum0 += a26*((lane_id < 58 )? tx0: ty0);
            
            OUT_3D(k,j  ,i) = sum0;
        
        }
    });
    fut.wait();
}

void Stencil_Hcc_Reg2_25Blk1Wf(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , DATA_TYPE a7 , DATA_TYPE a8 , DATA_TYPE a9 , 
        DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, DATA_TYPE a13, DATA_TYPE a14, 
        DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, DATA_TYPE a18, DATA_TYPE a19, 
        DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, DATA_TYPE a23, DATA_TYPE a24, 
        DATA_TYPE a25, DATA_TYPE a26, 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(4, m/2, n); 
    tiled_extent<3> comp_tile(comp_domain, 1, 4, 64);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        int i = tidx.global[2] + halo;
        int j = (((tidx.global[1])>>0)<<1) + halo;

        const int block_z = z / 4;
        int k = block_z * tidx.tile[0] + halo;
        const int k_end = k + block_z;

        int warp_id0 = (tidx.global[2])>>6;             // because the warp dimensions are 
        int warp_id1 = ((((tidx.global[1])>>0)<<1))>>0; // 1x1x64, warp_ids are division of these numbers

        const int lane_id = __lane_id();
        int lane_id_it = lane_id;
        DATA_TYPE t3_reg0, t3_reg1, t3_reg2, t3_reg3, t3_reg4;
        DATA_TYPE t2_reg0, t2_reg1, t2_reg2, t2_reg3, t2_reg4;
        DATA_TYPE t1_reg0, t1_reg1, t1_reg2, t1_reg3, t1_reg4;

        int new_id0 ;
        int new_id1 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        t3_reg0 = IN_3D(k  ,new_id1, new_id0) ;
        t2_reg0 = IN_3D(k-1,new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        t3_reg1 = IN_3D(k  ,new_id1, new_id0) ;
        t2_reg1 = IN_3D(k-1,new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        t3_reg2 = IN_3D(k  ,new_id1, new_id0) ;
        t2_reg2 = IN_3D(k-1,new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        t3_reg3 = IN_3D(k  ,new_id1, new_id0) ;
        t2_reg3 = IN_3D(k-1,new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
        new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
        t3_reg4 = IN_3D(k  ,new_id1, new_id0) ;
        t2_reg4 = IN_3D(k-1,new_id1, new_id0) ;

        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;
        int friend_id0;
        int friend_id1;
        DATA_TYPE tx0, ty0, tz0;
        DATA_TYPE tx1, ty1, tz1;

#pragma unroll 
        for(; k < k_end; ++k)
        {
            sum0 = 0.0;
            sum1 = 0.0;
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
            new_id0 = (warp_id0<<6) + lane_id_it%66 ;
            new_id1 = (warp_id1<<0) + lane_id_it/66 ;
            t3_reg0 = IN_3D(k+1,new_id1, new_id0) ;
            lane_id_it += 64 ;
            new_id0 = (warp_id0<<6) + lane_id_it%66 ;
            new_id1 = (warp_id1<<0) + lane_id_it/66 ;
            t3_reg1 = IN_3D(k+1,new_id1, new_id0) ;
            lane_id_it += 64 ;
            new_id0 = (warp_id0<<6) + lane_id_it%66 ;
            new_id1 = (warp_id1<<0) + lane_id_it/66 ;
            t3_reg2 = IN_3D(k+1,new_id1, new_id0) ;
            lane_id_it += 64 ;
            new_id0 = (warp_id0<<6) + lane_id_it%66 ;
            new_id1 = (warp_id1<<0) + lane_id_it/66 ;
            t3_reg3 = IN_3D(k+1,new_id1, new_id0) ;
            lane_id_it += 64 ;
            new_id0 = (warp_id0<<6) + lane_id_it%66 ;
            new_id1 = (warp_id1<<0) + lane_id_it/66 ;
            new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
            new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
            t3_reg4 = IN_3D(k+1,new_id1, new_id0) ;

            // process (0, 0, 0)
            friend_id0 = (lane_id+ 0)&63 ;
            tx0 = __shfl(t1_reg0, friend_id0);
            sum0 += a0 *(tx0);
            tx0 = __shfl(t2_reg0, friend_id0);
            sum0 += a9 *(tx0);
            tx0 = __shfl(t3_reg0, friend_id0);
            sum0 += a18*(tx0);
            friend_id1 = (lane_id+ 2)&63 ;
            tx1 = __shfl(t1_reg1, friend_id1);
            ty1 = __shfl(t1_reg2, friend_id1);
            sum1 += a0 *((lane_id < 62 )? tx1: ty1);
            tx1 = __shfl(t2_reg1, friend_id1);
            ty1 = __shfl(t2_reg2, friend_id1);
            sum1 += a9 *((lane_id < 62 )? tx1: ty1);
            tx1 = __shfl(t3_reg1, friend_id1);
            ty1 = __shfl(t3_reg2, friend_id1);
            sum1 += a18*((lane_id < 62 )? tx1: ty1);
            // process (1, 0, 0)
            friend_id0 = (lane_id+ 1)&63 ;
            tx0 = __shfl(t1_reg0, friend_id0);
            ty0 = __shfl(t1_reg1, friend_id0);
            sum0 += a1 *((lane_id < 63 )? tx0: ty0);
            tx0 = __shfl(t2_reg0, friend_id0);
            ty0 = __shfl(t2_reg1, friend_id0);
            sum0 += a10*((lane_id < 63 )? tx0: ty0);
            tx0 = __shfl(t3_reg0, friend_id0);
            ty0 = __shfl(t3_reg1, friend_id0);
            sum0 += a19*((lane_id < 63 )? tx0: ty0);
            friend_id1 = (lane_id+ 3)&63 ;
            tx1 = __shfl(t1_reg1, friend_id1);
            ty1 = __shfl(t1_reg2, friend_id1);
            sum1 += a1 *((lane_id < 61 )? tx1: ty1);
            tx1 = __shfl(t2_reg1, friend_id1);
            ty1 = __shfl(t2_reg2, friend_id1);
            sum1 += a10*((lane_id < 61 )? tx1: ty1);
            tx1 = __shfl(t3_reg1, friend_id1);
            ty1 = __shfl(t3_reg2, friend_id1);
            sum1 += a19*((lane_id < 61 )? tx1: ty1);
            // process (2, 0, 0)
            friend_id0 = (lane_id+ 2)&63 ;
            tx0 = __shfl(t1_reg0, friend_id0);
            ty0 = __shfl(t1_reg1, friend_id0);
            sum0 += a2 *((lane_id < 62 )? tx0: ty0);
            tx0 = __shfl(t2_reg0, friend_id0);
            ty0 = __shfl(t2_reg1, friend_id0);
            sum0 += a11*((lane_id < 62 )? tx0: ty0);
            tx0 = __shfl(t3_reg0, friend_id0);
            ty0 = __shfl(t3_reg1, friend_id0);
            sum0 += a20*((lane_id < 62 )? tx0: ty0);
            friend_id1 = (lane_id+ 4)&63 ;
            tx1 = __shfl(t1_reg1, friend_id1);
            ty1 = __shfl(t1_reg2, friend_id1);
            sum1 += a2 *((lane_id < 60 )? tx1: ty1);
            tx1 = __shfl(t2_reg1, friend_id1);
            ty1 = __shfl(t2_reg2, friend_id1);
            sum1 += a11*((lane_id < 60 )? tx1: ty1);
            tx1 = __shfl(t3_reg1, friend_id1);
            ty1 = __shfl(t3_reg2, friend_id1);
            sum1 += a20*((lane_id < 60 )? tx1: ty1);
            // process (0, 1, 0)
            friend_id0 = (lane_id+ 2)&63 ;
            tx0 = __shfl(t1_reg1, friend_id0);
            ty0 = __shfl(t1_reg2, friend_id0);
            sum0 += a3 *((lane_id < 62 )? tx0: ty0);
            tx0 = __shfl(t2_reg1, friend_id0);
            ty0 = __shfl(t2_reg2, friend_id0);
            sum0 += a12*((lane_id < 62 )? tx0: ty0);
            tx0 = __shfl(t3_reg1, friend_id0);
            ty0 = __shfl(t3_reg2, friend_id0);
            sum0 += a21*((lane_id < 62 )? tx0: ty0);
            friend_id1 = (lane_id+ 4)&63 ;
            tx1 = __shfl(t1_reg2, friend_id1);
            ty1 = __shfl(t1_reg3, friend_id1);
            sum1 += a3 *((lane_id < 60 )? tx1: ty1);
            tx1 = __shfl(t2_reg2, friend_id1);
            ty1 = __shfl(t2_reg3, friend_id1);
            sum1 += a12*((lane_id < 60 )? tx1: ty1);
            tx1 = __shfl(t3_reg2, friend_id1);
            ty1 = __shfl(t3_reg3, friend_id1);
            sum1 += a21*((lane_id < 60 )? tx1: ty1);
            // process (1, 1, 0)
            friend_id0 = (lane_id+ 3)&63 ;
            tx0 = __shfl(t1_reg1, friend_id0);
            ty0 = __shfl(t1_reg2, friend_id0);
            sum0 += a4 *((lane_id < 61 )? tx0: ty0);
            tx0 = __shfl(t2_reg1, friend_id0);
            ty0 = __shfl(t2_reg2, friend_id0);
            sum0 += a13*((lane_id < 61 )? tx0: ty0);
            tx0 = __shfl(t3_reg1, friend_id0);
            ty0 = __shfl(t3_reg2, friend_id0);
            sum0 += a22*((lane_id < 61 )? tx0: ty0);
            friend_id1 = (lane_id+ 5)&63 ;
            tx1 = __shfl(t1_reg2, friend_id1);
            ty1 = __shfl(t1_reg3, friend_id1);
            sum1 += a4 *((lane_id < 59 )? tx1: ty1);
            tx1 = __shfl(t2_reg2, friend_id1);
            ty1 = __shfl(t2_reg3, friend_id1);
            sum1 += a13*((lane_id < 59 )? tx1: ty1);
            tx1 = __shfl(t3_reg2, friend_id1);
            ty1 = __shfl(t3_reg3, friend_id1);
            sum1 += a22*((lane_id < 59 )? tx1: ty1);
            // process (2, 1, 0)
            friend_id0 = (lane_id+ 4)&63 ;
            tx0 = __shfl(t1_reg1, friend_id0);
            ty0 = __shfl(t1_reg2, friend_id0);
            sum0 += a5 *((lane_id < 60 )? tx0: ty0);
            tx0 = __shfl(t2_reg1, friend_id0);
            ty0 = __shfl(t2_reg2, friend_id0);
            sum0 += a14*((lane_id < 60 )? tx0: ty0);
            tx0 = __shfl(t3_reg1, friend_id0);
            ty0 = __shfl(t3_reg2, friend_id0);
            sum0 += a23*((lane_id < 60 )? tx0: ty0);
            friend_id1 = (lane_id+ 6)&63 ;
            tx1 = __shfl(t1_reg2, friend_id1);
            ty1 = __shfl(t1_reg3, friend_id1);
            sum1 += a5 *((lane_id < 58 )? tx1: ty1);
            tx1 = __shfl(t2_reg2, friend_id1);
            ty1 = __shfl(t2_reg3, friend_id1);
            sum1 += a14*((lane_id < 58 )? tx1: ty1);
            tx1 = __shfl(t3_reg2, friend_id1);
            ty1 = __shfl(t3_reg3, friend_id1);
            sum1 += a23*((lane_id < 58 )? tx1: ty1);
            // process (0, 2, 0)
            friend_id0 = (lane_id+ 4)&63 ;
            tx0 = __shfl(t1_reg2, friend_id0);
            ty0 = __shfl(t1_reg3, friend_id0);
            sum0 += a6 *((lane_id < 60 )? tx0: ty0);
            tx0 = __shfl(t2_reg2, friend_id0);
            ty0 = __shfl(t2_reg3, friend_id0);
            sum0 += a15*((lane_id < 60 )? tx0: ty0);
            tx0 = __shfl(t3_reg2, friend_id0);
            ty0 = __shfl(t3_reg3, friend_id0);
            sum0 += a24*((lane_id < 60 )? tx0: ty0);
            friend_id1 = (lane_id+ 6)&63 ;
            tx1 = __shfl(t1_reg3, friend_id1);
            ty1 = __shfl(t1_reg4, friend_id1);
            sum1 += a6 *((lane_id < 58 )? tx1: ty1);
            tx1 = __shfl(t2_reg3, friend_id1);
            ty1 = __shfl(t2_reg4, friend_id1);
            sum1 += a15*((lane_id < 58 )? tx1: ty1);
            tx1 = __shfl(t3_reg3, friend_id1);
            ty1 = __shfl(t3_reg4, friend_id1);
            sum1 += a24*((lane_id < 58 )? tx1: ty1);
            // process (1, 2, 0)
            friend_id0 = (lane_id+ 5)&63 ;
            tx0 = __shfl(t1_reg2, friend_id0);
            ty0 = __shfl(t1_reg3, friend_id0);
            sum0 += a7 *((lane_id < 59 )? tx0: ty0);
            tx0 = __shfl(t2_reg2, friend_id0);
            ty0 = __shfl(t2_reg3, friend_id0);
            sum0 += a16*((lane_id < 59 )? tx0: ty0);
            tx0 = __shfl(t3_reg2, friend_id0);
            ty0 = __shfl(t3_reg3, friend_id0);
            sum0 += a25*((lane_id < 59 )? tx0: ty0);
            friend_id1 = (lane_id+ 7)&63 ;
            tx1 = __shfl(t1_reg3, friend_id1);
            ty1 = __shfl(t1_reg4, friend_id1);
            sum1 += a7 *((lane_id < 57 )? tx1: ty1);
            tx1 = __shfl(t2_reg3, friend_id1);
            ty1 = __shfl(t2_reg4, friend_id1);
            sum1 += a16*((lane_id < 57 )? tx1: ty1);
            tx1 = __shfl(t3_reg3, friend_id1);
            ty1 = __shfl(t3_reg4, friend_id1);
            sum1 += a25*((lane_id < 57 )? tx1: ty1);
            // process (2, 2, 0)
            friend_id0 = (lane_id+ 6)&63 ;
            tx0 = __shfl(t1_reg2, friend_id0);
            ty0 = __shfl(t1_reg3, friend_id0);
            sum0 += a8 *((lane_id < 58 )? tx0: ty0);
            tx0 = __shfl(t2_reg2, friend_id0);
            ty0 = __shfl(t2_reg3, friend_id0);
            sum0 += a17*((lane_id < 58 )? tx0: ty0);
            tx0 = __shfl(t3_reg2, friend_id0);
            ty0 = __shfl(t3_reg3, friend_id0);
            sum0 += a26*((lane_id < 58 )? tx0: ty0);
            friend_id1 = (lane_id+ 8)&63 ;
            tx1 = __shfl(t1_reg3, friend_id1);
            ty1 = __shfl(t1_reg4, friend_id1);
            sum1 += a8 *((lane_id < 56 )? tx1: ty1);
            tx1 = __shfl(t2_reg3, friend_id1);
            ty1 = __shfl(t2_reg4, friend_id1);
            sum1 += a17*((lane_id < 56 )? tx1: ty1);
            tx1 = __shfl(t3_reg3, friend_id1);
            ty1 = __shfl(t3_reg4, friend_id1);
            sum1 += a26*((lane_id < 56 )? tx1: ty1);

            OUT_3D(k,j  ,i) = sum0;
            OUT_3D(k,j+1,i) = sum1;
        
        }
    });
    fut.wait();
}

void Stencil_Hcc_Reg4_25Blk1Wf(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , DATA_TYPE a7 , DATA_TYPE a8 , DATA_TYPE a9 , 
        DATA_TYPE a10, DATA_TYPE a11, DATA_TYPE a12, DATA_TYPE a13, DATA_TYPE a14, 
        DATA_TYPE a15, DATA_TYPE a16, DATA_TYPE a17, DATA_TYPE a18, DATA_TYPE a19, 
        DATA_TYPE a20, DATA_TYPE a21, DATA_TYPE a22, DATA_TYPE a23, DATA_TYPE a24, 
        DATA_TYPE a25, DATA_TYPE a26, 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(4, m/4, n); 
    tiled_extent<3> comp_tile(comp_domain, 1, 4, 64);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        int i = tidx.global[2] + halo;
        int j = (((tidx.global[1])>>0)<<2) + halo;

        const int block_z = z / 4;
        int k = block_z * tidx.tile[0] + halo;
        const int k_end = k + block_z;

        int warp_id0 = (tidx.global[2])>>6;             // because the warp dimensions are 
        int warp_id1 = ((((tidx.global[1])>>0)<<2))>>0; // 1x1x64, warp_ids are division of these numbers

        const int lane_id = __lane_id();
        int lane_id_it = lane_id;
        DATA_TYPE t3_reg0, t3_reg1, t3_reg2, t3_reg3, t3_reg4, t3_reg5, t3_reg6;
        DATA_TYPE t2_reg0, t2_reg1, t2_reg2, t2_reg3, t2_reg4, t2_reg5, t2_reg6;
        DATA_TYPE t1_reg0, t1_reg1, t1_reg2, t1_reg3, t1_reg4, t1_reg5, t1_reg6;
        int new_id0 ;
        int new_id1 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        t3_reg0 = IN_3D(k  , new_id1, new_id0) ;
        t2_reg0 = IN_3D(k-1, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        t3_reg1 = IN_3D(k  , new_id1, new_id0) ;
        t2_reg1 = IN_3D(k-1, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        t3_reg2 = IN_3D(k  , new_id1, new_id0) ;
        t2_reg2 = IN_3D(k-1, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        t3_reg3 = IN_3D(k  , new_id1, new_id0) ;
        t2_reg3 = IN_3D(k-1, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        t3_reg4 = IN_3D(k  , new_id1, new_id0) ;
        t2_reg4 = IN_3D(k-1, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        t3_reg5 = IN_3D(k  , new_id1, new_id0) ;
        t2_reg5 = IN_3D(k-1, new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
        new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
        t3_reg6 = IN_3D(k  , new_id1, new_id0) ;
        t2_reg6 = IN_3D(k-1, new_id1, new_id0) ;

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

#pragma unroll 
        for(; k < k_end; ++k)
        {
            sum0 = 0.0;
            sum1 = 0.0;
            sum2 = 0.0;
            sum3 = 0.0;

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
            new_id0 = (warp_id0<<6) + lane_id_it%66 ;
            new_id1 = (warp_id1<<0) + lane_id_it/66 ;
            t3_reg0 = IN_3D(k+1, new_id1, new_id0) ;
            lane_id_it += 64 ;
            new_id0 = (warp_id0<<6) + lane_id_it%66 ;
            new_id1 = (warp_id1<<0) + lane_id_it/66 ;
            t3_reg1 = IN_3D(k+1, new_id1, new_id0) ;
            lane_id_it += 64 ;
            new_id0 = (warp_id0<<6) + lane_id_it%66 ;
            new_id1 = (warp_id1<<0) + lane_id_it/66 ;
            t3_reg2 = IN_3D(k+1, new_id1, new_id0) ;
            lane_id_it += 64 ;
            new_id0 = (warp_id0<<6) + lane_id_it%66 ;
            new_id1 = (warp_id1<<0) + lane_id_it/66 ;
            t3_reg3 = IN_3D(k+1, new_id1, new_id0) ;
            lane_id_it += 64 ;
            new_id0 = (warp_id0<<6) + lane_id_it%66 ;
            new_id1 = (warp_id1<<0) + lane_id_it/66 ;
            t3_reg4 = IN_3D(k+1, new_id1, new_id0) ;
            lane_id_it += 64 ;
            new_id0 = (warp_id0<<6) + lane_id_it%66 ;
            new_id1 = (warp_id1<<0) + lane_id_it/66 ;
            t3_reg5 = IN_3D(k+1, new_id1, new_id0) ;
            lane_id_it += 64 ;
            new_id0 = (warp_id0<<6) + lane_id_it%66 ;
            new_id1 = (warp_id1<<0) + lane_id_it/66 ;
            new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
            new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
            t3_reg6 = IN_3D(k+1, new_id1, new_id0) ;

            // process (0, 0, 0)
            friend_id0 = (lane_id+ 0)&63 ;
            tx0 = __shfl(t1_reg0, friend_id0);
            sum0 += a0 *(tx0);
            tx0 = __shfl(t2_reg0, friend_id0);
            sum0 += a9 *(tx0);
            tx0 = __shfl(t3_reg0, friend_id0);
            sum0 += a18*(tx0);
            friend_id1 = (lane_id+ 2)&63 ;
            tx1 = __shfl(t1_reg1, friend_id1);
            ty1 = __shfl(t1_reg2, friend_id1);
            sum1 += a0 *((lane_id < 62 )? tx1: ty1);
            tx1 = __shfl(t2_reg1, friend_id1);
            ty1 = __shfl(t2_reg2, friend_id1);
            sum1 += a9 *((lane_id < 62 )? tx1: ty1);
            tx1 = __shfl(t3_reg1, friend_id1);
            ty1 = __shfl(t3_reg2, friend_id1);
            sum1 += a18*((lane_id < 62 )? tx1: ty1);
            friend_id2 = (lane_id+ 4)&63 ;
            tx2 = __shfl(t1_reg2, friend_id2);
            ty2 = __shfl(t1_reg3, friend_id2);
            sum2 += a0 *((lane_id < 60 )? tx2: ty2);
            tx2 = __shfl(t2_reg2, friend_id2);
            ty2 = __shfl(t2_reg3, friend_id2);
            sum2 += a9 *((lane_id < 60 )? tx2: ty2);
            tx2 = __shfl(t3_reg2, friend_id2);
            ty2 = __shfl(t3_reg3, friend_id2);
            sum2 += a18*((lane_id < 60 )? tx2: ty2);
            friend_id3 = (lane_id+ 6)&63 ;
            tx3 = __shfl(t1_reg3, friend_id3);
            ty3 = __shfl(t1_reg4, friend_id3);
            sum3 += a0 *((lane_id < 58 )? tx3: ty3);
            tx3 = __shfl(t2_reg3, friend_id3);
            ty3 = __shfl(t2_reg4, friend_id3);
            sum3 += a9 *((lane_id < 58 )? tx3: ty3);
            tx3 = __shfl(t3_reg3, friend_id3);
            ty3 = __shfl(t3_reg4, friend_id3);
            sum3 += a18*((lane_id < 58 )? tx3: ty3);
            // process (1, 0, 0)
            friend_id0 = (lane_id+ 1)&63 ;
            tx0 = __shfl(t1_reg0, friend_id0);
            ty0 = __shfl(t1_reg1, friend_id0);
            sum0 += a1 *((lane_id < 63 )? tx0: ty0);
            tx0 = __shfl(t2_reg0, friend_id0);
            ty0 = __shfl(t2_reg1, friend_id0);
            sum0 += a10*((lane_id < 63 )? tx0: ty0);
            tx0 = __shfl(t3_reg0, friend_id0);
            ty0 = __shfl(t3_reg1, friend_id0);
            sum0 += a19*((lane_id < 63 )? tx0: ty0);
            friend_id1 = (lane_id+ 3)&63 ;
            tx1 = __shfl(t1_reg1, friend_id1);
            ty1 = __shfl(t1_reg2, friend_id1);
            sum1 += a1 *((lane_id < 61 )? tx1: ty1);
            tx1 = __shfl(t2_reg1, friend_id1);
            ty1 = __shfl(t2_reg2, friend_id1);
            sum1 += a10*((lane_id < 61 )? tx1: ty1);
            tx1 = __shfl(t3_reg1, friend_id1);
            ty1 = __shfl(t3_reg2, friend_id1);
            sum1 += a19*((lane_id < 61 )? tx1: ty1);
            friend_id2 = (lane_id+ 5)&63 ;
            tx2 = __shfl(t1_reg2, friend_id2);
            ty2 = __shfl(t1_reg3, friend_id2);
            sum2 += a1 *((lane_id < 59 )? tx2: ty2);
            tx2 = __shfl(t2_reg2, friend_id2);
            ty2 = __shfl(t2_reg3, friend_id2);
            sum2 += a10*((lane_id < 59 )? tx2: ty2);
            tx2 = __shfl(t3_reg2, friend_id2);
            ty2 = __shfl(t3_reg3, friend_id2);
            sum2 += a19*((lane_id < 59 )? tx2: ty2);
            friend_id3 = (lane_id+ 7)&63 ;
            tx3 = __shfl(t1_reg3, friend_id3);
            ty3 = __shfl(t1_reg4, friend_id3);
            sum3 += a1 *((lane_id < 57 )? tx3: ty3);
            tx3 = __shfl(t2_reg3, friend_id3);
            ty3 = __shfl(t2_reg4, friend_id3);
            sum3 += a10*((lane_id < 57 )? tx3: ty3);
            tx3 = __shfl(t3_reg3, friend_id3);
            ty3 = __shfl(t3_reg4, friend_id3);
            sum3 += a19*((lane_id < 57 )? tx3: ty3);
            // process (2, 0, 0)
            friend_id0 = (lane_id+ 2)&63 ;
            tx0 = __shfl(t1_reg0, friend_id0);
            ty0 = __shfl(t1_reg1, friend_id0);
            sum0 += a2 *((lane_id < 62 )? tx0: ty0);
            tx0 = __shfl(t2_reg0, friend_id0);
            ty0 = __shfl(t2_reg1, friend_id0);
            sum0 += a11*((lane_id < 62 )? tx0: ty0);
            tx0 = __shfl(t3_reg0, friend_id0);
            ty0 = __shfl(t3_reg1, friend_id0);
            sum0 += a20*((lane_id < 62 )? tx0: ty0);
            friend_id1 = (lane_id+ 4)&63 ;
            tx1 = __shfl(t1_reg1, friend_id1);
            ty1 = __shfl(t1_reg2, friend_id1);
            sum1 += a2 *((lane_id < 60 )? tx1: ty1);
            tx1 = __shfl(t2_reg1, friend_id1);
            ty1 = __shfl(t2_reg2, friend_id1);
            sum1 += a11*((lane_id < 60 )? tx1: ty1);
            tx1 = __shfl(t3_reg1, friend_id1);
            ty1 = __shfl(t3_reg2, friend_id1);
            sum1 += a20*((lane_id < 60 )? tx1: ty1);
            friend_id2 = (lane_id+ 6)&63 ;
            tx2 = __shfl(t1_reg2, friend_id2);
            ty2 = __shfl(t1_reg3, friend_id2);
            sum2 += a2 *((lane_id < 58 )? tx2: ty2);
            tx2 = __shfl(t2_reg2, friend_id2);
            ty2 = __shfl(t2_reg3, friend_id2);
            sum2 += a11*((lane_id < 58 )? tx2: ty2);
            tx2 = __shfl(t3_reg2, friend_id2);
            ty2 = __shfl(t3_reg3, friend_id2);
            sum2 += a20*((lane_id < 58 )? tx2: ty2);
            friend_id3 = (lane_id+ 8)&63 ;
            tx3 = __shfl(t1_reg3, friend_id3);
            ty3 = __shfl(t1_reg4, friend_id3);
            sum3 += a2 *((lane_id < 56 )? tx3: ty3);
            tx3 = __shfl(t2_reg3, friend_id3);
            ty3 = __shfl(t2_reg4, friend_id3);
            sum3 += a11*((lane_id < 56 )? tx3: ty3);
            tx3 = __shfl(t3_reg3, friend_id3);
            ty3 = __shfl(t3_reg4, friend_id3);
            sum3 += a20*((lane_id < 56 )? tx3: ty3);
            // process (0, 1, 0)
            friend_id0 = (lane_id+ 2)&63 ;
            tx0 = __shfl(t1_reg1, friend_id0);
            ty0 = __shfl(t1_reg2, friend_id0);
            sum0 += a3 *((lane_id < 62 )? tx0: ty0);
            tx0 = __shfl(t2_reg1, friend_id0);
            ty0 = __shfl(t2_reg2, friend_id0);
            sum0 += a12*((lane_id < 62 )? tx0: ty0);
            tx0 = __shfl(t3_reg1, friend_id0);
            ty0 = __shfl(t3_reg2, friend_id0);
            sum0 += a21*((lane_id < 62 )? tx0: ty0);
            friend_id1 = (lane_id+ 4)&63 ;
            tx1 = __shfl(t1_reg2, friend_id1);
            ty1 = __shfl(t1_reg3, friend_id1);
            sum1 += a3 *((lane_id < 60 )? tx1: ty1);
            tx1 = __shfl(t2_reg2, friend_id1);
            ty1 = __shfl(t2_reg3, friend_id1);
            sum1 += a12*((lane_id < 60 )? tx1: ty1);
            tx1 = __shfl(t3_reg2, friend_id1);
            ty1 = __shfl(t3_reg3, friend_id1);
            sum1 += a21*((lane_id < 60 )? tx1: ty1);
            friend_id2 = (lane_id+ 6)&63 ;
            tx2 = __shfl(t1_reg3, friend_id2);
            ty2 = __shfl(t1_reg4, friend_id2);
            sum2 += a3 *((lane_id < 58 )? tx2: ty2);
            tx2 = __shfl(t2_reg3, friend_id2);
            ty2 = __shfl(t2_reg4, friend_id2);
            sum2 += a12*((lane_id < 58 )? tx2: ty2);
            tx2 = __shfl(t3_reg3, friend_id2);
            ty2 = __shfl(t3_reg4, friend_id2);
            sum2 += a21*((lane_id < 58 )? tx2: ty2);
            friend_id3 = (lane_id+ 8)&63 ;
            tx3 = __shfl(t1_reg4, friend_id3);
            ty3 = __shfl(t1_reg5, friend_id3);
            sum3 += a3 *((lane_id < 56 )? tx3: ty3);
            tx3 = __shfl(t2_reg4, friend_id3);
            ty3 = __shfl(t2_reg5, friend_id3);
            sum3 += a12*((lane_id < 56 )? tx3: ty3);
            tx3 = __shfl(t3_reg4, friend_id3);
            ty3 = __shfl(t3_reg5, friend_id3);
            sum3 += a21*((lane_id < 56 )? tx3: ty3);
            // process (1, 1, 0)
            friend_id0 = (lane_id+ 3)&63 ;
            tx0 = __shfl(t1_reg1, friend_id0);
            ty0 = __shfl(t1_reg2, friend_id0);
            sum0 += a4 *((lane_id < 61 )? tx0: ty0);
            tx0 = __shfl(t2_reg1, friend_id0);
            ty0 = __shfl(t2_reg2, friend_id0);
            sum0 += a13*((lane_id < 61 )? tx0: ty0);
            tx0 = __shfl(t3_reg1, friend_id0);
            ty0 = __shfl(t3_reg2, friend_id0);
            sum0 += a22*((lane_id < 61 )? tx0: ty0);
            friend_id1 = (lane_id+ 5)&63 ;
            tx1 = __shfl(t1_reg2, friend_id1);
            ty1 = __shfl(t1_reg3, friend_id1);
            sum1 += a4 *((lane_id < 59 )? tx1: ty1);
            tx1 = __shfl(t2_reg2, friend_id1);
            ty1 = __shfl(t2_reg3, friend_id1);
            sum1 += a13*((lane_id < 59 )? tx1: ty1);
            tx1 = __shfl(t3_reg2, friend_id1);
            ty1 = __shfl(t3_reg3, friend_id1);
            sum1 += a22*((lane_id < 59 )? tx1: ty1);
            friend_id2 = (lane_id+ 7)&63 ;
            tx2 = __shfl(t1_reg3, friend_id2);
            ty2 = __shfl(t1_reg4, friend_id2);
            sum2 += a4 *((lane_id < 57 )? tx2: ty2);
            tx2 = __shfl(t2_reg3, friend_id2);
            ty2 = __shfl(t2_reg4, friend_id2);
            sum2 += a13*((lane_id < 57 )? tx2: ty2);
            tx2 = __shfl(t3_reg3, friend_id2);
            ty2 = __shfl(t3_reg4, friend_id2);
            sum2 += a22*((lane_id < 57 )? tx2: ty2);
            friend_id3 = (lane_id+ 9)&63 ;
            tx3 = __shfl(t1_reg4, friend_id3);
            ty3 = __shfl(t1_reg5, friend_id3);
            sum3 += a4 *((lane_id < 55 )? tx3: ty3);
            tx3 = __shfl(t2_reg4, friend_id3);
            ty3 = __shfl(t2_reg5, friend_id3);
            sum3 += a13*((lane_id < 55 )? tx3: ty3);
            tx3 = __shfl(t3_reg4, friend_id3);
            ty3 = __shfl(t3_reg5, friend_id3);
            sum3 += a22*((lane_id < 55 )? tx3: ty3);
            // process (2, 1, 0)
            friend_id0 = (lane_id+ 4)&63 ;
            tx0 = __shfl(t1_reg1, friend_id0);
            ty0 = __shfl(t1_reg2, friend_id0);
            sum0 += a5 *((lane_id < 60 )? tx0: ty0);
            tx0 = __shfl(t2_reg1, friend_id0);
            ty0 = __shfl(t2_reg2, friend_id0);
            sum0 += a14*((lane_id < 60 )? tx0: ty0);
            tx0 = __shfl(t3_reg1, friend_id0);
            ty0 = __shfl(t3_reg2, friend_id0);
            sum0 += a23*((lane_id < 60 )? tx0: ty0);
            friend_id1 = (lane_id+ 6)&63 ;
            tx1 = __shfl(t1_reg2, friend_id1);
            ty1 = __shfl(t1_reg3, friend_id1);
            sum1 += a5 *((lane_id < 58 )? tx1: ty1);
            tx1 = __shfl(t2_reg2, friend_id1);
            ty1 = __shfl(t2_reg3, friend_id1);
            sum1 += a14*((lane_id < 58 )? tx1: ty1);
            tx1 = __shfl(t3_reg2, friend_id1);
            ty1 = __shfl(t3_reg3, friend_id1);
            sum1 += a23*((lane_id < 58 )? tx1: ty1);
            friend_id2 = (lane_id+ 8)&63 ;
            tx2 = __shfl(t1_reg3, friend_id2);
            ty2 = __shfl(t1_reg4, friend_id2);
            sum2 += a5 *((lane_id < 56 )? tx2: ty2);
            tx2 = __shfl(t2_reg3, friend_id2);
            ty2 = __shfl(t2_reg4, friend_id2);
            sum2 += a14*((lane_id < 56 )? tx2: ty2);
            tx2 = __shfl(t3_reg3, friend_id2);
            ty2 = __shfl(t3_reg4, friend_id2);
            sum2 += a23*((lane_id < 56 )? tx2: ty2);
            friend_id3 = (lane_id+10)&63 ;
            tx3 = __shfl(t1_reg4, friend_id3);
            ty3 = __shfl(t1_reg5, friend_id3);
            sum3 += a5 *((lane_id < 54 )? tx3: ty3);
            tx3 = __shfl(t2_reg4, friend_id3);
            ty3 = __shfl(t2_reg5, friend_id3);
            sum3 += a14*((lane_id < 54 )? tx3: ty3);
            tx3 = __shfl(t3_reg4, friend_id3);
            ty3 = __shfl(t3_reg5, friend_id3);
            sum3 += a23*((lane_id < 54 )? tx3: ty3);
            // process (0, 2, 0)
            friend_id0 = (lane_id+ 4)&63 ;
            tx0 = __shfl(t1_reg2, friend_id0);
            ty0 = __shfl(t1_reg3, friend_id0);
            sum0 += a6 *((lane_id < 60 )? tx0: ty0);
            tx0 = __shfl(t2_reg2, friend_id0);
            ty0 = __shfl(t2_reg3, friend_id0);
            sum0 += a15*((lane_id < 60 )? tx0: ty0);
            tx0 = __shfl(t3_reg2, friend_id0);
            ty0 = __shfl(t3_reg3, friend_id0);
            sum0 += a24*((lane_id < 60 )? tx0: ty0);
            friend_id1 = (lane_id+ 6)&63 ;
            tx1 = __shfl(t1_reg3, friend_id1);
            ty1 = __shfl(t1_reg4, friend_id1);
            sum1 += a6 *((lane_id < 58 )? tx1: ty1);
            tx1 = __shfl(t2_reg3, friend_id1);
            ty1 = __shfl(t2_reg4, friend_id1);
            sum1 += a15*((lane_id < 58 )? tx1: ty1);
            tx1 = __shfl(t3_reg3, friend_id1);
            ty1 = __shfl(t3_reg4, friend_id1);
            sum1 += a24*((lane_id < 58 )? tx1: ty1);
            friend_id2 = (lane_id+ 8)&63 ;
            tx2 = __shfl(t1_reg4, friend_id2);
            ty2 = __shfl(t1_reg5, friend_id2);
            sum2 += a6 *((lane_id < 56 )? tx2: ty2);
            tx2 = __shfl(t2_reg4, friend_id2);
            ty2 = __shfl(t2_reg5, friend_id2);
            sum2 += a15*((lane_id < 56 )? tx2: ty2);
            tx2 = __shfl(t3_reg4, friend_id2);
            ty2 = __shfl(t3_reg5, friend_id2);
            sum2 += a24*((lane_id < 56 )? tx2: ty2);
            friend_id3 = (lane_id+10)&63 ;
            tx3 = __shfl(t1_reg5, friend_id3);
            ty3 = __shfl(t1_reg6, friend_id3);
            sum3 += a6 *((lane_id < 54 )? tx3: ty3);
            tx3 = __shfl(t2_reg5, friend_id3);
            ty3 = __shfl(t2_reg6, friend_id3);
            sum3 += a15*((lane_id < 54 )? tx3: ty3);
            tx3 = __shfl(t3_reg5, friend_id3);
            ty3 = __shfl(t3_reg6, friend_id3);
            sum3 += a24*((lane_id < 54 )? tx3: ty3);
            // process (1, 2, 0)
            friend_id0 = (lane_id+ 5)&63 ;
            tx0 = __shfl(t1_reg2, friend_id0);
            ty0 = __shfl(t1_reg3, friend_id0);
            sum0 += a7 *((lane_id < 59 )? tx0: ty0);
            tx0 = __shfl(t2_reg2, friend_id0);
            ty0 = __shfl(t2_reg3, friend_id0);
            sum0 += a16*((lane_id < 59 )? tx0: ty0);
            tx0 = __shfl(t3_reg2, friend_id0);
            ty0 = __shfl(t3_reg3, friend_id0);
            sum0 += a25*((lane_id < 59 )? tx0: ty0);
            friend_id1 = (lane_id+ 7)&63 ;
            tx1 = __shfl(t1_reg3, friend_id1);
            ty1 = __shfl(t1_reg4, friend_id1);
            sum1 += a7 *((lane_id < 57 )? tx1: ty1);
            tx1 = __shfl(t2_reg3, friend_id1);
            ty1 = __shfl(t2_reg4, friend_id1);
            sum1 += a16*((lane_id < 57 )? tx1: ty1);
            tx1 = __shfl(t3_reg3, friend_id1);
            ty1 = __shfl(t3_reg4, friend_id1);
            sum1 += a25*((lane_id < 57 )? tx1: ty1);
            friend_id2 = (lane_id+ 9)&63 ;
            tx2 = __shfl(t1_reg4, friend_id2);
            ty2 = __shfl(t1_reg5, friend_id2);
            sum2 += a7 *((lane_id < 55 )? tx2: ty2);
            tx2 = __shfl(t2_reg4, friend_id2);
            ty2 = __shfl(t2_reg5, friend_id2);
            sum2 += a16*((lane_id < 55 )? tx2: ty2);
            tx2 = __shfl(t3_reg4, friend_id2);
            ty2 = __shfl(t3_reg5, friend_id2);
            sum2 += a25*((lane_id < 55 )? tx2: ty2);
            friend_id3 = (lane_id+11)&63 ;
            tx3 = __shfl(t1_reg5, friend_id3);
            ty3 = __shfl(t1_reg6, friend_id3);
            sum3 += a7 *((lane_id < 53 )? tx3: ty3);
            tx3 = __shfl(t2_reg5, friend_id3);
            ty3 = __shfl(t2_reg6, friend_id3);
            sum3 += a16*((lane_id < 53 )? tx3: ty3);
            tx3 = __shfl(t3_reg5, friend_id3);
            ty3 = __shfl(t3_reg6, friend_id3);
            sum3 += a25*((lane_id < 53 )? tx3: ty3);
            // process (2, 2, 0)
            friend_id0 = (lane_id+ 6)&63 ;
            tx0 = __shfl(t1_reg2, friend_id0);
            ty0 = __shfl(t1_reg3, friend_id0);
            sum0 += a8 *((lane_id < 58 )? tx0: ty0);
            tx0 = __shfl(t2_reg2, friend_id0);
            ty0 = __shfl(t2_reg3, friend_id0);
            sum0 += a17*((lane_id < 58 )? tx0: ty0);
            tx0 = __shfl(t3_reg2, friend_id0);
            ty0 = __shfl(t3_reg3, friend_id0);
            sum0 += a26*((lane_id < 58 )? tx0: ty0);
            friend_id1 = (lane_id+ 8)&63 ;
            tx1 = __shfl(t1_reg3, friend_id1);
            ty1 = __shfl(t1_reg4, friend_id1);
            sum1 += a8 *((lane_id < 56 )? tx1: ty1);
            tx1 = __shfl(t2_reg3, friend_id1);
            ty1 = __shfl(t2_reg4, friend_id1);
            sum1 += a17*((lane_id < 56 )? tx1: ty1);
            tx1 = __shfl(t3_reg3, friend_id1);
            ty1 = __shfl(t3_reg4, friend_id1);
            sum1 += a26*((lane_id < 56 )? tx1: ty1);
            friend_id2 = (lane_id+10)&63 ;
            tx2 = __shfl(t1_reg4, friend_id2);
            ty2 = __shfl(t1_reg5, friend_id2);
            sum2 += a8 *((lane_id < 54 )? tx2: ty2);
            tx2 = __shfl(t2_reg4, friend_id2);
            ty2 = __shfl(t2_reg5, friend_id2);
            sum2 += a17*((lane_id < 54 )? tx2: ty2);
            tx2 = __shfl(t3_reg4, friend_id2);
            ty2 = __shfl(t3_reg5, friend_id2);
            sum2 += a26*((lane_id < 54 )? tx2: ty2);
            friend_id3 = (lane_id+12)&63 ;
            tx3 = __shfl(t1_reg5, friend_id3);
            ty3 = __shfl(t1_reg6, friend_id3);
            sum3 += a8 *((lane_id < 52 )? tx3: ty3);
            tx3 = __shfl(t2_reg5, friend_id3);
            ty3 = __shfl(t2_reg6, friend_id3);
            sum3 += a17*((lane_id < 52 )? tx3: ty3);
            tx3 = __shfl(t3_reg5, friend_id3);
            ty3 = __shfl(t3_reg6, friend_id3);
            sum3 += a26*((lane_id < 52 )? tx3: ty3);           

            
            OUT_3D(k,j  ,i) = sum0;
            OUT_3D(k,j+1,i) = sum1;
            OUT_3D(k,j+2,i) = sum2;
            OUT_3D(k,j+3,i) = sum3;
        
        }
    });
    fut.wait();
}



int main(int argc, char **argv)
{
#ifdef __DEBUG
    int z = 4;
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
        std::swap(in, out_ref);
    }
    std::swap(in, out_ref);
    // Show_Me(out_ref, z, m, n, halo, "Output:");
    std::cout << "seq finished" << std::endl;


    extent<1> data_domain(total);
    array<DATA_TYPE>  in_d(data_domain);
    array<DATA_TYPE> out_d(data_domain);
    DATA_TYPE *out = new DATA_TYPE[total];
    float time_wo_pci;

    // Hcc version
    /////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed);
    Clear_Output_3D(out, z, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        // 4*8*8
        Stencil_Hcc_L1_3Blk(in_d, out_d,  
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    auto t2 = std::chrono::high_resolution_clock::now();
    double milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_L1_3Blk: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_L1_3Blk Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc Sweep version
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed);
    Clear_Output_3D(out, z, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        // 1*4*64
        Stencil_Hcc_L1_25Blk(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_L1_25Blk: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_L1_25Blk Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc 3D-Block with SM_Branch 
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed);
    Clear_Output_3D(out, z, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        //4*8*8
        Stencil_Hcc_Lds_3BlkBrc(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Lds_3BlkBrc: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Lds_3BlkBrc Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc 3D-Block with SM_Cyclic
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed);
    Clear_Output_3D(out, z, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        //4*8*8
        Stencil_Hcc_Lds_3BlkCyc(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Lds_3BlkCyc: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Lds_3BlkCyc Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc 2.5D-Block (Sweep) with SM_Branch
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed);
    Clear_Output_3D(out, z, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        // 1*4*64
        Stencil_Hcc_Lds_25BlkBrc(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Lds_25BlkBrc: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Lds_25BlkBrc Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc 2.5D-Block (Sweep) with SM_Cyclic
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed);
    Clear_Output_3D(out, z, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        // 1*4*64
        Stencil_Hcc_Lds_25BlkCyc(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Lds_25BlkCyc: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Lds_25BlkCyc Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc 3D-Block with Shfl 1-Point (2D-Warp)
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed);
    std::fill_n(out, total, 0);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        // 4*8*8
        Stencil_Hcc_Reg1_3Blk2Wf(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Reg1_3Blk2Wf: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Reg1_3Blk2Wf Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc 3D-Block with Shfl 2-Point (2D-Warp)
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed);
    Clear_Output_3D(out, z, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        // 4*8*8
        Stencil_Hcc_Reg2_3Blk2Wf(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Reg2_3Blk2Wf: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Reg2_3Blk2Wf Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc 3D-Block with Shfl 4-Point (2D-Warp)
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed);
    Clear_Output_3D(out, z, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        // 4*8*8
        Stencil_Hcc_Reg4_3Blk2Wf(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Reg4_3Blk2Wf: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Reg4_3Blk2Wf Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc 2.5D-Block with Shfl 1-Point (1D-Warp)
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed);
    Clear_Output_3D(out, z, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        // 1*4*64
        Stencil_Hcc_Reg1_25Blk1Wf(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Reg1_25Blk1Wf: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Reg1_25Blk1Wf Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc 2.5D-Block with Shfl 2-Point (1D-Warp)
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed);
    Clear_Output_3D(out, z, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        // 1*4*64
        Stencil_Hcc_Reg2_25Blk1Wf(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Reg2_25Blk1Wf: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Reg2_25Blk1Wf Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc 2.5D-Block with Shfl 4-Point (1D-Warp)
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed);
    Clear_Output_3D(out, z, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        // 1*4*64
        Stencil_Hcc_Reg4_25Blk1Wf(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], args[7 ], args[8 ], args[9 ], args[10], args[11], 
                args[12], args[13], args[14], args[15], args[16], args[17], 
                args[18], args[19], args[20], args[21], args[22], args[23], 
                args[24], args[25], args[26], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Reg4_25Blk1Wf: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Reg4_25Blk1Wf Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, OPS_3D27, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // std::cout << OUT_3D(1,1,1) << std::endl;
    delete[] in;
    delete[] out;
    delete[] out_ref;
}
