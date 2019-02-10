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

void Show_Me(DATA_TYPE *in, int m, int n, int halo, std::string prompt)
{
    std::cout << prompt << std::endl;
    for(int i = 0; i < m+2*halo; i++)
    {
        for(int j = 0; j < n+2*halo; j++)
        {
            std::cout << IN_2D(i,j) << ",";
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
        DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, 
        int m, int n, int halo)
{
#pragma omp parallel for
    for(int j = halo; j < m+halo; j++)
    {
        for(int i = halo; i < n+halo; i++)
        {
            OUT_2D(j,i) = a0 * IN_2D(j-1,i  ) +
                          a1 * IN_2D(j  ,i-1) +
                          a2 * IN_2D(j+1,i  ) +
                          a3 * IN_2D(j  ,i+1) +
                          a4 * IN_2D(j  ,i  ) ;
        }
    }
}

void Stencil_Hcc_L1_2Blk(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        int m, int n, int halo)
{
    extent<2> comp_domain(m, n); 
    tiled_extent<2> comp_tile(comp_domain, 8, 32);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<2> tidx) restrict(amp) {
        int i = tidx.global[1] + halo;
        int j = tidx.global[0] + halo;

        OUT_2D(j,i) = a0 * IN_2D(j-1,i  ) + 
                      a1 * IN_2D(j  ,i-1) + 
                      a2 * IN_2D(j+1,i  ) +
                      a3 * IN_2D(j  ,i+1) + 
                      a4 * IN_2D(j  ,i  ) ;
        
    });
    fut.wait();
}

void Stencil_Hcc_Lds_2BlkBrc(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        int m, int n, int halo)
{
    extent<2> comp_domain(m, n); 
    tiled_extent<2> comp_tile(comp_domain, 8, 32);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<2> tidx) restrict(amp) {
        int i = tidx.global[1] + halo;
        int j = tidx.global[0] + halo;

        int li = tidx.local[1] + 1;
        int lj = tidx.local[0] + 1;

        tile_static DATA_TYPE local[8+2][32+2];
        LOC_2D(lj,li) = IN_2D(j,i);
        if(li == halo)               LOC_2D(lj  ,li-1) = IN_2D(j  ,i-1);
        if(li == 32  )               LOC_2D(lj  ,li+1) = IN_2D(j  ,i+1);
        if(lj == halo)               LOC_2D(lj-1,li  ) = IN_2D(j-1,i  );
        if(lj == 8   )               LOC_2D(lj+1,li  ) = IN_2D(j+1,i  );
        tidx.barrier.wait();

        OUT_2D(j,i) = a0 * LOC_2D(lj-1,li  ) + 
                      a1 * LOC_2D(lj  ,li-1) + 
                      a2 * LOC_2D(lj+1,li  ) +
                      a3 * LOC_2D(lj  ,li+1) + 
                      a4 * LOC_2D(lj  ,li  ) ;

    });
    fut.wait();
}

void Stencil_Hcc_Lds_2BlkCyc(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        int m, int n, int halo)
{
    extent<2> comp_domain(m, n); 
    tiled_extent<2> comp_tile(comp_domain, 8, 32);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<2> tidx) restrict(amp) {
        int i = tidx.global[1] + halo;
        int j = tidx.global[0] + halo;

        int li = tidx.local[1] + 1;
        int lj = tidx.local[0] + 1;

        tile_static DATA_TYPE local[8+2][32+2];
        unsigned int lane_id = tidx.local[1] + tidx.local[0] * tidx.tile_dim[1];
        int blk_id_x = tidx.tile[1];
        int blk_id_y = tidx.tile[0];

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
        
        tidx.barrier.wait();

        OUT_2D(j,i) = a0 * LOC_2D(lj-1,li  ) + 
                      a1 * LOC_2D(lj  ,li-1) + 
                      a2 * LOC_2D(lj+1,li  ) +
                      a3 * LOC_2D(lj  ,li+1) + 
                      a4 * LOC_2D(lj  ,li  ) ;
    });
    fut.wait();
}

void Stencil_Hcc_Reg1_2Blk1Wf(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        int m, int n, int halo)
{
    extent<2> comp_domain(m, n); 
    tiled_extent<2> comp_tile(comp_domain, 4, 64);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<2> tidx) restrict(amp) {
        int i = tidx.global[1] + halo;
        int j = (((tidx.global[0])>>0)<<0) + halo;

        int warp_id0 = (tidx.global[1])>>6;             // because the warp dimensions are 
        int warp_id1 = ((((tidx.global[0])>>0)<<0))>>0; // 1x1x64, warp_ids are division of these numbers

        const int lane_id = __lane_id();
        int lane_id_it = lane_id;

        DATA_TYPE reg0, reg1, reg2, reg3;
        // load to regs: 
        int new_id0 ;
        int new_id1 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        reg0 = IN_2D(new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        reg1 = IN_2D(new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        reg2 = IN_2D(new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
        new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
        reg3 = IN_2D(new_id1, new_id0) ;

        DATA_TYPE sum0 = 0.0;
        int friend_id0;
        DATA_TYPE tx0, ty0, tz0;

        // process (1, 0, 0)
        friend_id0 = (lane_id+ 1)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a0 *((lane_id < 63 )? tx0: ty0);
        // process (0, 1, 0)
        friend_id0 = (lane_id+ 2)&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a1 *((lane_id < 62 )? tx0: ty0);
        // process (1, 1, 0)
        friend_id0 = (lane_id+ 3)&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a2 *((lane_id < 61 )? tx0: ty0);
        // process (2, 1, 0)
        friend_id0 = (lane_id+ 4)&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a3 *((lane_id < 60 )? tx0: ty0);
        // process (1, 2, 0)
        friend_id0 = (lane_id+ 5)&63 ;
        tx0 = __shfl(reg2, friend_id0);
        ty0 = __shfl(reg3, friend_id0);
        sum0 += a4 *((lane_id < 59 )? tx0: ty0);

        OUT_2D(j  ,i) = sum0;

    });
    fut.wait();
}

void Stencil_Hcc_Reg2_2Blk1Wf(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        int m, int n, int halo)
{
    extent<2> comp_domain(m/2, n); 
    tiled_extent<2> comp_tile(comp_domain, 4, 64);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<2> tidx) restrict(amp) {
        int i = tidx.global[1] + halo;
        int j = (((tidx.global[0])>>0)<<1) + halo;

        int warp_id0 = (tidx.global[1])>>6;             // because the warp dimensions are 
        int warp_id1 = ((((tidx.global[0])>>0)<<1))>>0; // 1x1x64, warp_ids are division of these numbers

        const int lane_id = __lane_id();
        int lane_id_it = lane_id;
        DATA_TYPE reg0, reg1, reg2, reg3, reg4;
        // load to regs: 
        int new_id0 ;
        int new_id1 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        reg0 = IN_2D(new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        reg1 = IN_2D(new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        reg2 = IN_2D(new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        reg3 = IN_2D(new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
        new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
        reg4 = IN_2D(new_id1, new_id0) ;
        
        
        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;
        int friend_id0;
        int friend_id1;
        DATA_TYPE tx0, ty0, tz0;
        DATA_TYPE tx1, ty1, tz1;

        // process (1, 0, 0)
        friend_id0 = (lane_id+ 1)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a0 *((lane_id < 63 )? tx0: ty0);
        friend_id1 = (lane_id+ 3)&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a0 *((lane_id < 61 )? tx1: ty1);
        // process (0, 1, 0)
        friend_id0 = (lane_id+ 2)&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a1 *((lane_id < 62 )? tx0: ty0);
        friend_id1 = (lane_id+ 4)&63 ;
        tx1 = __shfl(reg2, friend_id1);
        ty1 = __shfl(reg3, friend_id1);
        sum1 += a1 *((lane_id < 60 )? tx1: ty1);
        // process (1, 1, 0)
        friend_id0 = (lane_id+ 3)&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a2 *((lane_id < 61 )? tx0: ty0);
        friend_id1 = (lane_id+ 5)&63 ;
        tx1 = __shfl(reg2, friend_id1);
        ty1 = __shfl(reg3, friend_id1);
        sum1 += a2 *((lane_id < 59 )? tx1: ty1);
        // process (2, 1, 0)
        friend_id0 = (lane_id+ 4)&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a3 *((lane_id < 60 )? tx0: ty0);
        friend_id1 = (lane_id+ 6)&63 ;
        tx1 = __shfl(reg2, friend_id1);
        ty1 = __shfl(reg3, friend_id1);
        sum1 += a3 *((lane_id < 58 )? tx1: ty1);
        // process (1, 2, 0)
        friend_id0 = (lane_id+ 5)&63 ;
        tx0 = __shfl(reg2, friend_id0);
        ty0 = __shfl(reg3, friend_id0);
        sum0 += a4 *((lane_id < 59 )? tx0: ty0);
        friend_id1 = (lane_id+ 7)&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a4 *((lane_id < 57 )? tx1: ty1);

        OUT_2D(j  ,i) = sum0;
        OUT_2D(j+1,i) = sum1;
  
    });
    fut.wait();
}

void Stencil_Hcc_Reg4_2Blk1Wf(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        int m, int n, int halo)
{
    extent<2> comp_domain(m/4, n); 
    tiled_extent<2> comp_tile(comp_domain, 4, 64);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<2> tidx) restrict(amp) {
        int i = tidx.global[1] + halo;
        int j = (((tidx.global[0])>>0)<<2) + halo;

        int warp_id0 = (tidx.global[1])>>6;             // because the warp dimensions are 
        int warp_id1 = ((((tidx.global[0])>>0)<<2))>>0; // 1x1x64, warp_ids are division of these numbers

        const int lane_id = __lane_id();
        int lane_id_it = lane_id;
        DATA_TYPE reg0, reg1, reg2, reg3, reg4, reg5, reg6;
        // load to regs: 
        int new_id0 ;
        int new_id1 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        reg0 = IN_2D(new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        reg1 = IN_2D(new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        reg2 = IN_2D(new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        reg3 = IN_2D(new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        reg4 = IN_2D(new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        reg5 = IN_2D(new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + lane_id_it%66 ;
        new_id1 = (warp_id1<<0) + lane_id_it/66 ;
        new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
        new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
        reg6 = IN_2D(new_id1, new_id0) ;

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

        // process (1, 0, 0)
        friend_id0 = (lane_id+ 1)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a0 *((lane_id < 63 )? tx0: ty0);
        friend_id1 = (lane_id+ 3)&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a0 *((lane_id < 61 )? tx1: ty1);
        friend_id2 = (lane_id+ 5)&63 ;
        tx2 = __shfl(reg2, friend_id2);
        ty2 = __shfl(reg3, friend_id2);
        sum2 += a0 *((lane_id < 59 )? tx2: ty2);
        friend_id3 = (lane_id+ 7)&63 ;
        tx3 = __shfl(reg3, friend_id3);
        ty3 = __shfl(reg4, friend_id3);
        sum3 += a0 *((lane_id < 57 )? tx3: ty3);
        // process (0, 1, 0)
        friend_id0 = (lane_id+ 2)&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a1 *((lane_id < 62 )? tx0: ty0);
        friend_id1 = (lane_id+ 4)&63 ;
        tx1 = __shfl(reg2, friend_id1);
        ty1 = __shfl(reg3, friend_id1);
        sum1 += a1 *((lane_id < 60 )? tx1: ty1);
        friend_id2 = (lane_id+ 6)&63 ;
        tx2 = __shfl(reg3, friend_id2);
        ty2 = __shfl(reg4, friend_id2);
        sum2 += a1 *((lane_id < 58 )? tx2: ty2);
        friend_id3 = (lane_id+ 8)&63 ;
        tx3 = __shfl(reg4, friend_id3);
        ty3 = __shfl(reg5, friend_id3);
        sum3 += a1 *((lane_id < 56 )? tx3: ty3);
        // process (1, 1, 0)
        friend_id0 = (lane_id+ 3)&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a2 *((lane_id < 61 )? tx0: ty0);
        friend_id1 = (lane_id+ 5)&63 ;
        tx1 = __shfl(reg2, friend_id1);
        ty1 = __shfl(reg3, friend_id1);
        sum1 += a2 *((lane_id < 59 )? tx1: ty1);
        friend_id2 = (lane_id+ 7)&63 ;
        tx2 = __shfl(reg3, friend_id2);
        ty2 = __shfl(reg4, friend_id2);
        sum2 += a2 *((lane_id < 57 )? tx2: ty2);
        friend_id3 = (lane_id+ 9)&63 ;
        tx3 = __shfl(reg4, friend_id3);
        ty3 = __shfl(reg5, friend_id3);
        sum3 += a2 *((lane_id < 55 )? tx3: ty3);
        // process (2, 1, 0)
        friend_id0 = (lane_id+ 4)&63 ;
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a3 *((lane_id < 60 )? tx0: ty0);
        friend_id1 = (lane_id+ 6)&63 ;
        tx1 = __shfl(reg2, friend_id1);
        ty1 = __shfl(reg3, friend_id1);
        sum1 += a3 *((lane_id < 58 )? tx1: ty1);
        friend_id2 = (lane_id+ 8)&63 ;
        tx2 = __shfl(reg3, friend_id2);
        ty2 = __shfl(reg4, friend_id2);
        sum2 += a3 *((lane_id < 56 )? tx2: ty2);
        friend_id3 = (lane_id+10)&63 ;
        tx3 = __shfl(reg4, friend_id3);
        ty3 = __shfl(reg5, friend_id3);
        sum3 += a3 *((lane_id < 54 )? tx3: ty3);
        // process (1, 2, 0)
        friend_id0 = (lane_id+ 5)&63 ;
        tx0 = __shfl(reg2, friend_id0);
        ty0 = __shfl(reg3, friend_id0);
        sum0 += a4 *((lane_id < 59 )? tx0: ty0);
        friend_id1 = (lane_id+ 7)&63 ;
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum1 += a4 *((lane_id < 57 )? tx1: ty1);
        friend_id2 = (lane_id+ 9)&63 ;
        tx2 = __shfl(reg4, friend_id2);
        ty2 = __shfl(reg5, friend_id2);
        sum2 += a4 *((lane_id < 55 )? tx2: ty2);
        friend_id3 = (lane_id+11)&63 ;
        tx3 = __shfl(reg5, friend_id3);
        ty3 = __shfl(reg6, friend_id3);
        sum3 += a4 *((lane_id < 53 )? tx3: ty3);
        

        OUT_2D(j  ,i) = sum0;
        OUT_2D(j+1,i) = sum1;
        OUT_2D(j+2,i) = sum2;
        OUT_2D(j+3,i) = sum3;
     
  
    });
    fut.wait();
}

void Stencil_Hcc_Reg1_2Blk2Wf(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        int m, int n, int halo)
{
    extent<2> comp_domain(m, n); 
    tiled_extent<2> comp_tile(comp_domain, 32, 8);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<2> tidx) restrict(amp) {
        int i = tidx.global[1] + halo;
        int j = tidx.global[0] + halo;

        int warp_id0 = (tidx.global[1])>>3;             // because the warp dimensions are 
        int warp_id1 = (tidx.global[0])>>3; // 1x8x8, warp_ids are division of these numbers

        const int lane_id = __lane_id();
        int lane_id_it = lane_id;

        // num_regs: 2
        DATA_TYPE reg0 ;
        DATA_TYPE reg1 ;
        // load to regs: 
        int new_id0 ;
        int new_id1 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10 ;
        reg0 = IN_2D(new_id1, new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10 ;
        new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
        new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
        reg1 = IN_2D(new_id1, new_id0) ;

        DATA_TYPE sum = 0.0;
        int friend_id0;
        float tx0, ty0;
        // neighbor list: 1*3*3
        // job0:  0  1  2  | 10 11 12  | 20 21 22  | 
        // process (1, 0, 0)
        friend_id0 = (lane_id+ 1+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum += a0*((lane_id < 51 )? tx0: ty0);
        // process (0, 1, 0)
        friend_id0 = (lane_id+10+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum += a1*((lane_id < 44 )? tx0: ty0);
        // process (1, 1, 0)
        friend_id0 = (lane_id+11+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum += a2*((lane_id < 43 )? tx0: ty0);
        // process (2, 1, 0)
        friend_id0 = (lane_id+12+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum += a3*((lane_id < 42 )? tx0: ty0);
        // process (1, 2, 0)
        friend_id0 = (lane_id+21+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum += a4*((lane_id < 35 )? tx0: ty0);
        
        OUT_2D(j,i) = sum;

        
    });
    fut.wait();
}

void Stencil_Hcc_Reg2_2Blk2Wf(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        int m, int n, int halo)
{
    extent<2> comp_domain(m/2, n); 
    tiled_extent<2> comp_tile(comp_domain, 32, 8);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<2> tidx) restrict(amp) {
        const int lane_id = __lane_id();
        int i = tidx.global[1] + halo;
        int j = ((tidx.global[0]>>3)<<4) + (lane_id>>3) + halo;

        int warp_id0 = (tidx.global[1])>>3;             // because the warp dimensions are 
        int warp_id1 = (((tidx.global[0]>>3)<<4) + (lane_id>>3))>>3; // 1x8x8, warp_ids are division of these numbers

        int lane_id_it = lane_id;

        // num_regs: 3
        DATA_TYPE reg0 ;
        DATA_TYPE reg1 ;
        DATA_TYPE reg2 ;
        // load to regs: 
        int new_id0 ;
        int new_id1 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10 ;
        reg0 = IN_2D(new_id1, new_id0);
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10 ;
        reg1 = IN_2D(new_id1, new_id0);
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10 ;
        new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
        new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
        reg2 = IN_2D(new_id1, new_id0);

        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;
        int friend_id0, friend_id1;
        DATA_TYPE tx0, ty0, tx1, ty1;
        // neighbor list: 1*3*3
        // job0:  0  1  2  | 10 11 12  | 20 21 22  | 
        // job1: 16 17 18  | 26 27 28  | 36 37 38  | 
        // process (1, 0, 0)
        friend_id0 = (lane_id+ 1+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a0*((lane_id < 51 )? tx0: ty0);
        friend_id1 = (lane_id+17+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a0*((lane_id < 39 )? tx1: ty1);
        // process (0, 1, 0)
        friend_id0 = (lane_id+10+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a1*((lane_id < 44 )? tx0: ty0);
        friend_id1 = (lane_id+26+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a1*((lane_id < 32 )? tx1: ty1);
        // process (1, 1, 0)
        friend_id0 = (lane_id+11+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a2*((lane_id < 43 )? tx0: ty0);
        friend_id1 = (lane_id+27+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a2*((lane_id < 31 )? tx1: ty1);
        // process (2, 1, 0)
        friend_id0 = (lane_id+12+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a3*((lane_id < 42 )? tx0: ty0);
        friend_id1 = (lane_id+28+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a3*((lane_id < 30 )? tx1: ty1);
        // process (1, 2, 0)
        friend_id0 = (lane_id+21+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a4*((lane_id < 35 )? tx0: ty0);
        friend_id1 = (lane_id+37+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a4*((lane_id < 23 )? tx1: ty1);
        
        OUT_2D(j  ,i) = sum0;
        OUT_2D(j+8,i) = sum1;
        
    });
    fut.wait();
}

void Stencil_Hcc_Reg4_2Blk2Wf(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        int m, int n, int halo)
{
    extent<2> comp_domain(m/4, n); 
    tiled_extent<2> comp_tile(comp_domain, 32, 8);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<2> tidx) restrict(amp) {
        const int lane_id = __lane_id();
        int i = tidx.global[1] + halo;
        int j = ((tidx.global[0]>>3)<<5) + (lane_id>>3) + halo;

        int warp_id0 = (tidx.global[1])>>3;             // because the warp dimensions are 
        int warp_id1 = (((tidx.global[0]>>3)<<5) + (lane_id>>3))>>3; // 1x8x8, warp_ids are division of these numbers

        int lane_id_it = lane_id;

        // num_regs: 6
        DATA_TYPE reg0 ;
        DATA_TYPE reg1 ;
        DATA_TYPE reg2 ;
        DATA_TYPE reg3 ;
        DATA_TYPE reg4 ;
        DATA_TYPE reg5 ;
        // load to regs: 
        int new_id0 ;
        int new_id1 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10 ;
        reg0 = IN_2D(new_id1, new_id0);
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10 ;
        reg1 = IN_2D(new_id1, new_id0);
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10 ;
        reg2 = IN_2D(new_id1, new_id0);
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10 ;
        reg3 = IN_2D(new_id1, new_id0);
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10 ;
        reg4 = IN_2D(new_id1, new_id0);
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<3) + lane_id_it%10 ;
        new_id1 = (warp_id1<<3) + lane_id_it/10 ;
        new_id0 = (new_id0 < n+2)? new_id0 : n+1 ;
        new_id1 = (new_id1 < m+2)? new_id1 : m+1 ;
        reg5 = IN_2D(new_id1, new_id0);

        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;
        DATA_TYPE sum2 = 0.0;
        DATA_TYPE sum3 = 0.0;
        int friend_id0, friend_id1;
        int friend_id2, friend_id3;
        DATA_TYPE tx0, ty0, tz0, tx1, ty1, tz1;
        DATA_TYPE tx2, ty2, tz2, tx3, ty3, tz3;

        // neighbor list: 1*3*3
        // job0:  0  1  2  | 10 11 12  | 20 21 22  | 
        // job1: 16 17 18  | 26 27 28  | 36 37 38  | 
        // job2: 32 33 34  | 42 43 44  | 52 53 54  | 
        // job3: 48 49 50  | 58 59 60  |  4  5  6  | 
        // process (1, 0, 0)
        friend_id0 = (lane_id+ 1+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a0*((lane_id < 51 )? tx0: ty0);
        friend_id1 = (lane_id+17+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a0*((lane_id < 39 )? tx1: ty1);
        friend_id2 = (lane_id+33+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg2, friend_id2);
        ty2 = __shfl(reg3, friend_id2);
        sum2 += a0*((lane_id < 25 )? tx2: ty2);
        friend_id3 = (lane_id+49+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg3, friend_id3);
        ty3 = __shfl(reg4, friend_id3);
        sum3 += a0*((lane_id < 13 )? tx3: ty3);
        // process (0, 1, 0)
        friend_id0 = (lane_id+10+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a1*((lane_id < 44 )? tx0: ty0);
        friend_id1 = (lane_id+26+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a1*((lane_id < 32 )? tx1: ty1);
        friend_id2 = (lane_id+42+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg2, friend_id2);
        ty2 = __shfl(reg3, friend_id2);
        sum2 += a1*((lane_id < 18 )? tx2: ty2);
        friend_id3 = (lane_id+58+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg3, friend_id3);
        ty3 = __shfl(reg4, friend_id3);
        tz3 = __shfl(reg5, friend_id3);
        sum3 += a1*((lane_id < 6 )? tx3: ((lane_id < 56)? ty3: tz3));
        // process (1, 1, 0)
        friend_id0 = (lane_id+11+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a2*((lane_id < 43 )? tx0: ty0);
        friend_id1 = (lane_id+27+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a2*((lane_id < 31 )? tx1: ty1);
        friend_id2 = (lane_id+43+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg2, friend_id2);
        ty2 = __shfl(reg3, friend_id2);
        sum2 += a2*((lane_id < 17 )? tx2: ty2);
        friend_id3 = (lane_id+59+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg3, friend_id3);
        ty3 = __shfl(reg4, friend_id3);
        tz3 = __shfl(reg5, friend_id3);
        sum3 += a2*((lane_id < 5 )? tx3: ((lane_id < 56)? ty3: tz3));
        // process (2, 1, 0)
        friend_id0 = (lane_id+12+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a3*((lane_id < 42 )? tx0: ty0);
        friend_id1 = (lane_id+28+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a3*((lane_id < 30 )? tx1: ty1);
        friend_id2 = (lane_id+44+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg2, friend_id2);
        ty2 = __shfl(reg3, friend_id2);
        sum2 += a3*((lane_id < 16 )? tx2: ty2);
        friend_id3 = (lane_id+60+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg3, friend_id3);
        ty3 = __shfl(reg4, friend_id3);
        tz3 = __shfl(reg5, friend_id3);
        sum3 += a3*((lane_id < 4 )? tx3: ((lane_id < 56)? ty3: tz3));
        // process (1, 2, 0)
        friend_id0 = (lane_id+21+((lane_id>>3)*2))&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a4*((lane_id < 35 )? tx0: ty0);
        friend_id1 = (lane_id+37+((lane_id>>3)*2))&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a4*((lane_id < 23 )? tx1: ty1);
        friend_id2 = (lane_id+53+((lane_id>>3)*2))&63 ;
        tx2 = __shfl(reg2, friend_id2);
        ty2 = __shfl(reg3, friend_id2);
        tz2 = __shfl(reg4, friend_id2);
        sum2 += a4*((lane_id < 9 )? tx2: ((lane_id < 61)? ty2: tz2));
        friend_id3 = (lane_id+ 5+((lane_id>>3)*2))&63 ;
        tx3 = __shfl(reg4, friend_id3);
        ty3 = __shfl(reg5, friend_id3);
        sum3 += a4*((lane_id < 48 )? tx3: ty3);
        
        OUT_2D(j   ,i) = sum0;
        OUT_2D(j+8 ,i) = sum1;
        OUT_2D(j+16,i) = sum2;
        OUT_2D(j+24,i) = sum3;
        
    });
    fut.wait();
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
    const int K = 5;
#ifdef __DEBUG
    DATA_TYPE args[K] = {1.0, 1.0, 1.0, 1.0, 1.0};
#else
    DATA_TYPE args[K] = {0.20, 0.20, 0.20, 0.20, 0.20};
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
                m, n, halo);
        std::swap(in, out_ref);
    }
    std::swap(in, out_ref);
    // Show_Me(out_ref, m, n, halo, "Output:");

    extent<1> data_domain(total);
    array<DATA_TYPE>  in_d(data_domain);
    array<DATA_TYPE> out_d(data_domain);
    DATA_TYPE *out = new DATA_TYPE[total];
    float time_wo_pci;

    // Hcc version
    /////////////////////////////////////////
    Init_Input_2D(in, m, n, halo, seed);
    Clear_Output_2D(out, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_L1_2Blk(in_d, out_d,  
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ],
                m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    auto t2 = std::chrono::high_resolution_clock::now();
    double milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_L1_2Blk: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_L1_2Blk Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D5, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc Block with SM_Branch 
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, halo, seed);
    Clear_Output_2D(out, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Lds_2BlkBrc(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ],
                m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Lds_2BlkBrc: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Lds_2BlkBrc Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D5, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc Block with SM_Cyclic
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, halo, seed);
    Clear_Output_2D(out, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Lds_2BlkCyc(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], 
                m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Lds_2BlkCyc: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Lds_2BlkCyc Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D5, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc Shfl with 2D-Warp
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, halo, seed);
    Clear_Output_2D(out, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Reg1_2Blk2Wf(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ],
                m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Reg1_2Blk2Wf: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Reg1_2Blk2Wf Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D5, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc Shfl2 with 2D-Warp
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, halo, seed);
    Clear_Output_2D(out, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Reg2_2Blk2Wf(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ],
                m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Reg2_2Blk2Wf: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Reg2_2Blk2Wf Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D5, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc Shfl4 with 2D-Warp
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, halo, seed);
    Clear_Output_2D(out, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Reg4_2Blk2Wf(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ],
                m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Reg4_2Blk2Wf: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Reg4_2Blk2Wf Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D5, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));


    // Hcc Shfl with 1D-Warp
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, halo, seed);
    Clear_Output_2D(out, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Reg1_2Blk1Wf(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ],
                m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Reg1_2Blk1Wf: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Reg1_2Blk1Wf Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D5, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc Shfl2 with 1D-Warp
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, halo, seed);
    Clear_Output_2D(out, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Reg2_2Blk1Wf(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ],
                m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Reg2_2Blk1Wf: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Reg2_2Blk1Wf Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D5, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc Shfl4 with 1D-Warp
    /////////////////////////////////////////////////////////
    Init_Input_2D(in, m, n, halo, seed);
    Clear_Output_2D(out, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Reg4_2Blk1Wf(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ],
                m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Reg4_2Blk1Wf: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Reg4_2Blk1Wf Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m, n, ITER, OPS_2D5, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    delete[] in;
    delete[] out;
    delete[] out_ref;

}
