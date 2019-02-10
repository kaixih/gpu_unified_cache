#include <iostream>
#include <cmath>
#include <hc.hpp>
#include <metrics.h>

using namespace hc;

#define  IN_2D(_y,_x)  in[(_y)*(n+2*halo)+(_x)]
#define OUT_2D(_y,_x) out[(_y)*(n+2*halo)+(_x)]

#define LOC_2D(_y,_x) local[(_y)][(_x)]

#define warpSize 64

// #define __DEBUG

#ifdef __DEBUG
#define ITER 1
#else
#define ITER 100
#endif

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

void Stencil_Hcc(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
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

void Stencil_Hcc_Sm_Branch(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
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

void Stencil_Hcc_Sm_Cyclic(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
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

void Stencil_Hcc_Shfl_1DWarp(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        int m, int n, int halo)
{
    extent<2> comp_domain(m, n); 
    tiled_extent<2> comp_tile(comp_domain, 4, 64);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<2> tidx) restrict(amp) {
        int i = tidx.global[1] + halo;
        int j = (((tidx.global[0])>>0)<<0) + halo;

        int warp_id_x = (tidx.global[1])>>6;             // because the warp dimensions are 
        int warp_id_y = ((((tidx.global[0])>>0)<<0))>>0; // 1x1x64, warp_ids are division of these numbers

        const int lane_id = __lane_id();
        int lane_id_it = lane_id;
        int new_i, new_j;

        DATA_TYPE reg0, reg1, reg2, reg3;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        reg0 = IN_2D(new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        reg1 = IN_2D(new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        reg2 = IN_2D(new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
        reg3 = IN_2D(new_j, new_i);

        DATA_TYPE sum0 = 0.0;
        int friend_id0;
        DATA_TYPE tx0, ty0, tz0;

        friend_id0 = (lane_id+1 )&(warpSize-1);
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a0*((lane_id < 63)? tx0: ty0);
        
        friend_id0 = (lane_id+2 )&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a1*((lane_id < 62)? tx0: ty0);
        
        friend_id0 = (lane_id+3 )&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a4*((lane_id < 61)? tx0: ty0);
      
        friend_id0 = (lane_id+4 )&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a3*((lane_id < 60)? tx0: ty0);
     
        friend_id0 = (lane_id+5 )&(warpSize-1);
        tx0 = __shfl(reg2, friend_id0);
        ty0 = __shfl(reg3, friend_id0);
        sum0 += a2*((lane_id < 59)? tx0: ty0);

        OUT_2D(j  ,i) = sum0;

    });
    fut.wait();
}

void Stencil_Hcc_Shfl2_1DWarp(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        int m, int n, int halo)
{
    extent<2> comp_domain(m/2, n); 
    tiled_extent<2> comp_tile(comp_domain, 4, 64);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<2> tidx) restrict(amp) {
        int i = tidx.global[1] + halo;
        int j = (((tidx.global[0])>>0)<<1) + halo;

        int warp_id_x = (tidx.global[1])>>6;             // because the warp dimensions are 
        int warp_id_y = ((((tidx.global[0])>>0)<<1))>>0; // 1x1x64, warp_ids are division of these numbers

        const int lane_id = __lane_id();
        int lane_id_it = lane_id;
        int new_i, new_j;

        DATA_TYPE reg0, reg1, reg2, reg3, reg4;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        reg0 = IN_2D(new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        reg1 = IN_2D(new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        reg2 = IN_2D(new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        reg3 = IN_2D(new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
        reg4 = IN_2D(new_j, new_i);

        
        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;
        int friend_id0;
        int friend_id1;
        DATA_TYPE tx0, ty0, tz0;
        DATA_TYPE tx1, ty1, tz1;

        friend_id0 = (lane_id+1 )&(warpSize-1);
        friend_id1 = (lane_id+3 )&(warpSize-1);
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum0 += a0*((lane_id < 63)? tx0: ty0);
        sum1 += a0*((lane_id < 61)? tx1: ty1);
        
        friend_id0 = (lane_id+2 )&(warpSize-1);
        friend_id1 = (lane_id+4 )&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tx1 = __shfl(reg2, friend_id1);
        ty1 = __shfl(reg3, friend_id1);
        sum0 += a1*((lane_id < 62)? tx0: ty0);
        sum1 += a1*((lane_id < 60)? tx1: ty1);
        
        friend_id0 = (lane_id+3 )&(warpSize-1);
        friend_id1 = (lane_id+5 )&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tx1 = __shfl(reg2, friend_id1);
        ty1 = __shfl(reg3, friend_id1);
        sum0 += a4*((lane_id < 61)? tx0: ty0);
        sum1 += a4*((lane_id < 59)? tx1: ty1);

        friend_id0 = (lane_id+4 )&(warpSize-1);
        friend_id1 = (lane_id+6 )&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tx1 = __shfl(reg2, friend_id1);
        ty1 = __shfl(reg3, friend_id1);
        sum0 += a3*((lane_id < 60)? tx0: ty0);
        sum1 += a3*((lane_id < 58)? tx1: ty1);

        friend_id0 = (lane_id+5 )&(warpSize-1);
        friend_id1 = (lane_id+7 )&(warpSize-1);
        tx0 = __shfl(reg2, friend_id0);
        ty0 = __shfl(reg3, friend_id0);
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum0 += a2*((lane_id < 59)? tx0: ty0);
        sum1 += a2*((lane_id < 57)? tx1: ty1);

        OUT_2D(j  ,i) = sum0;
        OUT_2D(j+1,i) = sum1;
  
    });
    fut.wait();
}

void Stencil_Hcc_Shfl4_1DWarp(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        int m, int n, int halo)
{
    extent<2> comp_domain(m/4, n); 
    tiled_extent<2> comp_tile(comp_domain, 4, 64);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<2> tidx) restrict(amp) {
        int i = tidx.global[1] + halo;
        int j = (((tidx.global[0])>>0)<<2) + halo;

        int warp_id_x = (tidx.global[1])>>6;             // because the warp dimensions are 
        int warp_id_y = ((((tidx.global[0])>>0)<<2))>>0; // 1x1x64, warp_ids are division of these numbers

        const int lane_id = __lane_id();
        int lane_id_it = lane_id;
        int new_i, new_j;

        DATA_TYPE reg0, reg1, reg2, reg3, reg4, reg5, reg6;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        reg0 = IN_2D(new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        reg1 = IN_2D(new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        reg2 = IN_2D(new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        reg3 = IN_2D(new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        reg4 = IN_2D(new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        reg5 = IN_2D(new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
        reg6 = IN_2D(new_j, new_i);

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

        friend_id0 = (lane_id+1 )&(warpSize-1);
        friend_id1 = (lane_id+3 )&(warpSize-1);
        friend_id2 = (lane_id+5 )&(warpSize-1);
        friend_id3 = (lane_id+7 )&(warpSize-1);
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        tx2 = __shfl(reg2, friend_id2);
        ty2 = __shfl(reg3, friend_id2);
        tx3 = __shfl(reg3, friend_id3);
        ty3 = __shfl(reg4, friend_id3);
        sum0 += a0*((lane_id < 63)? tx0: ty0);
        sum1 += a0*((lane_id < 61)? tx1: ty1);
        sum2 += a0*((lane_id < 59)? tx2: ty2);
        sum3 += a0*((lane_id < 57)? tx3: ty3);
               
        friend_id0 = (lane_id+2 )&(warpSize-1);
        friend_id1 = (lane_id+4 )&(warpSize-1);
        friend_id2 = (lane_id+6 )&(warpSize-1);
        friend_id3 = (lane_id+8 )&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tx1 = __shfl(reg2, friend_id1);
        ty1 = __shfl(reg3, friend_id1);
        tx2 = __shfl(reg3, friend_id2);
        ty2 = __shfl(reg4, friend_id2);
        tx3 = __shfl(reg4, friend_id3);
        ty3 = __shfl(reg5, friend_id3);
        sum0 += a1*((lane_id < 62)? tx0: ty0);
        sum1 += a1*((lane_id < 60)? tx1: ty1);
        sum2 += a1*((lane_id < 58)? tx2: ty2);
        sum3 += a1*((lane_id < 56)? tx3: ty3);
        
        friend_id0 = (lane_id+3 )&(warpSize-1);
        friend_id1 = (lane_id+5 )&(warpSize-1);
        friend_id2 = (lane_id+7 )&(warpSize-1);
        friend_id3 = (lane_id+9 )&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tx1 = __shfl(reg2, friend_id1);
        ty1 = __shfl(reg3, friend_id1);
        tx2 = __shfl(reg3, friend_id2);
        ty2 = __shfl(reg4, friend_id2);
        tx3 = __shfl(reg4, friend_id3);
        ty3 = __shfl(reg5, friend_id3);
        sum0 += a4*((lane_id < 61)? tx0: ty0);
        sum1 += a4*((lane_id < 59)? tx1: ty1);
        sum2 += a4*((lane_id < 57)? tx2: ty2);
        sum3 += a4*((lane_id < 55)? tx3: ty3);

        friend_id0 = (lane_id+4 )&(warpSize-1);
        friend_id1 = (lane_id+6 )&(warpSize-1);
        friend_id2 = (lane_id+8 )&(warpSize-1);
        friend_id3 = (lane_id+10)&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tx1 = __shfl(reg2, friend_id1);
        ty1 = __shfl(reg3, friend_id1);
        tx2 = __shfl(reg3, friend_id2);
        ty2 = __shfl(reg4, friend_id2);
        tx3 = __shfl(reg4, friend_id3);
        ty3 = __shfl(reg5, friend_id3);
        sum0 += a3*((lane_id < 60)? tx0: ty0);
        sum1 += a3*((lane_id < 58)? tx1: ty1);
        sum2 += a3*((lane_id < 56)? tx2: ty2);
        sum3 += a3*((lane_id < 54)? tx3: ty3);

        friend_id0 = (lane_id+5 )&(warpSize-1);
        friend_id1 = (lane_id+7 )&(warpSize-1);
        friend_id2 = (lane_id+9 )&(warpSize-1);
        friend_id3 = (lane_id+11)&(warpSize-1);
        tx0 = __shfl(reg2, friend_id0);
        ty0 = __shfl(reg3, friend_id0);
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        tx2 = __shfl(reg4, friend_id2);
        ty2 = __shfl(reg5, friend_id2);
        tx3 = __shfl(reg5, friend_id3);
        ty3 = __shfl(reg6, friend_id3);
        sum0 += a2*((lane_id < 59)? tx0: ty0);
        sum1 += a2*((lane_id < 57)? tx1: ty1);
        sum2 += a2*((lane_id < 55)? tx2: ty2);
        sum3 += a2*((lane_id < 53)? tx3: ty3);

        OUT_2D(j  ,i) = sum0;
        OUT_2D(j+1,i) = sum1;
        OUT_2D(j+2,i) = sum2;
        OUT_2D(j+3,i) = sum3;
     
  
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
        Stencil_Hcc(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ],
                m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    auto t2 = std::chrono::high_resolution_clock::now();
    double milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m+2*halo, n+2*halo, ITER, OPS_2D5, time_wo_pci));
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
        Stencil_Hcc_Sm_Branch(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ],
                m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Sm_Branch: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Sm_Branch Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m+2*halo, n+2*halo, ITER, OPS_2D5, time_wo_pci));
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
        Stencil_Hcc_Sm_Cyclic(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], 
                m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Sm_Cyclic: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Sm_Cyclic Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m+2*halo, n+2*halo, ITER, OPS_2D5, time_wo_pci));
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
        Stencil_Hcc_Shfl_1DWarp(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ],
                m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Shfl_1DWarp: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Shfl_1DWarp Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m+2*halo, n+2*halo, ITER, OPS_2D5, time_wo_pci));
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
        Stencil_Hcc_Shfl2_1DWarp(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ],
                m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Shfl2_1DWarp: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Shfl2_1DWarp Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m+2*halo, n+2*halo, ITER, OPS_2D5, time_wo_pci));
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
        Stencil_Hcc_Shfl4_1DWarp(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ],
                m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Shfl4_1DWarp: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Shfl4_1DWarp Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, m+2*halo, n+2*halo, ITER, OPS_2D5, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    delete[] in;
    delete[] out;
    delete[] out_ref;

}
