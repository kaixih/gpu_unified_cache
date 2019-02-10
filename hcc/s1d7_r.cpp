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

void Show_Me(DATA_TYPE *in, int n, int halo, std::string prompt)
{
    std::cout << prompt << std::endl;
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


void Stencil_Hcc_L1_1Blk(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2,
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int n, int halo)
{
    extent<1> comp_domain(n); 
    tiled_extent<1> comp_tile(comp_domain, 256);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<1> tidx) restrict(amp) {
        int i = tidx.global[0] + halo;

        OUT_1D(i) = a0*IN_1D(i-3) + 
                    a1*IN_1D(i-2) + 
                    a2*IN_1D(i-1) + 
                    a3*IN_1D(i  ) + 
                    a4*IN_1D(i+1) + 
                    a5*IN_1D(i+2) + 
                    a6*IN_1D(i+3) ;
        
    });
    fut.wait();
}

void Stencil_Hcc_Lds_1BlkBrc(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2,
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int n, int halo)
{
    extent<1> comp_domain(n); 
    tiled_extent<1> comp_tile(comp_domain, 256);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<1> tidx) restrict(amp) {
        tile_static DATA_TYPE local[256+2*3];
        unsigned int tid = tidx.local[0];
        unsigned int gid = tidx.global[0] + halo;
        int local_id = tid + halo;
        local[local_id] = IN_1D(gid);
        if(tid == 0)
        {
            local[local_id-1] = IN_1D(gid-1);
            local[local_id-2] = IN_1D(gid-2);
            local[local_id-3] = IN_1D(gid-3);
        }
        if(tid == 255)
        {
            local[local_id+1] = IN_1D(gid+1);
            local[local_id+2] = IN_1D(gid+2);
            local[local_id+3] = IN_1D(gid+3);
        }
        tidx.barrier.wait();

        OUT_1D(gid) = a0*local[local_id-3] + 
                      a1*local[local_id-2] + 
                      a2*local[local_id-1] + 
                      a3*local[local_id  ] + 
                      a4*local[local_id+1] + 
                      a5*local[local_id+2] + 
                      a6*local[local_id+3] ;
        
    });
    fut.wait();
}

void Stencil_Hcc_Lds_1BlkCyc(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2,
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int n, int halo)
{
    extent<1> comp_domain(n); 
    tiled_extent<1> comp_tile(comp_domain, 256);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<1> tidx) restrict(amp) {
        tile_static DATA_TYPE local[256+2*3];
        unsigned int tid = tidx.local[0];
        unsigned int gid = tidx.global[0] + halo;
        int local_id = tid + halo;

        unsigned int lane_id = tidx.local[0];
        int lane_id_it = lane_id;
        int blk_id_x = tidx.tile[0];
        int new_i  = (blk_id_x<<8) + lane_id_it%262;
        int new_li = lane_id_it%262;
        local[new_li] = IN_1D(new_i);
        lane_id_it += 256;
        new_i  = (blk_id_x<<8) + (lane_id_it/262)*262 + lane_id_it%262;
        new_li = (lane_id_it/262)*262 + lane_id_it%262;
        if(new_li < 262)
            local[new_li] = IN_1D(new_i);
        tidx.barrier.wait();

        OUT_1D(gid) = a0*local[local_id-3] + 
                      a1*local[local_id-2] + 
                      a2*local[local_id-1] + 
                      a3*local[local_id  ] + 
                      a4*local[local_id+1] + 
                      a5*local[local_id+2] + 
                      a6*local[local_id+3] ;
        
    });
    fut.wait();
}

void Stencil_Hcc_Reg1_1Blk1Wf(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2,
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int n, int halo)
{
    extent<1> comp_domain(n); 
    tiled_extent<1> comp_tile(comp_domain, 256);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<1> tidx) restrict(amp) {
        unsigned int gid = tidx.global[0] + halo;
        unsigned int tid = tidx.local[0];
        unsigned int lane_id = __lane_id();

        int warp_id0 = (tidx.global[0])>>6;

        DATA_TYPE reg0, reg1;
        int lane_id_it = lane_id;
        // load to regs: 
        int new_id0 ;
        new_id0 = (warp_id0<<6) + lane_id_it%70 ;
        reg0 = IN_1D(new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + (lane_id_it/70)*70 + lane_id_it%70 ;
        new_id0 = (new_id0 < n+6)? new_id0 : n+5 ;
        reg1 = IN_1D(new_id0) ;

        DATA_TYPE sum0 = 0.0;
        int friend_id0;
        DATA_TYPE tx0, ty0;

        // process (0, 0, 0)
        friend_id0 = (lane_id+ 0)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        sum0 += a0*(tx0);
        // process (1, 0, 0)
        friend_id0 = (lane_id+ 1)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a1*((lane_id < 63 )? tx0: ty0);
        // process (2, 0, 0)
        friend_id0 = (lane_id+ 2)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a2*((lane_id < 62 )? tx0: ty0);
        // process (3, 0, 0)
        friend_id0 = (lane_id+ 3)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a3*((lane_id < 61 )? tx0: ty0);
        // process (4, 0, 0)
        friend_id0 = (lane_id+ 4)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a4*((lane_id < 60 )? tx0: ty0);
        // process (5, 0, 0)
        friend_id0 = (lane_id+ 5)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a5*((lane_id < 59 )? tx0: ty0);
        // process (6, 0, 0)
        friend_id0 = (lane_id+ 6)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a6*((lane_id < 58 )? tx0: ty0);
        
        OUT_1D(gid) = sum0;
        
    });
    fut.wait();
}

void Stencil_Hcc_Reg2_1Blk1Wf(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2,
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int n, int halo)
{
    extent<1> comp_domain(n/2); 
    tiled_extent<1> comp_tile(comp_domain, 256);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<1> tidx) restrict(amp) {
        unsigned int lane_id = __lane_id();
        unsigned int gid = (((tidx.global[0])>>6)<<7) + lane_id + halo;

        int warp_id0 = ((((tidx.global[0])>>6)<<7) + lane_id)>>6;

        DATA_TYPE reg0, reg1, reg2;
        int lane_id_it = lane_id;
        // load to regs: 
        int new_id0 ;
        new_id0 = (warp_id0<<6) + lane_id_it%70 ;
        reg0 = IN_1D(new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + (lane_id_it/70)*70 + lane_id_it%70 ;
        reg1 = IN_1D(new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + (lane_id_it/70)*70 + lane_id_it%70 ;
        new_id0 = (new_id0 < n+6)? new_id0 : n+5 ;
        reg2 = IN_1D(new_id0) ;

        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;
        int friend_id0;
        int friend_id1;
        DATA_TYPE tx0, ty0;
        DATA_TYPE tx1, ty1;

        // process (0, 0, 0)
        friend_id0 = (lane_id+ 0)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        sum0 += a0*(tx0);
        friend_id1 = (lane_id+ 0)&63 ;
        tx1 = __shfl(reg1, friend_id1);
        sum1 += a0*(tx1);
        // process (1, 0, 0)
        friend_id0 = (lane_id+ 1)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a1*((lane_id < 63 )? tx0: ty0);
        friend_id1 = (lane_id+ 1)&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a1*((lane_id < 63 )? tx1: ty1);
        // process (2, 0, 0)
        friend_id0 = (lane_id+ 2)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a2*((lane_id < 62 )? tx0: ty0);
        friend_id1 = (lane_id+ 2)&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a2*((lane_id < 62 )? tx1: ty1);
        // process (3, 0, 0)
        friend_id0 = (lane_id+ 3)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a3*((lane_id < 61 )? tx0: ty0);
        friend_id1 = (lane_id+ 3)&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a3*((lane_id < 61 )? tx1: ty1);
        // process (4, 0, 0)
        friend_id0 = (lane_id+ 4)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a4*((lane_id < 60 )? tx0: ty0);
        friend_id1 = (lane_id+ 4)&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a4*((lane_id < 60 )? tx1: ty1);
        // process (5, 0, 0)
        friend_id0 = (lane_id+ 5)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a5*((lane_id < 59 )? tx0: ty0);
        friend_id1 = (lane_id+ 5)&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a5*((lane_id < 59 )? tx1: ty1);
        // process (6, 0, 0)
        friend_id0 = (lane_id+ 6)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a6*((lane_id < 58 )? tx0: ty0);
        friend_id1 = (lane_id+ 6)&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a6*((lane_id < 58 )? tx1: ty1);

        OUT_1D(gid   ) = sum0;
        OUT_1D(gid+64) = sum1; 
        
    });
    fut.wait();
}

void Stencil_Hcc_Reg4_1Blk1Wf(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0, DATA_TYPE a1, DATA_TYPE a2,
        DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int n, int halo)
{
    extent<1> comp_domain(n/4); 
    tiled_extent<1> comp_tile(comp_domain, 256);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<1> tidx) restrict(amp) {
        unsigned int lane_id = __lane_id();
        unsigned int gid = (((tidx.global[0])>>6)<<8) + lane_id + halo;

        int warp_id0 = ((((tidx.global[0])>>6)<<8) + lane_id)>>6;

        int lane_id_it = lane_id;
        DATA_TYPE reg0, reg1, reg2, reg3, reg4;
        // load to regs: 
        int new_id0 ;
        new_id0 = (warp_id0<<6) + lane_id_it%70 ;
        reg0 = IN_1D(new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + (lane_id_it/70)*70 + lane_id_it%70 ;
        reg1 = IN_1D(new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + (lane_id_it/70)*70 + lane_id_it%70 ;
        reg2 = IN_1D(new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + (lane_id_it/70)*70 + lane_id_it%70 ;
        reg3 = IN_1D(new_id0) ;
        lane_id_it += 64 ;
        new_id0 = (warp_id0<<6) + (lane_id_it/70)*70 + lane_id_it%70 ;
        new_id0 = (new_id0 < n+6)? new_id0 : n+5 ;
        reg4 = IN_1D(new_id0) ;


        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;
        DATA_TYPE sum2 = 0.0;
        DATA_TYPE sum3 = 0.0;
        int friend_id0;
        int friend_id1;
        int friend_id2;
        int friend_id3;
        DATA_TYPE tx0, ty0, tx1, ty1;
        DATA_TYPE tx2, ty2, tx3, ty3;

        // process (0, 0, 0)
        friend_id0 = (lane_id+ 0)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        sum0 += a0 *(tx0);
        friend_id1 = (lane_id+ 0)&63 ;
        tx1 = __shfl(reg1, friend_id1);
        sum1 += a0 *(tx1);
        friend_id2 = (lane_id+ 0)&63 ;
        tx2 = __shfl(reg2, friend_id2);
        sum2 += a0 *(tx2);
        friend_id3 = (lane_id+ 0)&63 ;
        tx3 = __shfl(reg3, friend_id3);
        sum3 += a0 *(tx3);
        // process (1, 0, 0)
        friend_id0 = (lane_id+ 1)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a1 *((lane_id < 63 )? tx0: ty0);
        friend_id1 = (lane_id+ 1)&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a1 *((lane_id < 63 )? tx1: ty1);
        friend_id2 = (lane_id+ 1)&63 ;
        tx2 = __shfl(reg2, friend_id2);
        ty2 = __shfl(reg3, friend_id2);
        sum2 += a1 *((lane_id < 63 )? tx2: ty2);
        friend_id3 = (lane_id+ 1)&63 ;
        tx3 = __shfl(reg3, friend_id3);
        ty3 = __shfl(reg4, friend_id3);
        sum3 += a1 *((lane_id < 63 )? tx3: ty3);
        // process (2, 0, 0)
        friend_id0 = (lane_id+ 2)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a2 *((lane_id < 62 )? tx0: ty0);
        friend_id1 = (lane_id+ 2)&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a2 *((lane_id < 62 )? tx1: ty1);
        friend_id2 = (lane_id+ 2)&63 ;
        tx2 = __shfl(reg2, friend_id2);
        ty2 = __shfl(reg3, friend_id2);
        sum2 += a2 *((lane_id < 62 )? tx2: ty2);
        friend_id3 = (lane_id+ 2)&63 ;
        tx3 = __shfl(reg3, friend_id3);
        ty3 = __shfl(reg4, friend_id3);
        sum3 += a2 *((lane_id < 62 )? tx3: ty3);
        // process (3, 0, 0)
        friend_id0 = (lane_id+ 3)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a3 *((lane_id < 61 )? tx0: ty0);
        friend_id1 = (lane_id+ 3)&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a3 *((lane_id < 61 )? tx1: ty1);
        friend_id2 = (lane_id+ 3)&63 ;
        tx2 = __shfl(reg2, friend_id2);
        ty2 = __shfl(reg3, friend_id2);
        sum2 += a3 *((lane_id < 61 )? tx2: ty2);
        friend_id3 = (lane_id+ 3)&63 ;
        tx3 = __shfl(reg3, friend_id3);
        ty3 = __shfl(reg4, friend_id3);
        sum3 += a3 *((lane_id < 61 )? tx3: ty3);
        // process (4, 0, 0)
        friend_id0 = (lane_id+ 4)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a4 *((lane_id < 60 )? tx0: ty0);
        friend_id1 = (lane_id+ 4)&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a4 *((lane_id < 60 )? tx1: ty1);
        friend_id2 = (lane_id+ 4)&63 ;
        tx2 = __shfl(reg2, friend_id2);
        ty2 = __shfl(reg3, friend_id2);
        sum2 += a4 *((lane_id < 60 )? tx2: ty2);
        friend_id3 = (lane_id+ 4)&63 ;
        tx3 = __shfl(reg3, friend_id3);
        ty3 = __shfl(reg4, friend_id3);
        sum3 += a4 *((lane_id < 60 )? tx3: ty3);
        // process (5, 0, 0)
        friend_id0 = (lane_id+ 5)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a5 *((lane_id < 59 )? tx0: ty0);
        friend_id1 = (lane_id+ 5)&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a5 *((lane_id < 59 )? tx1: ty1);
        friend_id2 = (lane_id+ 5)&63 ;
        tx2 = __shfl(reg2, friend_id2);
        ty2 = __shfl(reg3, friend_id2);
        sum2 += a5 *((lane_id < 59 )? tx2: ty2);
        friend_id3 = (lane_id+ 5)&63 ;
        tx3 = __shfl(reg3, friend_id3);
        ty3 = __shfl(reg4, friend_id3);
        sum3 += a5 *((lane_id < 59 )? tx3: ty3);
        // process (6, 0, 0)
        friend_id0 = (lane_id+ 6)&63 ;
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a6 *((lane_id < 58 )? tx0: ty0);
        friend_id1 = (lane_id+ 6)&63 ;
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum1 += a6 *((lane_id < 58 )? tx1: ty1);
        friend_id2 = (lane_id+ 6)&63 ;
        tx2 = __shfl(reg2, friend_id2);
        ty2 = __shfl(reg3, friend_id2);
        sum2 += a6 *((lane_id < 58 )? tx2: ty2);
        friend_id3 = (lane_id+ 6)&63 ;
        tx3 = __shfl(reg3, friend_id3);
        ty3 = __shfl(reg4, friend_id3);
        sum3 += a6 *((lane_id < 58 )? tx3: ty3);

        OUT_1D(gid    ) = sum0; 
        OUT_1D(gid+64 ) = sum1; 
        OUT_1D(gid+128) = sum2; 
        OUT_1D(gid+192) = sum3; 

   
    });
    fut.wait();
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
        std::swap(in, out_ref);
    }
    std::swap(in, out_ref);
    // Show_Me(out_ref, n, halo, "Output:");

    extent<1> data_domain(total);
    array<DATA_TYPE>  in_d(data_domain);
    array<DATA_TYPE> out_d(data_domain);
    DATA_TYPE *out = new DATA_TYPE[total];
    float time_wo_pci;

    // Hcc version
    /////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_L1_1Blk(in_d, out_d,  
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    auto t2 = std::chrono::high_resolution_clock::now();
    double milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_L1_1Blk: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_L1_1Blk Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n, ITER, OPS_1D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc Shared Memory with Branch
    /////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Lds_1BlkBrc(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Lds_1BlkBrc: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Lds_1BlkBrc Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n, ITER, OPS_1D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc Shared Memory with Cyclic
    /////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Lds_1BlkCyc(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Lds_1BlkCyc: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Lds_1BlkCyc Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n, ITER, OPS_1D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc Shfl with 1D-Warp
    /////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Reg1_1Blk1Wf(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Reg1_1Blk1Wf: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Reg1_1Blk1Wf Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n, ITER, OPS_1D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc Shfl2 with 1D-Warp
    /////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Reg2_1Blk1Wf(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Reg2_1Blk1Wf: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Reg2_1Blk1Wf Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n, ITER, OPS_1D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc Shfl4 with 1D-Warp
    /////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Reg4_1Blk1Wf(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Reg4_1Blk1Wf: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Reg4_1Blk1Wf Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n, ITER, OPS_1D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));


  
    delete[] in;
    delete[] out;
    delete[] out_ref;

}
