#include <iostream>
#include <hc.hpp>
using namespace hc;
#define DATA_TYPE float
#define warpSize 64 
#define STEPS 100

#define START_TIME for(int _i = 0; _i < STEPS; _i++) {
#define END_TIME }

#define IN_2D(_x,_y) in[(_x)*(n+2*halo)+(_y)]
#define OUT_2D(_x,_y) out[(_x)*(n+2*halo)+(_y)]
#define LOC_2D(_x,_y) local[(_x)*(16+2*halo)+(_y)]

void Init_Input_2D(DATA_TYPE *in, int m, int n, int halo)
{
    srand(time(NULL));

    for(int i = 0; i < m+2*halo; i++)
    {
        for(int j = 0; j < n+2*halo; j++)
        {
            if(i<halo || j<halo || i>=m+halo || j>=n+halo)
                IN_2D(i,j) = 0.0;
            else
                IN_2D(i,j) = 1;//(DATA_TYPE)rand() * 100.0 / RAND_MAX;
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

void Stencil_Seq(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int halo)
{
    for(int j = halo; j < m+halo; j++)
    {
        for(int i = halo; i < n+halo; i++)
        {
            OUT_2D(j,i) = args[0] * IN_2D(j-1,i  ) +
                          args[1] * IN_2D(j  ,i-1) +
                          args[2] * IN_2D(j+1,i  ) +
                          args[3] * IN_2D(j  ,i+1) +
                          args[4] * IN_2D(j  ,i  ) ;
        }
    }

    for(int i = 0; i < m+2*halo; i++)
    {
        for(int j = 0; j < n+2*halo; j++)
        {
            if(i<halo || j<halo || i>=m+halo || j>=n+halo)
                OUT_2D(i,j) = 0.0;
        }
    }
}

void Stencil_Hcc(array<DATA_TYPE> &in, array<DATA_TYPE> &out, array<DATA_TYPE> &args, int m, int n, int halo) 
{
    extent<2> compute_domain(m+15, n+15); 
    tiled_extent<2> compute_tile(compute_domain, 16, 16);
    completion_future fut = parallel_for_each(compute_tile, [=, &in, &out, &args](tiled_index<2> tidx) restrict(amp) {
        int i = tidx.global[1] + halo;
        int j = tidx.global[0] + halo;
        if(j < m + halo && i < n + halo)
        {
            OUT_2D(j,i) = args[0]*IN_2D(j-1,i  ) + 
                          args[1]*IN_2D(j  ,i-1) + 
                          args[2]*IN_2D(j+1,i  ) +
                          args[3]*IN_2D(j  ,i+1) + 
                          args[4]*IN_2D(j  ,i  ) ;
            
        }
    });
    fut.wait();
}

void Stencil_Hcc_Sm(array<DATA_TYPE> &in, array<DATA_TYPE> &out, array<DATA_TYPE> &args, int m, int n, int halo) 
{
    extent<2> compute_domain(m+15, n+15); 
    tiled_extent<2> compute_tile(compute_domain, 16, 16);
    completion_future fut = parallel_for_each(compute_tile, [=, &in, &out, &args](tiled_index<2> tidx) restrict(amp) {
#define LM_SIZE 16 
        tile_static DATA_TYPE local[(LM_SIZE+2)*(LM_SIZE+2)];
        int i = tidx.global[1] + halo;
        int j = tidx.global[0] + halo;
        int li = tidx.local[1] + halo;
        int lj = tidx.local[0] + halo;
        LOC_2D(lj,li) = IN_2D(j,i);

        if(li == halo) LOC_2D(lj,li-1) = IN_2D(j,i-1);

        if(li == 16)   LOC_2D(lj,li+1) = IN_2D(j,i+1);
                
        if(lj == halo) LOC_2D(lj-1,li) = IN_2D(j-1,i);
                                                    
        if(lj == 16)   LOC_2D(lj+1,li) = IN_2D(j+1,i);

        tidx.barrier.wait();

        if(j < m+halo && i < n+halo)
        {
            
            OUT_2D(j,i) = args[0 ]*LOC_2D(lj-1,li  ) + 
                          args[1 ]*LOC_2D(lj  ,li-1) + 
                          args[2 ]*LOC_2D(lj+1,li  ) +
                          args[3 ]*LOC_2D(lj  ,li+1) + 
                          args[4 ]*LOC_2D(lj  ,li  ) ;
        }
        
    });
    fut.wait();
}

void Stencil_Hcc_Shfl(array<DATA_TYPE> &in, array<DATA_TYPE> &out, array<DATA_TYPE> &args, int m, int n, int halo) 
{
    extent<2> compute_domain(m+31, n+7); 
    tiled_extent<2> compute_tile(compute_domain, 32, 8);
    completion_future fut = parallel_for_each(compute_tile, [=, &in, &out, &args](tiled_index<2> tidx) restrict(amp) {
        int i = tidx.global[1] + halo;
        int j = tidx.global[0] + halo;
        int lane_id = __lane_id();
        int warp_id_x = (tidx.global[1])>>3;
        int warp_id_y = (tidx.global[0])>>3;
        int new_i = (warp_id_x<<3) + lane_id%10;
        int new_j = (warp_id_y<<3) + lane_id/10;
        DATA_TYPE threadInput0, threadInput1;
        threadInput0 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+4)%10;
        new_j = (warp_id_y<<3) + 6 + (lane_id+4)/10;
        if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput1 = IN_2D(new_j, new_i);
        DATA_TYPE sum = 0.0;
        int friend_id;
        // northwestern
        friend_id = (lane_id+((lane_id>>3)<<1))&(warpSize-1);
        DATA_TYPE tx, ty;
        // tx = args[0]*__shfl(threadInput0, friend_id, 64);
        // ty = args[0]*__shfl(threadInput1, friend_id, 64);
        // sum += (lane_id < 52)? tx: ty;

        // northern
        friend_id = (friend_id+1)&(warpSize-1);
        tx = args[0]*__shfl(threadInput0, friend_id, 64);
        ty = args[0]*__shfl(threadInput1, friend_id, 64);
        sum += (lane_id < 51)? tx: ty;

        // northeastern 
        friend_id = (friend_id+1)&(warpSize-1);
        // tx = args[2]*__shfl(threadInput0, friend_id, 64);
        // ty = args[2]*__shfl(threadInput1, friend_id, 64);
        // sum += (lane_id < 50)? tx: ty;

        // western 
        friend_id = (friend_id+8)&(warpSize-1);
        tx = args[1]*__shfl(threadInput0, friend_id, 64);
        ty = args[1]*__shfl(threadInput1, friend_id, 64);
        sum += (lane_id < 44)? tx: ty;

        // central 
        friend_id = (friend_id+1)&(warpSize-1);
        tx = args[4]*__shfl(threadInput0, friend_id, 64);
        ty = args[4]*__shfl(threadInput1, friend_id, 64);
        sum += (lane_id < 43)? tx: ty;

        // eastern 
        friend_id = (friend_id+1)&(warpSize-1);
        tx = args[3]*__shfl(threadInput0, friend_id, 64);
        ty = args[3]*__shfl(threadInput1, friend_id, 64);
        sum += (lane_id < 42)? tx: ty;

        // southwestern 
        friend_id = (friend_id+8)&(warpSize-1);
        // tx = args[6]*__shfl(threadInput0, friend_id, 64);
        // ty = args[6]*__shfl(threadInput1, friend_id, 64);
        // sum += (lane_id < 36)? tx: ty;

        // southern
        friend_id = (friend_id+1)&(warpSize-1);
        tx = args[2]*__shfl(threadInput0, friend_id, 64);
        ty = args[2]*__shfl(threadInput1, friend_id, 64);
        sum += (lane_id < 35)? tx: ty;

        // southeastern
        // friend_id = (friend_id+1)&(warpSize-1);
        // tx = args[8]*__shfl(threadInput0, friend_id, 64);
        // ty = args[8]*__shfl(threadInput1, friend_id, 64);
        // sum += (lane_id < 34)? tx: ty;

        if(j < m + halo && i < n + halo)
        {
            OUT_2D(j,i) = sum;
        }

    });
    fut.wait();
}

void Stencil_Hcc_Shfl2(array<DATA_TYPE> &in, array<DATA_TYPE> &out, array<DATA_TYPE> &args, int m, int n, int halo) 
{
    extent<2> compute_domain((m+1)/2+31, n+7); 
    tiled_extent<2> compute_tile(compute_domain, 32, 8);
    completion_future fut = parallel_for_each(compute_tile, [=, &in, &out, &args](tiled_index<2> tidx) restrict(amp) {
        int lane_id = __lane_id();
        int i = tidx.global[1] + halo;
        int j = ((tidx.global[0]>>3)<<4) + (lane_id>>3) + halo;
        int warp_id_x = (tidx.global[1])>>3;
        int warp_id_y = (((tidx.global[0]>>3)<<4) + (lane_id>>3))>>3;
        int new_i = (warp_id_x<<3) + lane_id%10;
        int new_j = (warp_id_y<<3) + lane_id/10;
        DATA_TYPE threadInput0, threadInput1, threadInput2;
        threadInput0 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+4)%10;
        new_j = (warp_id_y<<3) + 6 + (lane_id+4)/10;
        threadInput1 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+8)%10;
        new_j = (warp_id_y<<3) + 12 + (lane_id+8)/10;
        if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput2 = IN_2D(new_j, new_i);
        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;
        int friend_id0, friend_id1;
        // northwestern
        friend_id0 = (lane_id+   ((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
        DATA_TYPE tx0, ty0, tx1, ty1;
        // tx0 = args[0]*__shfl(threadInput0, friend_id0, 64);
        // ty0 = args[0]*__shfl(threadInput1, friend_id0, 64);
        // tx1 = args[0]*__shfl(threadInput1, friend_id1, 64);
        // ty1 = args[0]*__shfl(threadInput2, friend_id1, 64);
        // sum0 += (lane_id < 52)? tx0: ty0;
        // sum1 += (lane_id < 40)? tx1: ty1;

        // northern
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        tx0 = args[0]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[0]*__shfl(threadInput1, friend_id0, 64);
        tx1 = args[0]*__shfl(threadInput1, friend_id1, 64);
        ty1 = args[0]*__shfl(threadInput2, friend_id1, 64);
        sum0 += (lane_id < 51)? tx0: ty0;
        sum1 += (lane_id < 39)? tx1: ty1;

        // northeastern 
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        // tx0 = args[2]*__shfl(threadInput0, friend_id0, 64);
        // ty0 = args[2]*__shfl(threadInput1, friend_id0, 64);
        // tx1 = args[2]*__shfl(threadInput1, friend_id1, 64);
        // ty1 = args[2]*__shfl(threadInput2, friend_id1, 64);
        // sum0 += (lane_id < 50)? tx0: ty0;
        // sum1 += (lane_id < 38)? tx1: ty1;

        // western 
        friend_id0 = (friend_id0+8)&(warpSize-1);
        friend_id1 = (friend_id1+8)&(warpSize-1);
        tx0 = args[1]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[1]*__shfl(threadInput1, friend_id0, 64);
        tx1 = args[1]*__shfl(threadInput1, friend_id1, 64);
        ty1 = args[1]*__shfl(threadInput2, friend_id1, 64);
        sum0 += (lane_id < 44)? tx0: ty0;
        sum1 += (lane_id < 32)? tx1: ty1;

        // central
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        tx0 = args[4]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[4]*__shfl(threadInput1, friend_id0, 64);
        tx1 = args[4]*__shfl(threadInput1, friend_id1, 64);
        ty1 = args[4]*__shfl(threadInput2, friend_id1, 64);
        sum0 += (lane_id < 43)? tx0: ty0;
        sum1 += (lane_id < 31)? tx1: ty1;

        // eastern
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        tx0 = args[3]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[3]*__shfl(threadInput1, friend_id0, 64);
        tx1 = args[3]*__shfl(threadInput1, friend_id1, 64);
        ty1 = args[3]*__shfl(threadInput2, friend_id1, 64);
        sum0 += (lane_id < 42)? tx0: ty0;
        sum1 += (lane_id < 30)? tx1: ty1;

        // southeastern 
        friend_id0 = (friend_id0+8)&(warpSize-1);
        friend_id1 = (friend_id1+8)&(warpSize-1);
        // tx0 = args[6]*__shfl(threadInput0, friend_id0, 64);
        // ty0 = args[6]*__shfl(threadInput1, friend_id0, 64);
        // tx1 = args[6]*__shfl(threadInput1, friend_id1, 64);
        // ty1 = args[6]*__shfl(threadInput2, friend_id1, 64);
        // sum0 += (lane_id < 36)? tx0: ty0;
        // sum1 += (lane_id < 24)? tx1: ty1;

        // southern
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        tx0 = args[2]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[2]*__shfl(threadInput1, friend_id0, 64);
        tx1 = args[2]*__shfl(threadInput1, friend_id1, 64);
        ty1 = args[2]*__shfl(threadInput2, friend_id1, 64);
        sum0 += (lane_id < 35)? tx0: ty0;
        sum1 += (lane_id < 23)? tx1: ty1;

        // southeastern
        // friend_id0 = (friend_id0+1)&(warpSize-1);
        // friend_id1 = (friend_id1+1)&(warpSize-1);
        // tx0 = args[8]*__shfl(threadInput0, friend_id0, 64);
        // ty0 = args[8]*__shfl(threadInput1, friend_id0, 64);
        // tx1 = args[8]*__shfl(threadInput1, friend_id1, 64);
        // ty1 = args[8]*__shfl(threadInput2, friend_id1, 64);
        // sum0 += (lane_id < 34)? tx0: ty0;
        // sum1 += (lane_id < 22)? tx1: ty1;

        if(j < m + halo && i < n + halo)
            OUT_2D(j,i) = sum0;
        if(j+8 < m + halo && i < n + halo)
            OUT_2D(j+8,i) = sum1;

    });
    fut.wait();
}


void Stencil_Hcc_Shfl4(array<DATA_TYPE> &in, array<DATA_TYPE> &out, array<DATA_TYPE> &args, int m, int n, int halo) 
{
    extent<2> compute_domain((m+3)/4+31, n+7); 
    tiled_extent<2> compute_tile(compute_domain, 32, 8);
    completion_future fut = parallel_for_each(compute_tile, [=, &in, &out, &args](tiled_index<2> tidx) restrict(amp) {
        int lane_id = __lane_id();
        int i = tidx.global[1] + halo;
        int j = ((tidx.global[0]>>3)<<5) + (lane_id>>3) + halo;
        int warp_id_x = (tidx.global[1])>>3;
        int warp_id_y = (((tidx.global[0]>>3)<<5) + (lane_id>>3))>>3;
        int new_i = (warp_id_x<<3) + lane_id%10;
        int new_j = (warp_id_y<<3) + lane_id/10;
        DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5;
        threadInput0 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+4)%10;
        new_j = (warp_id_y<<3) + 6 + (lane_id+4)/10;
        threadInput1 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+8)%10;
        new_j = (warp_id_y<<3) + 12 + (lane_id+8)/10;
        threadInput2 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+2)%10;
        new_j = (warp_id_y<<3) + 19 + (lane_id+2)/10;
        threadInput3 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+6)%10;
        new_j = (warp_id_y<<3) + 25 + (lane_id+6)/10;
        threadInput4 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id)%10;
        new_j = (warp_id_y<<3) + 32 + (lane_id)/10;
        if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput5 = IN_2D(new_j, new_i);

        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;
        DATA_TYPE sum2 = 0.0;
        DATA_TYPE sum3 = 0.0;
        int friend_id0, friend_id1, friend_id2, friend_id3;
        // northwestern
        friend_id0 = (lane_id+   ((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+32+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+48+((lane_id>>3)<<1))&(warpSize-1);
        DATA_TYPE tx0, ty0, tx1, ty1, tx2, ty2, tz2, tx3, ty3, tz3;
        // tx0 = args[0]*__shfl(threadInput0, friend_id0, 64);
        // ty0 = args[0]*__shfl(threadInput1, friend_id0, 64);
        // tx1 = args[0]*__shfl(threadInput1, friend_id1, 64);
        // ty1 = args[0]*__shfl(threadInput2, friend_id1, 64);
        // tx2 = args[0]*__shfl(threadInput2, friend_id2, 64);
        // ty2 = args[0]*__shfl(threadInput3, friend_id2, 64);
        // tx3 = args[0]*__shfl(threadInput3, friend_id3, 64);
        // ty3 = args[0]*__shfl(threadInput4, friend_id3, 64);
        // sum0 += (lane_id < 52)? tx0: ty0;
        // sum1 += (lane_id < 40)? tx1: ty1;
        // sum2 += (lane_id < 26)? tx2: ty2;
        // sum3 += (lane_id < 14)? tx3: ty3;

        // northern
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        friend_id2 = (friend_id2+1)&(warpSize-1);
        friend_id3 = (friend_id3+1)&(warpSize-1);
        tx0 = args[0]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[0]*__shfl(threadInput1, friend_id0, 64);
        tx1 = args[0]*__shfl(threadInput1, friend_id1, 64);
        ty1 = args[0]*__shfl(threadInput2, friend_id1, 64);
        tx2 = args[0]*__shfl(threadInput2, friend_id2, 64);
        ty2 = args[0]*__shfl(threadInput3, friend_id2, 64);
        tx3 = args[0]*__shfl(threadInput3, friend_id3, 64);
        ty3 = args[0]*__shfl(threadInput4, friend_id3, 64);
        sum0 += (lane_id < 51)? tx0: ty0;
        sum1 += (lane_id < 39)? tx1: ty1;
        sum2 += (lane_id < 25)? tx2: ty2;
        sum3 += (lane_id < 13)? tx3: ty3;

        // northeastern 
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        friend_id2 = (friend_id2+1)&(warpSize-1);
        friend_id3 = (friend_id3+1)&(warpSize-1);
        // tx0 = args[2]*__shfl(threadInput0, friend_id0, 64);
        // ty0 = args[2]*__shfl(threadInput1, friend_id0, 64);
        // tx1 = args[2]*__shfl(threadInput1, friend_id1, 64);
        // ty1 = args[2]*__shfl(threadInput2, friend_id1, 64);
        // tx2 = args[2]*__shfl(threadInput2, friend_id2, 64);
        // ty2 = args[2]*__shfl(threadInput3, friend_id2, 64);
        // tx3 = args[2]*__shfl(threadInput3, friend_id3, 64);
        // ty3 = args[2]*__shfl(threadInput4, friend_id3, 64);
        // sum0 += (lane_id < 50)? tx0: ty0;
        // sum1 += (lane_id < 38)? tx1: ty1;
        // sum2 += (lane_id < 24)? tx2: ty2;
        // sum3 += (lane_id < 12)? tx3: ty3;

        // western 
        friend_id0 = (friend_id0+8)&(warpSize-1);
        friend_id1 = (friend_id1+8)&(warpSize-1);
        friend_id2 = (friend_id2+8)&(warpSize-1);
        friend_id3 = (friend_id3+8)&(warpSize-1);
        tx0 = args[1]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[1]*__shfl(threadInput1, friend_id0, 64);
        tx1 = args[1]*__shfl(threadInput1, friend_id1, 64);
        ty1 = args[1]*__shfl(threadInput2, friend_id1, 64);
        tx2 = args[1]*__shfl(threadInput2, friend_id2, 64);
        ty2 = args[1]*__shfl(threadInput3, friend_id2, 64);
        tx3 = args[1]*__shfl(threadInput3, friend_id3, 64);
        ty3 = args[1]*__shfl(threadInput4, friend_id3, 64);
        tz3 = args[1]*__shfl(threadInput5, friend_id3, 64);
        sum0 += (lane_id < 44)? tx0: ty0;
        sum1 += (lane_id < 32)? tx1: ty1;
        sum2 += (lane_id < 18)? tx2: ty2;
        sum3 += (lane_id < 6 )? tx3: ((lane_id < 56)? ty3: tz3);

        // central
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        friend_id2 = (friend_id2+1)&(warpSize-1);
        friend_id3 = (friend_id3+1)&(warpSize-1);
        tx0 = args[4]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[4]*__shfl(threadInput1, friend_id0, 64);
        tx1 = args[4]*__shfl(threadInput1, friend_id1, 64);
        ty1 = args[4]*__shfl(threadInput2, friend_id1, 64);
        tx2 = args[4]*__shfl(threadInput2, friend_id2, 64);
        ty2 = args[4]*__shfl(threadInput3, friend_id2, 64);
        tx3 = args[4]*__shfl(threadInput3, friend_id3, 64);
        ty3 = args[4]*__shfl(threadInput4, friend_id3, 64);
        tz3 = args[4]*__shfl(threadInput5, friend_id3, 64);
        sum0 += (lane_id < 43)? tx0: ty0;
        sum1 += (lane_id < 31)? tx1: ty1;
        sum2 += (lane_id < 17)? tx2: ty2;
        sum3 += (lane_id < 5 )? tx3: ((lane_id < 56)? ty3: tz3);

        // eastern
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        friend_id2 = (friend_id2+1)&(warpSize-1);
        friend_id3 = (friend_id3+1)&(warpSize-1);
        tx0 = args[3]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[3]*__shfl(threadInput1, friend_id0, 64);
        tx1 = args[3]*__shfl(threadInput1, friend_id1, 64);
        ty1 = args[3]*__shfl(threadInput2, friend_id1, 64);
        tx2 = args[3]*__shfl(threadInput2, friend_id2, 64);
        ty2 = args[3]*__shfl(threadInput3, friend_id2, 64);
        tx3 = args[3]*__shfl(threadInput3, friend_id3, 64);
        ty3 = args[3]*__shfl(threadInput4, friend_id3, 64);
        tz3 = args[3]*__shfl(threadInput5, friend_id3, 64);
        sum0 += (lane_id < 42)? tx0: ty0;
        sum1 += (lane_id < 30)? tx1: ty1;
        sum2 += (lane_id < 16)? tx2: ty2;
        sum3 += (lane_id < 4 )? tx3: ((lane_id < 56)? ty3: tz3);

        // southeastern 
        friend_id0 = (friend_id0+8)&(warpSize-1);
        friend_id1 = (friend_id1+8)&(warpSize-1);
        friend_id2 = (friend_id2+8)&(warpSize-1);
        friend_id3 = (friend_id3+8)&(warpSize-1);
        // tx0 = args[6]*__shfl(threadInput0, friend_id0, 64);
        // ty0 = args[6]*__shfl(threadInput1, friend_id0, 64);
        // tx1 = args[6]*__shfl(threadInput1, friend_id1, 64);
        // ty1 = args[6]*__shfl(threadInput2, friend_id1, 64);
        // tx2 = args[6]*__shfl(threadInput2, friend_id2, 64);
        // ty2 = args[6]*__shfl(threadInput3, friend_id2, 64);
        // tz2 = args[6]*__shfl(threadInput4, friend_id2, 64);
        // tx3 = args[6]*__shfl(threadInput4, friend_id3, 64);
        // ty3 = args[6]*__shfl(threadInput5, friend_id3, 64);
        // sum0 += (lane_id < 36)? tx0: ty0;
        // sum1 += (lane_id < 24)? tx1: ty1;
        // sum2 += (lane_id < 10)? tx2: ((lane_id < 62)? ty2: tz2);
        // sum3 += (lane_id < 48)? tx3: ty3;

        // southern
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        friend_id2 = (friend_id2+1)&(warpSize-1);
        friend_id3 = (friend_id3+1)&(warpSize-1);
        tx0 = args[2]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[2]*__shfl(threadInput1, friend_id0, 64);
        tx1 = args[2]*__shfl(threadInput1, friend_id1, 64);
        ty1 = args[2]*__shfl(threadInput2, friend_id1, 64);
        tx2 = args[2]*__shfl(threadInput2, friend_id2, 64);
        ty2 = args[2]*__shfl(threadInput3, friend_id2, 64);
        tz2 = args[2]*__shfl(threadInput4, friend_id2, 64);
        tx3 = args[2]*__shfl(threadInput4, friend_id3, 64);
        ty3 = args[2]*__shfl(threadInput5, friend_id3, 64);
        sum0 += (lane_id < 35)? tx0: ty0;
        sum1 += (lane_id < 23)? tx1: ty1;
        sum2 += (lane_id < 9 )? tx2: ((lane_id < 61)? ty2: tz2);
        sum3 += (lane_id < 48)? tx3: ty3;

        // southeastern
        // friend_id0 = (friend_id0+1)&(warpSize-1);
        // friend_id1 = (friend_id1+1)&(warpSize-1);
        // friend_id2 = (friend_id2+1)&(warpSize-1);
        // friend_id3 = (friend_id3+1)&(warpSize-1);
        // tx0 = args[8]*__shfl(threadInput0, friend_id0, 64);
        // ty0 = args[8]*__shfl(threadInput1, friend_id0, 64);
        // tx1 = args[8]*__shfl(threadInput1, friend_id1, 64);
        // ty1 = args[8]*__shfl(threadInput2, friend_id1, 64);
        // tx2 = args[8]*__shfl(threadInput2, friend_id2, 64);
        // ty2 = args[8]*__shfl(threadInput3, friend_id2, 64);
        // tz2 = args[8]*__shfl(threadInput4, friend_id2, 64);
        // tx3 = args[8]*__shfl(threadInput4, friend_id3, 64);
        // ty3 = args[8]*__shfl(threadInput5, friend_id3, 64);
        // sum0 += (lane_id < 34)? tx0: ty0;
        // sum1 += (lane_id < 22)? tx1: ty1;
        // sum2 += (lane_id < 8 )? tx2: ((lane_id < 60)? ty2: tz2);
        // sum3 += (lane_id < 48)? tx3: ty3;
        /*
        */

        if(j < m + halo && i < n + halo)
            OUT_2D(j,i) = sum0;
        if(j+8 < m + halo && i < n + halo)
            OUT_2D(j+8,i) = sum1;
        if(j+16 < m + halo && i < n + halo)
            OUT_2D(j+16,i) = sum2;
        if(j+24 < m + halo && i < n + halo)
            OUT_2D(j+24,i) = sum3;

    });
    fut.wait();
}

void Stencil_Hcc_Shfl4_2(array<DATA_TYPE> &in, array<DATA_TYPE> &out, array<DATA_TYPE> &args, int m, int n, int halo) 
{
    extent<2> compute_domain((m+1)/2+31, (n+1)/2+7); 
    // std::cout << "a:" << (m+1)/2+31 << std::endl;
    // std::cout << "b:" << (n+1)/2+7 << std::endl;
    tiled_extent<2> compute_tile(compute_domain, 32, 8);
    completion_future fut = parallel_for_each(compute_tile, [=, &in, &out, &args](tiled_index<2> tidx) restrict(amp) {
        int lane_id = __lane_id();
        int i = ((tidx.global[1]>>3)<<4) + (lane_id&7) + halo;
        int j = ((tidx.global[0]>>3)<<4) + (lane_id>>3) + halo;
        int warp_id_x = (((tidx.global[1]>>3)<<4) + (lane_id&7))>>3;
        int warp_id_y = (((tidx.global[0]>>3)<<4) + (lane_id>>3))>>3;
        int new_i = (warp_id_x<<3) + lane_id%18;
        int new_j = (warp_id_y<<3) + lane_id/18;
        DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5;
        // if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput0 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+10)%18;
        new_j = (warp_id_y<<3) + 3 + (lane_id+10)/18;
        // if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput1 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+2)%18;
        new_j = (warp_id_y<<3) + 7 + (lane_id+2)/18;
        // if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput2 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+12)%18;
        new_j = (warp_id_y<<3) + 10 + (lane_id+12)/18;
        // if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput3 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+4)%18;
        new_j = (warp_id_y<<3) + 14 + (lane_id+4)/18;
        // if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput4 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+14)%18;
        new_j = (warp_id_y<<3) + 17 + (lane_id+14)/18;
        if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput5 = IN_2D(new_j, new_i);

        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;
        DATA_TYPE sum2 = 0.0;
        DATA_TYPE sum3 = 0.0;
        int friend_id0, friend_id1, friend_id2, friend_id3;
        // northwestern
        friend_id0 = (lane_id+   ((lane_id>>3)*10))&(warpSize-1);
        friend_id1 = (lane_id+ 8+((lane_id>>3)*10))&(warpSize-1);
        friend_id2 = (lane_id+16+((lane_id>>3)*10))&(warpSize-1);
        friend_id3 = (lane_id+24+((lane_id>>3)*10))&(warpSize-1);
        DATA_TYPE tx0, ty0, tz0, tx1, ty1, tz1, tx2, ty2, tz2, tx3, ty3, tz3, ta3;
        // tx0 = args[0]*__shfl(threadInput0, friend_id0, 64);
        // ty0 = args[0]*__shfl(threadInput1, friend_id0, 64);
        // tz0 = args[0]*__shfl(threadInput2, friend_id0, 64);
        // tx1 = args[0]*__shfl(threadInput0, friend_id1, 64);
        // ty1 = args[0]*__shfl(threadInput1, friend_id1, 64);
        // tz1 = args[0]*__shfl(threadInput2, friend_id1, 64);
        // tx2 = args[0]*__shfl(threadInput2, friend_id2, 64);
        // ty2 = args[0]*__shfl(threadInput3, friend_id2, 64);
        // tz2 = args[0]*__shfl(threadInput4, friend_id2, 64);
        // tx3 = args[0]*__shfl(threadInput2, friend_id3, 64);
        // ty3 = args[0]*__shfl(threadInput3, friend_id3, 64);
        // tz3 = args[0]*__shfl(threadInput4, friend_id3, 64);
        // sum0 += (lane_id < 32)? tx0: ((lane_id < 58)? ty0: tz0);
        // sum1 += (lane_id < 26)? tx1: ((lane_id < 56)? ty1: tz1);
        // sum2 += (lane_id < 24)? tx2: ((lane_id < 52)? ty2: tz2);
        // sum3 += (lane_id < 20)? tx3: ((lane_id < 48)? ty3: tz3);

        // northern
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        friend_id2 = (friend_id2+1)&(warpSize-1);
        friend_id3 = (friend_id3+1)&(warpSize-1);
        tx0 = args[0]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[0]*__shfl(threadInput1, friend_id0, 64);
        tz0 = args[0]*__shfl(threadInput2, friend_id0, 64);
        tx1 = args[0]*__shfl(threadInput0, friend_id1, 64);
        ty1 = args[0]*__shfl(threadInput1, friend_id1, 64);
        tz1 = args[0]*__shfl(threadInput2, friend_id1, 64);
        tx2 = args[0]*__shfl(threadInput2, friend_id2, 64);
        ty2 = args[0]*__shfl(threadInput3, friend_id2, 64);
        tz2 = args[0]*__shfl(threadInput4, friend_id2, 64);
        tx3 = args[0]*__shfl(threadInput2, friend_id3, 64);
        ty3 = args[0]*__shfl(threadInput3, friend_id3, 64);
        tz3 = args[0]*__shfl(threadInput4, friend_id3, 64);
        sum0 += (lane_id < 32)? tx0: ((lane_id < 57)? ty0: tz0);
        sum1 += (lane_id < 25)? tx1: ((lane_id < 56)? ty1: tz1);
        sum2 += (lane_id < 24)? tx2: ((lane_id < 51)? ty2: tz2);
        sum3 += (lane_id < 19)? tx3: ((lane_id < 48)? ty3: tz3);

        // northeastern 
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        friend_id2 = (friend_id2+1)&(warpSize-1);
        friend_id3 = (friend_id3+1)&(warpSize-1);
        // tx0 = args[2]*__shfl(threadInput0, friend_id0, 64);
        // ty0 = args[2]*__shfl(threadInput1, friend_id0, 64);
        // tz0 = args[2]*__shfl(threadInput2, friend_id0, 64);
        // tx1 = args[2]*__shfl(threadInput0, friend_id1, 64);
        // ty1 = args[2]*__shfl(threadInput1, friend_id1, 64);
        // tz1 = args[2]*__shfl(threadInput2, friend_id1, 64);
        // tx2 = args[2]*__shfl(threadInput2, friend_id2, 64);
        // ty2 = args[2]*__shfl(threadInput3, friend_id2, 64);
        // tz2 = args[2]*__shfl(threadInput4, friend_id2, 64);
        // tx3 = args[2]*__shfl(threadInput2, friend_id3, 64);
        // ty3 = args[2]*__shfl(threadInput3, friend_id3, 64);
        // tz3 = args[2]*__shfl(threadInput4, friend_id3, 64);
        // sum0 += (lane_id < 32)? tx0: ((lane_id < 56)? ty0: tz0);
        // sum1 += (lane_id < 24)? tx1: ((lane_id < 56)? ty1: tz1);
        // sum2 += (lane_id < 24)? tx2: ((lane_id < 50)? ty2: tz2);
        // sum3 += (lane_id < 18)? tx3: ((lane_id < 48)? ty3: tz3);

        // western 
        friend_id0 = (friend_id0+16)&(warpSize-1);
        friend_id1 = (friend_id1+16)&(warpSize-1);
        friend_id2 = (friend_id2+16)&(warpSize-1);
        friend_id3 = (friend_id3+16)&(warpSize-1);
        tx0 = args[1]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[1]*__shfl(threadInput1, friend_id0, 64);
        tz0 = args[1]*__shfl(threadInput2, friend_id0, 64);
        tx1 = args[1]*__shfl(threadInput0, friend_id1, 64);
        ty1 = args[1]*__shfl(threadInput1, friend_id1, 64);
        tz1 = args[1]*__shfl(threadInput2, friend_id1, 64);
        tx2 = args[1]*__shfl(threadInput2, friend_id2, 64);
        ty2 = args[1]*__shfl(threadInput3, friend_id2, 64);
        tz2 = args[1]*__shfl(threadInput4, friend_id2, 64);
        tx3 = args[1]*__shfl(threadInput2, friend_id3, 64);
        ty3 = args[1]*__shfl(threadInput3, friend_id3, 64);
        tz3 = args[1]*__shfl(threadInput4, friend_id3, 64);
        sum0 += (lane_id < 24)? tx0: ((lane_id < 50)? ty0: tz0);
        sum1 += (lane_id < 18)? tx1: ((lane_id < 48)? ty1: tz1);
        sum2 += (lane_id < 16)? tx2: ((lane_id < 44)? ty2: tz2);
        sum3 += (lane_id < 12)? tx3: ((lane_id < 40)? ty3: tz3);

        // central
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        friend_id2 = (friend_id2+1)&(warpSize-1);
        friend_id3 = (friend_id3+1)&(warpSize-1);
        tx0 = args[4]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[4]*__shfl(threadInput1, friend_id0, 64);
        tz0 = args[4]*__shfl(threadInput2, friend_id0, 64);
        tx1 = args[4]*__shfl(threadInput0, friend_id1, 64);
        ty1 = args[4]*__shfl(threadInput1, friend_id1, 64);
        tz1 = args[4]*__shfl(threadInput2, friend_id1, 64);
        tx2 = args[4]*__shfl(threadInput2, friend_id2, 64);
        ty2 = args[4]*__shfl(threadInput3, friend_id2, 64);
        tz2 = args[4]*__shfl(threadInput4, friend_id2, 64);
        tx3 = args[4]*__shfl(threadInput2, friend_id3, 64);
        ty3 = args[4]*__shfl(threadInput3, friend_id3, 64);
        tz3 = args[4]*__shfl(threadInput4, friend_id3, 64);
        sum0 += (lane_id < 24)? tx0: ((lane_id < 49)? ty0: tz0);
        sum1 += (lane_id < 17)? tx1: ((lane_id < 48)? ty1: tz1);
        sum2 += (lane_id < 16)? tx2: ((lane_id < 43)? ty2: tz2);
        sum3 += (lane_id < 11)? tx3: ((lane_id < 40)? ty3: tz3);

        // eastern
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        friend_id2 = (friend_id2+1)&(warpSize-1);
        friend_id3 = (friend_id3+1)&(warpSize-1);
        tx0 = args[3]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[3]*__shfl(threadInput1, friend_id0, 64);
        tz0 = args[3]*__shfl(threadInput2, friend_id0, 64);
        tx1 = args[3]*__shfl(threadInput0, friend_id1, 64);
        ty1 = args[3]*__shfl(threadInput1, friend_id1, 64);
        tz1 = args[3]*__shfl(threadInput2, friend_id1, 64);
        tx2 = args[3]*__shfl(threadInput2, friend_id2, 64);
        ty2 = args[3]*__shfl(threadInput3, friend_id2, 64);
        tz2 = args[3]*__shfl(threadInput4, friend_id2, 64);
        tx3 = args[3]*__shfl(threadInput2, friend_id3, 64);
        ty3 = args[3]*__shfl(threadInput3, friend_id3, 64);
        tz3 = args[3]*__shfl(threadInput4, friend_id3, 64);
        sum0 += (lane_id < 24)? tx0: ((lane_id < 48)? ty0: tz0);
        sum1 += (lane_id < 16)? tx1: ((lane_id < 48)? ty1: tz1);
        sum2 += (lane_id < 16)? tx2: ((lane_id < 42)? ty2: tz2);
        sum3 += (lane_id < 10)? tx3: ((lane_id < 40)? ty3: tz3);

        // southeastern 
        friend_id0 = (friend_id0+16)&(warpSize-1);
        friend_id1 = (friend_id1+16)&(warpSize-1);
        friend_id2 = (friend_id2+16)&(warpSize-1);
        friend_id3 = (friend_id3+16)&(warpSize-1);
        // tx0 = args[6]*__shfl(threadInput0, friend_id0, 64);
        // ty0 = args[6]*__shfl(threadInput1, friend_id0, 64);
        // tz0 = args[6]*__shfl(threadInput2, friend_id0, 64);
        // tx1 = args[6]*__shfl(threadInput0, friend_id1, 64);
        // ty1 = args[6]*__shfl(threadInput1, friend_id1, 64);
        // tz1 = args[6]*__shfl(threadInput2, friend_id1, 64);
        // tx2 = args[6]*__shfl(threadInput2, friend_id2, 64);
        // ty2 = args[6]*__shfl(threadInput3, friend_id2, 64);
        // tz2 = args[6]*__shfl(threadInput4, friend_id2, 64);
        // tx3 = args[6]*__shfl(threadInput2, friend_id3, 64);
        // ty3 = args[6]*__shfl(threadInput3, friend_id3, 64);
        // tz3 = args[6]*__shfl(threadInput4, friend_id3, 64);
        // ta3 = args[6]*__shfl(threadInput5, friend_id3, 64);
        // sum0 += (lane_id < 16)? tx0: ((lane_id < 42)? ty0: tz0);
        // sum1 += (lane_id < 10)? tx1: ((lane_id < 40)? ty1: tz1);
        // sum2 += (lane_id < 8 )? tx2: ((lane_id < 36)? ty2: tz2);
        // sum3 += (lane_id < 4 )? tx3: ((lane_id < 32)? ty3: ((lane_id < 62)? tz3: ta3));

        // southern
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        friend_id2 = (friend_id2+1)&(warpSize-1);
        friend_id3 = (friend_id3+1)&(warpSize-1);
        tx0 = args[2]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[2]*__shfl(threadInput1, friend_id0, 64);
        tz0 = args[2]*__shfl(threadInput2, friend_id0, 64);
        tx1 = args[2]*__shfl(threadInput0, friend_id1, 64);
        ty1 = args[2]*__shfl(threadInput1, friend_id1, 64);
        tz1 = args[2]*__shfl(threadInput2, friend_id1, 64);
        tx2 = args[2]*__shfl(threadInput2, friend_id2, 64);
        ty2 = args[2]*__shfl(threadInput3, friend_id2, 64);
        tz2 = args[2]*__shfl(threadInput4, friend_id2, 64);
        tx3 = args[2]*__shfl(threadInput2, friend_id3, 64);
        ty3 = args[2]*__shfl(threadInput3, friend_id3, 64);
        tz3 = args[2]*__shfl(threadInput4, friend_id3, 64);
        ta3 = args[2]*__shfl(threadInput5, friend_id3, 64);
        sum0 += (lane_id < 16)? tx0: ((lane_id < 41)? ty0: tz0);
        sum1 += (lane_id < 9 )? tx1: ((lane_id < 40)? ty1: tz1);
        sum2 += (lane_id < 8 )? tx2: ((lane_id < 35)? ty2: tz2);
        sum3 += (lane_id < 3 )? tx3: ((lane_id < 32)? ty3: ((lane_id < 61)? tz3: ta3));

        // southeastern
        // friend_id0 = (friend_id0+1)&(warpSize-1);
        // friend_id1 = (friend_id1+1)&(warpSize-1);
        // friend_id2 = (friend_id2+1)&(warpSize-1);
        // friend_id3 = (friend_id3+1)&(warpSize-1);
        // tx0 = args[8]*__shfl(threadInput0, friend_id0, 64);
        // ty0 = args[8]*__shfl(threadInput1, friend_id0, 64);
        // tz0 = args[8]*__shfl(threadInput2, friend_id0, 64);
        // tx1 = args[8]*__shfl(threadInput0, friend_id1, 64);
        // ty1 = args[8]*__shfl(threadInput1, friend_id1, 64);
        // tz1 = args[8]*__shfl(threadInput2, friend_id1, 64);
        // tx2 = args[8]*__shfl(threadInput2, friend_id2, 64);
        // ty2 = args[8]*__shfl(threadInput3, friend_id2, 64);
        // tz2 = args[8]*__shfl(threadInput4, friend_id2, 64);
        // tx3 = args[8]*__shfl(threadInput2, friend_id3, 64);
        // ty3 = args[8]*__shfl(threadInput3, friend_id3, 64);
        // tz3 = args[8]*__shfl(threadInput4, friend_id3, 64);
        // ta3 = args[8]*__shfl(threadInput5, friend_id3, 64);
        // sum0 += (lane_id < 16)? tx0: ((lane_id < 40)? ty0: tz0);
        // sum1 += (lane_id < 8 )? tx1: ((lane_id < 40)? ty1: tz1);
        // sum2 += (lane_id < 8 )? tx2: ((lane_id < 34)? ty2: tz2);
        // sum3 += (lane_id < 2 )? tx3: ((lane_id < 32)? ty3: ((lane_id < 60)? tz3: ta3));
        /*
        */

        if(j < m + halo && i < n + halo)
            OUT_2D(j,i) = sum0;
        if(j < m + halo && i+8 < n + halo)
            OUT_2D(j,i+8) = sum1;
        if(j+8 < m + halo && i < n + halo)
            OUT_2D(j+8,i) = sum2;
        if(j+8 < m + halo && i+8 < n + halo)
            OUT_2D(j+8,i+8) = sum3;

    });
    fut.wait();
}

void Stencil_Hcc_Shfl8(array<DATA_TYPE> &in, array<DATA_TYPE> &out, array<DATA_TYPE> &args, int m, int n, int halo) 
{
    extent<2> compute_domain((m+7)/8+31, n+7); 
    tiled_extent<2> compute_tile(compute_domain, 32, 8);
    completion_future fut = parallel_for_each(compute_tile, [=, &in, &out, &args](tiled_index<2> tidx) restrict(amp) {
        int lane_id = __lane_id();
        int i = tidx.global[1] + halo;
        int j = ((tidx.global[0]>>3)<<6) + (lane_id>>3) + halo;
        int warp_id_x = (tidx.global[1])>>3;
        int warp_id_y = (((tidx.global[0]>>3)<<6) + (lane_id>>3))>>3;
        int new_i = (warp_id_x<<3) + lane_id%10;
        int new_j = (warp_id_y<<3) + lane_id/10;
        DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5;
        DATA_TYPE threadInput6, threadInput7, threadInput8, threadInput9, threadInput10;
        // if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput0 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+4)%10;
        new_j = (warp_id_y<<3) + 6 + (lane_id+4)/10;
        // if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput1 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+8)%10;
        new_j = (warp_id_y<<3) + 12 + (lane_id+8)/10;
        // if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput2 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+2)%10;
        new_j = (warp_id_y<<3) + 19 + (lane_id+2)/10;
        // if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput3 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+6)%10;
        new_j = (warp_id_y<<3) + 25 + (lane_id+6)/10;
        // if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput4 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id)%10;
        new_j = (warp_id_y<<3) + 32 + (lane_id)/10;
        // if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput5 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+4)%10;
        new_j = (warp_id_y<<3) + 38 + (lane_id+4)/10;
        // if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput6 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+8)%10;
        new_j = (warp_id_y<<3) + 44 + (lane_id+8)/10;
        // if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput7 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+2)%10;
        new_j = (warp_id_y<<3) + 51 + (lane_id+2)/10;
        // if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput8 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+6)%10;
        new_j = (warp_id_y<<3) + 57 + (lane_id+6)/10;
        // if(new_i < n+2*halo && new_j < m+2*halo)
            threadInput9 = IN_2D(new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id)%10;
        new_j = (warp_id_y<<3) + 64 + (lane_id)/10;
        if(new_i < n+2*halo && new_j < m+2*halo)
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
        // northwestern
        friend_id0 = (lane_id+   ((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+32+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+48+((lane_id>>3)<<1))&(warpSize-1);
        DATA_TYPE tx0, ty0, tx1, ty1, tx2, ty2, tz2, tx3, ty3, tz3;
        DATA_TYPE rx0, ry0, rx1, ry1, rx2, ry2, rz2, rx3, ry3, rz3;
        // tx0 = args[0]*__shfl(threadInput0, friend_id0, 64);
        // ty0 = args[0]*__shfl(threadInput1, friend_id0, 64);
        // tx1 = args[0]*__shfl(threadInput1, friend_id1, 64);
        // ty1 = args[0]*__shfl(threadInput2, friend_id1, 64);
        // tx2 = args[0]*__shfl(threadInput2, friend_id2, 64);
        // ty2 = args[0]*__shfl(threadInput3, friend_id2, 64);
        // tx3 = args[0]*__shfl(threadInput3, friend_id3, 64);
        // ty3 = args[0]*__shfl(threadInput4, friend_id3, 64);
        // rx0 = args[0]*__shfl(threadInput5, friend_id0, 64);
        // ry0 = args[0]*__shfl(threadInput6, friend_id0, 64);
        // rx1 = args[0]*__shfl(threadInput6, friend_id1, 64);
        // ry1 = args[0]*__shfl(threadInput7, friend_id1, 64);
        // rx2 = args[0]*__shfl(threadInput7, friend_id2, 64);
        // ry2 = args[0]*__shfl(threadInput8, friend_id2, 64);
        // rx3 = args[0]*__shfl(threadInput8, friend_id3, 64);
        // ry3 = args[0]*__shfl(threadInput9, friend_id3, 64);

        // sum0 += (lane_id < 52)? tx0: ty0;
        // sum1 += (lane_id < 40)? tx1: ty1;
        // sum2 += (lane_id < 26)? tx2: ty2;
        // sum3 += (lane_id < 14)? tx3: ty3;

        // sum4 += (lane_id < 52)? rx0: ry0;
        // sum5 += (lane_id < 40)? rx1: ry1;
        // sum6 += (lane_id < 26)? rx2: ry2;
        // sum7 += (lane_id < 14)? rx3: ry3;

        // northern
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        friend_id2 = (friend_id2+1)&(warpSize-1);
        friend_id3 = (friend_id3+1)&(warpSize-1);
        tx0 = args[0]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[0]*__shfl(threadInput1, friend_id0, 64);
        tx1 = args[0]*__shfl(threadInput1, friend_id1, 64);
        ty1 = args[0]*__shfl(threadInput2, friend_id1, 64);
        tx2 = args[0]*__shfl(threadInput2, friend_id2, 64);
        ty2 = args[0]*__shfl(threadInput3, friend_id2, 64);
        tx3 = args[0]*__shfl(threadInput3, friend_id3, 64);
        ty3 = args[0]*__shfl(threadInput4, friend_id3, 64);
        rx0 = args[0]*__shfl(threadInput5, friend_id0, 64);
        ry0 = args[0]*__shfl(threadInput6, friend_id0, 64);
        rx1 = args[0]*__shfl(threadInput6, friend_id1, 64);
        ry1 = args[0]*__shfl(threadInput7, friend_id1, 64);
        rx2 = args[0]*__shfl(threadInput7, friend_id2, 64);
        ry2 = args[0]*__shfl(threadInput8, friend_id2, 64);
        rx3 = args[0]*__shfl(threadInput8, friend_id3, 64);
        ry3 = args[0]*__shfl(threadInput9, friend_id3, 64);

        sum0 += (lane_id < 51)? tx0: ty0;
        sum1 += (lane_id < 39)? tx1: ty1;
        sum2 += (lane_id < 25)? tx2: ty2;
        sum3 += (lane_id < 13)? tx3: ty3;
        sum4 += (lane_id < 51)? rx0: ry0;
        sum5 += (lane_id < 39)? rx1: ry1;
        sum6 += (lane_id < 25)? rx2: ry2;
        sum7 += (lane_id < 13)? rx3: ry3;

        // northeastern 
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        friend_id2 = (friend_id2+1)&(warpSize-1);
        friend_id3 = (friend_id3+1)&(warpSize-1);
        // tx0 = args[2]*__shfl(threadInput0, friend_id0, 64);
        // ty0 = args[2]*__shfl(threadInput1, friend_id0, 64);
        // tx1 = args[2]*__shfl(threadInput1, friend_id1, 64);
        // ty1 = args[2]*__shfl(threadInput2, friend_id1, 64);
        // tx2 = args[2]*__shfl(threadInput2, friend_id2, 64);
        // ty2 = args[2]*__shfl(threadInput3, friend_id2, 64);
        // tx3 = args[2]*__shfl(threadInput3, friend_id3, 64);
        // ty3 = args[2]*__shfl(threadInput4, friend_id3, 64);
        // rx0 = args[2]*__shfl(threadInput5, friend_id0, 64);
        // ry0 = args[2]*__shfl(threadInput6, friend_id0, 64);
        // rx1 = args[2]*__shfl(threadInput6, friend_id1, 64);
        // ry1 = args[2]*__shfl(threadInput7, friend_id1, 64);
        // rx2 = args[2]*__shfl(threadInput7, friend_id2, 64);
        // ry2 = args[2]*__shfl(threadInput8, friend_id2, 64);
        // rx3 = args[2]*__shfl(threadInput8, friend_id3, 64);
        // ry3 = args[2]*__shfl(threadInput9, friend_id3, 64);

        // sum0 += (lane_id < 50)? tx0: ty0;
        // sum1 += (lane_id < 38)? tx1: ty1;
        // sum2 += (lane_id < 24)? tx2: ty2;
        // sum3 += (lane_id < 12)? tx3: ty3;
        // sum4 += (lane_id < 50)? rx0: ry0;
        // sum5 += (lane_id < 38)? rx1: ry1;
        // sum6 += (lane_id < 24)? rx2: ry2;
        // sum7 += (lane_id < 12)? rx3: ry3;

        // western 
        friend_id0 = (friend_id0+8)&(warpSize-1);
        friend_id1 = (friend_id1+8)&(warpSize-1);
        friend_id2 = (friend_id2+8)&(warpSize-1);
        friend_id3 = (friend_id3+8)&(warpSize-1);
        tx0 = args[1]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[1]*__shfl(threadInput1, friend_id0, 64);
        tx1 = args[1]*__shfl(threadInput1, friend_id1, 64);
        ty1 = args[1]*__shfl(threadInput2, friend_id1, 64);
        tx2 = args[1]*__shfl(threadInput2, friend_id2, 64);
        ty2 = args[1]*__shfl(threadInput3, friend_id2, 64);
        tx3 = args[1]*__shfl(threadInput3, friend_id3, 64);
        ty3 = args[1]*__shfl(threadInput4, friend_id3, 64);
        tz3 = args[1]*__shfl(threadInput5, friend_id3, 64);
        rx0 = args[1]*__shfl(threadInput5, friend_id0, 64);
        ry0 = args[1]*__shfl(threadInput6, friend_id0, 64);
        rx1 = args[1]*__shfl(threadInput6, friend_id1, 64);
        ry1 = args[1]*__shfl(threadInput7, friend_id1, 64);
        rx2 = args[1]*__shfl(threadInput7, friend_id2, 64);
        ry2 = args[1]*__shfl(threadInput8, friend_id2, 64);
        rx3 = args[1]*__shfl(threadInput8, friend_id3, 64);
        ry3 = args[1]*__shfl(threadInput9, friend_id3, 64);
        rz3 = args[1]*__shfl(threadInput10, friend_id3, 64);

        sum0 += (lane_id < 44)? tx0: ty0;
        sum1 += (lane_id < 32)? tx1: ty1;
        sum2 += (lane_id < 18)? tx2: ty2;
        sum3 += (lane_id < 6 )? tx3: ((lane_id < 56)? ty3: tz3);
        sum4 += (lane_id < 44)? rx0: ry0;
        sum5 += (lane_id < 32)? rx1: ry1;
        sum6 += (lane_id < 18)? rx2: ry2;
        sum7 += (lane_id < 6 )? rx3: ((lane_id < 56)? ry3: rz3);

        // central
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        friend_id2 = (friend_id2+1)&(warpSize-1);
        friend_id3 = (friend_id3+1)&(warpSize-1);
        tx0 = args[4]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[4]*__shfl(threadInput1, friend_id0, 64);
        tx1 = args[4]*__shfl(threadInput1, friend_id1, 64);
        ty1 = args[4]*__shfl(threadInput2, friend_id1, 64);
        tx2 = args[4]*__shfl(threadInput2, friend_id2, 64);
        ty2 = args[4]*__shfl(threadInput3, friend_id2, 64);
        tx3 = args[4]*__shfl(threadInput3, friend_id3, 64);
        ty3 = args[4]*__shfl(threadInput4, friend_id3, 64);
        tz3 = args[4]*__shfl(threadInput5, friend_id3, 64);

        rx0 = args[4]*__shfl(threadInput5, friend_id0, 64);
        ry0 = args[4]*__shfl(threadInput6, friend_id0, 64);
        rx1 = args[4]*__shfl(threadInput6, friend_id1, 64);
        ry1 = args[4]*__shfl(threadInput7, friend_id1, 64);
        rx2 = args[4]*__shfl(threadInput7, friend_id2, 64);
        ry2 = args[4]*__shfl(threadInput8, friend_id2, 64);
        rx3 = args[4]*__shfl(threadInput8, friend_id3, 64);
        ry3 = args[4]*__shfl(threadInput9, friend_id3, 64);
        rz3 = args[4]*__shfl(threadInput10, friend_id3, 64);

        sum0 += (lane_id < 43)? tx0: ty0;
        sum1 += (lane_id < 31)? tx1: ty1;
        sum2 += (lane_id < 17)? tx2: ty2;
        sum3 += (lane_id < 5 )? tx3: ((lane_id < 56)? ty3: tz3);
        sum4 += (lane_id < 43)? rx0: ry0;
        sum5 += (lane_id < 31)? rx1: ry1;
        sum6 += (lane_id < 17)? rx2: ry2;
        sum7 += (lane_id < 5 )? rx3: ((lane_id < 56)? ry3: rz3);

        // eastern
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        friend_id2 = (friend_id2+1)&(warpSize-1);
        friend_id3 = (friend_id3+1)&(warpSize-1);
        tx0 = args[3]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[3]*__shfl(threadInput1, friend_id0, 64);
        tx1 = args[3]*__shfl(threadInput1, friend_id1, 64);
        ty1 = args[3]*__shfl(threadInput2, friend_id1, 64);
        tx2 = args[3]*__shfl(threadInput2, friend_id2, 64);
        ty2 = args[3]*__shfl(threadInput3, friend_id2, 64);
        tx3 = args[3]*__shfl(threadInput3, friend_id3, 64);
        ty3 = args[3]*__shfl(threadInput4, friend_id3, 64);
        tz3 = args[3]*__shfl(threadInput5, friend_id3, 64);
        rx0 = args[3]*__shfl(threadInput5, friend_id0, 64);
        ry0 = args[3]*__shfl(threadInput6, friend_id0, 64);
        rx1 = args[3]*__shfl(threadInput6, friend_id1, 64);
        ry1 = args[3]*__shfl(threadInput7, friend_id1, 64);
        rx2 = args[3]*__shfl(threadInput7, friend_id2, 64);
        ry2 = args[3]*__shfl(threadInput8, friend_id2, 64);
        rx3 = args[3]*__shfl(threadInput8, friend_id3, 64);
        ry3 = args[3]*__shfl(threadInput9, friend_id3, 64);
        rz3 = args[3]*__shfl(threadInput10, friend_id3, 64);

        sum0 += (lane_id < 42)? tx0: ty0;
        sum1 += (lane_id < 30)? tx1: ty1;
        sum2 += (lane_id < 16)? tx2: ty2;
        sum3 += (lane_id < 4 )? tx3: ((lane_id < 56)? ty3: tz3);
        sum4 += (lane_id < 42)? rx0: ry0;
        sum5 += (lane_id < 30)? rx1: ry1;
        sum6 += (lane_id < 16)? rx2: ry2;
        sum7 += (lane_id < 4 )? rx3: ((lane_id < 56)? ry3: rz3);

        // southeastern 
        friend_id0 = (friend_id0+8)&(warpSize-1);
        friend_id1 = (friend_id1+8)&(warpSize-1);
        friend_id2 = (friend_id2+8)&(warpSize-1);
        friend_id3 = (friend_id3+8)&(warpSize-1);
        // tx0 = args[6]*__shfl(threadInput0, friend_id0, 64);
        // ty0 = args[6]*__shfl(threadInput1, friend_id0, 64);
        // tx1 = args[6]*__shfl(threadInput1, friend_id1, 64);
        // ty1 = args[6]*__shfl(threadInput2, friend_id1, 64);
        // tx2 = args[6]*__shfl(threadInput2, friend_id2, 64);
        // ty2 = args[6]*__shfl(threadInput3, friend_id2, 64);
        // tz2 = args[6]*__shfl(threadInput4, friend_id2, 64);
        // tx3 = args[6]*__shfl(threadInput4, friend_id3, 64);
        // ty3 = args[6]*__shfl(threadInput5, friend_id3, 64);

        // rx0 = args[6]*__shfl(threadInput5, friend_id0, 64);
        // ry0 = args[6]*__shfl(threadInput6, friend_id0, 64);
        // rx1 = args[6]*__shfl(threadInput6, friend_id1, 64);
        // ry1 = args[6]*__shfl(threadInput7, friend_id1, 64);
        // rx2 = args[6]*__shfl(threadInput7, friend_id2, 64);
        // ry2 = args[6]*__shfl(threadInput8, friend_id2, 64);
        // rz2 = args[6]*__shfl(threadInput9, friend_id2, 64);
        // rx3 = args[6]*__shfl(threadInput9, friend_id3, 64);
        // ry3 = args[6]*__shfl(threadInput10, friend_id3, 64);

        // sum0 += (lane_id < 36)? tx0: ty0;
        // sum1 += (lane_id < 24)? tx1: ty1;
        // sum2 += (lane_id < 10)? tx2: ((lane_id < 62)? ty2: tz2);
        // sum3 += (lane_id < 48)? tx3: ty3;
        // sum4 += (lane_id < 36)? rx0: ry0;
        // sum5 += (lane_id < 24)? rx1: ry1;
        // sum6 += (lane_id < 10)? rx2: ((lane_id < 62)? ry2: rz2);
        // sum7 += (lane_id < 48)? rx3: ry3;

        // southern
        friend_id0 = (friend_id0+1)&(warpSize-1);
        friend_id1 = (friend_id1+1)&(warpSize-1);
        friend_id2 = (friend_id2+1)&(warpSize-1);
        friend_id3 = (friend_id3+1)&(warpSize-1);
        tx0 = args[2]*__shfl(threadInput0, friend_id0, 64);
        ty0 = args[2]*__shfl(threadInput1, friend_id0, 64);
        tx1 = args[2]*__shfl(threadInput1, friend_id1, 64);
        ty1 = args[2]*__shfl(threadInput2, friend_id1, 64);
        tx2 = args[2]*__shfl(threadInput2, friend_id2, 64);
        ty2 = args[2]*__shfl(threadInput3, friend_id2, 64);
        tz2 = args[2]*__shfl(threadInput4, friend_id2, 64);
        tx3 = args[2]*__shfl(threadInput4, friend_id3, 64);
        ty3 = args[2]*__shfl(threadInput5, friend_id3, 64);
        rx0 = args[2]*__shfl(threadInput5, friend_id0, 64);
        ry0 = args[2]*__shfl(threadInput6, friend_id0, 64);
        rx1 = args[2]*__shfl(threadInput6, friend_id1, 64);
        ry1 = args[2]*__shfl(threadInput7, friend_id1, 64);
        rx2 = args[2]*__shfl(threadInput7, friend_id2, 64);
        ry2 = args[2]*__shfl(threadInput8, friend_id2, 64);
        rz2 = args[2]*__shfl(threadInput9, friend_id2, 64);
        rx3 = args[2]*__shfl(threadInput9, friend_id3, 64);
        ry3 = args[2]*__shfl(threadInput10, friend_id3, 64);

        sum0 += (lane_id < 35)? tx0: ty0;
        sum1 += (lane_id < 23)? tx1: ty1;
        sum2 += (lane_id < 9 )? tx2: ((lane_id < 61)? ty2: tz2);
        sum3 += (lane_id < 48)? tx3: ty3;
        sum4 += (lane_id < 35)? rx0: ry0;
        sum5 += (lane_id < 23)? rx1: ry1;
        sum6 += (lane_id < 9 )? rx2: ((lane_id < 61)? ry2: rz2);
        sum7 += (lane_id < 48)? rx3: ry3;

        // southeastern
        // friend_id0 = (friend_id0+1)&(warpSize-1);
        // friend_id1 = (friend_id1+1)&(warpSize-1);
        // friend_id2 = (friend_id2+1)&(warpSize-1);
        // friend_id3 = (friend_id3+1)&(warpSize-1);
        // tx0 = args[8]*__shfl(threadInput0, friend_id0, 64);
        // ty0 = args[8]*__shfl(threadInput1, friend_id0, 64);
        // tx1 = args[8]*__shfl(threadInput1, friend_id1, 64);
        // ty1 = args[8]*__shfl(threadInput2, friend_id1, 64);
        // tx2 = args[8]*__shfl(threadInput2, friend_id2, 64);
        // ty2 = args[8]*__shfl(threadInput3, friend_id2, 64);
        // tz2 = args[8]*__shfl(threadInput4, friend_id2, 64);
        // tx3 = args[8]*__shfl(threadInput4, friend_id3, 64);
        // ty3 = args[8]*__shfl(threadInput5, friend_id3, 64);
        // rx0 = args[8]*__shfl(threadInput5, friend_id0, 64);
        // ry0 = args[8]*__shfl(threadInput6, friend_id0, 64);
        // rx1 = args[8]*__shfl(threadInput6, friend_id1, 64);
        // ry1 = args[8]*__shfl(threadInput7, friend_id1, 64);
        // rx2 = args[8]*__shfl(threadInput7, friend_id2, 64);
        // ry2 = args[8]*__shfl(threadInput8, friend_id2, 64);
        // rz2 = args[8]*__shfl(threadInput9, friend_id2, 64);
        // rx3 = args[8]*__shfl(threadInput9, friend_id3, 64);
        // ry3 = args[8]*__shfl(threadInput10, friend_id3, 64);

        // sum0 += (lane_id < 34)? tx0: ty0;
        // sum1 += (lane_id < 22)? tx1: ty1;
        // sum2 += (lane_id < 8 )? tx2: ((lane_id < 60)? ty2: tz2);
        // sum3 += (lane_id < 48)? tx3: ty3;
        // sum4 += (lane_id < 34)? rx0: ry0;
        // sum5 += (lane_id < 22)? rx1: ry1;
        // sum6 += (lane_id < 8 )? rx2: ((lane_id < 60)? ry2: rz2);
        // sum7 += (lane_id < 48)? rx3: ry3;
        /*
        */

        if(j < m + halo && i < n + halo)
            OUT_2D(j   ,i) = sum0;
        if(j+8 < m + halo && i < n + halo)
            OUT_2D(j+8 ,i) = sum1;
        if(j+16 < m + halo && i < n + halo)
            OUT_2D(j+16,i) = sum2;
        if(j+24 < m + halo && i < n + halo)
            OUT_2D(j+24,i) = sum3;
        if(j+32 < m + halo && i < n + halo)
            OUT_2D(j+32,i) = sum4;
        if(j+40 < m + halo && i < n + halo)
            OUT_2D(j+40,i) = sum5;
        if(j+48 < m + halo && i < n + halo)
            OUT_2D(j+48,i) = sum6;
        if(j+56 < m + halo && i < n + halo)
            OUT_2D(j+56,i) = sum7;

    });
    fut.wait();
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

int main(int argc, char **argv)
{
    int m = 10008; // multiple of 16 (shfl4_2)
    int n = 10008; // multiple of 16 (shfl4_2)
    // int m = 64;
    // int n = 64;
    int halo = 1;
    int total = (m+2*halo)*(n+2*halo);
    const int K = 9;
    DATA_TYPE args[K] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    DATA_TYPE *in = new DATA_TYPE[total];
    DATA_TYPE *out_ref = new DATA_TYPE[total];
    Init_Input_2D(in, m, n, halo);

    // Show_Me(in, m, n, halo, "Input:");
    Stencil_Seq(in, out_ref, args, m, n, halo);
    // Show_Me(out_ref, m, n, halo, "Output:");

    DATA_TYPE *out = new DATA_TYPE[total];
    extent<1> data_domain(total);
    extent<1> args_domain(K);
    array<DATA_TYPE> in_d(data_domain, in, in+total);
    array<DATA_TYPE> out_d(data_domain);
    array<DATA_TYPE> args_d(args_domain, args, args+K);

    Stencil_Hcc(in_d, out_d, args_d, m, n, halo); // warmup

    Init_Input_2D(out, m, n, halo);
    copy(out, out_d);

    auto t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    Stencil_Hcc(in_d, out_d, args_d, m, n, halo);
    END_TIME;
    auto t2 = std::chrono::high_resolution_clock::now();

    copy(out_d, out);
    Fill_Halo_2D(out, m, n, halo);
    // Show_Me(out, m, n, halo, "Output(Device):");
    std::cout << "Verify Hcc: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout << "Hcc Time: " << timeInNS << std::endl;

    Init_Input_2D(out, m, n, halo);
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    Stencil_Hcc_Shfl(in_d, out_d, args_d, m, n, halo);
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();

    copy(out_d, out);
    Fill_Halo_2D(out, m, n, halo);
    // Show_Me(out, m, n, halo, "Output(Device):");
    std::cout << "Verify Hcc_Shfl: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout << "Hcc_Shfl Time: " << timeInNS << std::endl;

    Init_Input_2D(out, m, n, halo);
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    Stencil_Hcc_Shfl2(in_d, out_d, args_d, m, n, halo);
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();

    copy(out_d, out);
    Fill_Halo_2D(out, m, n, halo);
    // Show_Me(out, m, n, halo, "Output(Device):");
    std::cout << "Verify Hcc_Shfl2: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout << "Hcc_Shfl2 Time: " << timeInNS << std::endl;

    Init_Input_2D(out, m, n, halo);
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    Stencil_Hcc_Shfl4(in_d, out_d, args_d, m, n, halo);
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();

    copy(out_d, out);
    Fill_Halo_2D(out, m, n, halo);
    // Show_Me(out, m, n, halo, "Output(Device):");
    std::cout << "Verify Hcc_Shfl4: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout << "Hcc_Shfl4 Time: " << timeInNS << std::endl;

    Init_Input_2D(out, m, n, halo);
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    Stencil_Hcc_Shfl8(in_d, out_d, args_d, m, n, halo);
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();

    copy(out_d, out);
    Fill_Halo_2D(out, m, n, halo);
    // Show_Me(out, m, n, halo, "Output(Device):");
    std::cout << "Verify Hcc_Shfl8: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout << "Hcc_Shfl8 Time: " << timeInNS << std::endl;

    Init_Input_2D(out, m, n, halo);
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    Stencil_Hcc_Shfl4_2(in_d, out_d, args_d, m, n, halo);
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();

    copy(out_d, out);
    Fill_Halo_2D(out, m, n, halo);
    // Show_Me(out, m, n, halo, "Output(Device):");
    std::cout << "Verify Hcc_Shfl4_2: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout << "Hcc_Shfl4_2 Time: " << timeInNS << std::endl;



    Init_Input_2D(out, m, n, halo);
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    Stencil_Hcc_Sm(in_d, out_d, args_d, m, n, halo);
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();

    copy(out_d, out);
    Fill_Halo_2D(out, m, n, halo);
    // Show_Me(out, m, n, halo, "Output(Device):");
    std::cout << "Verify Hcc_Sm: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout << "Hcc_Sm Time: " << timeInNS << std::endl;


}
