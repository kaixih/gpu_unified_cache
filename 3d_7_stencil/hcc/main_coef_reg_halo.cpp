#include <iostream>
#include <cmath>
#include <hc.hpp>
#include <metrics.h>

using namespace hc;

#define  IN_3D(_z,_y,_x)  in[(_z)*(m+2*halo)*(n+2*halo)+(_y)*(n+2*halo)+(_x)]
#define OUT_3D(_z,_y,_x) out[(_z)*(m+2*halo)*(n+2*halo)+(_y)*(n+2*halo)+(_x)]

#define LOC_3D(_z,_y,_x) local[(_z)][(_y)][(_x)]

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
        DATA_TYPE a5 , DATA_TYPE a6 , 
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

void Stencil_Hcc(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(z, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 8, 4, 8);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        int i = tidx.global[2] + halo;
        int j = tidx.global[1] + halo;
        int k = tidx.global[0] + halo;

        OUT_3D(k,j,i) = a0 * IN_3D(k-1,j  ,i  ) +
                        a1 * IN_3D(k  ,j-1,i  ) +
                        a2 * IN_3D(k  ,j  ,i-1) +
                        a3 * IN_3D(k  ,j  ,i  ) +
                        a4 * IN_3D(k  ,j  ,i+1) +
                        a5 * IN_3D(k  ,j+1,i  ) +
                        a6 * IN_3D(k+1,j  ,i  ) ;
    });
    fut.wait();
}

void Stencil_Hcc_Sweep(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , 
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
            OUT_3D(k,j,i) = a0 * IN_3D(k-1,j  ,i  ) +
                            a1 * IN_3D(k  ,j-1,i  ) +
                            a2 * IN_3D(k  ,j  ,i-1) +
                            a3 * IN_3D(k  ,j  ,i  ) +
                            a4 * IN_3D(k  ,j  ,i+1) +
                            a5 * IN_3D(k  ,j+1,i  ) +
                            a6 * IN_3D(k+1,j  ,i  ) ;
        }
    });
    fut.wait();
}

void Stencil_Hcc_SweepX(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , 
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
            OUT_3D(k,j,i) = a0 * IN_3D(k-1,j  ,i  ) +
                            a1 * IN_3D(k  ,j-1,i  ) +
                            a2 * IN_3D(k  ,j  ,i-1) +
                            a3 * IN_3D(k  ,j  ,i  ) +
                            a4 * IN_3D(k  ,j  ,i+1) +
                            a5 * IN_3D(k  ,j+1,i  ) +
                            a6 * IN_3D(k+1,j  ,i  ) ;
        }
    });
    fut.wait();
}

void Stencil_Hcc_Sm_Branch(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , 
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
        
        tidx.barrier.wait();

        
        OUT_3D(k,j,i) = a0 * LOC_3D(lk-1,lj  ,li  ) +
                        a1 * LOC_3D(lk  ,lj-1,li  ) +
                        a2 * LOC_3D(lk  ,lj  ,li-1) +
                        a3 * LOC_3D(lk  ,lj  ,li  ) +
                        a4 * LOC_3D(lk  ,lj  ,li+1) +
                        a5 * LOC_3D(lk  ,lj+1,li  ) +
                        a6 * LOC_3D(lk+1,lj  ,li  ) ;
    });
    fut.wait();
}

void Stencil_Hcc_Sm_Cyclic(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , 
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

        OUT_3D(k,j,i) = a0 * LOC_3D(lk-1,lj  ,li  ) +
                        a1 * LOC_3D(lk  ,lj-1,li  ) +
                        a2 * LOC_3D(lk  ,lj  ,li-1) +
                        a3 * LOC_3D(lk  ,lj  ,li  ) +
                        a4 * LOC_3D(lk  ,lj  ,li+1) +
                        a5 * LOC_3D(lk  ,lj+1,li  ) +
                        a6 * LOC_3D(lk+1,lj  ,li  ) ;

    });
    fut.wait();
}

void Stencil_Hcc_Sweep_Sm_Branch(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , 
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

            OUT_3D(k,j,i) = a0 * LOC_3D(t1,lj  ,li  ) +
                            a1 * LOC_3D(t2,lj-1,li  ) + 
                            a2 * LOC_3D(t2,lj  ,li-1) + 
                            a3 * LOC_3D(t2,lj  ,li  ) + 
                            a4 * LOC_3D(t2,lj  ,li+1) + 
                            a5 * LOC_3D(t2,lj+1,li  ) + 
                            a6 * LOC_3D(t3,lj  ,li  ) ;
        }
    });
    fut.wait();
}

void Stencil_Hcc_Sweep_Sm_BranchX(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , 
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

            OUT_3D(k,j,i) = a0 * LOC_3D(t1,lj  ,li  ) +
                            a1 * LOC_3D(t2,lj-1,li  ) + 
                            a2 * LOC_3D(t2,lj  ,li-1) + 
                            a3 * LOC_3D(t2,lj  ,li  ) + 
                            a4 * LOC_3D(t2,lj  ,li+1) + 
                            a5 * LOC_3D(t2,lj+1,li  ) + 
                            a6 * LOC_3D(t3,lj  ,li  ) ;
        }
    });
    fut.wait();
}

void Stencil_Hcc_Sweep_Sm_Cyclic(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , 
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

            OUT_3D(k,j,i) = a0 * LOC_3D(t1,lj  ,li  ) +
                            a1 * LOC_3D(t2,lj-1,li  ) + 
                            a2 * LOC_3D(t2,lj  ,li-1) + 
                            a3 * LOC_3D(t2,lj  ,li  ) + 
                            a4 * LOC_3D(t2,lj  ,li+1) + 
                            a5 * LOC_3D(t2,lj+1,li  ) + 
                            a6 * LOC_3D(t3,lj  ,li  ) ;
        }
    });
    fut.wait();
}

void Stencil_Hcc_Sweep_Sm_CyclicX(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , 
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

            OUT_3D(k,j,i) = a0 * LOC_3D(t1,lj  ,li  ) +
                            a1 * LOC_3D(t2,lj-1,li  ) + 
                            a2 * LOC_3D(t2,lj  ,li-1) + 
                            a3 * LOC_3D(t2,lj  ,li  ) + 
                            a4 * LOC_3D(t2,lj  ,li+1) + 
                            a5 * LOC_3D(t2,lj+1,li  ) + 
                            a6 * LOC_3D(t3,lj  ,li  ) ;
        }
    });
    fut.wait();
}

void Stencil_Hcc_Shfl_2DWarp(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , 
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
        int warp_id_x = (tidx.global[2])>>3; // because the warp dimensions are 
        int warp_id_y = (tidx.global[1])>>3; // 1x8x8, warp_ids are division of 
        int warp_id_z = (tidx.global[0])>>0; // there numbers
        int new_i, new_j, new_k;
        DATA_TYPE reg0, reg1, reg2, reg3, reg4;   // ceil(10*10*3/64) = 5 
        new_i = (warp_id_x<<3) + lane_id_it%10;      // 10 is extended dimension of i
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10; // 10 is extended dimension of j 
        new_k = (warp_id_z<<0) + lane_id_it/100;     // 100 is extended area of ixj = 10x10
        reg0 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        reg1 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        reg2 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        reg3 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
        new_k = (new_k < z+2*halo)? new_k: z+2*halo-1;
        reg4 = IN_3D(new_k, new_j, new_i);

        DATA_TYPE sum0 = 0.0;
        int friend_id0;
        DATA_TYPE tx0, ty0, tz0;

        friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += a0*((lane_id < 43)? tx0: ty0);

        friend_id0 = (lane_id+37+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a1*((lane_id < 23)? tx0: ty0);

        friend_id0 = (lane_id+46+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a2*((lane_id < 16)? tx0: ty0);

        friend_id0 = (lane_id+47+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a3*((lane_id < 15)? tx0: ty0);

        friend_id0 = (lane_id+48+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        sum0 += a4*((lane_id < 14)? tx0: ty0);

        friend_id0 = (lane_id+57+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tz0 = __shfl(reg3, friend_id0);
        sum0 += a5*((lane_id < 7 )? tx0: ((lane_id < 57)? ty0: tz0));

        friend_id0 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        sum0 += a6*((lane_id < 37)? tx0: ty0);

        OUT_3D(k,j,i) = sum0;

    });
    fut.wait();
}

void Stencil_Hcc_Shfl2_2DWarp(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , 
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
        int warp_id_x = (tidx.global[2])>>3; // because the warp dimensions are 
        int warp_id_y = (tidx.global[1])>>3; // 1x8x8, warp_ids are division of 
        int warp_id_z = (((tidx.global[0])>>0)<<1)>>0; // there numbers
        int new_i, new_j, new_k;
        DATA_TYPE reg0, reg1, reg2, reg3, reg4, reg5, reg6;   // ceil(10*10*4/64) = 7
        new_i = (warp_id_x<<3) + lane_id_it%10;      // 10 is extended dimension of i
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10; // 10 is extended dimension of j 
        new_k = (warp_id_z<<0) + lane_id_it/100;     // 100 is extended area of ixj = 10x10
        reg0 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        reg1 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        reg2 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        reg3 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        reg4 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        reg5 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
        new_k = (new_k < z+2*halo)? new_k: z+2*halo-1;
        reg6 = IN_3D(new_k, new_j, new_i);

        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;
        int friend_id0;
        int friend_id1;
        DATA_TYPE tx0, ty0, tz0;
        DATA_TYPE tx1, ty1, tz1;

        friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+47+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        sum0 += a0*((lane_id < 43)? tx0: ty0);
        sum1 += a0*((lane_id < 15)? tx1: ty1);

        friend_id0 = (lane_id+37+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum0 += a1*((lane_id < 23)? tx0: ty0);
        sum1 += a1*((lane_id < 45)? tx1: ty1);

        friend_id0 = (lane_id+46+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum0 += a2*((lane_id < 16)? tx0: ty0);
        sum1 += a2*((lane_id < 38)? tx1: ty1);

        friend_id0 = (lane_id+47+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum0 += a3*((lane_id < 15)? tx0: ty0);
        sum1 += a3*((lane_id < 37)? tx1: ty1);

        friend_id0 = (lane_id+48+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum0 += a4*((lane_id < 14)? tx0: ty0);
        sum1 += a4*((lane_id < 36)? tx1: ty1);

        friend_id0 = (lane_id+57+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tz0 = __shfl(reg3, friend_id0);
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        sum0 += a5*((lane_id < 7 )? tx0: ((lane_id < 57)? ty0: tz0));
        sum1 += a5*((lane_id < 29)? tx1: ty1);

        friend_id0 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+55+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        tx1 = __shfl(reg4, friend_id1);
        ty1 = __shfl(reg5, friend_id1);
        tz1 = __shfl(reg6, friend_id1);
        sum0 += a6*((lane_id < 37)? tx0: ty0);
        sum1 += a6*((lane_id < 8 )? tx1: ((lane_id < 59)? ty1: tz1));

        OUT_3D(k  ,j,i) = sum0;
        OUT_3D(k+1,j,i) = sum1;

    });
    fut.wait();
}

void Stencil_Hcc_Shfl4_2DWarp(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , 
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
        int warp_id_x = (tidx.global[2])>>3; // because the warp dimensions are 
        int warp_id_y = (tidx.global[1])>>3; // 1x8x8, warp_ids are division of 
        int warp_id_z = (((tidx.global[0])>>0)<<2)>>0; // there numbers
        int new_i, new_j, new_k;
        DATA_TYPE reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9;   // ceil(10*10*6/64) = 10
        new_i = (warp_id_x<<3) + lane_id_it%10;      // 10 is extended dimension of i
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10; // 10 is extended dimension of j 
        new_k = (warp_id_z<<0) + lane_id_it/100;     // 100 is extended area of ixj = 10x10
        reg0 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        reg1 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        reg2 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        reg3 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        reg4 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        reg5 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        reg6 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        reg7 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        reg8 = IN_3D(new_k, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<3) + lane_id_it%10;
        new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
        new_k = (warp_id_z<<0) + lane_id_it/100;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
        new_k = (new_k < z+2*halo)? new_k: z+2*halo-1;
        reg9 = IN_3D(new_k, new_j, new_i);


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

        friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+47+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+55+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        tx1 = __shfl(reg1, friend_id1);
        ty1 = __shfl(reg2, friend_id1);
        tx2 = __shfl(reg3, friend_id2);
        ty2 = __shfl(reg4, friend_id2);
        tx3 = __shfl(reg4, friend_id3);
        ty3 = __shfl(reg5, friend_id3);
        tz3 = __shfl(reg6, friend_id3);
        sum0 += a0*((lane_id < 43)? tx0: ty0);
        sum1 += a0*((lane_id < 15)? tx1: ty1);
        sum2 += a0*((lane_id < 37)? tx2: ty2);
        sum3 += a0*((lane_id < 8 )? tx3: ((lane_id < 59)? ty3: tz3));

        friend_id0 = (lane_id+37+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+45+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        tx2 = __shfl(reg4, friend_id2);
        ty2 = __shfl(reg5, friend_id2);
        tx3 = __shfl(reg6, friend_id3);
        ty3 = __shfl(reg7, friend_id3);
        sum0 += a1*((lane_id < 23)? tx0: ty0);
        sum1 += a1*((lane_id < 45)? tx1: ty1);
        sum2 += a1*((lane_id < 16)? tx2: ty2);
        sum3 += a1*((lane_id < 39)? tx3: ty3);

        friend_id0 = (lane_id+46+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+54+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        tx2 = __shfl(reg4, friend_id2);
        ty2 = __shfl(reg5, friend_id2);
        tz2 = __shfl(reg6, friend_id2);
        tx3 = __shfl(reg6, friend_id3);
        ty3 = __shfl(reg7, friend_id3);
        sum0 += a2*((lane_id < 16)? tx0: ty0);
        sum1 += a2*((lane_id < 38)? tx1: ty1);
        sum2 += a2*((lane_id < 8 )? tx2: ((lane_id < 60)? ty2: tz2));
        sum3 += a2*((lane_id < 32)? tx3: ty3);

        friend_id0 = (lane_id+47+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+55+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        tx2 = __shfl(reg4, friend_id2);
        ty2 = __shfl(reg5, friend_id2);
        tz2 = __shfl(reg6, friend_id2);
        tx3 = __shfl(reg6, friend_id3);
        ty3 = __shfl(reg7, friend_id3);
        sum0 += a3*((lane_id < 15)? tx0: ty0);
        sum1 += a3*((lane_id < 37)? tx1: ty1);
        sum2 += a3*((lane_id < 8 )? tx2: ((lane_id < 59)? ty2: tz2));
        sum3 += a3*((lane_id < 31)? tx3: ty3);

        friend_id0 = (lane_id+48+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+56+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        tx2 = __shfl(reg4, friend_id2);
        ty2 = __shfl(reg5, friend_id2);
        tz2 = __shfl(reg6, friend_id2);
        tx3 = __shfl(reg6, friend_id3);
        ty3 = __shfl(reg7, friend_id3);
        sum0 += a4*((lane_id < 14)? tx0: ty0);
        sum1 += a4*((lane_id < 36)? tx1: ty1);
        sum2 += a4*((lane_id < 8 )? tx2: ((lane_id < 58)? ty2: tz2));
        sum3 += a4*((lane_id < 30)? tx3: ty3);

        friend_id0 = (lane_id+57+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+37+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg1, friend_id0);
        ty0 = __shfl(reg2, friend_id0);
        tz0 = __shfl(reg3, friend_id0);
        tx1 = __shfl(reg3, friend_id1);
        ty1 = __shfl(reg4, friend_id1);
        tx2 = __shfl(reg5, friend_id2);
        ty2 = __shfl(reg6, friend_id2);
        tx3 = __shfl(reg6, friend_id3);
        ty3 = __shfl(reg7, friend_id3);
        sum0 += a5*((lane_id < 7 )? tx0: ((lane_id < 57)? ty0: tz0));
        sum1 += a5*((lane_id < 29)? tx1: ty1);
        sum2 += a5*((lane_id < 51)? tx2: ty2);
        sum3 += a5*((lane_id < 23)? tx3: ty3);

        friend_id0 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
        friend_id1 = (lane_id+55+((lane_id>>3)<<1))&(warpSize-1);
        friend_id2 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
        friend_id3 = (lane_id+63+((lane_id>>3)<<1))&(warpSize-1);
        tx0 = __shfl(reg3, friend_id0);
        ty0 = __shfl(reg4, friend_id0);
        tx1 = __shfl(reg4, friend_id1);
        ty1 = __shfl(reg5, friend_id1);
        tz1 = __shfl(reg6, friend_id1);
        tx2 = __shfl(reg6, friend_id2);
        ty2 = __shfl(reg7, friend_id2);
        tx3 = __shfl(reg7, friend_id3);
        ty3 = __shfl(reg8, friend_id3);
        tz3 = __shfl(reg9, friend_id3);
        sum0 += a6*((lane_id < 37)? tx0: ty0);
        sum1 += a6*((lane_id < 8 )? tx1: ((lane_id < 59)? ty1: tz1));
        sum2 += a6*((lane_id < 31)? tx2: ty2);
        sum3 += a6*((lane_id < 1 )? tx3: ((lane_id < 53)? ty3: tz3));

        OUT_3D(k  ,j,i) = sum0;
        OUT_3D(k+1,j,i) = sum1;
        OUT_3D(k+2,j,i) = sum2;
        OUT_3D(k+3,j,i) = sum3;

    });
    fut.wait();
}

void Stencil_Hcc_Sweep_Shfl_1DWarp(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , 
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

        int warp_id_x = (tidx.global[2])>>6;             // because the warp dimensions are 
        int warp_id_y = ((((tidx.global[1])>>0)<<0))>>0; // 1x1x64, warp_ids are division of these numbers

        const int lane_id = __lane_id();
        int lane_id_it = lane_id;

        int new_i, new_j;
        DATA_TYPE t3_reg0, t3_reg1, t3_reg2, t3_reg3;
        DATA_TYPE t2_reg0, t2_reg1, t2_reg2, t2_reg3;
        DATA_TYPE t1_reg0, t1_reg1, t1_reg2, t1_reg3;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        t3_reg0 = IN_3D(k  , new_j, new_i);
        t2_reg0 = IN_3D(k-1, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        t3_reg1 = IN_3D(k  , new_j, new_i);
        t2_reg1 = IN_3D(k-1, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        t3_reg2 = IN_3D(k  , new_j, new_i);
        t2_reg2 = IN_3D(k-1, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
        t3_reg3 = IN_3D(k  , new_j, new_i);
        t2_reg3 = IN_3D(k-1, new_j, new_i);

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
            new_i = (warp_id_x<<6) + lane_id_it%66;
            new_j = (warp_id_y<<0) + lane_id_it/66;
            t3_reg0 = IN_3D(k+1, new_j, new_i);
            lane_id_it += warpSize;
            new_i = (warp_id_x<<6) + lane_id_it%66;
            new_j = (warp_id_y<<0) + lane_id_it/66;
            t3_reg1 = IN_3D(k+1, new_j, new_i);
            lane_id_it += warpSize;
            new_i = (warp_id_x<<6) + lane_id_it%66;
            new_j = (warp_id_y<<0) + lane_id_it/66;
            t3_reg2 = IN_3D(k+1, new_j, new_i);
            lane_id_it += warpSize;
            new_i = (warp_id_x<<6) + lane_id_it%66;
            new_j = (warp_id_y<<0) + lane_id_it/66;
            new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
            new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
            t3_reg3 = IN_3D(k+1, new_j, new_i);


            friend_id0 = (lane_id+1 )&(warpSize-1);
            tx0 = __shfl(t2_reg0, friend_id0);
            ty0 = __shfl(t2_reg1, friend_id0);
            sum0 += a1*((lane_id < 63)? tx0: ty0);
            
            friend_id0 = (lane_id+2 )&(warpSize-1);
            tx0 = __shfl(t2_reg1, friend_id0);
            ty0 = __shfl(t2_reg2, friend_id0);
            sum0 += a2*((lane_id < 62)? tx0: ty0);
        
            friend_id0 = (lane_id+3 )&(warpSize-1);
            tx0 = __shfl(t1_reg1, friend_id0);
            ty0 = __shfl(t1_reg2, friend_id0);
            sum0 += a0*((lane_id < 61)? tx0: ty0);
            tx0 = __shfl(t2_reg1, friend_id0);
            ty0 = __shfl(t2_reg2, friend_id0);
            sum0 += a3*((lane_id < 61)? tx0: ty0);
            tx0 = __shfl(t3_reg1, friend_id0);
            ty0 = __shfl(t3_reg2, friend_id0);
            sum0 += a6*((lane_id < 61)? tx0: ty0);

            friend_id0 = (lane_id+4 )&(warpSize-1);
            tx0 = __shfl(t2_reg1, friend_id0);
            ty0 = __shfl(t2_reg2, friend_id0);
            sum0 += a4*((lane_id < 60)? tx0: ty0);

            friend_id0 = (lane_id+5 )&(warpSize-1);
            tx0 = __shfl(t2_reg2, friend_id0);
            ty0 = __shfl(t2_reg3, friend_id0);
            sum0 += a5*((lane_id < 59)? tx0: ty0);

            OUT_3D(k,j  ,i) = sum0;
        
        }
    });
    fut.wait();
}

void Stencil_Hcc_Sweep_Shfl2_1DWarp(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , 
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

        int warp_id_x = (tidx.global[2])>>6;             // because the warp dimensions are 
        int warp_id_y = ((((tidx.global[1])>>0)<<1))>>0; // 1x1x64, warp_ids are division of these numbers

        const int lane_id = __lane_id();
        int lane_id_it = lane_id;

        int new_i, new_j;
        DATA_TYPE t3_reg0, t3_reg1, t3_reg2, t3_reg3, t3_reg4;
        DATA_TYPE t2_reg0, t2_reg1, t2_reg2, t2_reg3, t2_reg4;
        DATA_TYPE t1_reg0, t1_reg1, t1_reg2, t1_reg3, t1_reg4;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        t3_reg0 = IN_3D(k  , new_j, new_i);
        t2_reg0 = IN_3D(k-1, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        t3_reg1 = IN_3D(k  , new_j, new_i);
        t2_reg1 = IN_3D(k-1, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        t3_reg2 = IN_3D(k  , new_j, new_i);
        t2_reg2 = IN_3D(k-1, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        t3_reg3 = IN_3D(k  , new_j, new_i);
        t2_reg3 = IN_3D(k-1, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
        t3_reg4 = IN_3D(k  , new_j, new_i);
        t2_reg4 = IN_3D(k-1, new_j, new_i);

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
            new_i = (warp_id_x<<6) + lane_id_it%66;
            new_j = (warp_id_y<<0) + lane_id_it/66;
            t3_reg0 = IN_3D(k+1, new_j, new_i);
            lane_id_it += warpSize;
            new_i = (warp_id_x<<6) + lane_id_it%66;
            new_j = (warp_id_y<<0) + lane_id_it/66;
            t3_reg1 = IN_3D(k+1, new_j, new_i);
            lane_id_it += warpSize;
            new_i = (warp_id_x<<6) + lane_id_it%66;
            new_j = (warp_id_y<<0) + lane_id_it/66;
            t3_reg2 = IN_3D(k+1, new_j, new_i);
            lane_id_it += warpSize;
            new_i = (warp_id_x<<6) + lane_id_it%66;
            new_j = (warp_id_y<<0) + lane_id_it/66;
            t3_reg3 = IN_3D(k+1, new_j, new_i);
            lane_id_it += warpSize;
            new_i = (warp_id_x<<6) + lane_id_it%66;
            new_j = (warp_id_y<<0) + lane_id_it/66;
            new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
            new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
            t3_reg4 = IN_3D(k+1, new_j, new_i);

            friend_id0 = (lane_id+1 )&(warpSize-1);
            friend_id1 = (lane_id+3 )&(warpSize-1);
            tx0 = __shfl(t2_reg0, friend_id0);
            ty0 = __shfl(t2_reg1, friend_id0);
            tx1 = __shfl(t2_reg1, friend_id1);
            ty1 = __shfl(t2_reg2, friend_id1);
            sum0 += a1*((lane_id < 63)? tx0: ty0);
            sum1 += a1*((lane_id < 61)? tx1: ty1);
                        
            friend_id0 = (lane_id+2 )&(warpSize-1);
            friend_id1 = (lane_id+4 )&(warpSize-1);
            tx0 = __shfl(t2_reg1, friend_id0);
            ty0 = __shfl(t2_reg2, friend_id0);
            tx1 = __shfl(t2_reg2, friend_id1);
            ty1 = __shfl(t2_reg3, friend_id1);
            sum0 += a2*((lane_id < 62)? tx0: ty0);
            sum1 += a2*((lane_id < 60)? tx1: ty1);
        
            friend_id0 = (lane_id+3 )&(warpSize-1);
            friend_id1 = (lane_id+5 )&(warpSize-1);
            tx0 = __shfl(t1_reg1, friend_id0);
            ty0 = __shfl(t1_reg2, friend_id0);
            tx1 = __shfl(t1_reg2, friend_id1);
            ty1 = __shfl(t1_reg3, friend_id1);
            sum0 += a0*((lane_id < 61)? tx0: ty0);
            sum1 += a0*((lane_id < 59)? tx1: ty1);
            tx0 = __shfl(t2_reg1, friend_id0);
            ty0 = __shfl(t2_reg2, friend_id0);
            tx1 = __shfl(t2_reg2, friend_id1);
            ty1 = __shfl(t2_reg3, friend_id1);
            sum0 += a3*((lane_id < 61)? tx0: ty0);
            sum1 += a3*((lane_id < 59)? tx1: ty1);
            tx0 = __shfl(t3_reg1, friend_id0);
            ty0 = __shfl(t3_reg2, friend_id0);
            tx1 = __shfl(t3_reg2, friend_id1);
            ty1 = __shfl(t3_reg3, friend_id1);
            sum0 += a6*((lane_id < 61)? tx0: ty0);
            sum1 += a6*((lane_id < 59)? tx1: ty1);

            friend_id0 = (lane_id+4 )&(warpSize-1);
            friend_id1 = (lane_id+6 )&(warpSize-1);
            tx0 = __shfl(t2_reg1, friend_id0);
            ty0 = __shfl(t2_reg2, friend_id0);
            tx1 = __shfl(t2_reg2, friend_id1);
            ty1 = __shfl(t2_reg3, friend_id1);
            sum0 += a4*((lane_id < 60)? tx0: ty0);
            sum1 += a4*((lane_id < 58)? tx1: ty1);

            friend_id0 = (lane_id+5 )&(warpSize-1);
            friend_id1 = (lane_id+7 )&(warpSize-1);
            tx0 = __shfl(t2_reg2, friend_id0);
            ty0 = __shfl(t2_reg3, friend_id0);
            tx1 = __shfl(t2_reg3, friend_id1);
            ty1 = __shfl(t2_reg4, friend_id1);
            sum0 += a5*((lane_id < 59)? tx0: ty0);
            sum1 += a5*((lane_id < 57)? tx1: ty1);


            OUT_3D(k,j  ,i) = sum0;
            OUT_3D(k,j+1,i) = sum1;
        
        }
    });
    fut.wait();
}

void Stencil_Hcc_Sweep_Shfl4_1DWarp(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 ,
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

        int warp_id_x = (tidx.global[2])>>6;             // because the warp dimensions are 
        int warp_id_y = ((((tidx.global[1])>>0)<<2))>>0; // 1x1x64, warp_ids are division of these numbers

        const int lane_id = __lane_id();
        int lane_id_it = lane_id;

        int new_i, new_j;
        DATA_TYPE t3_reg0, t3_reg1, t3_reg2, t3_reg3, t3_reg4, t3_reg5, t3_reg6;
        DATA_TYPE t2_reg0, t2_reg1, t2_reg2, t2_reg3, t2_reg4, t2_reg5, t2_reg6;
        DATA_TYPE t1_reg0, t1_reg1, t1_reg2, t1_reg3, t1_reg4, t1_reg5, t1_reg6;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        t3_reg0 = IN_3D(k  , new_j, new_i);
        t2_reg0 = IN_3D(k-1, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        t3_reg1 = IN_3D(k  , new_j, new_i);
        t2_reg1 = IN_3D(k-1, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        t3_reg2 = IN_3D(k  , new_j, new_i);
        t2_reg2 = IN_3D(k-1, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        t3_reg3 = IN_3D(k  , new_j, new_i);
        t2_reg3 = IN_3D(k-1, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        t3_reg4 = IN_3D(k  , new_j, new_i);
        t2_reg4 = IN_3D(k-1, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        t3_reg5 = IN_3D(k  , new_j, new_i);
        t2_reg5 = IN_3D(k-1, new_j, new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + lane_id_it%66;
        new_j = (warp_id_y<<0) + lane_id_it/66;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
        new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
        t3_reg6 = IN_3D(k  , new_j, new_i);
        t2_reg6 = IN_3D(k-1, new_j, new_i);


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
            new_i = (warp_id_x<<6) + lane_id_it%66;
            new_j = (warp_id_y<<0) + lane_id_it/66;
            t3_reg0 = IN_3D(k+1, new_j, new_i);
            lane_id_it += warpSize;
            new_i = (warp_id_x<<6) + lane_id_it%66;
            new_j = (warp_id_y<<0) + lane_id_it/66;
            t3_reg1 = IN_3D(k+1, new_j, new_i);
            lane_id_it += warpSize;
            new_i = (warp_id_x<<6) + lane_id_it%66;
            new_j = (warp_id_y<<0) + lane_id_it/66;
            t3_reg2 = IN_3D(k+1, new_j, new_i);
            lane_id_it += warpSize;
            new_i = (warp_id_x<<6) + lane_id_it%66;
            new_j = (warp_id_y<<0) + lane_id_it/66;
            t3_reg3 = IN_3D(k+1, new_j, new_i);
            lane_id_it += warpSize;
            new_i = (warp_id_x<<6) + lane_id_it%66;
            new_j = (warp_id_y<<0) + lane_id_it/66;
            t3_reg4 = IN_3D(k+1, new_j, new_i);
            lane_id_it += warpSize;
            new_i = (warp_id_x<<6) + lane_id_it%66;
            new_j = (warp_id_y<<0) + lane_id_it/66;
            t3_reg5 = IN_3D(k+1, new_j, new_i);
            lane_id_it += warpSize;
            new_i = (warp_id_x<<6) + lane_id_it%66;
            new_j = (warp_id_y<<0) + lane_id_it/66;
            new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
            new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
            t3_reg6 = IN_3D(k+1, new_j, new_i);

            friend_id0 = (lane_id+1 )&(warpSize-1);
            friend_id1 = (lane_id+3 )&(warpSize-1);
            friend_id2 = (lane_id+5 )&(warpSize-1);
            friend_id3 = (lane_id+7 )&(warpSize-1);
            tx0 = __shfl(t2_reg0, friend_id0);
            ty0 = __shfl(t2_reg1, friend_id0);
            tx1 = __shfl(t2_reg1, friend_id1);
            ty1 = __shfl(t2_reg2, friend_id1);
            tx2 = __shfl(t2_reg2, friend_id2);
            ty2 = __shfl(t2_reg3, friend_id2);
            tx3 = __shfl(t2_reg3, friend_id3);
            ty3 = __shfl(t2_reg4, friend_id3);
            sum0 += a1*((lane_id < 63)? tx0: ty0);
            sum1 += a1*((lane_id < 61)? tx1: ty1);
            sum2 += a1*((lane_id < 59)? tx2: ty2);
            sum3 += a1*((lane_id < 57)? tx3: ty3);
                        
            friend_id0 = (lane_id+2 )&(warpSize-1);
            friend_id1 = (lane_id+4 )&(warpSize-1);
            friend_id2 = (lane_id+6 )&(warpSize-1);
            friend_id3 = (lane_id+8 )&(warpSize-1);
            tx0 = __shfl(t2_reg1, friend_id0);
            ty0 = __shfl(t2_reg2, friend_id0);
            tx1 = __shfl(t2_reg2, friend_id1);
            ty1 = __shfl(t2_reg3, friend_id1);
            tx2 = __shfl(t2_reg3, friend_id2);
            ty2 = __shfl(t2_reg4, friend_id2);
            tx3 = __shfl(t2_reg4, friend_id3);
            ty3 = __shfl(t2_reg5, friend_id3);
            sum0 += a2*((lane_id < 62)? tx0: ty0);
            sum1 += a2*((lane_id < 60)? tx1: ty1);
            sum2 += a2*((lane_id < 58)? tx2: ty2);
            sum3 += a2*((lane_id < 56)? tx3: ty3);
        
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
            sum0 += a0*((lane_id < 61)? tx0: ty0);
            sum1 += a0*((lane_id < 59)? tx1: ty1);
            sum2 += a0*((lane_id < 57)? tx2: ty2);
            sum3 += a0*((lane_id < 55)? tx3: ty3);
            tx0 = __shfl(t2_reg1, friend_id0);
            ty0 = __shfl(t2_reg2, friend_id0);
            tx1 = __shfl(t2_reg2, friend_id1);
            ty1 = __shfl(t2_reg3, friend_id1);
            tx2 = __shfl(t2_reg3, friend_id2);
            ty2 = __shfl(t2_reg4, friend_id2);
            tx3 = __shfl(t2_reg4, friend_id3);
            ty3 = __shfl(t2_reg5, friend_id3);
            sum0 += a3*((lane_id < 61)? tx0: ty0);
            sum1 += a3*((lane_id < 59)? tx1: ty1);
            sum2 += a3*((lane_id < 57)? tx2: ty2);
            sum3 += a3*((lane_id < 55)? tx3: ty3);
            tx0 = __shfl(t3_reg1, friend_id0);
            ty0 = __shfl(t3_reg2, friend_id0);
            tx1 = __shfl(t3_reg2, friend_id1);
            ty1 = __shfl(t3_reg3, friend_id1);
            tx2 = __shfl(t3_reg3, friend_id2);
            ty2 = __shfl(t3_reg4, friend_id2);
            tx3 = __shfl(t3_reg4, friend_id3);
            ty3 = __shfl(t3_reg5, friend_id3);
            sum0 += a6*((lane_id < 61)? tx0: ty0);
            sum1 += a6*((lane_id < 59)? tx1: ty1);
            sum2 += a6*((lane_id < 57)? tx2: ty2);
            sum3 += a6*((lane_id < 55)? tx3: ty3);

            friend_id0 = (lane_id+4 )&(warpSize-1);
            friend_id1 = (lane_id+6 )&(warpSize-1);
            friend_id2 = (lane_id+8 )&(warpSize-1);
            friend_id3 = (lane_id+10)&(warpSize-1);
            tx0 = __shfl(t2_reg1, friend_id0);
            ty0 = __shfl(t2_reg2, friend_id0);
            tx1 = __shfl(t2_reg2, friend_id1);
            ty1 = __shfl(t2_reg3, friend_id1);
            tx2 = __shfl(t2_reg3, friend_id2);
            ty2 = __shfl(t2_reg4, friend_id2);
            tx3 = __shfl(t2_reg4, friend_id3);
            ty3 = __shfl(t2_reg5, friend_id3);
            sum0 += a4*((lane_id < 60)? tx0: ty0);
            sum1 += a4*((lane_id < 58)? tx1: ty1);
            sum2 += a4*((lane_id < 56)? tx2: ty2);
            sum3 += a4*((lane_id < 54)? tx3: ty3);
            
            friend_id0 = (lane_id+5 )&(warpSize-1);
            friend_id1 = (lane_id+7 )&(warpSize-1);
            friend_id2 = (lane_id+9 )&(warpSize-1);
            friend_id3 = (lane_id+11)&(warpSize-1);
            tx0 = __shfl(t2_reg2, friend_id0);
            ty0 = __shfl(t2_reg3, friend_id0);
            tx1 = __shfl(t2_reg3, friend_id1);
            ty1 = __shfl(t2_reg4, friend_id1);
            tx2 = __shfl(t2_reg4, friend_id2);
            ty2 = __shfl(t2_reg5, friend_id2);
            tx3 = __shfl(t2_reg5, friend_id3);
            ty3 = __shfl(t2_reg6, friend_id3);
            sum0 += a5*((lane_id < 59)? tx0: ty0);
            sum1 += a5*((lane_id < 57)? tx1: ty1);
            sum2 += a5*((lane_id < 55)? tx2: ty2);
            sum3 += a5*((lane_id < 53)? tx3: ty3);

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
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], 
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
        Stencil_Hcc(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    auto t2 = std::chrono::high_resolution_clock::now();
    double milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
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
        Stencil_Hcc_SweepX(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Sweep: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Sweep Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
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
        Stencil_Hcc_Sm_Branch(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Sm_Branch: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Sm_Branch Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
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
        Stencil_Hcc_Sm_Cyclic(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Sm_Cyclic: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Sm_Cyclic Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
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
        Stencil_Hcc_Sweep_Sm_BranchX(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Sweep_Sm_Branch: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Sweep_Sm_Branch Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
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
        Stencil_Hcc_Sweep_Sm_CyclicX(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Sweep_Sm_Cyclic: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Sweep_Sm_Cyclic Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc 3D-Block with Shfl 1-Point (2D-Warp)
    /////////////////////////////////////////////////////////
    Init_Input_3D(in, z, m, n, halo, seed);
    Clear_Output_3D(out, z, m, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Shfl_2DWarp(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Shfl_2DWarp: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Shfl_2DWarp Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
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
        Stencil_Hcc_Shfl2_2DWarp(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Shfl2_2DWarp: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Shfl2_2DWarp Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
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
        Stencil_Hcc_Shfl4_2DWarp(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Shfl4_2DWarp: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Shfl4_2DWarp Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
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
        Stencil_Hcc_Sweep_Shfl_1DWarp(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Sweep_Shfl_1DWarp: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Sweep_Shfl_1DWarp Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
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
        Stencil_Hcc_Sweep_Shfl2_1DWarp(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Sweep_Shfl2_1DWarp: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Sweep_Shfl2_1DWarp Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
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
        Stencil_Hcc_Sweep_Shfl4_1DWarp(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Sweep_Shfl4_1DWarp: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Sweep_Shfl4_1DWarp Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // std::cout << OUT_3D(1,1,1) << std::endl;
    delete[] in;
    delete[] out;
    delete[] out_ref;
}
