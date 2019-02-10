#include <iostream>
#include <cmath>
#include <hc.hpp>
#include <metrics.h>

using namespace hc;

#define  IN_1D(_x)  in[_x]
#define OUT_1D(_x) out[_x]
#define ARG_1D(_l,_x) args[(_l)*(n+2*halo)+(_x)]

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

void Init_Args_1D(DATA_TYPE *args, int l, int n, int halo, DATA_TYPE val)
{
    for(int k = 0; k < l; k++)
    {
        for(int i = 0; i < n+2*halo; i++)
        {
            ARG_1D(k,i) = val; 
        }
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
        DATA_TYPE *args, 
        int n, int halo)
{
#pragma omp parallel for
    for(int i = halo; i < n+halo; i++)
    {
        OUT_1D(i) = ARG_1D(0,i)*IN_1D(i-1) + 
                    ARG_1D(1,i)*IN_1D(i  ) + 
                    ARG_1D(2,i)*IN_1D(i+1) ;
    }
}


void Stencil_Hcc(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        array<DATA_TYPE> &args, 
        int n, int halo)
{
    extent<1> comp_domain(n); 
    tiled_extent<1> comp_tile(comp_domain, 256);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out, &args](tiled_index<1> tidx) restrict(amp) {
        int i = tidx.global[0] + halo;

        OUT_1D(i) = ARG_1D(0,i)*IN_1D(i-1) + 
                    ARG_1D(1,i)*IN_1D(i  ) + 
                    ARG_1D(2,i)*IN_1D(i+1) ;
    });
    fut.wait();
}

void Stencil_Hcc_Sm_Branch(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        array<DATA_TYPE> &args, 
        int n, int halo)
{
    extent<1> comp_domain(n); 
    tiled_extent<1> comp_tile(comp_domain, 256);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out, &args](tiled_index<1> tidx) restrict(amp) {
        tile_static DATA_TYPE local[256+2];
        unsigned int tid = tidx.local[0];
        unsigned int gid = tidx.global[0] + halo;
        int local_id = tid + halo;

        local[local_id] = IN_1D(gid);
        if(tid == 0)
        {
            local[local_id-1] = IN_1D(gid-1);
        }
        if(tid == 255)
        {
            local[local_id+1] = IN_1D(gid+1);
        }
        tidx.barrier.wait();

        OUT_1D(gid) = ARG_1D(0,gid)*local[local_id-1] + 
                      ARG_1D(1,gid)*local[local_id  ] + 
                      ARG_1D(2,gid)*local[local_id+1] ;
        
    });
    fut.wait();
}

void Stencil_Hcc_Sm_Cyclic(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        array<DATA_TYPE> &args,
        int n, int halo)
{
    extent<1> comp_domain(n); 
    tiled_extent<1> comp_tile(comp_domain, 256);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out, &args](tiled_index<1> tidx) restrict(amp) {
        tile_static DATA_TYPE local[256+2];
        unsigned int tid = tidx.local[0];
        unsigned int gid = tidx.global[0] + halo;
        int local_id = tid + halo;

        unsigned int lane_id = tidx.local[0];
        int lane_id_it = lane_id;
        int blk_id_x = tidx.tile[0];
        int new_i  = (blk_id_x<<8) + lane_id_it%258;
        int new_li = lane_id_it%258;
        local[new_li] = IN_1D(new_i);
        lane_id_it += 256;
        new_i  = (blk_id_x<<8) + (lane_id_it/258)*258 + lane_id_it%258;
        new_li = (lane_id_it/258)*258 + lane_id_it%258;
        if(new_li < 258)
            local[new_li] = IN_1D(new_i);
        tidx.barrier.wait();

        OUT_1D(gid) = ARG_1D(0,gid)*local[local_id-1] + 
                      ARG_1D(1,gid)*local[local_id  ] + 
                      ARG_1D(2,gid)*local[local_id+1] ;
    });
    fut.wait();
}

void Stencil_Hcc_Shfl(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        array<DATA_TYPE> &args,
        int n, int halo)
{
    extent<1> comp_domain(n); 
    tiled_extent<1> comp_tile(comp_domain, 256);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out, &args](tiled_index<1> tidx) restrict(amp) {
        unsigned int gid = tidx.global[0] + halo;
        unsigned int tid = tidx.local[0];
        unsigned int lane_id = __lane_id();

        int warp_id_x = (tidx.global[0])>>6;

        DATA_TYPE reg0, reg1;
        int lane_id_it = lane_id;
        int new_i = (warp_id_x<<6) + lane_id_it%70;
        reg0 = IN_1D(new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + (lane_id_it/70)*70 + lane_id_it%70;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
        reg1 = IN_1D(new_i);

        DATA_TYPE sum0 = 0.0;
        int friend_id0;
        DATA_TYPE tx0, ty0;

        friend_id0 = (lane_id+0 )&(warpSize-1);
        tx0 = __shfl(reg0, friend_id0);
        sum0 += ARG_1D(0,gid)*tx0;

        friend_id0 = (lane_id+1 )&(warpSize-1);
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += ARG_1D(1,gid)*((lane_id < 63)? tx0: ty0);

        friend_id0 = (lane_id+2 )&(warpSize-1);
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        sum0 += ARG_1D(2,gid)*((lane_id < 62)? tx0: ty0);

        OUT_1D(gid) = sum0;
        
    });
    fut.wait();
}

void Stencil_Hcc_Shfl2(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        array<DATA_TYPE> &args,
        int n, int halo)
{
    extent<1> comp_domain(n/2); 
    tiled_extent<1> comp_tile(comp_domain, 256);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out, &args](tiled_index<1> tidx) restrict(amp) {
        unsigned int lane_id = __lane_id();
        unsigned int gid = (((tidx.global[0])>>6)<<7) + lane_id + halo;

        int warp_id_x = ((((tidx.global[0])>>6)<<7) + lane_id)>>6;

        DATA_TYPE reg0, reg1, reg2;
        int lane_id_it = lane_id;
        int new_i = (warp_id_x<<6) + lane_id_it%70;
        reg0 = IN_1D(new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + (lane_id_it/70)*70 + lane_id_it%70;
        reg1 = IN_1D(new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + (lane_id_it/70)*70 + lane_id_it%70;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
        reg2 = IN_1D(new_i);

        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;
        int friend_id0;
        DATA_TYPE tx0, ty0;
        DATA_TYPE tx1, ty1;

        friend_id0 = (lane_id+0 )&(warpSize-1);
        tx0 = __shfl(reg0, friend_id0);
        tx1 = __shfl(reg1, friend_id0);
        sum0 += ARG_1D(0,gid   )*tx0;
        sum1 += ARG_1D(0,gid+64)*tx1;

        friend_id0 = (lane_id+1 )&(warpSize-1);
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        tx1 = __shfl(reg1, friend_id0);
        ty1 = __shfl(reg2, friend_id0);
        sum0 += ARG_1D(1,gid   )*((lane_id < 63)? tx0: ty0);
        sum1 += ARG_1D(1,gid+64)*((lane_id < 63)? tx1: ty1);

        friend_id0 = (lane_id+2 )&(warpSize-1);
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        tx1 = __shfl(reg1, friend_id0);
        ty1 = __shfl(reg2, friend_id0);
        sum0 += ARG_1D(2,gid   )*((lane_id < 62)? tx0: ty0);
        sum1 += ARG_1D(2,gid+64)*((lane_id < 62)? tx1: ty1);

        OUT_1D(gid   ) = sum0;
        OUT_1D(gid+64) = sum1; 
        
    });
    fut.wait();
}

void Stencil_Hcc_Shfl4(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        array<DATA_TYPE> &args,
        int n, int halo)
{
    extent<1> comp_domain(n/4); 
    tiled_extent<1> comp_tile(comp_domain, 256);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out, &args](tiled_index<1> tidx) restrict(amp) {
        unsigned int lane_id = __lane_id();
        unsigned int gid = (((tidx.global[0])>>6)<<8) + lane_id + halo;

        int warp_id_x = ((((tidx.global[0])>>6)<<8) + lane_id)>>6;

        DATA_TYPE reg0, reg1, reg2, reg3, reg4;
        int lane_id_it = lane_id;
        int new_i = (warp_id_x<<6) + lane_id_it%70;
        reg0 = IN_1D(new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + (lane_id_it/70)*70 + lane_id_it%70;
        reg1 = IN_1D(new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + (lane_id_it/70)*70 + lane_id_it%70;
        reg2 = IN_1D(new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + (lane_id_it/70)*70 + lane_id_it%70;
        reg3 = IN_1D(new_i);
        lane_id_it += warpSize;
        new_i = (warp_id_x<<6) + (lane_id_it/70)*70 + lane_id_it%70;
        new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
        reg4 = IN_1D(new_i);

        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;
        DATA_TYPE sum2 = 0.0;
        DATA_TYPE sum3 = 0.0;
        int friend_id0;
        // int friend_id1;
        DATA_TYPE tx0, ty0, tx1, ty1;
        DATA_TYPE tx2, ty2, tx3, ty3;

        friend_id0 = (lane_id+0 )&(warpSize-1);
        tx0 = __shfl(reg0, friend_id0);
        tx1 = __shfl(reg1, friend_id0);
        tx2 = __shfl(reg2, friend_id0);
        tx3 = __shfl(reg3, friend_id0);
        sum0 += ARG_1D(0,gid    )*tx0;
        sum1 += ARG_1D(0,gid+64 )*tx1;
        sum2 += ARG_1D(0,gid+128)*tx2;
        sum3 += ARG_1D(0,gid+192)*tx3;

        friend_id0 = (lane_id+1 )&(warpSize-1);
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        tx1 = __shfl(reg1, friend_id0);
        ty1 = __shfl(reg2, friend_id0);
        tx2 = __shfl(reg2, friend_id0);
        ty2 = __shfl(reg3, friend_id0);
        tx3 = __shfl(reg3, friend_id0);
        ty3 = __shfl(reg4, friend_id0);
        sum0 += ARG_1D(1,gid    )*((lane_id < 63)? tx0: ty0);
        sum1 += ARG_1D(1,gid+64 )*((lane_id < 63)? tx1: ty1);
        sum2 += ARG_1D(1,gid+128)*((lane_id < 63)? tx2: ty2);
        sum3 += ARG_1D(1,gid+192)*((lane_id < 63)? tx3: ty3);

        friend_id0 = (lane_id+2 )&(warpSize-1);
        tx0 = __shfl(reg0, friend_id0);
        ty0 = __shfl(reg1, friend_id0);
        tx1 = __shfl(reg1, friend_id0);
        ty1 = __shfl(reg2, friend_id0);
        tx2 = __shfl(reg2, friend_id0);
        ty2 = __shfl(reg3, friend_id0);
        tx3 = __shfl(reg3, friend_id0);
        ty3 = __shfl(reg4, friend_id0);
        sum0 += ARG_1D(2,gid    )*((lane_id < 62)? tx0: ty0);
        sum1 += ARG_1D(2,gid+64 )*((lane_id < 62)? tx1: ty1);
        sum2 += ARG_1D(2,gid+128)*((lane_id < 62)? tx2: ty2);
        sum3 += ARG_1D(2,gid+192)*((lane_id < 62)? tx3: ty3);

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
    int halo = 1; 
    int total = (n+2*halo);
    int K = total*3;
    DATA_TYPE *args = new DATA_TYPE[K];
#ifdef __DEBUG
    Init_Args_1D(args, 3, n, halo, 1.0);
#else
    Init_Args_1D(args, 3, n, halo, 0.33);
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
                args, 
                n, halo);
        std::swap(in, out_ref);
    }
    std::swap(in, out_ref);
    // Show_Me(out_ref, n, halo, "Output:");

    extent<1> data_domain(total);
    extent<1> args_domain(K);
    array<DATA_TYPE>  in_d(data_domain);
    array<DATA_TYPE> out_d(data_domain);
    array<DATA_TYPE> args_d(args_domain);
    DATA_TYPE *out = new DATA_TYPE[total];
    copy(args, args_d);
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
        Stencil_Hcc(in_d, out_d, 
                args_d, 
                n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    auto t2 = std::chrono::high_resolution_clock::now();
    double milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(ARGC_1D3, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc Shared Memory with Branch
    /////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Sm_Branch(in_d, out_d, 
                args_d, 
                n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Sm_Branch: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Sm_Branch Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(ARGC_1D3, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc Shared Memory with Cyclic
    /////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Sm_Cyclic(in_d, out_d, 
                args_d, 
                n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Sm_Cyclic: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Sm_Cyclic Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(ARGC_1D3, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc Shfl with 1D-Warp
    /////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Shfl(in_d, out_d, 
                args_d, 
                n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Shfl: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Shfl Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(ARGC_1D3, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc Shfl2 with 1D-Warp
    /////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Shfl2(in_d, out_d, 
                args_d, 
                n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Shfl2: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Shfl2 Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(ARGC_1D3, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc Shfl4 with 1D-Warp
    /////////////////////////////////////////
    Init_Input_1D(in, n, halo, seed);
    Clear_Output_1D(out, n, halo);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Shfl4(in_d, out_d, 
                args_d, 
                n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Shfl4: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Shfl4 Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(1, 1, n+2*halo, ITER, OPS_1D3, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(ARGC_1D3, 1, 1, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

  
    delete[] in;
    delete[] out;
    delete[] out_ref;

}
