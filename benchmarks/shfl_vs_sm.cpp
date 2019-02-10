#include <iostream>
#include <algorithm>
#include <cmath>
#include <hc.hpp>

using namespace hc;
// using namespace std;

// #define DATA_TYPE float 
#define warpSize 64

// #define __shfl(a, b) amdgcn_ds_bpermute((b)<<2, a)

//*
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
inline double __xamdgcn_ds_bpermute(double src, int lane) restrict(amp)
{
    long val = *((long*)(&src));
    int src1 =  (int)val&0xffffffff;
    int t1 = __amdgcn_ds_bpermute(lane<<2, src1);
    int src2 =  (int)(val>>32)&0xffffffff;
    int t2 = __amdgcn_ds_bpermute(lane<<2, src2);
    long ans = t2;
    ans = (ans<<32)|t1;
    return *((double*)(&ans));
}
inline float __xamdgcn_ds_bpermute(float src, int lane) restrict(amp)
{
    int t = __amdgcn_ds_bpermute(lane<<2, *((int*)(&src)));
    return *((float*)(&t));
}
// */

void kern_shfl_load(array<DATA_TYPE> &in_d, array<DATA_TYPE> &out_d, int n)
{
    extent<1> comp_domain(n); 
    tiled_extent<1> comp_tile(comp_domain, 256);
    completion_future fut = parallel_for_each(comp_tile, [=, &in_d, &out_d](tiled_index<1> tidx) restrict(amp) {
        int i = tidx.global[0] ;
        unsigned int lane_id = __lane_id();

        DATA_TYPE reg = in_d[i];
        DATA_TYPE sum = 0;
        int friend_id = (lane_id+1)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+2)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+3)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+4)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+5)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+6)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+7)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+8)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        
        friend_id = (lane_id+9)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        
        friend_id = (lane_id+10)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+11)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+12)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+13)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+14)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+15)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+16)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+17)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+18)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        
        friend_id = (lane_id+19)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+20)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+21)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+22)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+23)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+24)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+25)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+26)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+27)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+28)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        
        friend_id = (lane_id+29)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+30)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        friend_id = (lane_id+31)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;

        out_d[i] = sum;
        
    });
    fut.wait();
}

void kern_shfl_sync(array<DATA_TYPE> &in_d, array<DATA_TYPE> &out_d, int n)
{
    extent<1> comp_domain(n); 
    tiled_extent<1> comp_tile(comp_domain, 256);
    completion_future fut = parallel_for_each(comp_tile, [=, &in_d, &out_d](tiled_index<1> tidx) restrict(amp) {
        int i = tidx.global[0] ;
        unsigned int lane_id = __lane_id();
        
        DATA_TYPE reg = in_d[i];
        DATA_TYPE sum = 0;
        int friend_id = (lane_id+1)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+2)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+3)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+4)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+5)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+6)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+7)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+8)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;
        
        friend_id = (lane_id+9)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;
        
        friend_id = (lane_id+10)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+11)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+12)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+13)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+14)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+15)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+16)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+17)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+18)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;
        
        friend_id = (lane_id+19)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+20)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+21)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+22)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+23)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+24)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+25)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+26)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+27)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+28)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;
        
        friend_id = (lane_id+29)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+30)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        friend_id = (lane_id+31)&(warpSize-1);
        sum += __xamdgcn_ds_bpermute(reg, friend_id) ;
        reg = sum;

        out_d[i] = reg;
        
    });
    fut.wait();
}

void kern_sm_load(array<DATA_TYPE> &in_d, array<DATA_TYPE> &out_d, int n)
{
    extent<1> comp_domain(n); 
    tiled_extent<1> comp_tile(comp_domain, 256);
    completion_future fut = parallel_for_each(comp_tile, [=, &in_d, &out_d](tiled_index<1> tidx) restrict(amp) {
        int i = tidx.global[0] ;
        unsigned int tid = tidx.local[0]; 
        unsigned int lane_id = __lane_id();
        tile_static DATA_TYPE sm[256];

        sm[tid] = in_d[i];
        tidx.barrier.wait();

        int friend_id = (lane_id+1)&(warpSize-1);
        DATA_TYPE sum = 0 ;
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+2)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+3)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+4)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+5)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        
        friend_id = (lane_id+6)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+7)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+8)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+9)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+10)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+11)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+12)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+13)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+14)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+15)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+16)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+17)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+18)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+19)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+20)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+21)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+22)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+23)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+24)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+25)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+26)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+27)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+28)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+29)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+30)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        friend_id = (lane_id+31)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;

        out_d[i] = sum;
        
    });
    fut.wait();
}

void kern_sm_sync(array<DATA_TYPE> &in_d, array<DATA_TYPE> &out_d, int n)
{
    extent<1> comp_domain(n); 
    tiled_extent<1> comp_tile(comp_domain, 256);
    completion_future fut = parallel_for_each(comp_tile, [=, &in_d, &out_d](tiled_index<1> tidx) restrict(amp) {
        int i = tidx.global[0] ;
        unsigned int tid = tidx.local[0]; 
        unsigned int lane_id = __lane_id();
        tile_static DATA_TYPE sm[256];

        sm[tid] = in_d[i];
        tidx.barrier.wait();

        int friend_id = (lane_id+1)&(warpSize-1);
        DATA_TYPE sum = 0 ;
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+2)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+3)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+4)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+5)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();
        
        friend_id = (lane_id+6)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+7)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+8)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+9)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+10)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+11)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+12)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+13)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+14)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+15)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+16)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+17)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+18)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+19)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+20)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+21)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+22)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+23)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+24)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+25)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+26)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+27)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+28)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+29)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+30)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        friend_id = (lane_id+31)&(warpSize-1);
        sum += sm[((tid>>6)<<6)+friend_id] ;
        sm[tid] = sum;
        tidx.barrier.wait();

        out_d[i] = sm[tid];
        
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
            std::cout << "wrong at " << i << " test:" << test[i] << " (ref: " << ref[i] << ")";
            std::cout << std::endl;
            flag = false;
            break;
        }
    }
    return flag;
}

int quick_pow(int a, int b)
{
    int ans = 1;
    while(b>0)
    {
        if(b&1) ans = ans*a;
        a = a*a;
        b>>=1;
    }
    return ans;
}

int main(int argc, char **argv)
{
    int n = 33554432;
    if(argc!=1)
        n = quick_pow(2, atoi(argv[1]));
    std::cout << "size: " << n << std::endl;
    int total = n;

    DATA_TYPE *in = new DATA_TYPE[total];
    DATA_TYPE *out1 = new DATA_TYPE[total];
    DATA_TYPE *out2 = new DATA_TYPE[total];
    for(int i = 0; i < n; i++)
    {
        in[i] = i&(warpSize-1);
    }
    std::fill_n(out1, total, 0);
    std::fill_n(out2, total, 0);

    extent<1> data_domain(total);
    array<DATA_TYPE>  in_d(data_domain);
    array<DATA_TYPE> out_d(data_domain);
    copy(in , in_d );
    auto t1 = std::chrono::high_resolution_clock::now();
    kern_shfl_load(in_d, out_d, n);
    auto t2 = std::chrono::high_resolution_clock::now();
    double milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    std::cout << "shfl_load time(ms): " << milliseconds << std::endl;
    copy(out_d, out1);

    t1 = std::chrono::high_resolution_clock::now();
    kern_sm_load(in_d, out_d, n);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    std::cout << "sm_load time(ms): " << milliseconds << std::endl;
    copy(out_d, out2);
    std::cout << "verify load results: " << std::boolalpha << Verify(out1, out2, total) << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    kern_shfl_sync(in_d, out_d, n);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    std::cout << "shfl_sync time(ms): " << milliseconds << std::endl;
    copy(out_d, out1);

    t1 = std::chrono::high_resolution_clock::now();
    kern_sm_sync(in_d, out_d, n);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    std::cout << "sm_sync time(ms): " << milliseconds << std::endl;
    copy(out_d, out2);
    std::cout << "verify sync results: " << std::boolalpha << Verify(out1, out2, total) << std::endl;


}
