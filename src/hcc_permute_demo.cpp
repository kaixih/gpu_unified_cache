#include <iostream>
#include <algorithm>
#include <cmath>
#include <hc.hpp>

using namespace hc;

#define DTYPE double


inline double __xamdgcn_ds_bpermute(int lane, double src) restrict(amp)
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

inline float __xamdgcn_ds_bpermute(int lane, float src) restrict(amp)
{
    int t = __amdgcn_ds_bpermute(lane<<2, *((int*)(&src)));
    return *((float*)(&t));
}

int main(int argc, char **argv)
{
    int n = 64;
    DTYPE *in = new DTYPE[n];
    DTYPE *out = new DTYPE[n];
    DTYPE i = 0.0;
    std::generate(in, in+n, [&](){ DTYPE v = i; i += 1.0; return v;});
    std::fill(out, out+n, 0.0);
    std::cout << "in:" << std::endl;
    std::for_each(in, in+n, [](DTYPE v){std::cout << v << ",";});
    std::cout << std::endl;

    extent<1> data_domain(n);
    array<DTYPE> in_d(data_domain);
    array<DTYPE> out_d(data_domain);
    copy(in, in_d);
    extent<1> comp_domain(n); 
    tiled_extent<1> comp_tile(comp_domain, 64);
    completion_future fut = parallel_for_each(comp_tile, [=, &in_d, &out_d](tiled_index<1> tidx) restrict(amp) {
        int i = __lane_id();
        DTYPE val = in_d[i];
        DTYPE t = __xamdgcn_ds_bpermute(i, val);
        out_d[i] = t;
    });
    fut.wait();
    copy(out_d, out);
    
    std::cout << "out:" << std::endl;
    std::for_each(out, out+n, [](DTYPE v){std::cout << v << ",";});
    std::cout << std::endl;


    delete[] in;
    delete[] out;

}

