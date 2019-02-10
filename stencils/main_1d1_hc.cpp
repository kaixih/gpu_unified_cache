#include <iostream>
#include <cmath>
#include <hc.hpp>

using namespace hc;
// using namespace std;

#define DATA_TYPE float
#define warpSize 64 
#define STEPS 100

#define START_TIME for(int _i = 0; _i < STEPS; _i++) {
#define END_TIME }

#define IN_1D(_x) in[_x]
#define OUT_1D(_x) out[_x]

void Init_Input_1D(DATA_TYPE *in, int n, int halo)
{
    srand(time(NULL));
    for(int i = 0; i < halo; i++) IN_1D(i) = 0.0;
    for(int i = halo; i < n+halo; i++) 
        IN_1D(i) = (DATA_TYPE)rand() * 100.0 / RAND_MAX;
    for(int i = n+halo; i < n+2*halo; i++) IN_1D(i) = 0.0;
}

void Fill_Halo_1D(DATA_TYPE *in, int n, int halo)
{
    for(int i = 0; i < halo; i++) IN_1D(i) = 0.0;
    for(int i = n+halo; i < n+2*halo; i++) IN_1D(i) = 0.0;
}

void Show_Me(DATA_TYPE *array, int n, std::string prompt)
{
    std::cout << prompt << std::endl;
    for(int i = 0; i < n; i++)
        std::cout << array[i] << ",";
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
    double precision = tol_finder(8);

    for(int i = 0; i < n; i++)
    {
        if(fabs(test[i]-ref[i])>precision)
        {
            std::cout << "wrong at " << i << " test:" << test[i] << " (ref: " << ref[i] << ")";
            std::cout << std::endl;
            flag = false;
            break;
        }
    }
    return flag;
}

void Stencil_Seq(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int halo)
{
    for(int i = 0; i < halo; i++) OUT_1D(i) = 0.0;

    for(int i = halo; i < n+halo; i++)
    {
        OUT_1D(i) = args[0]*IN_1D(i-1) + 
                    args[1]*IN_1D(i  ) + 
                    args[2]*IN_1D(i+1) ;
    }

    for(int i = n+halo; i < n+2*halo; i++) OUT_1D(i) = 0.0;
}

void Stencil_Hcc(array<DATA_TYPE> &in, array<DATA_TYPE> &out, array<DATA_TYPE> &args, int n, int halo) 
{
    extent<1> compute_domain(n+255); 
    tiled_extent<1> compute_tile(compute_domain, 256);
    completion_future fut = parallel_for_each(compute_tile, [=, &in, &out, &args](tiled_index<1> tidx) restrict(amp) {
        int i = tidx.global[0] + halo;
        if(i < n + halo)
        {
            OUT_1D(i) = args[0]*IN_1D(i-1) + 
                        args[1]*IN_1D(i  ) + 
                        args[2]*IN_1D(i+1) ;
        }
    });
    fut.wait();
}

void Stencil_Hcc_Shfl(array<DATA_TYPE> &in, array<DATA_TYPE> &out, array<DATA_TYPE> &args, int n, int halo) 
{
    extent<1> compute_domain(n+255); 
    tiled_extent<1> compute_tile(compute_domain, 256);
    completion_future fut = parallel_for_each(compute_tile, [=, &in, &out, &args](tiled_index<1> tidx) restrict(amp) {
        int gid = tidx.global[0] + halo;
        int lane_id = __lane_id();
        DATA_TYPE threadInput0, threadInput1;
        int lowIdx = gid - halo;
        int highIdx = lowIdx + warpSize;
        threadInput0 = IN_1D(lowIdx);
        if(highIdx < n + 2*halo)
            threadInput1 = IN_1D(highIdx);

        DATA_TYPE sum = 0.0;
        sum += args[0]*threadInput0;

        int friend_id = (lane_id + 1) % warpSize;
        DATA_TYPE reg_id = lane_id == 0 ? threadInput1: threadInput0;
        sum += args[1]*__shfl(reg_id, friend_id, warpSize);

        friend_id = (lane_id + 2) % warpSize;
        reg_id    = (lane_id == 0 || lane_id == 1) ? threadInput1: threadInput0;
        sum += args[2]*__shfl(reg_id, friend_id, warpSize);

        if(gid < n + halo)
        {
            OUT_1D(gid) = sum; 
        }
    });
    fut.wait();
}

void Stencil_Hcc_Shfl_Direct(array<DATA_TYPE> &in, array<DATA_TYPE> &out, array<DATA_TYPE> &args, int n, int halo) 
{
    extent<1> compute_domain(n+255); 
    tiled_extent<1> compute_tile(compute_domain, 256);
    completion_future fut = parallel_for_each(compute_tile, [=, &in, &out, &args](tiled_index<1> tidx) restrict(amp) {
        int gid = tidx.global[0] + halo;
        int lane_id = __lane_id();
        DATA_TYPE threadInput0, threadInput1;
        int lowIdx = gid - halo;
        int highIdx = lowIdx + warpSize;
        threadInput0 = IN_1D(lowIdx);
        if(highIdx < n + 2*halo)
            threadInput1 = IN_1D(highIdx);

        DATA_TYPE sum = 0.0;
        sum += args[0]*threadInput0;

        int friend_id = (lane_id + 1) % warpSize;
        DATA_TYPE reg_id = lane_id == 0 ? threadInput1: threadInput0;
        // sum += args[1]*__shfl(reg_id, friend_id, warpSize);
        __u tmp; 
        tmp.f =  reg_id;
        tmp.i = amdgcn_ds_bpermute(friend_id<<2, tmp.i);
        sum += args[1]*tmp.f;

        friend_id = (lane_id + 2) % warpSize;
        reg_id    = (lane_id == 0 || lane_id == 1) ? threadInput1: threadInput0;
        // sum += args[2]*__shfl(reg_id, friend_id, warpSize);
        tmp.f =  reg_id;
        tmp.i = amdgcn_ds_bpermute(friend_id<<2, tmp.i);
        sum += args[2]*tmp.f;

        if(gid < n + halo)
        {
            OUT_1D(gid) = sum; 
        }
    });
    fut.wait();
}

void Stencil_Hcc_Shfl2(array<DATA_TYPE> &in, array<DATA_TYPE> &out, array<DATA_TYPE> &args, int n, int halo) 
{
    extent<1> compute_domain((n+1)/2+255); 
    tiled_extent<1> compute_tile(compute_domain, 256);
    completion_future fut = parallel_for_each(compute_tile, [=, &in, &out, &args](tiled_index<1> tidx) restrict(amp) {
        int lane_id = __lane_id();
        int gid = (tidx.global[0])/warpSize*(2*warpSize) + lane_id + halo;
        DATA_TYPE threadInput0, threadInput1, threadInput2;
        int lowIdx1 = gid - halo;
        int lowIdx2 = lowIdx1 + warpSize;
        int highIdx = lowIdx2 + warpSize;
        threadInput0 = IN_1D(lowIdx1);
        threadInput1 = IN_1D(lowIdx2);
        if(highIdx < n + 2*halo)
            threadInput2 = IN_1D(highIdx);

        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;
        sum0 += args[0]*threadInput0;
        sum1 += args[0]*threadInput1;

        int friend_id = (lane_id + 1) % warpSize;
        DATA_TYPE reg_id0 = lane_id == 0 ? threadInput1: threadInput0;
        DATA_TYPE reg_id1 = lane_id == 0 ? threadInput2: threadInput1;
        sum0 += args[1]*__shfl(reg_id0, friend_id);
        sum1 += args[1]*__shfl(reg_id1, friend_id);

        friend_id = (lane_id + 2) % warpSize;
        reg_id0    = (lane_id == 0 || lane_id == 1) ? threadInput1: threadInput0;
        reg_id1    = (lane_id == 0 || lane_id == 1) ? threadInput2: threadInput1;
        sum0 += args[2]*__shfl(reg_id0, friend_id);
        sum1 += args[2]*__shfl(reg_id1, friend_id);

        if(gid < n + halo)
        {
            OUT_1D(gid) = sum0; 
        }

        if(gid + warpSize < n + halo)
        {
            OUT_1D(gid+warpSize) = sum1;
        }
        
    });
    fut.wait();
}

void Stencil_Hcc_Shfl2_Direct(array<DATA_TYPE> &in, array<DATA_TYPE> &out, array<DATA_TYPE> &args, int n, int halo) 
{
    extent<1> compute_domain((n+1)/2+255); 
    tiled_extent<1> compute_tile(compute_domain, 256);
    completion_future fut = parallel_for_each(compute_tile, [=, &in, &out, &args](tiled_index<1> tidx) restrict(amp) {
        int lane_id = __lane_id();
        int gid = (tidx.global[0])/warpSize*(2*warpSize) + lane_id + halo;
        DATA_TYPE threadInput0, threadInput1, threadInput2;
        int lowIdx1 = gid - halo;
        int lowIdx2 = lowIdx1 + warpSize;
        int highIdx = lowIdx2 + warpSize;
        threadInput0 = IN_1D(lowIdx1);
        threadInput1 = IN_1D(lowIdx2);
        if(highIdx < n + 2*halo)
            threadInput2 = IN_1D(highIdx);

        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;
        sum0 += args[0]*threadInput0;
        sum1 += args[0]*threadInput1;

        int friend_id = (lane_id + 1) % warpSize;
        DATA_TYPE reg_id0 = lane_id == 0 ? threadInput1: threadInput0;
        DATA_TYPE reg_id1 = lane_id == 0 ? threadInput2: threadInput1;
        // sum0 += args[1]*__shfl(reg_id0, friend_id);
        // sum1 += args[1]*__shfl(reg_id1, friend_id);
        __u tmp0, tmp1; 
        tmp0.f =  reg_id0;
        tmp1.f =  reg_id1;
        tmp0.i = amdgcn_ds_bpermute(friend_id<<2, tmp0.i);
        tmp1.i = amdgcn_ds_bpermute(friend_id<<2, tmp1.i);
        sum0 += args[1]*tmp0.f;
        sum1 += args[1]*tmp1.f;


        friend_id = (lane_id + 2) % warpSize;
        reg_id0    = (lane_id == 0 || lane_id == 1) ? threadInput1: threadInput0;
        reg_id1    = (lane_id == 0 || lane_id == 1) ? threadInput2: threadInput1;
        // sum0 += args[2]*__shfl(reg_id0, friend_id);
        // sum1 += args[2]*__shfl(reg_id1, friend_id);
        tmp0.f =  reg_id0;
        tmp1.f =  reg_id1;
        tmp0.i = amdgcn_ds_bpermute(friend_id<<2, tmp0.i);
        tmp1.i = amdgcn_ds_bpermute(friend_id<<2, tmp1.i);
        sum0 += args[2]*tmp0.f;
        sum1 += args[2]*tmp1.f;


        if(gid < n + halo)
        {
            OUT_1D(gid) = sum0; 
        }

        if(gid + warpSize < n + halo)
        {
            OUT_1D(gid+warpSize) = sum1;
        }
        
    });
    fut.wait();
}

void Stencil_Hcc_Shfl4(array<DATA_TYPE> &in, array<DATA_TYPE> &out, array<DATA_TYPE> &args, int n, int halo) 
{
    extent<1> compute_domain((n+3)/4+255); 
    tiled_extent<1> compute_tile(compute_domain, 256);
    completion_future fut = parallel_for_each(compute_tile, [=, &in, &out, &args](tiled_index<1> tidx) restrict(amp) {
        int lane_id = __lane_id();
        int gid = (tidx.global[0])/warpSize*(4*warpSize) + lane_id + halo;

        DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4; 
        int lowIdx1 = gid - halo;
        int lowIdx2 = lowIdx1 + warpSize;
        int lowIdx3 = lowIdx2 + warpSize;
        int lowIdx4 = lowIdx3 + warpSize;
        int highIdx = lowIdx4 + warpSize;
        threadInput0 = IN_1D(lowIdx1);
        threadInput1 = IN_1D(lowIdx2);
        threadInput2 = IN_1D(lowIdx3);
        threadInput3 = IN_1D(lowIdx4);
        if(highIdx < n + 2*halo)
            threadInput4 = IN_1D(highIdx);

        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;
        DATA_TYPE sum2 = 0.0;
        DATA_TYPE sum3 = 0.0;
        sum0 += args[0]*threadInput0;
        sum1 += args[0]*threadInput1;
        sum2 += args[0]*threadInput2;
        sum3 += args[0]*threadInput3;

        int friend_id = (lane_id + 1) % warpSize;
        DATA_TYPE reg_id0 = lane_id == 0 ? threadInput1: threadInput0;
        DATA_TYPE reg_id1 = lane_id == 0 ? threadInput2: threadInput1;
        DATA_TYPE reg_id2 = lane_id == 0 ? threadInput3: threadInput2;
        DATA_TYPE reg_id3 = lane_id == 0 ? threadInput4: threadInput3;
        sum0 += args[1]*__shfl(reg_id0, friend_id);
        sum1 += args[1]*__shfl(reg_id1, friend_id);
        sum2 += args[1]*__shfl(reg_id2, friend_id);
        sum3 += args[1]*__shfl(reg_id3, friend_id);

        friend_id = (lane_id + 2) % warpSize;
        reg_id0    = (lane_id == 0 || lane_id == 1) ? threadInput1: threadInput0;
        reg_id1    = (lane_id == 0 || lane_id == 1) ? threadInput2: threadInput1;
        reg_id2    = (lane_id == 0 || lane_id == 1) ? threadInput3: threadInput2;
        reg_id3    = (lane_id == 0 || lane_id == 1) ? threadInput4: threadInput3;
        sum0 += args[2]*__shfl(reg_id0, friend_id);
        sum1 += args[2]*__shfl(reg_id1, friend_id);
        sum2 += args[2]*__shfl(reg_id2, friend_id);
        sum3 += args[2]*__shfl(reg_id3, friend_id);

        if(gid < n + halo)
        {
            OUT_1D(gid) = sum0; 
        }

        if(gid + 1*warpSize < n + halo)
        {
            OUT_1D(gid+1*warpSize) = sum1;
        }

        if(gid + 2*warpSize < n + halo)
        {
            OUT_1D(gid+2*warpSize) = sum2;
        }

        if(gid + 3*warpSize < n + halo)
        {
            OUT_1D(gid+3*warpSize) = sum3;
        }
        
        
    });
    fut.wait();
}

void Stencil_Hcc_Shfl8(array<DATA_TYPE> &in, array<DATA_TYPE> &out, array<DATA_TYPE> &args, int n, int halo) 
{
    extent<1> compute_domain((n+7)/8+255); 
    tiled_extent<1> compute_tile(compute_domain, 256);
    completion_future fut = parallel_for_each(compute_tile, [=, &in, &out, &args](tiled_index<1> tidx) restrict(amp) {
        int lane_id = __lane_id();
        int gid = (tidx.global[0])/warpSize*(8*warpSize) + lane_id + halo;
        DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5, 
              threadInput6, threadInput7, threadInput8;
        int lowIdx1 = gid - halo;
        int lowIdx2 = lowIdx1 + warpSize;
        int lowIdx3 = lowIdx2 + warpSize;
        int lowIdx4 = lowIdx3 + warpSize;
        int lowIdx5 = lowIdx4 + warpSize;
        int lowIdx6 = lowIdx5 + warpSize;
        int lowIdx7 = lowIdx6 + warpSize;
        int lowIdx8 = lowIdx7 + warpSize;
        int highIdx = lowIdx8 + warpSize;
        threadInput0 = IN_1D(lowIdx1);
        threadInput1 = IN_1D(lowIdx2);
        threadInput2 = IN_1D(lowIdx3);
        threadInput3 = IN_1D(lowIdx4);
        threadInput4 = IN_1D(lowIdx5);
        threadInput5 = IN_1D(lowIdx6);
        threadInput6 = IN_1D(lowIdx7);
        threadInput7 = IN_1D(lowIdx8);
        if(highIdx < n + 2*halo)
            threadInput8 = IN_1D(highIdx);

        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;
        DATA_TYPE sum2 = 0.0;
        DATA_TYPE sum3 = 0.0;
        DATA_TYPE sum4 = 0.0;
        DATA_TYPE sum5 = 0.0;
        DATA_TYPE sum6 = 0.0;
        DATA_TYPE sum7 = 0.0;
        sum0 += args[0]*threadInput0;
        sum1 += args[0]*threadInput1;
        sum2 += args[0]*threadInput2;
        sum3 += args[0]*threadInput3;
        sum4 += args[0]*threadInput4;
        sum5 += args[0]*threadInput5;
        sum6 += args[0]*threadInput6;
        sum7 += args[0]*threadInput7;

        int friend_id = (lane_id + 1) % warpSize;
        DATA_TYPE reg_id0 = lane_id == 0 ? threadInput1: threadInput0;
        DATA_TYPE reg_id1 = lane_id == 0 ? threadInput2: threadInput1;
        DATA_TYPE reg_id2 = lane_id == 0 ? threadInput3: threadInput2;
        DATA_TYPE reg_id3 = lane_id == 0 ? threadInput4: threadInput3;
        DATA_TYPE reg_id4 = lane_id == 0 ? threadInput5: threadInput4;
        DATA_TYPE reg_id5 = lane_id == 0 ? threadInput6: threadInput5;
        DATA_TYPE reg_id6 = lane_id == 0 ? threadInput7: threadInput6;
        DATA_TYPE reg_id7 = lane_id == 0 ? threadInput8: threadInput7;
        sum0 += args[1]*__shfl(reg_id0, friend_id);
        sum1 += args[1]*__shfl(reg_id1, friend_id);
        sum2 += args[1]*__shfl(reg_id2, friend_id);
        sum3 += args[1]*__shfl(reg_id3, friend_id);
        sum4 += args[1]*__shfl(reg_id4, friend_id);
        sum5 += args[1]*__shfl(reg_id5, friend_id);
        sum6 += args[1]*__shfl(reg_id6, friend_id);
        sum7 += args[1]*__shfl(reg_id7, friend_id);

        friend_id = (lane_id + 2) % warpSize;
        reg_id0    = (lane_id == 0 || lane_id == 1) ? threadInput1: threadInput0;
        reg_id1    = (lane_id == 0 || lane_id == 1) ? threadInput2: threadInput1;
        reg_id2    = (lane_id == 0 || lane_id == 1) ? threadInput3: threadInput2;
        reg_id3    = (lane_id == 0 || lane_id == 1) ? threadInput4: threadInput3;
        reg_id4    = (lane_id == 0 || lane_id == 1) ? threadInput5: threadInput4;
        reg_id5    = (lane_id == 0 || lane_id == 1) ? threadInput6: threadInput5;
        reg_id6    = (lane_id == 0 || lane_id == 1) ? threadInput7: threadInput6;
        reg_id7    = (lane_id == 0 || lane_id == 1) ? threadInput8: threadInput7;
        sum0 += args[2]*__shfl(reg_id0, friend_id);
        sum1 += args[2]*__shfl(reg_id1, friend_id);
        sum2 += args[2]*__shfl(reg_id2, friend_id);
        sum3 += args[2]*__shfl(reg_id3, friend_id);
        sum4 += args[2]*__shfl(reg_id4, friend_id);
        sum5 += args[2]*__shfl(reg_id5, friend_id);
        sum6 += args[2]*__shfl(reg_id6, friend_id);
        sum7 += args[2]*__shfl(reg_id7, friend_id);

        if(gid < n + halo)
        {
            OUT_1D(gid) = sum0; 
        }

        if(gid + warpSize < n + halo)
        {
            OUT_1D(gid+warpSize) = sum1;
        }

        if(gid + 2*warpSize < n + halo)
        {
            OUT_1D(gid+2*warpSize) = sum2;
        }

        if(gid + 3*warpSize < n + halo)
        {
            OUT_1D(gid+3*warpSize) = sum3;
        }

        if(gid + 4*warpSize < n + halo)
        {
            OUT_1D(gid+4*warpSize) = sum4;
        }

        if(gid + 5*warpSize < n + halo)
        {
            OUT_1D(gid+5*warpSize) = sum5;
        }

        if(gid + 6*warpSize < n + halo)
        {
            OUT_1D(gid+6*warpSize) = sum6;
        }

        if(gid + 7*warpSize < n + halo)
        {
            OUT_1D(gid+7*warpSize) = sum7;
        }
        
    });
    fut.wait();
}

void Stencil_Hcc_Shfl16(array<DATA_TYPE> &in, array<DATA_TYPE> &out, array<DATA_TYPE> &args, int n, int halo) 
{
    extent<1> compute_domain((n+15)/16+255); 
    tiled_extent<1> compute_tile(compute_domain, 256);
    completion_future fut = parallel_for_each(compute_tile, [=, &in, &out, &args](tiled_index<1> tidx) restrict(amp) {
        int lane_id = __lane_id();
        int gid = (tidx.global[0])/warpSize*(16*warpSize) + lane_id + halo;

        DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3,
                  threadInput4, threadInput5, threadInput6, threadInput7,
                  threadInput8, threadInput9, threadInput10, threadInput11, 
                  threadInput12, threadInput13, threadInput14, threadInput15,
                  threadInput16;
        int lowIdx1  = gid - halo;
        int lowIdx2  = lowIdx1  + warpSize;
        int lowIdx3  = lowIdx2  + warpSize;
        int lowIdx4  = lowIdx3  + warpSize;
        int lowIdx5  = lowIdx4  + warpSize;
        int lowIdx6  = lowIdx5  + warpSize;
        int lowIdx7  = lowIdx6  + warpSize;
        int lowIdx8  = lowIdx7  + warpSize;
        int lowIdx9  = lowIdx8  + warpSize;
        int lowIdx10 = lowIdx9  + warpSize;
        int lowIdx11 = lowIdx10 + warpSize;
        int lowIdx12 = lowIdx11 + warpSize;
        int lowIdx13 = lowIdx12 + warpSize;
        int lowIdx14 = lowIdx13 + warpSize;
        int lowIdx15 = lowIdx14 + warpSize;
        int lowIdx16 = lowIdx15 + warpSize;
        int highIdx  = lowIdx16 + warpSize;
        threadInput0  = IN_1D(lowIdx1 );
        threadInput1  = IN_1D(lowIdx2 );
        threadInput2  = IN_1D(lowIdx3 );
        threadInput3  = IN_1D(lowIdx4 );
        threadInput4  = IN_1D(lowIdx5 );
        threadInput5  = IN_1D(lowIdx6 );
        threadInput6  = IN_1D(lowIdx7 );
        threadInput7  = IN_1D(lowIdx8 );
        threadInput8  = IN_1D(lowIdx9 );
        threadInput9  = IN_1D(lowIdx10);
        threadInput10 = IN_1D(lowIdx11);
        threadInput11 = IN_1D(lowIdx12);
        threadInput12 = IN_1D(lowIdx13);
        threadInput13 = IN_1D(lowIdx14);
        threadInput14 = IN_1D(lowIdx15);
        threadInput15 = IN_1D(lowIdx16);
        if(highIdx < n + 2*halo)
            threadInput16 = IN_1D(highIdx);

        DATA_TYPE sum0  = 0.0;
        DATA_TYPE sum1  = 0.0;
        DATA_TYPE sum2  = 0.0;
        DATA_TYPE sum3  = 0.0;
        DATA_TYPE sum4  = 0.0;
        DATA_TYPE sum5  = 0.0;
        DATA_TYPE sum6  = 0.0;
        DATA_TYPE sum7  = 0.0;
        DATA_TYPE sum8  = 0.0;
        DATA_TYPE sum9  = 0.0;
        DATA_TYPE sum10 = 0.0;
        DATA_TYPE sum11 = 0.0;
        DATA_TYPE sum12 = 0.0;
        DATA_TYPE sum13 = 0.0;
        DATA_TYPE sum14 = 0.0;
        DATA_TYPE sum15 = 0.0;
        sum0  += args[0]*threadInput0 ;
        sum1  += args[0]*threadInput1 ;
        sum2  += args[0]*threadInput2 ;
        sum3  += args[0]*threadInput3 ;
        sum4  += args[0]*threadInput4 ;
        sum5  += args[0]*threadInput5 ;
        sum6  += args[0]*threadInput6 ;
        sum7  += args[0]*threadInput7 ;
        sum8  += args[0]*threadInput8 ;
        sum9  += args[0]*threadInput9 ;
        sum10 += args[0]*threadInput10;
        sum11 += args[0]*threadInput11;
        sum12 += args[0]*threadInput12;
        sum13 += args[0]*threadInput13;
        sum14 += args[0]*threadInput14;
        sum15 += args[0]*threadInput15;

        int friend_id = (lane_id + 1) % warpSize;
        DATA_TYPE reg_id0     = lane_id == 0 ? threadInput1 : threadInput0 ;
        DATA_TYPE reg_id1     = lane_id == 0 ? threadInput2 : threadInput1 ;
        DATA_TYPE reg_id2     = lane_id == 0 ? threadInput3 : threadInput2 ;
        DATA_TYPE reg_id3     = lane_id == 0 ? threadInput4 : threadInput3 ;
        DATA_TYPE reg_id4     = lane_id == 0 ? threadInput5 : threadInput4 ;
        DATA_TYPE reg_id5     = lane_id == 0 ? threadInput6 : threadInput5 ;
        DATA_TYPE reg_id6     = lane_id == 0 ? threadInput7 : threadInput6 ;
        DATA_TYPE reg_id7     = lane_id == 0 ? threadInput8 : threadInput7 ;
        DATA_TYPE reg_id8     = lane_id == 0 ? threadInput9 : threadInput8 ;
        DATA_TYPE reg_id9     = lane_id == 0 ? threadInput10: threadInput9 ;
        DATA_TYPE reg_id10    = lane_id == 0 ? threadInput11: threadInput10;
        DATA_TYPE reg_id11    = lane_id == 0 ? threadInput12: threadInput11;
        DATA_TYPE reg_id12    = lane_id == 0 ? threadInput13: threadInput12;
        DATA_TYPE reg_id13    = lane_id == 0 ? threadInput14: threadInput13;
        DATA_TYPE reg_id14    = lane_id == 0 ? threadInput15: threadInput14;
        DATA_TYPE reg_id15    = lane_id == 0 ? threadInput16: threadInput15;
        sum0  += args[1]*__shfl(reg_id0 , friend_id);
        sum1  += args[1]*__shfl(reg_id1 , friend_id);
        sum2  += args[1]*__shfl(reg_id2 , friend_id);
        sum3  += args[1]*__shfl(reg_id3 , friend_id);
        sum4  += args[1]*__shfl(reg_id4 , friend_id);
        sum5  += args[1]*__shfl(reg_id5 , friend_id);
        sum6  += args[1]*__shfl(reg_id6 , friend_id);
        sum7  += args[1]*__shfl(reg_id7 , friend_id);
        sum8  += args[1]*__shfl(reg_id8 , friend_id);
        sum9  += args[1]*__shfl(reg_id9 , friend_id);
        sum10 += args[1]*__shfl(reg_id10, friend_id);
        sum11 += args[1]*__shfl(reg_id11, friend_id);
        sum12 += args[1]*__shfl(reg_id12, friend_id);
        sum13 += args[1]*__shfl(reg_id13, friend_id);
        sum14 += args[1]*__shfl(reg_id14, friend_id);
        sum15 += args[1]*__shfl(reg_id15, friend_id);

        friend_id = (lane_id + 2) % warpSize;
        reg_id0     = (lane_id == 0 || lane_id == 1) ? threadInput1 : threadInput0 ;
        reg_id1     = (lane_id == 0 || lane_id == 1) ? threadInput2 : threadInput1 ;
        reg_id2     = (lane_id == 0 || lane_id == 1) ? threadInput3 : threadInput2 ;
        reg_id3     = (lane_id == 0 || lane_id == 1) ? threadInput4 : threadInput3 ;
        reg_id4     = (lane_id == 0 || lane_id == 1) ? threadInput5 : threadInput4 ;
        reg_id5     = (lane_id == 0 || lane_id == 1) ? threadInput6 : threadInput5 ;
        reg_id6     = (lane_id == 0 || lane_id == 1) ? threadInput7 : threadInput6 ;
        reg_id7     = (lane_id == 0 || lane_id == 1) ? threadInput8 : threadInput7 ;
        reg_id8     = (lane_id == 0 || lane_id == 1) ? threadInput9 : threadInput8 ;
        reg_id9     = (lane_id == 0 || lane_id == 1) ? threadInput10: threadInput9 ;
        reg_id10    = (lane_id == 0 || lane_id == 1) ? threadInput11: threadInput10;
        reg_id11    = (lane_id == 0 || lane_id == 1) ? threadInput12: threadInput11;
        reg_id12    = (lane_id == 0 || lane_id == 1) ? threadInput13: threadInput12;
        reg_id13    = (lane_id == 0 || lane_id == 1) ? threadInput14: threadInput13;
        reg_id14    = (lane_id == 0 || lane_id == 1) ? threadInput15: threadInput14;
        reg_id15    = (lane_id == 0 || lane_id == 1) ? threadInput16: threadInput15;
        sum0  += args[2]*__shfl(reg_id0 , friend_id);
        sum1  += args[2]*__shfl(reg_id1 , friend_id);
        sum2  += args[2]*__shfl(reg_id2 , friend_id);
        sum3  += args[2]*__shfl(reg_id3 , friend_id);
        sum4  += args[2]*__shfl(reg_id4 , friend_id);
        sum5  += args[2]*__shfl(reg_id5 , friend_id);
        sum6  += args[2]*__shfl(reg_id6 , friend_id);
        sum7  += args[2]*__shfl(reg_id7 , friend_id);
        sum8  += args[2]*__shfl(reg_id8 , friend_id);
        sum9  += args[2]*__shfl(reg_id9 , friend_id);
        sum10 += args[2]*__shfl(reg_id10, friend_id);
        sum11 += args[2]*__shfl(reg_id11, friend_id);
        sum12 += args[2]*__shfl(reg_id12, friend_id);
        sum13 += args[2]*__shfl(reg_id13, friend_id);
        sum14 += args[2]*__shfl(reg_id14, friend_id);
        sum15 += args[2]*__shfl(reg_id15, friend_id);

        if(gid < n + halo)
        {
            OUT_1D(gid) = sum0; 
        }

        if(gid + warpSize < n + halo)
        {
            OUT_1D(gid+warpSize) = sum1;
        }

        if(gid + 2*warpSize < n + halo)
        {
            OUT_1D(gid+2*warpSize) = sum2;
        }

        if(gid + 3*warpSize < n + halo)
        {
            OUT_1D(gid+3*warpSize) = sum3;
        }

        if(gid + 4*warpSize < n + halo)
        {
            OUT_1D(gid+4*warpSize) = sum4;
        }

        if(gid + 5*warpSize < n + halo)
        {
            OUT_1D(gid+5*warpSize) = sum5;
        }

        if(gid + 6*warpSize < n + halo)
        {
            OUT_1D(gid+6*warpSize) = sum6;
        }

        if(gid + 7*warpSize < n + halo)
        {
            OUT_1D(gid+7*warpSize) = sum7;
        }

        if(gid + 8*warpSize < n + halo)
        {
            OUT_1D(gid+8*warpSize) = sum8; 
        }

        if(gid + 9*warpSize < n + halo)
        {
            OUT_1D(gid+9*warpSize) = sum9;
        }

        if(gid + 10*warpSize < n + halo)
        {
            OUT_1D(gid+10*warpSize) = sum10;
        }

        if(gid + 11*warpSize < n + halo)
        {
            OUT_1D(gid+11*warpSize) = sum11;
        }

        if(gid + 12*warpSize < n + halo)
        {
            OUT_1D(gid+12*warpSize) = sum12;
        }

        if(gid + 13*warpSize < n + halo)
        {
            OUT_1D(gid+13*warpSize) = sum13;
        }

        if(gid + 14*warpSize < n + halo)
        {
            OUT_1D(gid+14*warpSize) = sum14;
        }

        if(gid + 15*warpSize < n + halo)
        {
            OUT_1D(gid+15*warpSize) = sum15;
        }  
        
    });
    fut.wait();
}

void Stencil_Hcc_Sm(array<DATA_TYPE> &in, array<DATA_TYPE> &out, array<DATA_TYPE> &args, int n, int halo) 
{
    extent<1> compute_domain(n+255); 
    tiled_extent<1> compute_tile(compute_domain, 256);
#define LM_SIZE 256
    completion_future fut = parallel_for_each(compute_tile, [=, &in, &out, &args](tiled_index<1> tidx) restrict(amp) {
        tile_static DATA_TYPE local[LM_SIZE+2];
        int gid = tidx.global[0] + halo;
        unsigned int tid = tidx.local[0];
        int local_id = tid + halo;
        local[local_id] = IN_1D(gid);
        if(tid == 0)
            local[local_id-halo] = IN_1D(gid-halo);
        if(tid == LM_SIZE - 1)
            local[local_id+halo] = IN_1D(gid+halo);
        tidx.barrier.wait();

        if(gid < n + halo)
        {
            OUT_1D(gid) = args[0]*local[local_id-1] + 
                          args[1]*local[local_id  ] + 
                          args[2]*local[local_id+1] ;
        }
    });
    fut.wait();
}

int main(int argc, char **argv)
{
    int n = 100000000;
    // int n = 64;
    int halo = 1; 
    const int K = 3;
    DATA_TYPE args[K] = {1.0, 1.0, 1.0};
    DATA_TYPE *in = new DATA_TYPE[n+2*halo];
    DATA_TYPE *out_ref = new DATA_TYPE[n+2*halo];
    Init_Input_1D(in, n, halo);

    // Show_Me(in, n+2*halo, "Input:");
    Stencil_Seq(in, out_ref, args, n, halo);
    // Show_Me(out_ref, n+2*halo, "Output:");

    DATA_TYPE *out = new DATA_TYPE[n+2*halo];
    extent<1> data_domain(n+2*halo);
    extent<1> args_domain(K);
    array<DATA_TYPE> in_d(data_domain, in, in+n+2*halo);
    array<DATA_TYPE> out_d(data_domain);
    array<DATA_TYPE> args_d(args_domain, args, args+K);

    Stencil_Hcc(in_d, out_d, args_d, n, halo); // warmup

    Init_Input_1D(out, n, halo);
    copy(out, out_d);

    auto t1 = std::chrono::high_resolution_clock::now();
    START_TIME; 
    Stencil_Hcc(in_d, out_d, args_d, n, halo); 
    END_TIME;
    auto t2 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    Fill_Halo_1D(out, n, halo);
    // Show_Me(out, n+2*halo, "Output(Device):");
    std::cout << "Verify Hcc: " << std::boolalpha << Verify(out, out_ref, n+2*halo) << std::endl;
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout << "Hcc Time: " << timeInNS << std::endl;

    Init_Input_1D(out, n, halo);
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME; 
    Stencil_Hcc_Shfl(in_d, out_d, args_d, n, halo); 
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    Fill_Halo_1D(out, n, halo);
    // Show_Me(out, n+2*halo, "Output(Device):");
    std::cout << "Verify Hcc_Shfl: " << std::boolalpha << Verify(out, out_ref, n+2*halo) << std::endl;
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout << "Hcc_Shfl Time: " << timeInNS << std::endl;

    Init_Input_1D(out, n, halo);
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME; 
    Stencil_Hcc_Shfl_Direct(in_d, out_d, args_d, n, halo); 
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    Fill_Halo_1D(out, n, halo);
    // Show_Me(out, n+2*halo, "Output(Shfl_Direct):");
    std::cout << "Verify Hcc_Shfl_Direct: " << std::boolalpha << Verify(out, out_ref, n+2*halo) << std::endl;
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout << "Hcc_Shfl_Direct Time: " << timeInNS << std::endl;


    Init_Input_1D(out, n, halo);
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME; 
    Stencil_Hcc_Shfl2(in_d, out_d, args_d, n, halo); 
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    Fill_Halo_1D(out, n, halo);
    // Show_Me(out, n+2*halo, "Output(Device):");
    std::cout << "Verify Hcc_Shfl2: " << std::boolalpha << Verify(out, out_ref, n+2*halo) << std::endl;
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout << "Hcc_Shfl2 Time: " << timeInNS << std::endl;

    Init_Input_1D(out, n, halo);
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME; 
    Stencil_Hcc_Shfl2_Direct(in_d, out_d, args_d, n, halo); 
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    Fill_Halo_1D(out, n, halo);
    // Show_Me(out, n+2*halo, "Output(Shfl2_Direct):");
    std::cout << "Verify Hcc_Shfl2_Direct: " << std::boolalpha << Verify(out, out_ref, n+2*halo) << std::endl;
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout << "Hcc_Shfl2_Direct Time: " << timeInNS << std::endl;


    Init_Input_1D(out, n, halo);
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME; 
    Stencil_Hcc_Shfl4(in_d, out_d, args_d, n, halo); 
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    Fill_Halo_1D(out, n, halo);
    // Show_Me(out, n+2*halo, "Output(Device):");
    std::cout << "Verify Hcc_Shfl4: " << std::boolalpha << Verify(out, out_ref, n+2*halo) << std::endl;
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout << "Hcc_Shfl4 Time: " << timeInNS << std::endl;

    Init_Input_1D(out, n, halo);
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME; 
    Stencil_Hcc_Shfl8(in_d, out_d, args_d, n, halo); 
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    Fill_Halo_1D(out, n, halo);
    // Show_Me(out, n+2*halo, "Output(Device):");
    std::cout << "Verify Hcc_Shfl8: " << std::boolalpha << Verify(out, out_ref, n+2*halo) << std::endl;
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout << "Hcc_Shfl8 Time: " << timeInNS << std::endl;

    Init_Input_1D(out, n, halo);
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME; 
    Stencil_Hcc_Shfl16(in_d, out_d, args_d, n, halo); 
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    Fill_Halo_1D(out, n, halo);
    // Show_Me(out, n+2*halo, "Output(Device):");
    std::cout << "Verify Hcc_Shfl16: " << std::boolalpha << Verify(out, out_ref, n+2*halo) << std::endl;
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout << "Hcc_Shfl16 Time: " << timeInNS << std::endl;

    Init_Input_1D(out, n, halo);
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME; 
    Stencil_Hcc_Sm(in_d, out_d, args_d, n, halo); 
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    Fill_Halo_1D(out, n, halo);
    // Show_Me(out, n+2*halo, "Output(Device):");
    std::cout << "Verify Hcc_Sm: " << std::boolalpha << Verify(out, out_ref, n+2*halo) << std::endl;
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout << "Hcc_Sm Time: " << timeInNS << std::endl;

    delete[] in;
    delete[] out;
    delete[] out_ref;
}

