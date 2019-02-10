#include <iostream>
#include <cmath>
#include <hc.hpp>

using namespace hc;

#define DATA_TYPE float
#define STEPS 100
#define START_TIME for(int _i = 0; _i < STEPS; _i++) {
#define END_TIME }

double tol_finder(int error_tol)
{
    double val = 1.0;
    for(; error_tol > 0; error_tol--)
        val *= 10;
    return 1.0/(double)val;
}

void stencil_seq(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int wrapper_size)
{
    for(int i = wrapper_size/2; i < n+wrapper_size/2; i++)
    {
        out[i] = args[0 ]*in[i-7] + 
                 args[1 ]*in[i-6] + 
                 args[2 ]*in[i-5] +
                 args[3 ]*in[i-4] + 
                 args[4 ]*in[i-3] + 
                 args[5 ]*in[i-2] +
                 args[6 ]*in[i-1] + 
                 args[7 ]*in[i  ] + 
                 args[8 ]*in[i+1] +
                 args[9 ]*in[i+2] + 
                 args[10]*in[i+3] + 
                 args[11]*in[i+4] +
                 args[12]*in[i+5] + 
                 args[13]*in[i+6] + 
                 args[14]*in[i+7] ;
    }
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = 0;
    out[6] = 0;
    out[n+wrapper_size/2+0] = 0;
    out[n+wrapper_size/2+1] = 0;
    out[n+wrapper_size/2+2] = 0;
    out[n+wrapper_size/2+3] = 0;
    out[n+wrapper_size/2+4] = 0;
    out[n+wrapper_size/2+5] = 0;
    out[n+wrapper_size/2+6] = 0;
}

void stencil_hcc(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int wrapper_size)
{
    extent<1> ct_domain(n+wrapper_size);
    extent<1> cp_domain(n); 
    tiled_extent<1> cp_tile(cp_domain, 256);

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + n + wrapper_size);
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(wrapper_size+1, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<1> tidx) restrict(amp) {
        int i = tidx.global[0] + wrapper_size/2;
        out_d[i] = args_d[0 ]*in_d[i-7] + 
                   args_d[1 ]*in_d[i-6] + 
                   args_d[2 ]*in_d[i-5] +
                   args_d[3 ]*in_d[i-4] + 
                   args_d[4 ]*in_d[i-3] + 
                   args_d[5 ]*in_d[i-2] +
                   args_d[6 ]*in_d[i-1] + 
                   args_d[7 ]*in_d[i  ] + 
                   args_d[8 ]*in_d[i+1] +
                   args_d[9 ]*in_d[i+2] + 
                   args_d[10]*in_d[i+3] + 
                   args_d[11]*in_d[i+4] +
                   args_d[12]*in_d[i+5] + 
                   args_d[13]*in_d[i+6] + 
                   args_d[14]*in_d[i+7] ;
    });
    fut.wait();
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "kern: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "d2h: %lg ns\n", timeInNS);
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = 0;
    out[6] = 0;
    out[n+wrapper_size/2+0] = 0;
    out[n+wrapper_size/2+1] = 0;
    out[n+wrapper_size/2+2] = 0;
    out[n+wrapper_size/2+3] = 0;
    out[n+wrapper_size/2+4] = 0;
    out[n+wrapper_size/2+5] = 0;
    out[n+wrapper_size/2+6] = 0;
}

void stencil_hcc_shfl(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int wrapper_size)
{
    extent<1> ct_domain(n+wrapper_size);
    extent<1> cp_domain(n); 
    tiled_extent<1> cp_tile(cp_domain, 256);

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + n + wrapper_size);
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(wrapper_size+1, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<1> tidx) restrict(amp) {

        int gid = tidx.global[0] + wrapper_size/2;
        int wid = __lane_id();
        DATA_TYPE threadInput[8];
        int lowIdx1 = gid - 7;
        int highIdx = lowIdx1 + 64;
        threadInput[0] = in_d[lowIdx1];
        if(highIdx < n + wrapper_size)
            threadInput[1] = in_d[highIdx];

        DATA_TYPE sum = 0.0f;
        sum += args_d[0]*threadInput[0];
        int new_wid = (wid + 1) % 64;
        int new_reg = wid == 0 ? 1: 0;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 1: 0;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 1: 0;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 1: 0;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 1: 0;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 1: 0;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 1: 0;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 1: 0;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 1: 0;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 1: 0;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 1: 0;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 1: 0;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 1: 0;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 1: 0;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid] = sum;

        // out_d[i] = args_d[0 ]*in_d[i-7] + 
                   // args_d[1 ]*in_d[i-6] + 
                   // args_d[2 ]*in_d[i-5] +
                   // args_d[3 ]*in_d[i-4] + 
                   // args_d[4 ]*in_d[i-3] + 
                   // args_d[5 ]*in_d[i-2] +
                   // args_d[6 ]*in_d[i-1] + 
                   // args_d[7 ]*in_d[i  ] + 
                   // args_d[8 ]*in_d[i+1] +
                   // args_d[9 ]*in_d[i+2] + 
                   // args_d[10]*in_d[i+3] + 
                   // args_d[11]*in_d[i+4] +
                   // args_d[12]*in_d[i+5] + 
                   // args_d[13]*in_d[i+6] + 
                   // args_d[14]*in_d[i+7] ;

    });
    fut.wait();
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "kern: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "d2h: %lg ns\n", timeInNS);
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = 0;
    out[6] = 0;
    out[n+wrapper_size/2+0] = 0;
    out[n+wrapper_size/2+1] = 0;
    out[n+wrapper_size/2+2] = 0;
    out[n+wrapper_size/2+3] = 0;
    out[n+wrapper_size/2+4] = 0;
    out[n+wrapper_size/2+5] = 0;
    out[n+wrapper_size/2+6] = 0;
}

void stencil_hcc_shfl2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int wrapper_size)
{
    extent<1> ct_domain(n+wrapper_size);
    extent<1> cp_domain(n/2); 
    tiled_extent<1> cp_tile(cp_domain, 256);

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + n + wrapper_size);
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(wrapper_size+1, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<1> tidx) restrict(amp) {

        int wid = __lane_id();
        int gid = (tidx.global[0]/64)*128 + wid + wrapper_size/2;
        DATA_TYPE threadInput[3];
        int lowIdx1 = gid - 7;
        int lowIdx2 = lowIdx1 + 64;
        int highIdx = lowIdx2 + 64;
        threadInput[0] = in_d[lowIdx1];
        threadInput[1] = in_d[lowIdx2];
        if(highIdx < n + wrapper_size)
            threadInput[2] = in_d[highIdx];

        DATA_TYPE sum = 0.0f;
        sum += args_d[0]*threadInput[0];
        int new_wid = (wid + 1) % 64;
        int new_reg = wid == 0 ? 1: 0;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 1: 0;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 1: 0;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 1: 0;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 1: 0;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 1: 0;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 1: 0;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 1: 0;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 1: 0;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 1: 0;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 1: 0;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 1: 0;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 1: 0;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 1: 0;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[1];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 2: 1;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 2: 1;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 2: 1;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 2: 1;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 2: 1;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 2: 1;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 2: 1;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 2: 1;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 2: 1;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 2: 1;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 2: 1;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 2: 1;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 2: 1;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 2: 1;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+64] = sum;

        // out_d[i] = args_d[0 ]*in_d[i-7] + 
                   // args_d[1 ]*in_d[i-6] + 
                   // args_d[2 ]*in_d[i-5] +
                   // args_d[3 ]*in_d[i-4] + 
                   // args_d[4 ]*in_d[i-3] + 
                   // args_d[5 ]*in_d[i-2] +
                   // args_d[6 ]*in_d[i-1] + 
                   // args_d[7 ]*in_d[i  ] + 
                   // args_d[8 ]*in_d[i+1] +
                   // args_d[9 ]*in_d[i+2] + 
                   // args_d[10]*in_d[i+3] + 
                   // args_d[11]*in_d[i+4] +
                   // args_d[12]*in_d[i+5] + 
                   // args_d[13]*in_d[i+6] + 
                   // args_d[14]*in_d[i+7] ;

    });
    fut.wait();
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "kern: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "d2h: %lg ns\n", timeInNS);
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = 0;
    out[6] = 0;
    out[n+wrapper_size/2+0] = 0;
    out[n+wrapper_size/2+1] = 0;
    out[n+wrapper_size/2+2] = 0;
    out[n+wrapper_size/2+3] = 0;
    out[n+wrapper_size/2+4] = 0;
    out[n+wrapper_size/2+5] = 0;
    out[n+wrapper_size/2+6] = 0;
}

void stencil_hcc_shfl4(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int wrapper_size)
{
    extent<1> ct_domain(n+wrapper_size);
    extent<1> cp_domain(n/4); 
    tiled_extent<1> cp_tile(cp_domain, 256);

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + n + wrapper_size);
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(wrapper_size+1, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<1> tidx) restrict(amp) {

        int wid = __lane_id();
        int gid = (tidx.global[0]/64)*256 + wid + wrapper_size/2;
        DATA_TYPE threadInput[5];
        int lowIdx1 = gid - 7;
        int lowIdx2 = lowIdx1 + 64;
        int lowIdx3 = lowIdx2 + 64;
        int lowIdx4 = lowIdx3 + 64;
        int highIdx = lowIdx4 + 64;
        threadInput[0] = in_d[lowIdx1];
        threadInput[1] = in_d[lowIdx2];
        threadInput[2] = in_d[lowIdx3];
        threadInput[3] = in_d[lowIdx4];
        if(highIdx < n + wrapper_size)
            threadInput[4] = in_d[highIdx];

        DATA_TYPE sum = 0.0f;
        sum += args_d[0]*threadInput[0];
        int new_wid = (wid + 1) % 64;
        int new_reg = wid == 0 ? 1: 0;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 1: 0;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 1: 0;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 1: 0;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 1: 0;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 1: 0;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 1: 0;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 1: 0;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 1: 0;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 1: 0;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 1: 0;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 1: 0;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 1: 0;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 1: 0;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[1];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 2: 1;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 2: 1;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 2: 1;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 2: 1;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 2: 1;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 2: 1;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 2: 1;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 2: 1;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 2: 1;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 2: 1;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 2: 1;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 2: 1;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 2: 1;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 2: 1;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+64] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[2];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 3: 2;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 3: 2;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 3: 2;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 3: 2;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 3: 2;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 3: 2;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 3: 2;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 3: 2;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 3: 2;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 3: 2;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 3: 2;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 3: 2;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 3: 2;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 3: 2;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+128] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[3];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 4: 3;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 4: 3;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 4: 3;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 4: 3;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 4: 3;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 4: 3;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 4: 3;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 4: 3;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 4: 3;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 4: 3;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 4: 3;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 4: 3;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 4: 3;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 4: 3;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+192] = sum;

        // out_d[i] = args_d[0 ]*in_d[i-7] + 
                   // args_d[1 ]*in_d[i-6] + 
                   // args_d[2 ]*in_d[i-5] +
                   // args_d[3 ]*in_d[i-4] + 
                   // args_d[4 ]*in_d[i-3] + 
                   // args_d[5 ]*in_d[i-2] +
                   // args_d[6 ]*in_d[i-1] + 
                   // args_d[7 ]*in_d[i  ] + 
                   // args_d[8 ]*in_d[i+1] +
                   // args_d[9 ]*in_d[i+2] + 
                   // args_d[10]*in_d[i+3] + 
                   // args_d[11]*in_d[i+4] +
                   // args_d[12]*in_d[i+5] + 
                   // args_d[13]*in_d[i+6] + 
                   // args_d[14]*in_d[i+7] ;

    });
    fut.wait();
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "kern: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "d2h: %lg ns\n", timeInNS);
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = 0;
    out[6] = 0;
    out[n+wrapper_size/2+0] = 0;
    out[n+wrapper_size/2+1] = 0;
    out[n+wrapper_size/2+2] = 0;
    out[n+wrapper_size/2+3] = 0;
    out[n+wrapper_size/2+4] = 0;
    out[n+wrapper_size/2+5] = 0;
    out[n+wrapper_size/2+6] = 0;
}

void stencil_hcc_shfl8(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int wrapper_size)
{
    extent<1> ct_domain(n+wrapper_size);
    extent<1> cp_domain(n/8); 
    tiled_extent<1> cp_tile(cp_domain, 256);

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + n + wrapper_size);
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(wrapper_size+1, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<1> tidx) restrict(amp) {

        int wid = __lane_id();
        int gid = (tidx.global[0]/64)*512 + wid + wrapper_size/2;
        DATA_TYPE threadInput[9];
        int lowIdx1 = gid - 7;
        int lowIdx2 = lowIdx1 + 64;
        int lowIdx3 = lowIdx2 + 64;
        int lowIdx4 = lowIdx3 + 64;
        int lowIdx5 = lowIdx4 + 64;
        int lowIdx6 = lowIdx5 + 64;
        int lowIdx7 = lowIdx6 + 64;
        int lowIdx8 = lowIdx7 + 64;
        int highIdx = lowIdx8 + 64;
        threadInput[0] = in_d[lowIdx1];
        threadInput[1] = in_d[lowIdx2];
        threadInput[2] = in_d[lowIdx3];
        threadInput[3] = in_d[lowIdx4];
        threadInput[4] = in_d[lowIdx5];
        threadInput[5] = in_d[lowIdx6];
        threadInput[6] = in_d[lowIdx7];
        threadInput[7] = in_d[lowIdx8];
        if(highIdx < n + wrapper_size)
            threadInput[8] = in_d[highIdx];

        DATA_TYPE sum = 0.0f;
        sum += args_d[0]*threadInput[0];
        int new_wid = (wid + 1) % 64;
        int new_reg = wid == 0 ? 1: 0;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 1: 0;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 1: 0;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 1: 0;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 1: 0;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 1: 0;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 1: 0;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 1: 0;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 1: 0;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 1: 0;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 1: 0;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 1: 0;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 1: 0;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 1: 0;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[1];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 2: 1;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 2: 1;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 2: 1;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 2: 1;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 2: 1;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 2: 1;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 2: 1;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 2: 1;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 2: 1;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 2: 1;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 2: 1;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 2: 1;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 2: 1;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 2: 1;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+64] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[2];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 3: 2;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 3: 2;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 3: 2;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 3: 2;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 3: 2;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 3: 2;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 3: 2;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 3: 2;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 3: 2;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 3: 2;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 3: 2;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 3: 2;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 3: 2;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 3: 2;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+128] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[3];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 4: 3;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 4: 3;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 4: 3;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 4: 3;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 4: 3;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 4: 3;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 4: 3;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 4: 3;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 4: 3;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 4: 3;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 4: 3;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 4: 3;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 4: 3;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 4: 3;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+192] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[4];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 5: 4;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 5: 4;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 5: 4;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 5: 4;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 5: 4;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 5: 4;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 5: 4;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 5: 4;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 5: 4;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 5: 4;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 5: 4;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 5: 4;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 5: 4;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 5: 4;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+256] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[5];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 6: 5;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 6: 5;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 6: 5;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 6: 5;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 6: 5;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 6: 5;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 6: 5;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 6: 5;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 6: 5;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 6: 5;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 6: 5;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 6: 5;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 6: 5;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 6: 5;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+320] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[6];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 7: 6;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 7: 6;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 7: 6;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 7: 6;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 7: 6;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 7: 6;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 7: 6;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 7: 6;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 7: 6;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 7: 6;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 7: 6;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 7: 6;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 7: 6;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 7: 6;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+384] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[7];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 8: 7;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 8: 7;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 8: 7;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 8: 7;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 8: 7;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 8: 7;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 8: 7;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 8: 7;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 8: 7;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 8: 7;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 8: 7;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 8: 7;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 8: 7;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 8: 7;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+448] = sum;




        // out_d[i] = args_d[0 ]*in_d[i-7] + 
                   // args_d[1 ]*in_d[i-6] + 
                   // args_d[2 ]*in_d[i-5] +
                   // args_d[3 ]*in_d[i-4] + 
                   // args_d[4 ]*in_d[i-3] + 
                   // args_d[5 ]*in_d[i-2] +
                   // args_d[6 ]*in_d[i-1] + 
                   // args_d[7 ]*in_d[i  ] + 
                   // args_d[8 ]*in_d[i+1] +
                   // args_d[9 ]*in_d[i+2] + 
                   // args_d[10]*in_d[i+3] + 
                   // args_d[11]*in_d[i+4] +
                   // args_d[12]*in_d[i+5] + 
                   // args_d[13]*in_d[i+6] + 
                   // args_d[14]*in_d[i+7] ;

    });
    fut.wait();
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "kern: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "d2h: %lg ns\n", timeInNS);
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = 0;
    out[6] = 0;
    out[n+wrapper_size/2+0] = 0;
    out[n+wrapper_size/2+1] = 0;
    out[n+wrapper_size/2+2] = 0;
    out[n+wrapper_size/2+3] = 0;
    out[n+wrapper_size/2+4] = 0;
    out[n+wrapper_size/2+5] = 0;
    out[n+wrapper_size/2+6] = 0;
}

void stencil_hcc_shfl16(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int wrapper_size)
{
    extent<1> ct_domain(n+wrapper_size);
    extent<1> cp_domain(n/16); 
    tiled_extent<1> cp_tile(cp_domain, 256);

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + n + wrapper_size);
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(wrapper_size+1, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<1> tidx) restrict(amp) {

        int wid = __lane_id();
        int gid = (tidx.global[0]/64)*1024 + wid + wrapper_size/2;
        DATA_TYPE threadInput[17];
        int lowIdx1  = gid - 7;
        int lowIdx2  = lowIdx1  + 64;
        int lowIdx3  = lowIdx2  + 64;
        int lowIdx4  = lowIdx3  + 64;
        int lowIdx5  = lowIdx4  + 64;
        int lowIdx6  = lowIdx5  + 64;
        int lowIdx7  = lowIdx6  + 64;
        int lowIdx8  = lowIdx7  + 64;
        int lowIdx9  = lowIdx8  + 64;
        int lowIdx10 = lowIdx9  + 64;
        int lowIdx11 = lowIdx10 + 64;
        int lowIdx12 = lowIdx11 + 64;
        int lowIdx13 = lowIdx12 + 64;
        int lowIdx14 = lowIdx13 + 64;
        int lowIdx15 = lowIdx14 + 64;
        int lowIdx16 = lowIdx15 + 64;
        int highIdx  = lowIdx16 + 64;
        threadInput[0] = in_d[lowIdx1];
        threadInput[1] = in_d[lowIdx2];
        threadInput[2] = in_d[lowIdx3];
        threadInput[3] = in_d[lowIdx4];
        threadInput[4] = in_d[lowIdx5];
        threadInput[5] = in_d[lowIdx6];
        threadInput[6] = in_d[lowIdx7];
        threadInput[7] = in_d[lowIdx8];
        threadInput[8] = in_d[lowIdx9];
        threadInput[9] = in_d[lowIdx10];
        threadInput[10] = in_d[lowIdx11];
        threadInput[11] = in_d[lowIdx12];
        threadInput[12] = in_d[lowIdx13];
        threadInput[13] = in_d[lowIdx14];
        threadInput[14] = in_d[lowIdx15];
        threadInput[15] = in_d[lowIdx16];
        if(highIdx < n + wrapper_size)
            threadInput[16] = in_d[highIdx];

        DATA_TYPE sum = 0.0f;
        sum += args_d[0]*threadInput[0];
        int new_wid = (wid + 1) % 64;
        int new_reg = wid == 0 ? 1: 0;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 1: 0;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 1: 0;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 1: 0;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 1: 0;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 1: 0;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 1: 0;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 1: 0;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 1: 0;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 1: 0;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 1: 0;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 1: 0;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 1: 0;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 1: 0;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[1];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 2: 1;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 2: 1;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 2: 1;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 2: 1;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 2: 1;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 2: 1;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 2: 1;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 2: 1;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 2: 1;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 2: 1;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 2: 1;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 2: 1;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 2: 1;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 2: 1;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+64] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[2];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 3: 2;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 3: 2;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 3: 2;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 3: 2;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 3: 2;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 3: 2;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 3: 2;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 3: 2;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 3: 2;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 3: 2;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 3: 2;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 3: 2;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 3: 2;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 3: 2;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+128] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[3];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 4: 3;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 4: 3;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 4: 3;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 4: 3;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 4: 3;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 4: 3;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 4: 3;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 4: 3;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 4: 3;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 4: 3;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 4: 3;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 4: 3;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 4: 3;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 4: 3;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+192] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[4];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 5: 4;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 5: 4;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 5: 4;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 5: 4;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 5: 4;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 5: 4;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 5: 4;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 5: 4;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 5: 4;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 5: 4;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 5: 4;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 5: 4;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 5: 4;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 5: 4;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+256] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[5];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 6: 5;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 6: 5;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 6: 5;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 6: 5;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 6: 5;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 6: 5;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 6: 5;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 6: 5;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 6: 5;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 6: 5;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 6: 5;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 6: 5;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 6: 5;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 6: 5;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+320] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[6];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 7: 6;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 7: 6;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 7: 6;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 7: 6;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 7: 6;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 7: 6;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 7: 6;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 7: 6;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 7: 6;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 7: 6;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 7: 6;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 7: 6;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 7: 6;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 7: 6;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+384] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[7];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 8: 7;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 8: 7;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 8: 7;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 8: 7;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 8: 7;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 8: 7;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 8: 7;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 8: 7;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 8: 7;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 8: 7;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 8: 7;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 8: 7;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 8: 7;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 8: 7;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+448] = sum;


        // half
        sum = 0.0f;
        sum += args_d[0]*threadInput[8];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 9: 8;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 9: 8;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 9: 8;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 9: 8;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 9: 8;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 9: 8;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 9: 8;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 9: 8;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 9: 8;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 9: 8;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 9: 8;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 9: 8;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 9: 8;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 9: 8;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+512] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[9];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 10: 9;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 10: 9;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 10: 9;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 10: 9;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 10: 9;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 10: 9;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 10: 9;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 10: 9;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 10: 9;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 10: 9;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 10: 9;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 10: 9;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 10: 9;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 10: 9;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+576] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[10];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 11: 10;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 11: 10;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 11: 10;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 11: 10;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 11: 10;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 11: 10;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 11: 10;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 11: 10;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 11: 10;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 11: 10;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 11: 10;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 11: 10;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 11: 10;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 11: 10;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+640] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[11];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 12: 11;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 12: 11;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 12: 11;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 12: 11;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 12: 11;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 12: 11;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 12: 11;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 12: 11;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 12: 11;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 12: 11;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 12: 11;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 12: 11;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 12: 11;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 12: 11;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+704] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[12];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 13: 12;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 13: 12;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 13: 12;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 13: 12;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 13: 12;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 13: 12;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 13: 12;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 13: 12;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 13: 12;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 13: 12;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 13: 12;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 13: 12;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 13: 12;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 13: 12;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+768] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[13];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 14: 13;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 14: 13;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 14: 13;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 14: 13;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 14: 13;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 14: 13;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 14: 13;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 14: 13;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 14: 13;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 14: 13;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 14: 13;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 14: 13;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 14: 13;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 14: 13;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+832] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[14];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 15: 14;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 15: 14;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 15: 14;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 15: 14;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 15: 14;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 15: 14;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 15: 14;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 15: 14;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 15: 14;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 15: 14;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 15: 14;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 15: 14;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 15: 14;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 15: 14;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+896] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[15];
        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 16: 15;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 2) % 64;
        new_reg = (0<=wid  && wid <=1) ? 16: 15;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 3) % 64;
        new_reg = (0<=wid  && wid <=2) ? 16: 15;
        sum += args_d[3]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 4) % 64;
        new_reg = (0<=wid  && wid <=3) ? 16: 15;
        sum += args_d[4]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 5) % 64;
        new_reg = (0<=wid  && wid <=4) ? 16: 15;
        sum += args_d[5]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 6) % 64;
        new_reg = (0<=wid  && wid <=5) ? 16: 15;
        sum += args_d[6]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 7) % 64;
        new_reg = (0<=wid  && wid <=6) ? 16: 15;
        sum += args_d[7]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 8) % 64;
        new_reg = (0<=wid  && wid <=7) ? 16: 15;
        sum += args_d[8]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 9) % 64;
        new_reg = (0<=wid  && wid <=8) ? 16: 15;
        sum += args_d[9]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 10) % 64;
        new_reg = (0<=wid  && wid <=9) ? 16: 15;
        sum += args_d[10]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 11) % 64;
        new_reg = (0<=wid  && wid <=10) ? 16: 15;
        sum += args_d[11]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 12) % 64;
        new_reg = (0<=wid  && wid <=11) ? 16: 15;
        sum += args_d[12]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 13) % 64;
        new_reg = (0<=wid && wid  <=12) ? 16: 15;
        sum += args_d[13]*__shfl(threadInput[new_reg], new_wid, 64);
        new_wid = (wid + 14) % 64;
        new_reg = (0<=wid && wid <=13) ? 16: 15;
        sum += args_d[14]*__shfl(threadInput[new_reg], new_wid, 64);
        out_d[gid+960] = sum;


        // out_d[i] = args_d[0 ]*in_d[i-7] + 
                   // args_d[1 ]*in_d[i-6] + 
                   // args_d[2 ]*in_d[i-5] +
                   // args_d[3 ]*in_d[i-4] + 
                   // args_d[4 ]*in_d[i-3] + 
                   // args_d[5 ]*in_d[i-2] +
                   // args_d[6 ]*in_d[i-1] + 
                   // args_d[7 ]*in_d[i  ] + 
                   // args_d[8 ]*in_d[i+1] +
                   // args_d[9 ]*in_d[i+2] + 
                   // args_d[10]*in_d[i+3] + 
                   // args_d[11]*in_d[i+4] +
                   // args_d[12]*in_d[i+5] + 
                   // args_d[13]*in_d[i+6] + 
                   // args_d[14]*in_d[i+7] ;

    });
    fut.wait();
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "kern: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "d2h: %lg ns\n", timeInNS);
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = 0;
    out[6] = 0;
    out[n+wrapper_size/2+0] = 0;
    out[n+wrapper_size/2+1] = 0;
    out[n+wrapper_size/2+2] = 0;
    out[n+wrapper_size/2+3] = 0;
    out[n+wrapper_size/2+4] = 0;
    out[n+wrapper_size/2+5] = 0;
    out[n+wrapper_size/2+6] = 0;
}

void stencil_hcc_sm(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int wrapper_size)
{
    extent<1> ct_domain(n+wrapper_size);
    extent<1> cp_domain(n); 
    tiled_extent<1> cp_tile(cp_domain, 256);

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + n + wrapper_size);
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(wrapper_size+1, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<1> tidx) restrict(amp) {
        tile_static DATA_TYPE local[256+14];
        int i = tidx.global[0] + wrapper_size/2;
        int lid = tidx.local[0] + wrapper_size/2;
        local[lid] = in_d[i];
        if(lid == wrapper_size/2)
        {
            local[lid-1] = in_d[i-1];
            local[lid-2] = in_d[i-2];
            local[lid-3] = in_d[i-3];
            local[lid-4] = in_d[i-4];
            local[lid-5] = in_d[i-5];
            local[lid-6] = in_d[i-6];
            local[lid-7] = in_d[i-7];
        }
        if(lid == 255+wrapper_size/2)
        {
            local[lid+1] = in_d[i+1];
            local[lid+2] = in_d[i+2];
            local[lid+3] = in_d[i+3];
            local[lid+4] = in_d[i+4];
            local[lid+5] = in_d[i+5];
            local[lid+6] = in_d[i+6];
            local[lid+7] = in_d[i+7];
        }
        tidx.barrier.wait();

        out_d[i] = args_d[0 ]*local[lid-7] + 
                   args_d[1 ]*local[lid-6] + 
                   args_d[2 ]*local[lid-5] +
                   args_d[3 ]*local[lid-4] + 
                   args_d[4 ]*local[lid-3] + 
                   args_d[5 ]*local[lid-2] +
                   args_d[6 ]*local[lid-1] + 
                   args_d[7 ]*local[lid  ] + 
                   args_d[8 ]*local[lid+1] +
                   args_d[9 ]*local[lid+2] + 
                   args_d[10]*local[lid+3] + 
                   args_d[11]*local[lid+4] +
                   args_d[12]*local[lid+5] + 
                   args_d[13]*local[lid+6] + 
                   args_d[14]*local[lid+7] ;
    });
    fut.wait();
    END_TIME;
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "kern: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "d2h: %lg ns\n", timeInNS);
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = 0;
    out[6] = 0;
    out[n+wrapper_size/2+0] = 0;
    out[n+wrapper_size/2+1] = 0;
    out[n+wrapper_size/2+2] = 0;
    out[n+wrapper_size/2+3] = 0;
    out[n+wrapper_size/2+4] = 0;
    out[n+wrapper_size/2+5] = 0;
    out[n+wrapper_size/2+6] = 0;
}

bool verify(DATA_TYPE *test, DATA_TYPE *ref, int n)
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

void default_properties() 
{
    std::cout << "Default Device Info: " << "\n";
    accelerator default_acc;
    std::wcout << default_acc.get_device_path() << "\n";
    std::wcout << default_acc.get_dedicated_memory() << "\n";
    std::wcout << (default_acc.get_supports_cpu_shared_memory() ? 
        "CPU shared memory: true" : "CPU shared memory: false") << "\n";
    std::wcout << (default_acc.get_supports_double_precision() ? 
        "double precision: true" : "double precision: false") << "\n";
    std::wcout << (default_acc.get_supports_limited_double_precision() ? 
        "limited double precision: true" : "limited double precision: false") << "\n";
    bool success = false;
    success = accelerator::set_default(default_acc.get_device_path());
    std::cout << std::boolalpha << success << std::endl;

}

int main(int argc, char **argv)
{
    int n = 96000000;
    int wrapper_size = 14;
    DATA_TYPE *in = new DATA_TYPE[n+wrapper_size];
    DATA_TYPE *out_ref = new DATA_TYPE[n+wrapper_size];
    DATA_TYPE *out_tst = new DATA_TYPE[n+wrapper_size];
    DATA_TYPE args[15];
    args[0 ] = 1.0/15;
    args[1 ] = 1.0/15;
    args[2 ] = 1.0/15;
    args[3 ] = 1.0/15;
    args[4 ] = 1.0/15;
    args[5 ] = 1.0/15;
    args[6 ] = 1.0/15;
    args[7 ] = 1.0/15;
    args[8 ] = 1.0/15;
    args[9 ] = 1.0/15;
    args[10] = 1.0/15;
    args[11] = 1.0/15;
    args[12] = 1.0/15;
    args[13] = 1.0/15;
    args[14] = 1.0/15;


    srand(1);
    for(int i = wrapper_size/2; i < n+wrapper_size/2; i++)
    {
        in[i] = (DATA_TYPE)((DATA_TYPE)rand() * 100 / (DATA_TYPE)(RAND_MAX));
    }
    in[0] = 0;
    in[1] = 0;
    in[2] = 0;
    in[3] = 0;
    in[4] = 0;
    in[5] = 0;
    in[6] = 0;
    in[n+wrapper_size/2+0] = 0;
    in[n+wrapper_size/2+1] = 0;
    in[n+wrapper_size/2+2] = 0;
    in[n+wrapper_size/2+3] = 0;
    in[n+wrapper_size/2+4] = 0;
    in[n+wrapper_size/2+5] = 0;
    in[n+wrapper_size/2+6] = 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    stencil_seq(in, out_ref, args, n, wrapper_size);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "seq: %lg ns\n", timeInNS);

    default_properties();
    stencil_hcc(in, out_tst, args, n, wrapper_size); // warmup
    
    t1 = std::chrono::high_resolution_clock::now();
    stencil_hcc(in, out_tst, args, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc: %lg ns\n", timeInNS);

    std::cout << "Verify hcc: " << std::boolalpha << verify(out_tst, out_ref, n+wrapper_size) << std::endl;
    std::cout << std::noboolalpha;

    t1 = std::chrono::high_resolution_clock::now();
    stencil_hcc_shfl(in, out_tst, args, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc_shf: %lg ns\n", timeInNS);

    std::cout << "Verify hcc_shf: " << std::boolalpha << verify(out_tst, out_ref, n+wrapper_size) << std::endl;
    std::cout << std::noboolalpha;

    t1 = std::chrono::high_resolution_clock::now();
    stencil_hcc_shfl2(in, out_tst, args, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc_shf2: %lg ns\n", timeInNS);

    std::cout << "Verify hcc_shf2: " << std::boolalpha << verify(out_tst, out_ref, n+wrapper_size) << std::endl;
    std::cout << std::noboolalpha;

    t1 = std::chrono::high_resolution_clock::now();
    stencil_hcc_shfl4(in, out_tst, args, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc_shf4: %lg ns\n", timeInNS);

    std::cout << "Verify hcc_shf4: " << std::boolalpha << verify(out_tst, out_ref, n+wrapper_size) << std::endl;
    std::cout << std::noboolalpha;

    t1 = std::chrono::high_resolution_clock::now();
    stencil_hcc_shfl8(in, out_tst, args, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc_shf8: %lg ns\n", timeInNS);

    std::cout << "Verify hcc_shf8: " << std::boolalpha << verify(out_tst, out_ref, n+wrapper_size) << std::endl;
    std::cout << std::noboolalpha;

    t1 = std::chrono::high_resolution_clock::now();
    stencil_hcc_shfl16(in, out_tst, args, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc_shf16: %lg ns\n", timeInNS);

    std::cout << "Verify hcc_shf16: " << std::boolalpha << verify(out_tst, out_ref, n+wrapper_size) << std::endl;
    std::cout << std::noboolalpha;


    t1 = std::chrono::high_resolution_clock::now();
    stencil_hcc_sm(in, out_tst, args, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc_sm: %lg ns\n", timeInNS);

    std::cout << "Verify hcc_sm: " << std::boolalpha << verify(out_tst, out_ref, n+wrapper_size) << std::endl;
    std::cout << std::noboolalpha;



    delete []in;
    delete []out_ref;
    delete []out_tst;
}

