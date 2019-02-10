#include <iostream>
#include <cmath>
#include <hc.hpp>

using namespace hc;

#define DATA_TYPE float

double tol_finder(int error_tol)
{
    double val = 1.0;
    for(; error_tol > 0; error_tol--)
        val *= 10;
    return 1.0/(double)val;
}

void stencil_seq(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int wrapper_size)
{
    for(int i = 1; i <= n; i++)
    {
        out[i] = args[0]*in[i-1] + args[1]*in[i] + args[2]*in[i+1];
    }
    out[0] = 0;
    out[n+1] = 0;
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
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<1> tidx) restrict(amp) {
        int i = tidx.global[0] + 1;
        out_d[i] = args_d[0]*in_d[i-1] + args_d[1]*in_d[i] + args_d[2]*in_d[i+1];
    });
    fut.wait();
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "kern: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "d2h: %lg ns\n", timeInNS);
    out[0] = 0;
    out[n+1] = 0;
    
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
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<1> tidx) restrict(amp) {

        int gid = tidx.global[0] + 1;
        int wid = __lane_id();
        DATA_TYPE threadInput[2];
        int lowIdx = gid - 1;
        int highIdx = lowIdx + 64;
        threadInput[0] = in_d[lowIdx];
        if(highIdx < n + wrapper_size)
            threadInput[1] = in_d[highIdx];
        else
            threadInput[1] = -1;

        DATA_TYPE sum = 0.0f;
        // use local data
        sum += args_d[0]*threadInput[0];

        // 1st communication
        int new_wid = (wid + 1) % 64;
        int new_reg = wid == 0 ? 1: 0;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);

        // 2nd communication
        new_wid = (wid + 2) % 64;
        new_reg = (wid == 0 || wid ==1) ? 1: 0;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);

        out_d[gid] = sum;

    });
    fut.wait();
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "kern: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "d2h: %lg ns\n", timeInNS);
    out[0] = 0;
    out[n+1] = 0;
    
}

void stencil_hcc_shfl2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int wrapper_size)
{
    extent<1> ct_domain(n+wrapper_size);
    extent<1> cp_domain(n/2); 
    tiled_extent<1> cp_tile(cp_domain, 256); // tested 128 is the best

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + n + wrapper_size);
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(wrapper_size+1, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<1> tidx) restrict(amp) {
        
        int wid = __lane_id();
        int gid = (tidx.global[0]/64)*128 + wid + 1;
        DATA_TYPE threadInput[3];
        int lowIdx1 = gid - 1;
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
        new_reg = (wid == 0 || wid ==1) ? 1: 0;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);

        out_d[gid] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[1];

        new_wid = (wid + 1) % 64;
        new_reg = wid == 0 ? 2: 1;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);

        new_wid = (wid + 2) % 64;
        new_reg = (wid == 0 || wid ==1) ? 2: 1;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);

        out_d[gid+64] = sum;

    });
    fut.wait();
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "kern: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "d2h: %lg ns\n", timeInNS);
    out[0] = 0;
    out[n+1] = 0;
    
}

void stencil_hcc_shfl4(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int wrapper_size)
{
    extent<1> ct_domain(n+wrapper_size);
    extent<1> cp_domain(n/4); 
    tiled_extent<1> cp_tile(cp_domain, 256); //

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + n + wrapper_size);
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(wrapper_size+1, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<1> tidx) restrict(amp) {
        
        int wid = __lane_id();
        int gid = (tidx.global[0]/64)*256 + wid + 1;
        DATA_TYPE threadInput[5];
        int lowIdx1 = gid - 1;
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
        int new_reg = (wid == 0) ? 1: 0;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);

        new_wid = (wid + 2) % 64;
        new_reg = (wid == 0 || wid ==1) ? 1: 0;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);

        out_d[gid] = sum;
        // out_d[gid] = new_reg;

        sum = 0.0f;
        sum += args_d[0]*threadInput[1];

        new_wid = (wid + 1) % 64;
        new_reg = (wid == 0) ? 2: 1;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);

        new_wid = (wid + 2) % 64;
        new_reg = (wid == 0 || wid ==1) ? 2: 1;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);

        out_d[gid+64] = sum;
        // out_d[gid+64] = new_reg;

        sum = 0.0f;
        sum += args_d[0]*threadInput[2];

        new_wid = (wid + 1) % 64;
        new_reg = (wid == 0) ? 3: 2;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);

        new_wid = (wid + 2) % 64;
        new_reg = (wid == 0 || wid ==1) ? 3: 2;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);

        out_d[gid+128] = sum;
        // out_d[gid+128] = new_reg;

        sum = 0.0f;
        sum += args_d[0]*threadInput[3];

        new_wid = (wid + 1) % 64;
        new_reg = (wid == 0) ? 4: 3;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);

        new_wid = (wid + 2) % 64;
        new_reg = (wid == 0 || wid ==1) ? 4: 3;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);

        out_d[gid+192] = sum;
        // out_d[gid+192] = new_reg;

    });
    fut.wait();
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "kern: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "d2h: %lg ns\n", timeInNS);
    out[0] = 0;
    out[n+1] = 0;
    
}

void stencil_hcc_shfl8(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int n, int wrapper_size)
{
    extent<1> ct_domain(n+wrapper_size);
    extent<1> cp_domain(n/8); 
    tiled_extent<1> cp_tile(cp_domain, 256); //

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + n + wrapper_size);
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(wrapper_size+1, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<1> tidx) restrict(amp) {
        
        int wid = __lane_id();
        int gid = (tidx.global[0]/64)*512 + wid + 1;
        DATA_TYPE threadInput[9];
        int lowIdx1 = gid - 1;
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
        int new_reg = (wid == 0) ? 1: 0;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);

        new_wid = (wid + 2) % 64;
        new_reg = (wid == 0 || wid ==1) ? 1: 0;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);

        out_d[gid] = sum;
        // out_d[gid] = new_reg;

        sum = 0.0f;
        sum += args_d[0]*threadInput[1];

        new_wid = (wid + 1) % 64;
        new_reg = (wid == 0) ? 2: 1;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);

        new_wid = (wid + 2) % 64;
        new_reg = (wid == 0 || wid ==1) ? 2: 1;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);

        out_d[gid+64] = sum;
        // out_d[gid+64] = new_reg;

        sum = 0.0f;
        sum += args_d[0]*threadInput[2];

        new_wid = (wid + 1) % 64;
        new_reg = (wid == 0) ? 3: 2;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);

        new_wid = (wid + 2) % 64;
        new_reg = (wid == 0 || wid ==1) ? 3: 2;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);

        out_d[gid+128] = sum;
        // out_d[gid+128] = new_reg;

        sum = 0.0f;
        sum += args_d[0]*threadInput[3];

        new_wid = (wid + 1) % 64;
        new_reg = (wid == 0) ? 4: 3;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);

        new_wid = (wid + 2) % 64;
        new_reg = (wid == 0 || wid ==1) ? 4: 3;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);

        out_d[gid+192] = sum;
        // out_d[gid+192] = new_reg;

        sum = 0.0f;
        sum += args_d[0]*threadInput[4];

        new_wid = (wid + 1) % 64;
        new_reg = (wid == 0) ? 5: 4;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);

        new_wid = (wid + 2) % 64;
        new_reg = (wid == 0 || wid ==1) ? 5: 4;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);

        out_d[gid+256] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[5];

        new_wid = (wid + 1) % 64;
        new_reg = (wid == 0) ? 6: 5;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);

        new_wid = (wid + 2) % 64;
        new_reg = (wid == 0 || wid ==1) ? 6: 5;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);

        out_d[gid+320] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[6];

        new_wid = (wid + 1) % 64;
        new_reg = (wid == 0) ? 7: 6;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);

        new_wid = (wid + 2) % 64;
        new_reg = (wid == 0 || wid ==1) ? 7: 6;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);

        out_d[gid+384] = sum;

        sum = 0.0f;
        sum += args_d[0]*threadInput[7];

        new_wid = (wid + 1) % 64;
        new_reg = (wid == 0) ? 8: 7;
        sum += args_d[1]*__shfl(threadInput[new_reg], new_wid, 64);

        new_wid = (wid + 2) % 64;
        new_reg = (wid == 0 || wid ==1) ? 8: 7;
        sum += args_d[2]*__shfl(threadInput[new_reg], new_wid, 64);

        out_d[gid+448] = sum;


    });
    fut.wait();
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "kern: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "d2h: %lg ns\n", timeInNS);
    out[0] = 0;
    out[n+1] = 0;
    
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
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<1> tidx) restrict(amp) {
        tile_static DATA_TYPE local[256+2];
        int i = tidx.global[0] + 1;
        int lid = tidx.local[0] + 1;
        local[lid] = in_d[i];
        if(lid == 1)
            local[lid-1] = in_d[i-1];
        if(lid == 256)
            local[lid+1] = in_d[i+1];
        tidx.barrier.wait();
        out_d[i] = args_d[0]*local[lid-1] + args_d[1]*local[lid] + args_d[2]*local[lid+1];
    });
    fut.wait();
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "kern: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    copy(out_d, out);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "d2h: %lg ns\n", timeInNS);
    out[0] = 0;
    out[n+1] = 0;
    
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
    // int n = 2048;
    // int n = 64;
    int wrapper_size = 2;
    DATA_TYPE *in = new DATA_TYPE[n+wrapper_size];
    DATA_TYPE *out_ref = new DATA_TYPE[n+wrapper_size];
    DATA_TYPE *out_tst = new DATA_TYPE[n+wrapper_size];
    DATA_TYPE args[3];
    args[0] = 1.0;
    args[1] = 1.0;
    args[2] = 1.0;

    srand(1);
    for(int i = 1; i < n+1; i++)
    {
        in[i] = (DATA_TYPE)((DATA_TYPE)rand() * 100 / (DATA_TYPE)(RAND_MAX));
    }
    in[0] = 0;
    in[n+1] = 0;

    // std::cout << "input:" << std::endl;
    // for(int i = 0; i < n+wrapper_size; i++) std::cout << in[i] << ","; std::cout << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    stencil_seq(in, out_ref, args, n, wrapper_size);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "seq: %lg ns\n", timeInNS);
    // std::cout << "output:" << std::endl;
    // for(int i = 0; i < n+wrapper_size; i++) std::cout << out_ref[i] << ","; std::cout << std::endl;

    default_properties();
    stencil_hcc(in, out_tst, args, n, wrapper_size); // warmup

    t1 = std::chrono::high_resolution_clock::now();
    stencil_hcc_shfl(in, out_tst, args, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc_shfl: %lg ns\n", timeInNS);
    // std::cout << "output:" << std::endl;
    // for(int i = 0; i < n+wrapper_size; i++) std::cout << out_tst[i] << ","; std::cout << std::endl;

    std::cout << "Verify hcc_shfl: " << std::boolalpha << verify(out_tst, out_ref, n+wrapper_size) << std::endl;
    std::cout << std::noboolalpha;

    t1 = std::chrono::high_resolution_clock::now();
    stencil_hcc_shfl2(in, out_tst, args, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc_shfl2: %lg ns\n", timeInNS);

    std::cout << "Verify hcc_shfl2: " << std::boolalpha << verify(out_tst, out_ref, n+wrapper_size) << std::endl;
    std::cout << std::noboolalpha;
    // std::cout << "output2:" << std::endl;
    // for(int i = 0; i < n+wrapper_size; i++) std::cout << out_tst[i] << ","; std::cout << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    stencil_hcc_shfl4(in, out_tst, args, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc_shfl4: %lg ns\n", timeInNS);

    std::cout << "Verify hcc_shfl4: " << std::boolalpha << verify(out_tst, out_ref, n+wrapper_size) << std::endl;
    std::cout << std::noboolalpha;
    // std::cout << "output4:" << std::endl;
    // for(int i = 0; i < n+wrapper_size; i++) std::cout << out_tst[i] << ","; std::cout << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    stencil_hcc_shfl8(in, out_tst, args, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc_shfl8: %lg ns\n", timeInNS);

    std::cout << "Verify hcc_shfl8: " << std::boolalpha << verify(out_tst, out_ref, n+wrapper_size) << std::endl;
    std::cout << std::noboolalpha;
    // std::cout << "output8:" << std::endl;
    // for(int i = 0; i < n+wrapper_size; i++) std::cout << out_tst[i] << ","; std::cout << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    stencil_hcc_sm(in, out_tst, args, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc_sm: %lg ns\n", timeInNS);

    std::cout << "Verify hcc_sm: " << std::boolalpha << verify(out_tst, out_ref, n+wrapper_size) << std::endl;
    std::cout << std::noboolalpha;

    t1 = std::chrono::high_resolution_clock::now();
    stencil_hcc(in, out_tst, args, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc: %lg ns\n", timeInNS);

    std::cout << "Verify hcc: " << std::boolalpha << verify(out_tst, out_ref, n+wrapper_size) << std::endl;
    std::cout << std::noboolalpha;



    delete []in;
    delete []out_ref;
    delete []out_tst;
}

