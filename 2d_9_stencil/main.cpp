#include <iostream>
#include <cmath>
#include <hc.hpp>

using namespace hc;

#define DATA_TYPE float
#define STEPS 100
#define fetch_in(_x,_y) in[(_x)*(n+wrapper_size)+(_y)]
#define fetch_out(_x,_y) out[(_x)*(n+wrapper_size)+(_y)]
#define fetch_out_ref(_x,_y) out_ref[(_x)*(n+wrapper_size)+(_y)]

#define fetch_in_d(_x,_y) in_d[(_x)*(n+wrapper_size)+(_y)]
#define fetch_out_d(_x,_y) out_d[(_x)*(n+wrapper_size)+(_y)]
#define fetch_out_tst(_x,_y) out_tst[(_x)*(n+wrapper_size)+(_y)]

#define fetch_local(_x,_y) local[(_x)*(16+wrapper_size)+(_y)]

double tol_finder(int error_tol)
{
    double val = 1.0;
    for(; error_tol > 0; error_tol--)
        val *= 10;
    return 1.0/(double)val;
}

void stencil_seq(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int wrapper_size)
{
    for(int i = wrapper_size/2; i < m+wrapper_size/2; i++)
    {
        for(int j = wrapper_size/2; j < n+wrapper_size/2; j++)
        {
            fetch_out(i,j) = args[0] * fetch_in(i-1,j-1) +
                             args[1] * fetch_in(i-1,j  ) +
                             args[2] * fetch_in(i-1,j+1) +
                             args[3] * fetch_in(i  ,j-1) +
                             args[4] * fetch_in(i  ,j  ) +
                             args[5] * fetch_in(i  ,j+1) +
                             args[6] * fetch_in(i+1,j-1) +
                             args[7] * fetch_in(i+1,j  ) +
                             args[8] * fetch_in(i+1,j+1) ;
        }
    }
    
    for(int i = 0; i < m+wrapper_size; i++)
    {
        for(int j = 0; j < n+wrapper_size; j++)
        {
            if(i<wrapper_size/2 || j<wrapper_size/2 || i>=m+wrapper_size/2 || j>=n+wrapper_size/2)
            {
                fetch_out(i,j) = 0;
            }
        }
    }

}
#define START_TIME for(int _i = 0; _i < STEPS; _i++) {
#define END_TIME }

void stencil_hcc(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int wrapper_size)
{
    extent<1> ct_domain((m+wrapper_size)*(n+wrapper_size));
    extent<2> cp_domain(m, n); 
    tiled_extent<2> cp_tile(cp_domain, 16, 16);

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + (m+wrapper_size)*(n+wrapper_size));
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(9, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<2> tidx) restrict(amp) {
        int i = tidx.global[0] + wrapper_size/2;
        int j = tidx.global[1] + wrapper_size/2;
        if(i <= m+wrapper_size/2 && j <= n+wrapper_size/2)
        {
            fetch_out_d(i,j) =  args_d[0 ]*fetch_in_d(i-1,j-1) + 
                                args_d[1 ]*fetch_in_d(i-1,j  ) + 
                                args_d[2 ]*fetch_in_d(i-1,j+1) +
                                args_d[3 ]*fetch_in_d(i  ,j-1) + 
                                args_d[4 ]*fetch_in_d(i  ,j  ) + 
                                args_d[5 ]*fetch_in_d(i  ,j+1) +
                                args_d[6 ]*fetch_in_d(i+1,j-1) + 
                                args_d[7 ]*fetch_in_d(i+1,j  ) + 
                                args_d[8 ]*fetch_in_d(i+1,j+1) ;
        }
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
    
    for(int i = 0; i < m+wrapper_size; i++)
    {
        for(int j = 0; j < n+wrapper_size; j++)
        {
            if(i<wrapper_size/2 || j<wrapper_size/2 || i>=m+wrapper_size/2 || j>=n+wrapper_size/2)
            {
                fetch_out(i,j) = 0;
            }
        }
    }
}

void stencil_hcc_shfl(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int wrapper_size)
{
    extent<1> ct_domain((m+wrapper_size)*(n+wrapper_size));
    extent<2> cp_domain(m, n); 
    tiled_extent<2> cp_tile(cp_domain, 32, 8);

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + (m+wrapper_size)*(n+wrapper_size));
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(9, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<2> tidx) restrict(amp) {
        int wid = __lane_id();
        // int gid_x = ((tidx.global[0]&0xfffffff8)+1) - 1 + wid%10;
        // int gid_y = ((tidx.global[1]&0xfffffff8)+1) - 1 + wid/10;
        int gid_x = tidx.global[0]+1;
        int gid_y = tidx.global[1]+1;
        int gtile_id_x = tidx.global[0]>>3;
        int gtile_id_y = tidx.global[1]>>3;
        int new_gid_x = (gtile_id_y<<3) + wid%10;
        int new_gid_y = (gtile_id_x<<3) + wid/10;
        DATA_TYPE threadInput[2];
        threadInput[0] = fetch_in_d(new_gid_y, new_gid_x);
        // fetch_out_d(gid_x,gid_y) = threadInput[0];
        // 4 is new x location; 6 is new y location
        new_gid_x = (gtile_id_y<<3) + (wid+4)%10;
        new_gid_y = (gtile_id_x<<3) + 6 + (wid+4)/10;
        if(new_gid_x < n+wrapper_size && new_gid_y < m+wrapper_size)
            threadInput[1] = fetch_in_d(new_gid_y, new_gid_x);
        // else
            // threadInput[1] = 9;
        // fetch_out_d(gid_x,gid_y) = new_gid_x;

        //*
        DATA_TYPE sum = 0.0f;
        int new_wid;
        // northwestern

        // if(wid <= 51)
            // new_wid = wid+((wid>>3)<<1);
        // else
            // new_wid = wid-64+((wid>>3)<<1);
        new_wid = (wid+((wid>>3)<<1))&63;

        DATA_TYPE tmp0, tmp1;
        tmp0 = args_d[0]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[0]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 52) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // northern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[1]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[1]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 51) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // northeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[1]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[1]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 50) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // western 
        new_wid = ((new_wid+8)&63);
        tmp0 = args_d[1]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[1]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 44) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // central
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[1]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[1]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 43) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // eastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[1]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[1]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 42) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // southwestern
        new_wid = ((new_wid+8)&63);
        tmp0 = args_d[1]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[1]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 36) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // southern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[1]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[1]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 35) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // southeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[1]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[1]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 34) 
            sum += tmp0;
        else
            sum += tmp1;
        fetch_out_d(gid_x,gid_y) = sum;
        // */

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
    
    for(int i = 0; i < m+wrapper_size; i++)
    {
        for(int j = 0; j < n+wrapper_size; j++)
        {
            if(i<wrapper_size/2 || j<wrapper_size/2 || i>=m+wrapper_size/2 || j>=n+wrapper_size/2)
            {
                fetch_out(i,j) = 0;
            }
        }
    }
}

void stencil_hcc_shfl2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int wrapper_size)
{
    extent<1> ct_domain((m+wrapper_size)*(n+wrapper_size));
    extent<2> cp_domain(m/2, n); 
    tiled_extent<2> cp_tile(cp_domain, 32, 8);

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + (m+wrapper_size)*(n+wrapper_size));
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(9, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<2> tidx) restrict(amp) {
        int wid = __lane_id();

        int gid_x = ((tidx.global[0]>>3)<<4) + (wid>>3) + 1;
        int gid_y = tidx.global[1]+1;
        int gtile_id_x = (((tidx.global[0]>>3)<<4)+(wid>>3))>>3;
        int gtile_id_y = tidx.global[1]>>3;
        int new_gid_x = (gtile_id_y<<3) + wid%10;
        int new_gid_y = (gtile_id_x<<3) + wid/10;
        DATA_TYPE threadInput[3];
        threadInput[0] = fetch_in_d(new_gid_y, new_gid_x);
        // fetch_out_d(gid_x,gid_y) = threadInput[0];
        // 4 is new x location; 6 is new y location
        new_gid_x = (gtile_id_y<<3) + (wid+4)%10;
        new_gid_y = (gtile_id_x<<3) + 6 + (wid+4)/10;
        threadInput[1] = fetch_in_d(new_gid_y, new_gid_x);
        new_gid_x = (gtile_id_y<<3) + (wid+8)%10;
        new_gid_y = (gtile_id_x<<3) + 12 + (wid+8)/10;
        if(new_gid_x < n+wrapper_size && new_gid_y < m+wrapper_size)
            threadInput[2] = fetch_in_d(new_gid_y, new_gid_x);
        // else
            // threadInput[2] = -1;
        // fetch_out_d(gid_x,gid_y) =threadInput[2];

        //*
        DATA_TYPE sum = 0.0f;
        int new_wid;
        // northwestern
        // if((wid>>3)==6 && (wid>>2)==13)
        // {
            // new_wid = wid-52;
        // } else if((wid>>3)==7)
        // {
            // new_wid = wid-50;
        // } else
        // {
            // new_wid = wid+((wid>>3)<<1);
        // }
        new_wid = (wid+((wid>>3)<<1))&63;

        int new_reg = (wid < 52 )? 0: 1;
        DATA_TYPE tmp0, tmp1;
        tmp0 = args_d[0]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[0]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 52) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // northern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[1]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[1]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 51) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // northeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[2]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[2]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 50) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // western 
        new_wid = ((new_wid+8)&63);
        tmp0 = args_d[3]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[3]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 44) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // central
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[4]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[4]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 43) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // eastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[5]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[5]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 42) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // southwestern
        new_wid = ((new_wid+8)&63);
        tmp0 = args_d[6]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[6]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 36) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // southern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[7]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[7]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 35) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // southeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[8]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[8]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 34) 
            sum += tmp0;
        else
            sum += tmp1;
        fetch_out_d(gid_x,gid_y) = sum;

        // extended
        sum = 0.0f;
        // northwestern
        // if((wid>>3)<=4)
        // {
            // new_wid = wid+16+((wid>>3)<<1);
        // } else 
        // {
            // new_wid = wid-48+((wid>>3)<<1);
        // }
        new_wid = (wid+16+((wid>>3)<<1))&63;
        tmp0 = args_d[0]*__shfl(threadInput[1], new_wid, 64);
        tmp1 = args_d[0]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 40) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // northern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[1]*__shfl(threadInput[1], new_wid, 64);
        tmp1 = args_d[1]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 39) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // northeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[2]*__shfl(threadInput[1], new_wid, 64);
        tmp1 = args_d[2]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 38) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // western 
        new_wid = ((new_wid+8)&63);
        tmp0 = args_d[3]*__shfl(threadInput[1], new_wid, 64);
        tmp1 = args_d[3]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 32) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // central
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[4]*__shfl(threadInput[1], new_wid, 64);
        tmp1 = args_d[4]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 31) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // eastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[5]*__shfl(threadInput[1], new_wid, 64);
        tmp1 = args_d[5]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 30) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // southwestern
        new_wid = ((new_wid+8)&63);
        tmp0 = args_d[6]*__shfl(threadInput[1], new_wid, 64);
        tmp1 = args_d[6]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 24) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // southern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[7]*__shfl(threadInput[1], new_wid, 64);
        tmp1 = args_d[7]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 23) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // southeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[8]*__shfl(threadInput[1], new_wid, 64);
        tmp1 = args_d[8]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 22) 
            sum += tmp0;
        else
            sum += tmp1;
        fetch_out_d(gid_x+8,gid_y) = sum;

        // */

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
    
    for(int i = 0; i < m+wrapper_size; i++)
    {
        for(int j = 0; j < n+wrapper_size; j++)
        {
            if(i<wrapper_size/2 || j<wrapper_size/2 || i>=m+wrapper_size/2 || j>=n+wrapper_size/2)
            {
                fetch_out(i,j) = 0;
            }
        }
    }
}

void stencil_hcc_shfl4(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int wrapper_size)
{
    extent<1> ct_domain((m+wrapper_size)*(n+wrapper_size));
    extent<2> cp_domain(m/4, n); 
    tiled_extent<2> cp_tile(cp_domain, 32, 8);

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + (m+wrapper_size)*(n+wrapper_size));
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(9, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<2> tidx) restrict(amp) {
        int wid = __lane_id();
        // int gid_x = ((tidx.global[0]&0xfffffff8)+1) - 1 + wid%10;
        // int gid_y = ((tidx.global[1]&0xfffffff8)+1) - 1 + wid/10;

        int gid_x = ((tidx.global[0]>>3)<<5) + (wid>>3) + 1;
        int gid_y = tidx.global[1]+1;
        int gtile_id_x = (((tidx.global[0]>>3)<<5)+(wid>>3))>>3;
        int gtile_id_y = tidx.global[1]>>3;
        int new_gid_x = (gtile_id_y<<3) + wid%10;
        int new_gid_y = (gtile_id_x<<3) + wid/10;
        DATA_TYPE threadInput[6];
        threadInput[0] = fetch_in_d(new_gid_y, new_gid_x);
        new_gid_x = (gtile_id_y<<3) + (wid+4)%10;
        new_gid_y = (gtile_id_x<<3) + 6 + (wid+4)/10;
        threadInput[1] = fetch_in_d(new_gid_y, new_gid_x);
        new_gid_x = (gtile_id_y<<3) + (wid+8)%10;
        new_gid_y = (gtile_id_x<<3) + 12 + (wid+8)/10;
        threadInput[2] = fetch_in_d(new_gid_y, new_gid_x);
        new_gid_x = (gtile_id_y<<3) + (wid+2)%10;
        new_gid_y = (gtile_id_x<<3) + 19 + (wid+2)/10;
        threadInput[3] = fetch_in_d(new_gid_y, new_gid_x);
        new_gid_x = (gtile_id_y<<3) + (wid+6)%10;
        new_gid_y = (gtile_id_x<<3) + 25 + (wid+6)/10;
        threadInput[4] = fetch_in_d(new_gid_y, new_gid_x);
        new_gid_x = (gtile_id_y<<3) + (wid)%10;
        new_gid_y = (gtile_id_x<<3) + 32 + (wid)/10;
        if(new_gid_x < n+wrapper_size && new_gid_y < m+wrapper_size)
            threadInput[5] = fetch_in_d(new_gid_y, new_gid_x);
        // else
            // threadInput[2] = -1;
        // fetch_out_d(gid_x,gid_y) =threadInput[2];

        //*
        DATA_TYPE sum = 0.0f;
        int new_wid;
        // northwestern
        // if((wid>>3)==6 && (wid>>2)==13)
        // {
            // new_wid = wid-52;
        // } else if((wid>>3)==7)
        // {
            // new_wid = wid-50;
        // } else
        // {
            // new_wid = wid+((wid>>3)<<1);
        // }
        new_wid = (wid+((wid>>3)<<1))&63;
        int new_reg = (wid < 52 )? 0: 1;
        DATA_TYPE tmp0, tmp1, tmp2;
        tmp0 = args_d[0]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[0]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 52) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // northern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[1]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[1]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 51) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // northeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[2]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[2]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 50) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // western 
        new_wid = ((new_wid+8)&63);
        tmp0 = args_d[3]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[3]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 44) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // central
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[4]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[4]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 43) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // eastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[5]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[5]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 42) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // southwestern
        new_wid = ((new_wid+8)&63);
        tmp0 = args_d[6]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[6]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 36) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // southern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[7]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[7]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 35) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x,gid_y) = sum;

        // southeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[8]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[8]*__shfl(threadInput[1], new_wid, 64);
        if(wid < 34) 
            sum += tmp0;
        else
            sum += tmp1;
        fetch_out_d(gid_x,gid_y) = sum;

        // extended to 2
        sum = 0.0f;
        // northwestern
        // if((wid>>3)<=4)
        // {
            // new_wid = wid+16+((wid>>3)<<1);
        // } else 
        // {
            // new_wid = wid-48+((wid>>3)<<1);
        // }
        new_wid = (wid+16+((wid>>3)<<1))&63;
        tmp0 = args_d[0]*__shfl(threadInput[1], new_wid, 64);
        tmp1 = args_d[0]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 40) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // northern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[1]*__shfl(threadInput[1], new_wid, 64);
        tmp1 = args_d[1]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 39) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // northeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[2]*__shfl(threadInput[1], new_wid, 64);
        tmp1 = args_d[2]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 38) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // western 
        new_wid = ((new_wid+8)&63);
        tmp0 = args_d[3]*__shfl(threadInput[1], new_wid, 64);
        tmp1 = args_d[3]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 32) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // central
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[4]*__shfl(threadInput[1], new_wid, 64);
        tmp1 = args_d[4]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 31) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // eastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[5]*__shfl(threadInput[1], new_wid, 64);
        tmp1 = args_d[5]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 30) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // southwestern
        new_wid = ((new_wid+8)&63);
        tmp0 = args_d[6]*__shfl(threadInput[1], new_wid, 64);
        tmp1 = args_d[6]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 24) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // southern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[7]*__shfl(threadInput[1], new_wid, 64);
        tmp1 = args_d[7]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 23) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // southeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[8]*__shfl(threadInput[1], new_wid, 64);
        tmp1 = args_d[8]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 22) 
            sum += tmp0;
        else
            sum += tmp1;
        fetch_out_d(gid_x+8,gid_y) = sum;

        // extended to 3
        sum = 0.0f;
        // northwestern
        // if(wid>=26)
        // {
            // new_wid = wid-32+((wid>>3)<<1);
        // } else 
        // {
            // new_wid = wid+32+((wid>>3)<<1);
        // } 
        new_wid = (wid+32+((wid>>3)<<1))&63;
        tmp0 = args_d[0]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[0]*__shfl(threadInput[3], new_wid, 64);
        if(wid < 26) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+16,gid_y) = sum;

        // northern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[1]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[1]*__shfl(threadInput[3], new_wid, 64);
        if(wid < 25) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+16,gid_y) = sum;

        // northeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[2]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[2]*__shfl(threadInput[3], new_wid, 64);
        if(wid < 24) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+16,gid_y) = sum;

        // western 
        new_wid = ((new_wid+8)&63);
        tmp0 = args_d[3]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[3]*__shfl(threadInput[3], new_wid, 64);
        if(wid < 18) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+16,gid_y) = sum;

        // central
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[4]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[4]*__shfl(threadInput[3], new_wid, 64);
        if(wid < 17) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+16,gid_y) = sum;

        // eastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[5]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[5]*__shfl(threadInput[3], new_wid, 64);
        if(wid < 16) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+16,gid_y) = sum;

        // southwestern
        new_wid = ((new_wid+8)&63);
        tmp0 = args_d[6]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[6]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[6]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 10) 
            sum += tmp0;
        else if(wid < 62) 
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x+16,gid_y) = sum;

        // southern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[7]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[7]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[7]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 9) 
            sum += tmp0;
        else if(wid < 61)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x+16,gid_y) = sum;

        // southeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[8]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[8]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[8]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 8) 
            sum += tmp0;
        else if(wid < 60)
            sum += tmp1;
        else
            sum += tmp2;
        fetch_out_d(gid_x+16,gid_y) = sum;

        // extended to 4
        sum = 0.0f;
        // northwestern
        // if(wid>=14)
        // {
            // new_wid = wid-16+((wid>>3)<<1);
        // } else 
        // {
            // new_wid = wid+48+((wid>>3)<<1);
        // } 
        new_wid = (wid+48+((wid>>3)<<1))&63;
        // fetch_out_d(gid_x+24,gid_y) = new_wid;

        tmp0 = args_d[0]*__shfl(threadInput[3], new_wid, 64);
        tmp1 = args_d[0]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 14) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+24,gid_y) = sum;

        // northern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[1]*__shfl(threadInput[3], new_wid, 64);
        tmp1 = args_d[1]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 13) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+24,gid_y) = sum;

        // northeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[2]*__shfl(threadInput[3], new_wid, 64);
        tmp1 = args_d[2]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 12) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+24,gid_y) = sum;

        // western 
        new_wid = ((new_wid+8)&63);
        tmp0 = args_d[3]*__shfl(threadInput[3], new_wid, 64);
        tmp1 = args_d[3]*__shfl(threadInput[4], new_wid, 64);
        tmp2 = args_d[3]*__shfl(threadInput[5], new_wid, 64);
        if(wid < 6) 
            sum += tmp0;
        else if(wid < 56)
            sum += tmp1;
        else 
            sum += tmp2;
        // fetch_out_d(gid_x+24,gid_y) = sum;

        // central
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[4]*__shfl(threadInput[3], new_wid, 64);
        tmp1 = args_d[4]*__shfl(threadInput[4], new_wid, 64);
        tmp2 = args_d[4]*__shfl(threadInput[5], new_wid, 64);
        if(wid < 5) 
            sum += tmp0;
        else if(wid < 56)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x+24,gid_y) = sum;

        // eastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[5]*__shfl(threadInput[3], new_wid, 64);
        tmp1 = args_d[5]*__shfl(threadInput[4], new_wid, 64);
        tmp2 = args_d[5]*__shfl(threadInput[5], new_wid, 64);
        if(wid < 4) 
            sum += tmp0;
        else if(wid < 56)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x+24,gid_y) = sum;

        // southwestern
        new_wid = ((new_wid+8)&63);
        tmp0 = args_d[6]*__shfl(threadInput[4], new_wid, 64);
        tmp1 = args_d[6]*__shfl(threadInput[5], new_wid, 64);
        if(wid < 48) 
            sum += tmp0;
        else  
            sum += tmp1;
        // fetch_out_d(gid_x+24,gid_y) = sum;

        // southern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[7]*__shfl(threadInput[4], new_wid, 64);
        tmp1 = args_d[7]*__shfl(threadInput[5], new_wid, 64);
        if(wid < 48) 
            sum += tmp0;
        else
            sum += tmp1;
        // fetch_out_d(gid_x+24,gid_y) = sum;

        // southeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[8]*__shfl(threadInput[4], new_wid, 64);
        tmp1 = args_d[8]*__shfl(threadInput[5], new_wid, 64);
        if(wid < 48) 
            sum += tmp0;
        else
            sum += tmp1;
        fetch_out_d(gid_x+24,gid_y) = sum;
        /*
        // */

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
    
    for(int i = 0; i < m+wrapper_size; i++)
    {
        for(int j = 0; j < n+wrapper_size; j++)
        {
            if(i<wrapper_size/2 || j<wrapper_size/2 || i>=m+wrapper_size/2 || j>=n+wrapper_size/2)
            {
                fetch_out(i,j) = 0;
            }
        }
    }
}

void stencil_hcc_shfl4_2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int wrapper_size)
{
    extent<1> ct_domain((m+wrapper_size)*(n+wrapper_size));
    extent<2> cp_domain(m/2, n/2); 
    tiled_extent<2> cp_tile(cp_domain, 32, 8);

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + (m+wrapper_size)*(n+wrapper_size));
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(9, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<2> tidx) restrict(amp) {
        int wid = __lane_id();
        // int gid_x = ((tidx.global[0]&0xfffffff8)+1) - 1 + wid%10;
        // int gid_y = ((tidx.global[1]&0xfffffff8)+1) - 1 + wid/10;

        int gid_x = ((tidx.global[0]>>3)<<4) + (wid>>3) + 1;
        int gid_y = ((tidx.global[1]>>3)<<4) + (wid&7) + 1;
        int gtile_id_x = (((tidx.global[0]>>3)<<4)+(wid>>3))>>3;
        int gtile_id_y = (((tidx.global[1]>>3)<<4)+(wid&7))>>3;
        int new_gid_x = (gtile_id_y<<3) + wid%18;
        int new_gid_y = (gtile_id_x<<3) + wid/18;
        // fetch_out_d(gid_x,gid_y) = new_gid_y;

        DATA_TYPE threadInput[6];
        threadInput[0] = fetch_in_d(new_gid_y, new_gid_x);
        new_gid_x = (gtile_id_y<<3) + (wid+10)%18;
        new_gid_y = (gtile_id_x<<3) + 3 + (wid+10)/18;
        threadInput[1] = fetch_in_d(new_gid_y, new_gid_x);
        new_gid_x = (gtile_id_y<<3) + (wid+2)%18;
        new_gid_y = (gtile_id_x<<3) + 7 + (wid+2)/18;
        threadInput[2] = fetch_in_d(new_gid_y, new_gid_x);
        new_gid_x = (gtile_id_y<<3) + (wid+12)%18;
        new_gid_y = (gtile_id_x<<3) + 10 + (wid+12)/18;
        threadInput[3] = fetch_in_d(new_gid_y, new_gid_x);
        new_gid_x = (gtile_id_y<<3) + (wid+4)%18;
        new_gid_y = (gtile_id_x<<3) + 14 + (wid+4)/18;
        threadInput[4] = fetch_in_d(new_gid_y, new_gid_x);
        new_gid_x = (gtile_id_y<<3) + (wid+14)%18;
        new_gid_y = (gtile_id_x<<3) + 17 + (wid+14)/18;
        if(new_gid_x < n+wrapper_size && new_gid_y < m+wrapper_size)
            threadInput[5] = fetch_in_d(new_gid_y, new_gid_x);
        // else
            // threadInput[5] = -1;
        // fetch_out_d(gid_x,gid_y) =threadInput[5];

        DATA_TYPE sum = 0.0f;
        int new_wid;
        // northwestern
        // if((wid>>3)==6 && (wid>>2)==13)
        // {
            // new_wid = wid-52;
        // } else if((wid>>3)==7)
        // {
            // new_wid = wid-50;
        // } else
        // {
            // new_wid = wid+((wid>>3)<<1);
        // }
        new_wid = (wid+((wid>>3)*10))&63;
        DATA_TYPE tmp0, tmp1, tmp2, tmp3;
        tmp0 = args_d[0]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[0]*__shfl(threadInput[1], new_wid, 64);
        tmp2 = args_d[0]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 32) 
            sum += tmp0;
        else if(wid < 58)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x,gid_y) = sum;

        // northern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[1]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[1]*__shfl(threadInput[1], new_wid, 64);
        tmp2 = args_d[1]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 32) 
            sum += tmp0;
        else if(wid < 57)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x,gid_y) = sum;

        // northeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[2]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[2]*__shfl(threadInput[1], new_wid, 64);
        tmp2 = args_d[2]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 32) 
            sum += tmp0;
        else if(wid < 56)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x,gid_y) = sum;

        // western 
        new_wid = ((new_wid+16)&63);
        tmp0 = args_d[3]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[3]*__shfl(threadInput[1], new_wid, 64);
        tmp2 = args_d[3]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 24) 
            sum += tmp0;
        else if(wid < 50)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x,gid_y) = sum;

        // central
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[4]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[4]*__shfl(threadInput[1], new_wid, 64);
        tmp2 = args_d[4]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 24) 
            sum += tmp0;
        else if(wid < 49)
            sum += tmp1;
        else 
            sum += tmp2;
        // fetch_out_d(gid_x,gid_y) = sum;

        // eastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[5]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[5]*__shfl(threadInput[1], new_wid, 64);
        tmp2 = args_d[5]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 24) 
            sum += tmp0;
        else if(wid < 48)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x,gid_y) = sum;

        // southwestern
        new_wid = ((new_wid+16)&63);
        tmp0 = args_d[6]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[6]*__shfl(threadInput[1], new_wid, 64);
        tmp2 = args_d[6]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 16) 
            sum += tmp0;
        else if(wid < 42)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x,gid_y) = sum;

        // southern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[7]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[7]*__shfl(threadInput[1], new_wid, 64);
        tmp2 = args_d[7]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 16) 
            sum += tmp0;
        else if(wid < 41)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x,gid_y) = sum;

        // southeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[8]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[8]*__shfl(threadInput[1], new_wid, 64);
        tmp2 = args_d[8]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 16) 
            sum += tmp0;
        else if(wid < 40)
            sum += tmp1;
        else
            sum += tmp2;
        fetch_out_d(gid_x,gid_y) = sum;

        // extended to 2
        sum = 0.0f;
        // northwestern
        // if((wid>>3)<=4)
        // {
            // new_wid = wid+16+((wid>>3)<<1);
        // } else 
        // {
            // new_wid = wid-48+((wid>>3)<<1);
        // }
        new_wid = (wid+8+((wid>>3)*10))&63;
        tmp0 = args_d[0]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[0]*__shfl(threadInput[1], new_wid, 64);
        tmp2 = args_d[0]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 26) 
            sum += tmp0;
        else if(wid < 56)
            sum += tmp1;
        else 
            sum += tmp2;
        // fetch_out_d(gid_x,gid_y+8) = sum;

        // northern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[1]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[1]*__shfl(threadInput[1], new_wid, 64);
        tmp2 = args_d[1]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 25) 
            sum += tmp0;
        else if(wid < 56)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x,gid_y+8) = sum;

        // northeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[2]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[2]*__shfl(threadInput[1], new_wid, 64);
        tmp2 = args_d[2]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 24) 
            sum += tmp0;
        else if(wid < 56)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x,gid_y+8) = sum;

        // western 
        new_wid = ((new_wid+16)&63);
        tmp0 = args_d[3]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[3]*__shfl(threadInput[1], new_wid, 64);
        tmp2 = args_d[3]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 18) 
            sum += tmp0;
        else if(wid < 48)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x,gid_y+8) = sum;

        // central
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[4]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[4]*__shfl(threadInput[1], new_wid, 64);
        tmp2 = args_d[4]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 17) 
            sum += tmp0;
        else if(wid < 48)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x,gid_y+8) = sum;

        // eastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[5]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[5]*__shfl(threadInput[1], new_wid, 64);
        tmp2 = args_d[5]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 16) 
            sum += tmp0;
        else if(wid < 48)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x,gid_y+8) = sum;

        // southwestern
        new_wid = ((new_wid+16)&63);
        tmp0 = args_d[6]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[6]*__shfl(threadInput[1], new_wid, 64);
        tmp2 = args_d[6]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 10) 
            sum += tmp0;
        else if(wid < 40)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x,gid_y+8) = sum;

        // southern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[7]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[7]*__shfl(threadInput[1], new_wid, 64);
        tmp2 = args_d[7]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 9) 
            sum += tmp0;
        else if(wid < 40)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x,gid_y+8) = sum;

        // southeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[8]*__shfl(threadInput[0], new_wid, 64);
        tmp1 = args_d[8]*__shfl(threadInput[1], new_wid, 64);
        tmp2 = args_d[8]*__shfl(threadInput[2], new_wid, 64);
        if(wid < 8) 
            sum += tmp0;
        else if(wid < 40)
            sum += tmp1;
        else
            sum += tmp2;
        fetch_out_d(gid_x,gid_y+8) = sum;

        // extended to 3
        sum = 0.0f;
        // northwestern
        // if(wid>=26)
        // {
            // new_wid = wid-32+((wid>>3)<<1);
        // } else 
        // {
            // new_wid = wid+32+((wid>>3)<<1);
        // } 
        new_wid = (wid+16+((wid>>3)*10))&63;
        tmp0 = args_d[0]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[0]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[0]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 24) 
            sum += tmp0;
        else if(wid < 52)
            sum += tmp1;
        else 
            sum += tmp2;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // northern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[1]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[1]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[1]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 24) 
            sum += tmp0;
        else if(wid < 51)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // northeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[2]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[2]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[2]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 24) 
            sum += tmp0;
        else if(wid < 50)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // western 
        new_wid = ((new_wid+16)&63);
        tmp0 = args_d[3]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[3]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[3]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 16) 
            sum += tmp0;
        else if(wid < 44)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // central
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[4]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[4]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[4]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 16) 
            sum += tmp0;
        else if(wid < 43)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // eastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[5]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[5]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[5]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 16) 
            sum += tmp0;
        else if(wid < 42)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // southwestern
        new_wid = ((new_wid+16)&63);
        tmp0 = args_d[6]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[6]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[6]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 8) 
            sum += tmp0;
        else if(wid < 36) 
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // southern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[7]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[7]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[7]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 8) 
            sum += tmp0;
        else if(wid < 35)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x+8,gid_y) = sum;

        // southeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[8]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[8]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[8]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 8) 
            sum += tmp0;
        else if(wid < 34)
            sum += tmp1;
        else
            sum += tmp2;
        fetch_out_d(gid_x+8,gid_y) = sum;

        // extended to 4
        sum = 0.0f;
        // northwestern
        // if(wid>=14)
        // {
            // new_wid = wid-16+((wid>>3)<<1);
        // } else 
        // {
            // new_wid = wid+48+((wid>>3)<<1);
        // } 
        new_wid = (wid+24+((wid>>3)*10))&63;
        // fetch_out_d(gid_x+24,gid_y) = new_wid;

        tmp0 = args_d[0]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[0]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[0]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 20) 
            sum += tmp0;
        else if(wid < 48)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x+8,gid_y+8) = sum;

        // northern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[1]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[1]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[1]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 19) 
            sum += tmp0;
        else if(wid < 48)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x+8,gid_y+8) = sum;

        // northeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[2]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[2]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[2]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 18) 
            sum += tmp0;
        else if(wid < 48)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x+8,gid_y+8) = sum;

        // western 
        new_wid = ((new_wid+16)&63);
        tmp0 = args_d[3]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[3]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[3]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 12) 
            sum += tmp0;
        else if(wid < 40)
            sum += tmp1;
        else 
            sum += tmp2;
        // fetch_out_d(gid_x+8,gid_y+8) = sum;

        // central
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[4]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[4]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[4]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 11) 
            sum += tmp0;
        else if(wid < 40)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x+8,gid_y+8) = sum;

        // eastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[5]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[5]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[5]*__shfl(threadInput[4], new_wid, 64);
        if(wid < 10) 
            sum += tmp0;
        else if(wid < 40)
            sum += tmp1;
        else
            sum += tmp2;
        // fetch_out_d(gid_x+8,gid_y+8) = sum;

        // southwestern
        new_wid = ((new_wid+16)&63);
        tmp0 = args_d[6]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[6]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[6]*__shfl(threadInput[4], new_wid, 64);
        tmp3 = args_d[6]*__shfl(threadInput[5], new_wid, 64);
        if(wid < 4) 
            sum += tmp0;
        else if(wid < 32)
            sum += tmp1;
        else if(wid < 62)
            sum += tmp2;
        else 
            sum += tmp3;
        // fetch_out_d(gid_x+8,gid_y+8) = sum;

        // southern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[7]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[7]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[7]*__shfl(threadInput[4], new_wid, 64);
        tmp3 = args_d[7]*__shfl(threadInput[5], new_wid, 64);
        if(wid < 3) 
            sum += tmp0;
        else if(wid < 32)
            sum += tmp1;
        else if(wid < 61)
            sum += tmp2;
        else
            sum += tmp3;
        // fetch_out_d(gid_x+8,gid_y+8) = sum;

        // southeastern
        new_wid = ((new_wid+1)&63);
        tmp0 = args_d[8]*__shfl(threadInput[2], new_wid, 64);
        tmp1 = args_d[8]*__shfl(threadInput[3], new_wid, 64);
        tmp2 = args_d[8]*__shfl(threadInput[4], new_wid, 64);
        tmp3 = args_d[8]*__shfl(threadInput[5], new_wid, 64);
        if(wid < 2) 
            sum += tmp0;
        else if(wid < 32)
            sum += tmp1;
        else if(wid < 60)
            sum += tmp2;
        else
            sum += tmp3;
        fetch_out_d(gid_x+8,gid_y+8) = sum;
        /*
        // */

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
    
    for(int i = 0; i < m+wrapper_size; i++)
    {
        for(int j = 0; j < n+wrapper_size; j++)
        {
            if(i<wrapper_size/2 || j<wrapper_size/2 || i>=m+wrapper_size/2 || j>=n+wrapper_size/2)
            {
                fetch_out(i,j) = 0;
            }
        }
    }
}

void stencil_hcc_sm(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int wrapper_size)
{
    extent<1> ct_domain((m+wrapper_size)*(n+wrapper_size));
    extent<2> cp_domain(m, n); 
    tiled_extent<2> cp_tile(cp_domain, 16, 16);

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + (m+wrapper_size)*(n+wrapper_size));
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(9, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    START_TIME;
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<2> tidx) restrict(amp) {
        tile_static DATA_TYPE local[16+2][16+2];
        int i = tidx.global[0] + wrapper_size/2;
        int j = tidx.global[1] + wrapper_size/2;
        int li = tidx.local[0] + wrapper_size/2;
        int lj = tidx.local[1] + wrapper_size/2;

        if(i <= m+wrapper_size/2 && j <= n+wrapper_size/2)
        {
            local[li][lj] = fetch_in_d(i,j);

            if(li == wrapper_size/2)
            {
                local[li-1][lj] = fetch_in_d(i-1,j);
            }

            if(li == 15+wrapper_size/2)
            {
                local[li+1][lj] = fetch_in_d(i+1,j);
            }
            
            if(lj == wrapper_size/2)
            {
                local[li][lj-1] = fetch_in_d(i,j-1);
            }

            if(lj == 15+wrapper_size/2)
            {
                local[li][lj+1] = fetch_in_d(i,j+1);
            }

            if(li == wrapper_size/2 && lj == wrapper_size/2)
            {
                local[li-1][lj-1] = fetch_in_d(i-1,j-1);
            }

            if(li == wrapper_size/2 && lj == 15+wrapper_size/2)
            {
                local[li-1][lj+1] = fetch_in_d(i-1,j+1);
            }

            if(li == 15+wrapper_size/2 && lj == wrapper_size/2)
            {
                local[li+1][lj-1] = fetch_in_d(i+1,j-1);
            }

            if(li == 15+wrapper_size/2 && lj == 15+wrapper_size/2)
            {
                local[li+1][lj+1] = fetch_in_d(i+1,j+1);
            }
        }
        tidx.barrier.wait();

        if(i <= m+wrapper_size/2 && j <= n+wrapper_size/2)
        {
            fetch_out_d(i,j) = args_d[0 ]*local[li-1][lj-1] + 
                               args_d[1 ]*local[li-1][lj  ] + 
                               args_d[2 ]*local[li-1][lj+1] +
                               args_d[3 ]*local[li  ][lj-1] + 
                               args_d[4 ]*local[li  ][lj  ] + 
                               args_d[5 ]*local[li  ][lj+1] +
                               args_d[6 ]*local[li+1][lj-1] + 
                               args_d[7 ]*local[li+1][lj  ] + 
                               args_d[8 ]*local[li+1][lj+1] ;
        }
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

    for(int i = 0; i < m+wrapper_size; i++)
    {
        for(int j = 0; j < n+wrapper_size; j++)
        {
            if(i<wrapper_size/2 || j<wrapper_size/2 || i>=m+wrapper_size/2 || j>=n+wrapper_size/2)
            {
                fetch_out(i,j) = 0;
            }
        }
    }
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
    int n = 10016;
    int m = 10016;
    // int n = 32;
    // int m = 32;
    int wrapper_size = 2;
    DATA_TYPE *in = new DATA_TYPE[(n+wrapper_size)*(m+wrapper_size)];
    DATA_TYPE *out_ref = new DATA_TYPE[(n+wrapper_size)*(m+wrapper_size)];
    DATA_TYPE *out_tst = new DATA_TYPE[(n+wrapper_size)*(m+wrapper_size)];
    DATA_TYPE args[9];
    args[0 ] = 1;//-1;
    args[1 ] = 1;//-2;
    args[2 ] = 1;//-1;
    args[3 ] = 1;//0;
    args[4 ] = 1;//0;
    args[5 ] = 1;//0;
    args[6 ] = 1;//1;
    args[7 ] = 1;//2;
    args[8 ] = 1;//1;


    srand(1);
    for(int i = 0; i < m+wrapper_size; i++)
    {
        for(int j = 0; j < n+wrapper_size; j++)
        {
            if(i<wrapper_size/2 || j<wrapper_size/2 || i>=m+wrapper_size/2 || j>=n+wrapper_size/2)
            {
                fetch_in(i,j) = 0;
            } else
            {
                fetch_in(i,j) = (DATA_TYPE)((DATA_TYPE)rand() * 100 / (DATA_TYPE)(RAND_MAX));
            }
        }
    }
    /*
    for(int i = 0; i < m+wrapper_size; i++)
    {
        for(int j = 0; j < n+wrapper_size; j++)
        {
            std::cout << fetch_in(i,j) << ",";
        }
        std::cout << std::endl;
    }
    // */
    auto t1 = std::chrono::high_resolution_clock::now();
    stencil_seq(in, out_ref, args, m, n, wrapper_size);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "seq: %lg ns\n", timeInNS);
    /*
    for(int i = 0; i < m+wrapper_size; i++)
    {
        for(int j = 0; j < n+wrapper_size; j++)
        {
            std::cout << fetch_out_ref(i,j) << ",";
        }
        std::cout << std::endl;
    }
    // */
    default_properties();

    t1 = std::chrono::high_resolution_clock::now();
    // for(int i = 0; i < STEPS; i++)
        stencil_hcc(in, out_tst, args, m, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc: %lg ns\n", timeInNS);
    /*
    for(int i = 0; i < m+wrapper_size; i++)
    {
        for(int j = 0; j < n+wrapper_size; j++)
        {
            std::cout << fetch_out_tst(i,j) << ",";
        }
        std::cout << std::endl;
    }
    // */
    std::cout << "Verify hcc: " << std::boolalpha << verify(out_tst, out_ref, (m+wrapper_size)*(n+wrapper_size)) << std::endl;
    std::cout << std::noboolalpha;

    t1 = std::chrono::high_resolution_clock::now();
    // for(int i = 0; i < STEPS; i++)
        stencil_hcc_shfl(in, out_tst, args, m, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc_shfl: %lg ns\n", timeInNS);
    /*
    for(int i = 0; i < m+wrapper_size; i++)
    {
        for(int j = 0; j < n+wrapper_size; j++)
        {
            std::cout << fetch_out_tst(i,j) << ",";
        }
        std::cout << std::endl;
    }
    // */
    std::cout << "Verify hcc_shfl: " << std::boolalpha << verify(out_tst, out_ref, (m+wrapper_size)*(n+wrapper_size)) << std::endl;
    std::cout << std::noboolalpha;

    t1 = std::chrono::high_resolution_clock::now();
    // for(int i = 0; i < STEPS; i++)
        stencil_hcc_shfl2(in, out_tst, args, m, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc_shfl2: %lg ns\n", timeInNS);
    /*
    for(int i = 0; i < m+wrapper_size; i++)
    {
        for(int j = 0; j < n+wrapper_size; j++)
        {
            std::cout << fetch_out_tst(i,j) << ",";
        }
        std::cout << std::endl;
    }
    // */
    std::cout << "Verify hcc_shfl2: " << std::boolalpha << verify(out_tst, out_ref, (m+wrapper_size)*(n+wrapper_size)) << std::endl;
    std::cout << std::noboolalpha;

    t1 = std::chrono::high_resolution_clock::now();
    // for(int i = 0; i < STEPS; i++)
        stencil_hcc_shfl4(in, out_tst, args, m, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc_shfl4: %lg ns\n", timeInNS);
    std::cout << "Verify hcc_shfl4: " << std::boolalpha << verify(out_tst, out_ref, (m+wrapper_size)*(n+wrapper_size)) << std::endl;
    std::cout << std::noboolalpha;

    t1 = std::chrono::high_resolution_clock::now();
    // for(int i = 0; i < STEPS; i++)
        stencil_hcc_shfl4_2(in, out_tst, args, m, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc_shfl4_2: %lg ns\n", timeInNS);
    /*
    for(int i = 0; i < m+wrapper_size; i++)
    {
        for(int j = 0; j < n+wrapper_size; j++)
        {
            std::cout << fetch_out_tst(i,j) << ",";
        }
        std::cout << std::endl;
    }
    // */
    std::cout << "Verify hcc_shfl4_2: " << std::boolalpha << verify(out_tst, out_ref, (m+wrapper_size)*(n+wrapper_size)) << std::endl;
    std::cout << std::noboolalpha;

    

    t1 = std::chrono::high_resolution_clock::now();
    // for(int i = 0; i < STEPS; i++)
        stencil_hcc_sm(in, out_tst, args, m, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc_sm: %lg ns\n", timeInNS);

    std::cout << "Verify hcc_sm: " << std::boolalpha << verify(out_tst, out_ref, n+wrapper_size) << std::endl;
    std::cout << std::noboolalpha;



    delete []in;
    delete []out_ref;
    delete []out_tst;
}

