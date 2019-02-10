#include <iostream>
#include <cmath>
#include <hc.hpp>

using namespace hc;

#define DATA_TYPE float
#define fetch_in(_z,_x,_y) in[(_z)*((n+wrapper_size)*(m+wrapper_size))+(_x)*(n+wrapper_size)+(_y)]
#define fetch_out(_z,_x,_y) out[(_z)*((n+wrapper_size)*(m+wrapper_size))+(_x)*(n+wrapper_size)+(_y)]
#define fetch_out_ref(_z,_x,_y) out_ref[(_z)*((n+wrapper_size)*(m+wrapper_size))+(_x)*(n+wrapper_size)+(_y)]

#define fetch_in_d(_z,_x,_y) in_d[(_z)*((n+wrapper_size)*(m+wrapper_size))+(_x)*(n+wrapper_size)+(_y)]
#define fetch_out_d(_z,_x,_y) out_d[(_z)*((n+wrapper_size)*(m+wrapper_size))+(_x)*(n+wrapper_size)+(_y)]
#define fetch_out_tst(_z,_x,_y) out_tst[(_z)*((n+wrapper_size)*(m+wrapper_size))+(_x)*(n+wrapper_size)+(_y)]

#define fetch_local(_x,_y) local[(_x)*(16+wrapper_size)+(_y)]

double tol_finder(int error_tol)
{
    double val = 1.0;
    for(; error_tol > 0; error_tol--)
        val *= 10;
    return 1.0/(double)val;
}

void stencil_seq(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int z, int m, int n, int wrapper_size)
{
    for(int k = wrapper_size/2; k < z+wrapper_size/2; k++)
    {
        for(int i = wrapper_size/2; i < m+wrapper_size/2; i++)
        {
            for(int j = wrapper_size/2; j < n+wrapper_size/2; j++)
            {
                fetch_out(k,i,j) = args[0] * fetch_in(k  ,i  ,j-1) +
                                   args[1] * fetch_in(k  ,i  ,j+1) +
                                   args[2] * fetch_in(k  ,i-1,j  ) +
                                   args[3] * fetch_in(k  ,i+1,j  ) +
                                   args[4] * fetch_in(k-1,i  ,j  ) + 
                                   args[5] * fetch_in(k+1,i  ,j  ) + 
                                   args[6] * fetch_in(k  ,i  ,j  ) ; 
            }
        }
    }
    
    
    for(int k = 0; k < z+wrapper_size; k++)
    {
        for(int i = 0; i < m+wrapper_size; i++)
        {
            for(int j = 0; j < n+wrapper_size; j++)
            {
                if(k<wrapper_size/2 || i<wrapper_size/2 || j<wrapper_size/2 || k>=z+wrapper_size/2 || i>=m+wrapper_size/2 || j>=n+wrapper_size/2)
                {
                    fetch_out(k,i,j) = 0;
                }
            }
        }
    }

}

void stencil_hcc(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int z, int m, int n, int wrapper_size)
{
    int total = (z+wrapper_size)*(m+wrapper_size)*(n+wrapper_size);
    extent<1> ct_domain(total);
    extent<3> cp_domain(z, m, n); 
    tiled_extent<3> cp_tile(cp_domain, 8, 8, 8);

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + total);
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(7, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<3> tidx) restrict(amp) {
        int k = tidx.global[0] + wrapper_size/2;
        int i = tidx.global[1] + wrapper_size/2;
        int j = tidx.global[2] + wrapper_size/2;
        if(k <= z+wrapper_size/2 && i <= m+wrapper_size/2 && j <= n+wrapper_size/2)
        {
            fetch_out_d(k,i,j) =  args_d[0 ]*fetch_in_d(  k, i  ,j-1) + 
                                  args_d[1 ]*fetch_in_d(  k, i  ,j+1) + 
                                  args_d[2 ]*fetch_in_d(  k, i-1,j  ) +
                                  args_d[3 ]*fetch_in_d(  k, i+1,j  ) + 
                                  args_d[4 ]*fetch_in_d(k-1, i  ,j  ) +
                                  args_d[5 ]*fetch_in_d(k+1, i  ,j  ) + 
                                  args_d[6 ]*fetch_in_d(  k, i  ,j  ) ;
        }
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
    
    for(int k = 0; k < z+wrapper_size; k++)
    {
        for(int i = 0; i < m+wrapper_size; i++)
        {
            for(int j = 0; j < n+wrapper_size; j++)
            {
                if(k<wrapper_size/2 || i<wrapper_size/2 || j<wrapper_size/2 || k>=z+wrapper_size/2 || i>=m+wrapper_size/2 || j>=n+wrapper_size/2)
                {
                    fetch_out(k,i,j) = 0;
                }
            }
        }
    }
}

void stencil_hcc_sweep(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int z, int m, int n, int wrapper_size)
{
    int block_z_size = 4;
    int block_real_z_size = 8;
    int total = (z+wrapper_size)*(m+wrapper_size)*(n+wrapper_size);
    extent<1> ct_domain(total);
    extent<3> cp_domain(z/block_real_z_size, m, n); 
    tiled_extent<3> cp_tile(cp_domain, block_z_size, 8, 8);

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + total);
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(7, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<3> tidx) restrict(amp) {
        // int k = tidx.global[0] + wrapper_size/2;
        int k = tidx.tile[0]*block_real_z_size + wrapper_size/2;
        int k_end = k + block_real_z_size;
        k = k + tidx.local[0];

        int i = tidx.global[1] + wrapper_size/2;
        int j = tidx.global[2] + wrapper_size/2;
        if(k <= z+wrapper_size/2 && i <= m+wrapper_size/2 && j <= n+wrapper_size/2)
        {
#pragma unroll
            for(; k < k_end; k+=block_z_size)
            {
                fetch_out_d(k,i,j) =  args_d[0 ]*fetch_in_d(  k, i  ,j-1) + 
                                      args_d[1 ]*fetch_in_d(  k, i  ,j+1) + 
                                      args_d[2 ]*fetch_in_d(  k, i-1,j  ) +
                                      args_d[3 ]*fetch_in_d(  k, i+1,j  ) + 
                                      args_d[4 ]*fetch_in_d(k-1, i  ,j  ) +
                                      args_d[5 ]*fetch_in_d(k+1, i  ,j  ) + 
                                      args_d[6 ]*fetch_in_d(  k, i  ,j  ) ;
            }
        }
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
    
    for(int k = 0; k < z+wrapper_size; k++)
    {
        for(int i = 0; i < m+wrapper_size; i++)
        {
            for(int j = 0; j < n+wrapper_size; j++)
            {
                if(k<wrapper_size/2 || i<wrapper_size/2 || j<wrapper_size/2 || k>=z+wrapper_size/2 || i>=m+wrapper_size/2 || j>=n+wrapper_size/2)
                {
                    fetch_out(k,i,j) = 0;
                }
            }
        }
    }
}

void stencil_hcc_sm(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int z, int m, int n, int wrapper_size)
{
    int total = (z+wrapper_size)*(m+wrapper_size)*(n+wrapper_size);
    extent<1> ct_domain(total);
    extent<3> cp_domain(z, m, n); 
    tiled_extent<3> cp_tile(cp_domain, 8, 8, 8);

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + total);
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(7, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<3> tidx) restrict(amp) {
        tile_static DATA_TYPE local[8+2][8+2][8+2];
        int k = tidx.global[0] + wrapper_size/2;
        int i = tidx.global[1] + wrapper_size/2;
        int j = tidx.global[2] + wrapper_size/2;
        int lk = tidx.local[0] + wrapper_size/2;
        int li = tidx.local[1] + wrapper_size/2;
        int lj = tidx.local[2] + wrapper_size/2;

        if(k <= z+wrapper_size/2 && i <= m+wrapper_size/2 && j <= n+wrapper_size/2)
        {
            local[lk][li][lj] = fetch_in_d(k,i,j);

            if(lk == wrapper_size/2)
            {
                local[lk-1][li][lj] = fetch_in_d(k-1,i,j);
            }

            if(lk == 8 + wrapper_size/2 - 1)
            {
                local[lk+1][li][lj] = fetch_in_d(k+1,i,j);
            }

            if(li == wrapper_size/2)
            {
                local[lk][li-1][lj] = fetch_in_d(k,i-1,j);
            }

            if(li == 8 + wrapper_size/2 - 1)
            {
                local[lk][li+1][lj] = fetch_in_d(k,i+1,j);
            }
            
            if(lj == wrapper_size/2)
            {
                local[lk][li][lj-1] = fetch_in_d(k,i,j-1);
            }

            if(lj == 8 + wrapper_size/2 - 1)
            {
                local[lk][li][lj+1] = fetch_in_d(k,i,j+1);
            }
        }
        tidx.barrier.wait();

        if(k <= z+wrapper_size/2 && i <= m+wrapper_size/2 && j <= n+wrapper_size/2)
        {
            fetch_out_d(k,i,j) = args_d[0 ]*local[lk  ][li  ][lj-1] + 
                                 args_d[1 ]*local[lk  ][li  ][lj+1] + 
                                 args_d[2 ]*local[lk  ][li-1][lj  ] +
                                 args_d[3 ]*local[lk  ][li+1][lj  ] + 
                                 args_d[4 ]*local[lk-1][li  ][lj  ] +
                                 args_d[5 ]*local[lk+1][li  ][lj  ] + 
                                 args_d[6 ]*local[lk  ][li  ][lj  ] ;
        }
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

    for(int k = 0; k < z+wrapper_size; k++)
    {
        for(int i = 0; i < m+wrapper_size; i++)
        {
            for(int j = 0; j < n+wrapper_size; j++)
            {
                if(k<wrapper_size/2 || i<wrapper_size/2 || j<wrapper_size/2 || k>=z+wrapper_size/2 || i>=m+wrapper_size/2 || j>=n+wrapper_size/2)
                {
                    fetch_out(k,i,j) = 0;
                }
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
    // TODO: n, m, z need to be mutiples of 8 
    int n =  128;
    int m =  128;
    int z =  128;
    int wrapper_size = 2;

    int total = (n+wrapper_size)*(m+wrapper_size)*(z+wrapper_size);
    DATA_TYPE *in      = new DATA_TYPE[total];
    DATA_TYPE *out_ref = new DATA_TYPE[total];
    DATA_TYPE *out_tst = new DATA_TYPE[total];
    DATA_TYPE args[7];
    args[0 ] = 1;
    args[1 ] = 1;
    args[2 ] = 1;
    args[3 ] = 1;
    args[4 ] = 1;
    args[5 ] = 1;
    args[6 ] = 1;

    srand(1);
    for(int k = 0; k < z+wrapper_size; k++)
        for(int i = 0; i < m+wrapper_size; i++)
            for(int j = 0; j < n+wrapper_size; j++)
                if(k<wrapper_size/2 || i<wrapper_size/2 || j<wrapper_size/2 || k>=z+wrapper_size/2 || i>=m+wrapper_size/2 || j>=n+wrapper_size/2)
                    fetch_in(k,i,j) = 0;
                else
                    fetch_in(k,i,j) = 1; // (DATA_TYPE)((DATA_TYPE)rand() * 100 / (DATA_TYPE)(RAND_MAX));
    /*
    for(int k = 0; k < z+wrapper_size; k++)
        for(int i = 0; i < m+wrapper_size; i++)
        {
            for(int j = 0; j < n+wrapper_size; j++)
                std::cout << fetch_in(k,i,j) << ",";
            std::cout << std::endl;
        }
    // */
    auto t1 = std::chrono::high_resolution_clock::now();
    stencil_seq(in, out_ref, args, z, m, n, wrapper_size);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "seq: %lg ns\n", timeInNS);
    /*
    for(int k = 0; k < z+wrapper_size; k++)
        for(int i = 0; i < m+wrapper_size; i++)
        {
            for(int j = 0; j < n+wrapper_size; j++)
                std::cout << fetch_out_ref(k,i,j) << ",";
            std::cout << std::endl;
        }
    // */

    default_properties();
    t1 = std::chrono::high_resolution_clock::now();
    stencil_hcc(in, out_tst, args, z, m, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc: %lg ns\n", timeInNS);
    /*
    for(int k = 0; k < z+wrapper_size; k++)
        for(int i = 0; i < m+wrapper_size; i++)
        {
            for(int j = 0; j < n+wrapper_size; j++)
                std::cout << fetch_out_tst(k,i,j) << ",";
            std::cout << std::endl;
        }
    // */
    std::cout << "Verify hcc: " << std::boolalpha << verify(out_tst, out_ref, total) << std::endl;
    std::cout << std::noboolalpha;

    t1 = std::chrono::high_resolution_clock::now();
    stencil_hcc_sweep(in, out_tst, args, z, m, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc_sweep: %lg ns\n", timeInNS);
    /*
    for(int k = 0; k < z+wrapper_size; k++)
        for(int i = 0; i < m+wrapper_size; i++)
        {
            for(int j = 0; j < n+wrapper_size; j++)
                std::cout << fetch_out_tst(k,i,j) << ",";
            std::cout << std::endl;
        }
    // */

    std::cout << "Verify hcc_sweep: " << std::boolalpha << verify(out_tst, out_ref, total) << std::endl;
    std::cout << std::noboolalpha;

    t1 = std::chrono::high_resolution_clock::now();
    stencil_hcc_sm(in, out_tst, args, z, m, n, wrapper_size);
    t2 = std::chrono::high_resolution_clock::now();
    timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "hcc_sm: %lg ns\n", timeInNS);
    /*
    for(int k = 0; k < z+wrapper_size; k++)
        for(int i = 0; i < m+wrapper_size; i++)
        {
            for(int j = 0; j < n+wrapper_size; j++)
                std::cout << fetch_out_tst(k,i,j) << ",";
            std::cout << std::endl;
        }
    // */

    std::cout << "Verify hcc_sm: " << std::boolalpha << verify(out_tst, out_ref, total) << std::endl;
    std::cout << std::noboolalpha;



    delete []in;
    delete []out_ref;
    delete []out_tst;
}

