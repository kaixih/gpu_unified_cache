#include <iostream>
#include <cmath>
#include <hc.hpp>

using namespace hc;

#define DATA_TYPE float
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
            fetch_out(i,j) = args[0] * fetch_in(i  ,j-1) +
                             args[1] * fetch_in(i  ,j+1) +
                             args[2] * fetch_in(i-1,j  ) +
                             args[3] * fetch_in(i+1,j  ) +
                             args[4] * fetch_in(i  ,j  ) ; 
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

void stencil_hcc(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int wrapper_size)
{
    extent<1> ct_domain((m+wrapper_size)*(n+wrapper_size));
    extent<2> cp_domain(m, n); 
    tiled_extent<2> cp_tile(cp_domain, 16, 16);

    auto t1 = std::chrono::high_resolution_clock::now();
    array<DATA_TYPE,1> in_d(ct_domain, in, in + (m+wrapper_size)*(n+wrapper_size));
    array<DATA_TYPE,1> out_d(ct_domain);
    array_view<DATA_TYPE> args_d(5, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
    completion_future fut = parallel_for_each(cp_tile, [=, &in_d, &out_d](tiled_index<2> tidx) restrict(amp) {
        int i = tidx.global[0] + wrapper_size/2;
        int j = tidx.global[1] + wrapper_size/2;
        if(i <= m+wrapper_size/2 && j <= n+wrapper_size/2)
        {
            fetch_out_d(i,j) =  args_d[0 ]*fetch_in_d(i  ,j-1) + 
                                args_d[1 ]*fetch_in_d(i  ,j+1) + 
                                args_d[2 ]*fetch_in_d(i-1,j  ) +
                                args_d[3 ]*fetch_in_d(i+1,j  ) + 
                                args_d[4 ]*fetch_in_d(i  ,j  ) ;
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
    array_view<DATA_TYPE> args_d(5, args);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeInNS = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fprintf(stdout, "h2d: %lg ns\n", timeInNS);

    t1 = std::chrono::high_resolution_clock::now();
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
        }
        tidx.barrier.wait();

        if(i <= m+wrapper_size/2 && j <= n+wrapper_size/2)
        {
            fetch_out_d(i,j) = args_d[0 ]*local[li  ][lj-1] + 
                               args_d[1 ]*local[li  ][lj+1] + 
                               args_d[2 ]*local[li-1][lj  ] +
                               args_d[3 ]*local[li+1][lj  ] + 
                               args_d[4 ]*local[li  ][lj  ] ;
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
    int n = 10000;
    int m = 10000;
    int wrapper_size = 2;
    DATA_TYPE *in = new DATA_TYPE[(n+wrapper_size)*(m+wrapper_size)];
    DATA_TYPE *out_ref = new DATA_TYPE[(n+wrapper_size)*(m+wrapper_size)];
    DATA_TYPE *out_tst = new DATA_TYPE[(n+wrapper_size)*(m+wrapper_size)];
    DATA_TYPE args[5];
    args[0 ] = 1;
    args[1 ] = 1;
    args[2 ] = 1;
    args[3 ] = 1;
    args[4 ] = -4;


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

