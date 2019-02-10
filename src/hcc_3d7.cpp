#include <iostream>
#include <cmath>
#include <hc.hpp>
#include "tools.h"
#include "sten_macro.h"
#include "hcc_ubuf_3d7.h"
#include "hcc_ubuf_3d7_2d.h"
#include "metrics.h"

using namespace hc;

void Stencil_Seq(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int z, int m, int n, int halo)
{

#pragma omp parallel for 
    for(int k = halo; k < z+halo; k++)
    {
        for(int j = halo; j < m+halo; j++)
        {
            for(int i = halo; i < n+halo; i++)
            {
                ACC_3D(out,k,j,i) = 
                    a0 * ACC_3D(in,k-1,j  ,i  ) +
                    a1 * ACC_3D(in,k  ,j-1,i  ) +
                    a2 * ACC_3D(in,k  ,j  ,i-1) +
                    a3 * ACC_3D(in,k  ,j  ,i  ) +
                    a4 * ACC_3D(in,k  ,j  ,i+1) +
                    a5 * ACC_3D(in,k  ,j+1,i  ) +
                    a6 * ACC_3D(in,k+1,j  ,i  ) ;
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
    tiled_extent<3> comp_tile(comp_domain, 4, 8, 8);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        L1Buffer<DATA_TYPE> buf(z, m, n, halo, tidx);
        buf.glb2buf(in);

        ACC_3D(out,buf.gidz,buf.gidy,buf.gidx) = 
            a0 * Sten_L1M_Fetch(buf,-1, 0, 0)+
            a1 * Sten_L1M_Fetch(buf, 0,-1, 0)+
            a2 * Sten_L1M_Fetch(buf, 0, 0,-1)+
            a3 * Sten_L1M_Fetch(buf, 0, 0, 0)+
            a4 * Sten_L1M_Fetch(buf, 0, 0, 1)+
            a5 * Sten_L1M_Fetch(buf, 0, 1, 0)+
            a6 * Sten_L1M_Fetch(buf, 1, 0, 0);
    });
    fut.wait();
}

void Stencil_Hcc_Sm(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(z, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 4, 8, 8);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        LDSBuffer<DATA_TYPE> buf(z, m, n, halo, tidx);
        buf.glb2buf(in, BRANCH);

        ACC_3D(out,buf.gidz,buf.gidy,buf.gidx) = 
            a0 * Sten_LDS_Fetch(buf, -1, 0, 0)+
            a1 * Sten_LDS_Fetch(buf, 0,-1, 0)+
            a2 * Sten_LDS_Fetch(buf, 0, 0,-1)+
            a3 * Sten_LDS_Fetch(buf, 0, 0, 0)+
            a4 * Sten_LDS_Fetch(buf, 0, 0, 1)+
            a5 * Sten_LDS_Fetch(buf, 0, 1, 0)+
            a6 * Sten_LDS_Fetch(buf, 1, 0, 0);
    });
    fut.wait();
}

void Stencil_Hcc_Shfl_2DWarp(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(z/4, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 4, 8, 8);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {
        REGBuffer<DATA_TYPE> buf(z, m, n, halo, tidx, 4);
        buf.glb2buf(in, CYCLIC);
        ACC_3D(out,buf.gidz,buf.gidy,buf.gidx) = 
            a0  * Sten_REG_Fetch(buf,-1, 0, 0,0)+
            a1  * Sten_REG_Fetch(buf, 0,-1, 0,0)+
            a2  * Sten_REG_Fetch(buf, 0, 0,-1,0)+
            a3  * Sten_REG_Fetch(buf, 0, 0, 0,0)+
            a4  * Sten_REG_Fetch(buf, 0, 0, 1,0)+
            a5  * Sten_REG_Fetch(buf, 0, 1, 0,0)+
            a6  * Sten_REG_Fetch(buf, 1, 0, 0,0);
        ACC_3D(out,buf.gidz+1,buf.gidy,buf.gidx) = 
            a0  * Sten_REG_Fetch(buf,-1, 0, 0,1)+
            a1  * Sten_REG_Fetch(buf, 0,-1, 0,1)+
            a2  * Sten_REG_Fetch(buf, 0, 0,-1,1)+
            a3  * Sten_REG_Fetch(buf, 0, 0, 0,1)+
            a4  * Sten_REG_Fetch(buf, 0, 0, 1,1)+
            a5  * Sten_REG_Fetch(buf, 0, 1, 0,1)+
            a6  * Sten_REG_Fetch(buf, 1, 0, 0,1);
        ACC_3D(out,buf.gidz+2,buf.gidy,buf.gidx) = 
            a0  * Sten_REG_Fetch(buf,-1, 0, 0,2)+
            a1  * Sten_REG_Fetch(buf, 0,-1, 0,2)+
            a2  * Sten_REG_Fetch(buf, 0, 0,-1,2)+
            a3  * Sten_REG_Fetch(buf, 0, 0, 0,2)+
            a4  * Sten_REG_Fetch(buf, 0, 0, 1,2)+
            a5  * Sten_REG_Fetch(buf, 0, 1, 0,2)+
            a6  * Sten_REG_Fetch(buf, 1, 0, 0,2);

        ACC_3D(out,buf.gidz+3,buf.gidy,buf.gidx) = 
            a0  * Sten_REG_Fetch(buf,-1, 0, 0,3)+
            a1  * Sten_REG_Fetch(buf, 0,-1, 0,3)+
            a2  * Sten_REG_Fetch(buf, 0, 0,-1,3)+
            a3  * Sten_REG_Fetch(buf, 0, 0, 0,3)+
            a4  * Sten_REG_Fetch(buf, 0, 0, 1,3)+
            a5  * Sten_REG_Fetch(buf, 0, 1, 0,3)+
            a6  * Sten_REG_Fetch(buf, 1, 0, 0,3);



    });
    fut.wait();
}

void Stencil_Hcc_Sweep(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(4, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 1, 4, 64);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {

        // int i = tidx.global[2] + halo;
        // int j = tidx.global[1] + halo;

        const int block_z = z / 4;
        int k = block_z * tidx.tile[0] + halo;
        const int k_end = k + block_z;

        L1Buffer2D<DATA_TYPE> buf3(m, n, halo, tidx);
        int off = (k)*(m+2*halo)*(n+2*halo);
        buf3.glb2buf(in, off);

        L1Buffer2D<DATA_TYPE> buf2(m, n, halo, tidx);
        off = (k-1)*(m+2*halo)*(n+2*halo);
        buf2.glb2buf(in, off);

        L1Buffer2D<DATA_TYPE> buf1(m, n, halo, tidx);

#pragma unroll 
        for(; k < k_end; ++k)
        {
            // std::swap(buf2, buf1);
            // std::swap(buf3, buf2);
            buf1 = buf2;
            buf2 = buf3;
            off = (k+1)*(m+2*halo)*(n+2*halo);
            buf3.glb2buf(in, off);

            ACC_3D(out,k,buf2.gidy,buf2.gidx) = 
                a0 * Sten_L1M_Fetch2D(buf1, 0, 0)+
                a1 * Sten_L1M_Fetch2D(buf2,-1, 0)+
                a2 * Sten_L1M_Fetch2D(buf2, 0,-1)+
                a3 * Sten_L1M_Fetch2D(buf2, 0, 0)+
                a4 * Sten_L1M_Fetch2D(buf2, 0, 1)+
                a5 * Sten_L1M_Fetch2D(buf2, 1, 0)+
                a6 * Sten_L1M_Fetch2D(buf3, 0, 0);

        }
    });
    fut.wait();
}

void Stencil_Hcc_Sweep_Sm(array<DATA_TYPE> &in, array<DATA_TYPE> &out, 
        DATA_TYPE a0 , DATA_TYPE a1 , DATA_TYPE a2 , DATA_TYPE a3 , DATA_TYPE a4 , 
        DATA_TYPE a5 , DATA_TYPE a6 , 
        int z, int m, int n, int halo)
{
    extent<3> comp_domain(4, m, n); 
    tiled_extent<3> comp_tile(comp_domain, 1, 4, 64);
    completion_future fut = parallel_for_each(comp_tile, [=, &in, &out](tiled_index<3> tidx) restrict(amp) {

        // int i = tidx.global[2] + halo;
        // int j = tidx.global[1] + halo;

        const int block_z = z / 4;
        int k = block_z * tidx.tile[0] + halo;
        const int k_end = k + block_z;

        LDSBuffer2D<DATA_TYPE> buf3(m, n, halo, tidx);
        int off = (k)*(m+2*halo)*(n+2*halo);
        buf3.glb2buf(in, off, BRANCH);

        LDSBuffer2D<DATA_TYPE> buf2(m, n, halo, tidx);
        off = (k-1)*(m+2*halo)*(n+2*halo);
        buf2.glb2buf(in, off, BRANCH);

        LDSBuffer2D<DATA_TYPE> buf1(m, n, halo, tidx);

#pragma unroll 
        for(; k < k_end; ++k)
        {
            // std::swap(buf2, buf1);
            // std::swap(buf3, buf2);
            // std::swap(buf3, buf1);
            // buf1 = buf2;
            // buf2 = buf3;
            // buf3 = buf1;
            // off = (k+1)*(m+2*halo)*(n+2*halo);
            // buf3.glb2buf(in, off, BRANCH);

            ACC_3D(out,k,buf2.gidy,buf2.gidx) =  
                a0 * Sten_LDS_Fetch2D(buf3, 0, 0);
                // a1 * Sten_LDS_Fetch2D(buf2,-1, 0)+
                // a2 * Sten_LDS_Fetch2D(buf2, 0,-1)+
                // a3 * Sten_LDS_Fetch2D(buf2, 0, 0)+
                // a4 * Sten_LDS_Fetch2D(buf2, 0, 1)+
                // a5 * Sten_LDS_Fetch2D(buf2, 1, 0)+
                // a6 * Sten_LDS_Fetch2D(buf3, 0, 0);

        }
    });
    fut.wait();
}

int main(int argc, char **argv)
{
#ifdef __DEBUG
    int z = 4;
    int m = 4;
    int n = 64;
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
    std::fill_n(in, total, 0);
    std::fill_n(out_ref, total, 0);
    Init_Input(in, z, m, n, halo, seed);

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
    std::cout << "Baseline done." << std::endl;

    extent<1> data_domain(total);
    array<DATA_TYPE>  in_d(data_domain);
    array<DATA_TYPE> out_d(data_domain);
    DATA_TYPE *out = new DATA_TYPE[total];
    float time_wo_pci;

    // Hcc version
    /////////////////////////////////////////
    Init_Input(in, z, m, n, halo, seed);
    std::fill_n(out, total, 0);
    copy(in , in_d );
    copy(out, out_d);

    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        // 4*8*8
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

    // Hcc version with SM
    /////////////////////////////////////////
    Init_Input(in, z, m, n, halo, seed);
    std::fill_n(out, total, 0);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        // 4*8*8
        Stencil_Hcc_Sm(in_d, out_d,  
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Sm: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Sm Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Hcc version with Shfl 
    /////////////////////////////////////////
    Init_Input(in, z, m, n, halo, seed);
    std::fill_n(out, total, 0);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        // 4*8*8
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

    // Hcc Sweep version
    /////////////////////////////////////////////////////////
    Init_Input(in, z, m, n, halo, seed);
    std::fill_n(out, total, 0);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Sweep(in_d, out_d, 
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
    // Show_Me(out, z, m, n, halo, "Output:");

    // Hcc Sweep version with Sm
    /////////////////////////////////////////////////////////
    Init_Input(in, z, m, n, halo, seed);
    std::fill_n(out, total, 0);
    copy(in , in_d );
    copy(out, out_d);

    t1 = std::chrono::high_resolution_clock::now();
    for(int i =0; i< ITER; i++)
    {
        Stencil_Hcc_Sweep_Sm(in_d, out_d, 
                args[0 ], args[1 ], args[2 ], args[3 ], args[4 ], args[5 ], 
                args[6 ], 
                z, m, n, halo); 
        std::swap(in_d, out_d);
    }
    std::swap(in_d, out_d);
    t2 = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*1.0e-03;
    copy(out_d, out);
    std::cout << "Verify Hcc_Sweep_Sm: " << std::boolalpha << Verify(out, out_ref, total) << std::endl;
    std::cout << "Hcc_Sweep_Sm Time: " << milliseconds << std::endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));
    Show_Me(out, z, m, n, halo, "Output:");

}

