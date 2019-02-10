#include <iostream>
#include "sten_macro.h"
#include "cuda_ubuf_3d7.h"
#include "cuda_ubuf_3d7_2d.h"
#include "tools.h"
#include "metrics.h"

using namespace std;

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

__global__ void Stencil_Cuda(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int z, int m, int n, int halo) 
{
    L1Buffer<DATA_TYPE> buf(z, m, n, halo);
    buf.glb2buf(in);
    ACC_3D(out,buf.gidz,buf.gidy,buf.gidx) = 
        a0 * Sten_L1M_Fetch(buf,-1, 0, 0)+
        a1 * Sten_L1M_Fetch(buf, 0,-1, 0)+
        a2 * Sten_L1M_Fetch(buf, 0, 0,-1)+
        a3 * Sten_L1M_Fetch(buf, 0, 0, 0)+
        a4 * Sten_L1M_Fetch(buf, 0, 0, 1)+
        a5 * Sten_L1M_Fetch(buf, 0, 1, 0)+
        a6 * Sten_L1M_Fetch(buf, 1, 0, 0);
}

__global__ void Stencil_Cuda_Sm_Branch(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int z, int m, int n, int halo) 
{

    LDSBuffer<DATA_TYPE> buf(z, m, n, halo);
    buf.glb2buf(in, CYCLIC);
    ACC_3D(out,buf.gidz,buf.gidy,buf.gidx) = 
        a0 * Sten_LDS_Fetch(buf,-1, 0, 0)+
        a1 * Sten_LDS_Fetch(buf, 0,-1, 0)+
        a2 * Sten_LDS_Fetch(buf, 0, 0,-1)+
        a3 * Sten_LDS_Fetch(buf, 0, 0, 0)+
        a4 * Sten_LDS_Fetch(buf, 0, 0, 1)+
        a5 * Sten_LDS_Fetch(buf, 0, 1, 0)+
        a6 * Sten_LDS_Fetch(buf, 1, 0, 0);

}

__global__ void Stencil_Cuda_Shfl_2DWarp(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int z, int m, int n, int halo)
{
    REGBuffer<DATA_TYPE> buf(z, m, n, halo, 4);
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

}

__global__ void Stencil_Cuda_Sweep(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int z, int m, int n, int halo) 
{

    const int block_z = z / 4;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;

    L1Buffer2D<DATA_TYPE> buf3(m, n, halo);
    int off = (k)*(m+2*halo)*(n+2*halo);
    buf3.glb2buf(in, off);

    L1Buffer2D<DATA_TYPE> buf2(m, n, halo);
    off = (k-1)*(m+2*halo)*(n+2*halo);
    buf2.glb2buf(in, off);

    L1Buffer2D<DATA_TYPE> buf1(m, n, halo);

#pragma unroll 
    for(; k < k_end; ++k)
    {
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


}

__global__ void Stencil_Cuda_Sweep_Sm(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int z, int m, int n, int halo) 
{

    const int block_z = z / 4;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;

    int t3=2, t2=1, t1=0;
    LDSBuffer2D<DATA_TYPE> buf(m, n, halo, 3);
    buf.setLayer(t3);
    int off = (k)*(m+2*halo)*(n+2*halo);
    buf.glb2buf(in, off, CYCLIC);

    buf.setLayer(t2);
    off = (k-1)*(m+2*halo)*(n+2*halo);
    buf.glb2buf(in, off, CYCLIC);


#pragma unroll 
    for(; k < k_end; ++k)
    {
        t1 = t2;
        t2 = t3;
        t3 = (t3+1)%3;
        buf.setLayer(t3);
        off = (k+1)*(m+2*halo)*(n+2*halo);
        buf.glb2buf(in, off, CYCLIC);

        ACC_3D(out,k,buf.gidy,buf.gidx) = 
            a0 * Sten_LDS_Fetch2D(buf, 0, 0, t1)+
            a1 * Sten_LDS_Fetch2D(buf,-1, 0, t2)+
            a2 * Sten_LDS_Fetch2D(buf, 0,-1, t2)+
            a3 * Sten_LDS_Fetch2D(buf, 0, 0, t2)+
            a4 * Sten_LDS_Fetch2D(buf, 0, 1, t2)+
            a5 * Sten_LDS_Fetch2D(buf, 1, 0, t2)+
            a6 * Sten_LDS_Fetch2D(buf, 0, 0, t3);
        __syncthreads();
    }


}

__global__ void Stencil_Cuda_Sweep_Shfl(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE a0, DATA_TYPE a1, 
        DATA_TYPE a2, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE a5, DATA_TYPE a6, 
        int z, int m, int n, int halo) 
{

    const int block_z = z / 4;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;

    int t3=2, t2=1, t1=0;
    REGBuffer2D<DATA_TYPE> buf(m, n, halo, 3, 4);
    buf.setLayer(t3);
    int off = (k)*(m+2*halo)*(n+2*halo);
    buf.glb2buf(in, off, CYCLIC);

    buf.setLayer(t2);
    off = (k-1)*(m+2*halo)*(n+2*halo);
    buf.glb2buf(in, off, CYCLIC);


#pragma unroll 
    for(; k < k_end; ++k)
    {
        t1 = t2;
        t2 = t3;
        t3 = (t3+1)%3;
        buf.setLayer(t3);
        off = (k+1)*(m+2*halo)*(n+2*halo);
        buf.glb2buf(in, off, CYCLIC);

        ACC_3D(out,k,buf.gidy,buf.gidx) = 
            a0 * buf.Sten_REG_Fetch2D( 0, 0, t1, 0)+
            a1 * buf.Sten_REG_Fetch2D(-1, 0, t2, 0)+
            a2 * buf.Sten_REG_Fetch2D( 0,-1, t2, 0)+
            a3 * buf.Sten_REG_Fetch2D( 0, 0, t2, 0)+
            a4 * buf.Sten_REG_Fetch2D( 0, 1, t2, 0)+
            a5 * buf.Sten_REG_Fetch2D( 1, 0, t2, 0)+
            a6 * buf.Sten_REG_Fetch2D( 0, 0, t3, 0);
        ACC_3D(out,k,buf.gidy+1,buf.gidx) = 
            a0 * buf.Sten_REG_Fetch2D( 0, 0, t1, 1)+
            a1 * buf.Sten_REG_Fetch2D(-1, 0, t2, 1)+
            a2 * buf.Sten_REG_Fetch2D( 0,-1, t2, 1)+
            a3 * buf.Sten_REG_Fetch2D( 0, 0, t2, 1)+
            a4 * buf.Sten_REG_Fetch2D( 0, 1, t2, 1)+
            a5 * buf.Sten_REG_Fetch2D( 1, 0, t2, 1)+
            a6 * buf.Sten_REG_Fetch2D( 0, 0, t3, 1);
        ACC_3D(out,k,buf.gidy+2,buf.gidx) = 
            a0 * buf.Sten_REG_Fetch2D( 0, 0, t1, 2)+
            a1 * buf.Sten_REG_Fetch2D(-1, 0, t2, 2)+
            a2 * buf.Sten_REG_Fetch2D( 0,-1, t2, 2)+
            a3 * buf.Sten_REG_Fetch2D( 0, 0, t2, 2)+
            a4 * buf.Sten_REG_Fetch2D( 0, 1, t2, 2)+
            a5 * buf.Sten_REG_Fetch2D( 1, 0, t2, 2)+
            a6 * buf.Sten_REG_Fetch2D( 0, 0, t3, 2);
        ACC_3D(out,k,buf.gidy+3,buf.gidx) = 
            a0 * buf.Sten_REG_Fetch2D( 0, 0, t1, 3)+
            a1 * buf.Sten_REG_Fetch2D(-1, 0, t2, 3)+
            a2 * buf.Sten_REG_Fetch2D( 0,-1, t2, 3)+
            a3 * buf.Sten_REG_Fetch2D( 0, 0, t2, 3)+
            a4 * buf.Sten_REG_Fetch2D( 0, 1, t2, 3)+
            a5 * buf.Sten_REG_Fetch2D( 1, 0, t2, 3)+
            a6 * buf.Sten_REG_Fetch2D( 0, 0, t3, 3);
    }


}

int main(int argc, char **argv)
{
#ifdef __DEBUG
    int z = 4;
    int m = 8;
    int n = 32;
#else
    int z = 256; 
    int m = 256;
    int n = 256; 
#endif
    int halo = 1;
    int order = 2*halo;
    int total = (z+order)*(m+order)*(n+order);
    const int K = 7;
#ifdef __DEBUG
    DATA_TYPE args[K] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
#else
    DATA_TYPE args[K] = {0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14};
#endif

    DATA_TYPE *in = new DATA_TYPE[total];
    DATA_TYPE *out_ref = new DATA_TYPE[total];
    unsigned int seed = time(NULL);
    fill_n(in, total, 0);
    fill_n(out_ref, total, 0);
    Init_Input(in, z, m, n, halo, seed);

    // Show_Me(in, z, m, n, halo, "Input:");
    for(int i = 0; i < ITER; i++)
    {
        Stencil_Seq(in, out_ref, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo);
        swap(in, out_ref);
    }
    swap(in, out_ref);
    // Show_Me(out_ref, z, m, n, halo, "Output:");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    float time_wo_pci;

    DATA_TYPE *in_d;
    DATA_TYPE *out_d;
    DATA_TYPE *out = new DATA_TYPE[total];
    cudaMalloc((void**)&in_d, total*sizeof(DATA_TYPE));
    cudaMalloc((void**)&out_d, total*sizeof(DATA_TYPE));
    dim3 dimGrid;
    dim3 dimBlock;

    // Cuda: Default
    /////////////////////////////////////////////////////////
    Init_Input(in, z, m, n, halo, seed);
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemset(out_d, 0, total*sizeof(DATA_TYPE));

    dimGrid.x = (n)/8; 
    dimGrid.y = (m)/4; 
    dimGrid.z = (z)/8;
    dimBlock.x = 8; 
    dimBlock.y = 4; 
    dimBlock.z = 8;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo); 
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));


    // Cuda 3D-Block with SM_Branch
    /////////////////////////////////////////////////////////
    Init_Input(in, z, m, n, halo, seed); // reset input
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemset(out_d, 0, total*sizeof(DATA_TYPE));

    dimGrid.x = (n)/8; 
    dimGrid.y = (m)/4; 
    dimGrid.z = (z)/8;
    dimBlock.x = 8; 
    dimBlock.y = 4; 
    dimBlock.z = 8;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sm_Branch<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sm_Branch: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Sm_Branch Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda 3D-Block with Shfl 1-Point (2D-Warp)
    /////////////////////////////////////////////////////////
    Init_Input(in, z, m, n, halo, seed); 
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemset(out_d, 0, total*sizeof(DATA_TYPE));

    dimGrid.x = (n)/8; 
    dimGrid.y = (m)/4; 
    dimGrid.z = (z)/(8*4);
    dimBlock.x = 8; 
    dimBlock.y = 4; 
    dimBlock.z = 8;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Shfl_2DWarp<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Shfl_2DWarp: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Shfl_2DWarp Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda 2.5D-Block
    /////////////////////////////////////////////////////////
    Init_Input(in, z, m, n, halo, seed); 
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemset(out_d, 0, total*sizeof(DATA_TYPE));

    dimGrid.x = (n)/32; 
    dimGrid.y = (m)/8; 
    dimGrid.z = 4;
    dimBlock.x = 32; 
    dimBlock.y = 8; 
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sweep<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Shfl_Sweep Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda 2.5D-Block with SM
    /////////////////////////////////////////////////////////
    Init_Input(in, z, m, n, halo, seed); 
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemset(out_d, 0, total*sizeof(DATA_TYPE));

    dimGrid.x = (n)/32; 
    dimGrid.y = (m)/8; 
    dimGrid.z = 4;
    dimBlock.x = 32; 
    dimBlock.y = 8; 
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sweep_Sm<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep_Sm: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Shfl_Sweep_Sm Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));

    // Cuda 2.5D-Block with reg
    /////////////////////////////////////////////////////////
    Init_Input(in, z, m, n, halo, seed); 
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemset(out_d, 0, total*sizeof(DATA_TYPE));

    dimGrid.x = (n)/32; 
    dimGrid.y = (m)/(8*4); 
    dimGrid.z = 4;
    dimBlock.x = 32; 
    dimBlock.y = 8; 
    dimBlock.z = 1;
    cudaEventRecord(start);
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sweep_Shfl<<<dimGrid, dimBlock>>>(in_d, out_d, 
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], 
                z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Verify Cuda_Sweep_Shfl: " << boolalpha << Verify(out, out_ref, total) << endl;
    cout << "Cuda_Shfl_Sweep_Shfl Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z+2*halo, m+2*halo, n+2*halo, ITER, OPS_3D7, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(0, z+2*halo, m+2*halo, n+2*halo, ITER, time_wo_pci, sizeof(DATA_TYPE)));
    // Show_Me(out, z, m, n, halo, "Output:");




}
