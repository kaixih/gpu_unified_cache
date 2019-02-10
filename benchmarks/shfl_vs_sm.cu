#include<iostream>
#include<algorithm>

using namespace std;

// #define DATA_TYPE float 
#define warpSize 32 

__global__ void kern_shfl_load(DATA_TYPE *in_d, DATA_TYPE *out_d, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;  
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % warpSize;

    DATA_TYPE reg = in_d[i];
    DATA_TYPE sum = 0;
    int friend_id = (lane_id+1)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+2)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+3)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+4)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+5)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+6)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+7)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+8)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    
    friend_id = (lane_id+9)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    
    friend_id = (lane_id+10)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+11)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+12)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+13)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+14)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+15)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+16)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+17)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+18)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    
    friend_id = (lane_id+19)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+20)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+21)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+22)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+23)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+24)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+25)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+26)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+27)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+28)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    
    friend_id = (lane_id+29)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+30)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    friend_id = (lane_id+31)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;

    out_d[i] = sum;
}

__global__ void kern_shfl_sync(DATA_TYPE *in_d, DATA_TYPE *out_d, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;  
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % warpSize;

    DATA_TYPE reg = in_d[i];
    DATA_TYPE sum = 0;
    int friend_id = (lane_id+1)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+2)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+3)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+4)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+5)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+6)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+7)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+8)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;
    
    friend_id = (lane_id+9)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;
    
    friend_id = (lane_id+10)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+11)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+12)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+13)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+14)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+15)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+16)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+17)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+18)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;
    
    friend_id = (lane_id+19)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+20)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+21)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+22)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+23)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+24)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+25)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+26)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+27)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+28)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;
    
    friend_id = (lane_id+29)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+30)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    friend_id = (lane_id+31)&(warpSize-1);
    sum += __shfl(reg, friend_id) ;
    reg = sum;

    out_d[i] = reg;
}

__global__ void kern_sm_load(DATA_TYPE *in_d, DATA_TYPE *out_d, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;  
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % warpSize;
    __shared__ DATA_TYPE sm[256];

    sm[tid] = in_d[i];
    __syncthreads();

    int friend_id = (lane_id+1)&(warpSize-1);
    DATA_TYPE sum = 0 ;
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+2)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+3)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+4)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+5)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    
    friend_id = (lane_id+6)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+7)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+8)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+9)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+10)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+11)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+12)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+13)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+14)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+15)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+16)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+17)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+18)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+19)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+20)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+21)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+22)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+23)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+24)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+25)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+26)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+27)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+28)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+29)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+30)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    friend_id = (lane_id+31)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;

    out_d[i] = sum;
}



__global__ void kern_sm_sync(DATA_TYPE *in_d, DATA_TYPE *out_d, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;  
    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % warpSize;
    __shared__ DATA_TYPE sm[256];

    sm[tid] = in_d[i];
    __syncthreads();

    int friend_id = (lane_id+1)&(warpSize-1);
    DATA_TYPE sum = 0 ;
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+2)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+3)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+4)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+5)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();
    
    friend_id = (lane_id+6)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+7)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+8)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+9)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+10)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+11)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+12)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+13)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+14)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+15)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+16)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+17)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+18)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+19)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+20)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+21)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+22)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+23)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+24)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+25)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+26)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+27)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+28)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+29)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+30)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    friend_id = (lane_id+31)&(warpSize-1);
    sum += sm[((tid>>5)<<5)+friend_id] ;
    sm[tid] = sum;
    __syncthreads();

    out_d[i] = sm[tid];
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
    if(argc != 1)
        n = quick_pow(2, atoi(argv[1]));
    
    cout << "size: " << n << endl;
    int total = n;
    

    DATA_TYPE *in = new DATA_TYPE[total];
    DATA_TYPE *out1 = new DATA_TYPE[total];
    DATA_TYPE *out2 = new DATA_TYPE[total];
    for(int i = 0; i < n; i++)
    {
        in[i] = i&(warpSize-1);
    }
    fill_n(out1, total, 0);
    fill_n(out2, total, 0);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    DATA_TYPE *in_d;
    DATA_TYPE *out_d;
    cudaMalloc((void**)&in_d, total*sizeof(DATA_TYPE));
    cudaMalloc((void**)&out_d, total*sizeof(DATA_TYPE));
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = (n)/256;
    dimGrid.y = 1;
    dimGrid.z = 1;
    dimBlock.x = 256;
    dimBlock.y = 1;
    dimBlock.z = 1;

    cudaMemset(out_d, 0, total);
    cudaEventRecord(start);
    kern_shfl_load<<<dimGrid, dimBlock>>>(in_d, out_d, n);
    cudaEventRecord(stop);
    cudaMemcpy(out1, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "shfl_load time(ms): " << milliseconds << endl;
   
    cudaMemset(out_d, 0, total);
    cudaEventRecord(start);
    kern_sm_load<<<dimGrid, dimBlock>>>(in_d, out_d, n);
    cudaEventRecord(stop);
    cudaMemcpy(out2, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "sm_load time(ms): " << milliseconds << endl;
    cout << "verify load results: " << boolalpha << Verify(out1, out2, total) << endl;
    
    cudaMemset(out_d, 0, total);
    cudaEventRecord(start);
    kern_shfl_sync<<<dimGrid, dimBlock>>>(in_d, out_d, n);
    cudaEventRecord(stop);
    cudaMemcpy(out1, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "shfl_sync time(ms): " << milliseconds << endl;

    cudaMemset(out_d, 0, total);
    cudaEventRecord(start);
    kern_sm_sync<<<dimGrid, dimBlock>>>(in_d, out_d, n);
    cudaEventRecord(stop);
    cudaMemcpy(out2, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "sm_sync time(ms): " << milliseconds << endl;
    cout << "verify sync results: " << boolalpha << Verify(out1, out2, total) << endl;
    // cout << out1[0] << endl;

}
