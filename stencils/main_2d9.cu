#include <iostream>
using namespace std;
#define DATA_TYPE float
#define warpSize 32 

#define IN_2D(_x,_y) in[(_x)*(n+2*halo)+(_y)]
#define OUT_2D(_x,_y) out[(_x)*(n+2*halo)+(_y)]
#define LOC_2D(_x,_y) local[(_x)*(16+2*halo)+(_y)]

void Init_Input_2D(DATA_TYPE *in, int m, int n, int halo)
{
    srand(time(NULL));

    for(int i = 0; i < m+2*halo; i++)
    {
        for(int j = 0; j < n+2*halo; j++)
        {
            if(i<halo || j<halo || i>=m+halo || j>=n+halo)
                IN_2D(i,j) = 0.0;
            else
                IN_2D(i,j) = (DATA_TYPE)rand() * 100.0 / RAND_MAX;
        }
    }
}

void Fill_Halo_2D(DATA_TYPE *in, int m, int n, int halo)
{
    for(int i = 0; i < m+2*halo; i++)
    {
        for(int j = 0; j < n+2*halo; j++)
        {
            if(i<halo || j<halo || i>=m+halo || j>=n+halo)
                IN_2D(i,j) = 0.0;
        }
    }
}

void Show_Me(DATA_TYPE *in, int m, int n, int halo, string prompt)
{
    cout << prompt << endl;
    for(int i = 0; i < m+2*halo; i++)
    {
        for(int j = 0; j < n+2*halo; j++)
        {
            std::cout << IN_2D(i,j) << ",";
        }
        std::cout << std::endl;
    }
}

void Stencil_Seq(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int halo)
{
    for(int j = halo; j < m+halo; j++)
    {
        for(int i = halo; i < n+halo; i++)
        {
            OUT_2D(j,i) = args[0] * IN_2D(j-1,i-1) +
                          args[1] * IN_2D(j  ,i-1) +
                          args[2] * IN_2D(j+1,i-1) +
                          args[3] * IN_2D(j-1,i  ) +
                          args[4] * IN_2D(j  ,i  ) +
                          args[5] * IN_2D(j+1,i  ) +
                          args[6] * IN_2D(j-1,i+1) +
                          args[7] * IN_2D(j  ,i+1) +
                          args[8] * IN_2D(j+1,i+1) ;
        }
    }

    for(int i = 0; i < m+2*halo; i++)
    {
        for(int j = 0; j < n+2*halo; j++)
        {
            if(i<halo || j<halo || i>=m+halo || j>=n+halo)
                OUT_2D(i,j) = 0.0;
        }
    }
}

__global__ void Stencil_Cuda(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    if(j < m + halo && i < n + halo)
    {
        OUT_2D(j,i) = args[0]*IN_2D(j-1,i-1) + 
                      args[1]*IN_2D(j  ,i-1) + 
                      args[2]*IN_2D(j+1,i-1) +
                      args[3]*IN_2D(j-1,i  ) + 
                      args[4]*IN_2D(j  ,i  ) + 
                      args[5]*IN_2D(j+1,i  ) +
                      args[6]*IN_2D(j-1,i+1) + 
                      args[7]*IN_2D(j  ,i+1) + 
                      args[8]*IN_2D(j+1,i+1) ;
    }
}

__global__ void Stencil_Cuda_Shfl(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;
    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3;
    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2;
    int new_i = (warp_id_x<<3) + lane_id%10;
    int new_j = (warp_id_y<<2) + lane_id/10;
    DATA_TYPE threadInput0, threadInput1;
    threadInput0 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+2)%10;
    new_j = (warp_id_y<<2) + 3 + (lane_id+2)/10;
    if(new_i < n+2*halo && new_j < m+2*halo)
        threadInput1 = IN_2D(new_j, new_i);
    // else
        // threadInput1 = -1; //IN_2D(new_i, new_j);

    DATA_TYPE sum = 0.0;
    int friend_id;
    friend_id = (lane_id+((lane_id>>3)<<1))&(warpSize-1);
    DATA_TYPE tx, ty;
    tx = args[0]*__shfl(threadInput0, friend_id);
    ty = args[0]*__shfl(threadInput1, friend_id);
    if(lane_id < 26) 
        sum += tx;
    else
        sum += ty;

    friend_id = ((friend_id+1)&(warpSize-1));
    tx = args[1]*__shfl(threadInput0, friend_id);
    ty = args[1]*__shfl(threadInput1, friend_id);
    if(lane_id < 25) 
        sum += tx;
    else
        sum += ty;

    friend_id = ((friend_id+1)&(warpSize-1));
    tx = args[2]*__shfl(threadInput0, friend_id);
    ty = args[2]*__shfl(threadInput1, friend_id);
    if(lane_id < 24) 
        sum += tx;
    else
        sum += ty;

    friend_id = ((friend_id+8)&(warpSize-1));
    tx = args[3]*__shfl(threadInput0, friend_id);
    ty = args[3]*__shfl(threadInput1, friend_id);
    if(lane_id < 18) 
        sum += tx;
    else
        sum += ty;

    friend_id = ((friend_id+1)&(warpSize-1));
    tx = args[4]*__shfl(threadInput0, friend_id);
    ty = args[4]*__shfl(threadInput1, friend_id);
    if(lane_id < 17) 
        sum += tx;
    else
        sum += ty;

    friend_id = ((friend_id+1)&(warpSize-1));
    tx = args[5]*__shfl(threadInput0, friend_id);
    ty = args[5]*__shfl(threadInput1, friend_id);
    if(lane_id < 16) 
        sum += tx;
    else
        sum += ty;

    friend_id = ((friend_id+8)&(warpSize-1));
    tx = args[6]*__shfl(threadInput0, friend_id);
    ty = args[6]*__shfl(threadInput1, friend_id);
    if(lane_id < 10) 
        sum += tx;
    else
        sum += ty;

    friend_id = ((friend_id+1)&(warpSize-1));
    tx = args[7]*__shfl(threadInput0, friend_id);
    ty = args[7]*__shfl(threadInput1, friend_id);
    if(lane_id < 9) 
        sum += tx;
    else
        sum += ty;

    friend_id = ((friend_id+1)&(warpSize-1));
    tx = args[8]*__shfl(threadInput0, friend_id);
    ty = args[8]*__shfl(threadInput1, friend_id);
    if(lane_id < 8) 
        sum += tx;
    else
        sum += ty;


    if(j < m + halo && i < n + halo)
    {
        OUT_2D(j,i) = sum; //args[0]*IN_2D(i-1,j-1) + 
                      // args[1]*IN_2D(i-1,j  ) + 
                      // args[2]*IN_2D(i-1,j+1) +
                      // args[3]*IN_2D(i  ,j-1) + 
                      // args[4]*IN_2D(i  ,j  ) + 
                      // args[5]*IN_2D(i  ,j+1) +
                      // args[6]*IN_2D(i+1,j-1) + 
                      // args[7]*IN_2D(i+1,j  ) + 
                      // args[8]*IN_2D(i+1,j+1) ;
    }
}

__global__ void Stencil_Cuda_Shfl2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int halo) 
{
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<3) + (lane_id>>3)  + halo;

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3;
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<3) + (lane_id>>3))>>2;
    int new_i = (warp_id_x<<3) + lane_id%10;
    int new_j = (warp_id_y<<2) + lane_id/10;
    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3;
    threadInput0 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+2)%10;
    new_j = (warp_id_y<<2) + 3 + (lane_id+2)/10;
    threadInput1 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+4)%10;
    new_j = (warp_id_y<<2) + 6 + (lane_id+4)/10;
    threadInput2 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+6)%10;
    new_j = (warp_id_y<<2) + 9 + (lane_id+6)/10;
    if(new_i < n+2*halo && new_j < m+2*halo)
        threadInput3 = IN_2D(new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    int friend_id0, friend_id1;
    friend_id0 = (lane_id+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+8+((lane_id>>3)<<1))&(warpSize-1);
    DATA_TYPE tx0, ty0, tx1, ty1, tz1;
    tx0 = args[0]*__shfl(threadInput0, friend_id0);
    ty0 = args[0]*__shfl(threadInput1, friend_id0);
    tx1 = args[0]*__shfl(threadInput1, friend_id1);
    ty1 = args[0]*__shfl(threadInput2, friend_id1);

    sum0 += (lane_id < 26)? tx0: ty0;
    sum1 += (lane_id < 20)? tx1: ty1;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    tx0 = args[1]*__shfl(threadInput0, friend_id0);
    ty0 = args[1]*__shfl(threadInput1, friend_id0);
    tx1 = args[1]*__shfl(threadInput1, friend_id1);
    ty1 = args[1]*__shfl(threadInput2, friend_id1);
    sum0 += (lane_id < 25)? tx0: ty0;
    sum1 += (lane_id < 19)? tx1: ty1;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    tx0 = args[2]*__shfl(threadInput0, friend_id0);
    ty0 = args[2]*__shfl(threadInput1, friend_id0);
    tx1 = args[2]*__shfl(threadInput1, friend_id1);
    ty1 = args[2]*__shfl(threadInput2, friend_id1);
    sum0 += (lane_id < 24)? tx0: ty0;
    sum1 += (lane_id < 18)? tx1: ty1;

    friend_id0 = ((friend_id0+8)&(warpSize-1));
    friend_id1 = ((friend_id1+8)&(warpSize-1));
    tx0 = args[3]*__shfl(threadInput0, friend_id0);
    ty0 = args[3]*__shfl(threadInput1, friend_id0);
    tx1 = args[3]*__shfl(threadInput1, friend_id1);
    ty1 = args[3]*__shfl(threadInput2, friend_id1);
    sum0 += (lane_id < 18)? tx0: ty0;
    sum1 += (lane_id < 12)? tx1: ty1;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    tx0 = args[4]*__shfl(threadInput0, friend_id0);
    ty0 = args[4]*__shfl(threadInput1, friend_id0);
    tx1 = args[4]*__shfl(threadInput1, friend_id1);
    ty1 = args[4]*__shfl(threadInput2, friend_id1);
    sum0 += (lane_id < 17)? tx0: ty0;
    sum1 += (lane_id < 11)? tx1: ty1;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    tx0 = args[5]*__shfl(threadInput0, friend_id0);
    ty0 = args[5]*__shfl(threadInput1, friend_id0);
    tx1 = args[5]*__shfl(threadInput1, friend_id1);
    ty1 = args[5]*__shfl(threadInput2, friend_id1);
    sum0 += (lane_id < 16)? tx0: ty0;
    sum1 += (lane_id < 10)? tx1: ty1;

    friend_id0 = ((friend_id0+8)&(warpSize-1));
    friend_id1 = ((friend_id1+8)&(warpSize-1));
    tx0 = args[6]*__shfl(threadInput0, friend_id0);
    ty0 = args[6]*__shfl(threadInput1, friend_id0);
    tx1 = args[6]*__shfl(threadInput1, friend_id1);
    ty1 = args[6]*__shfl(threadInput2, friend_id1);
    tz1 = args[6]*__shfl(threadInput3, friend_id1);
    sum0 += (lane_id < 10)? tx0: ty0;
    sum1 += (lane_id < 4)? tx1: ((lane_id < 30)? ty1: tz1);

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    tx0 = args[7]*__shfl(threadInput0, friend_id0);
    ty0 = args[7]*__shfl(threadInput1, friend_id0);
    tx1 = args[7]*__shfl(threadInput1, friend_id1);
    ty1 = args[7]*__shfl(threadInput2, friend_id1);
    tz1 = args[7]*__shfl(threadInput3, friend_id1);
    sum0 += (lane_id < 9)? tx0: ty0;
    sum1 += (lane_id < 3)? tx1: ((lane_id < 29)? ty1: tz1);

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    tx0 = args[8]*__shfl(threadInput0, friend_id0);
    ty0 = args[8]*__shfl(threadInput1, friend_id0);
    tx1 = args[8]*__shfl(threadInput1, friend_id1);
    ty1 = args[8]*__shfl(threadInput2, friend_id1);
    tz1 = args[8]*__shfl(threadInput3, friend_id1);
    sum0 += (lane_id < 8)? tx0: ty0;
    sum1 += (lane_id < 2)? tx1: ((lane_id < 28)? ty1: tz1);


    if(j < m + halo && i < n + halo)
    {
        OUT_2D(j,i) = sum0; //args[0]*IN_2D(i-1,j-1) + 
                      // args[1]*IN_2D(i-1,j  ) + 
                      // args[2]*IN_2D(i-1,j+1) +
                      // args[3]*IN_2D(i  ,j-1) + 
                      // args[4]*IN_2D(i  ,j  ) + 
                      // args[5]*IN_2D(i  ,j+1) +
                      // args[6]*IN_2D(i+1,j-1) + 
                      // args[7]*IN_2D(i+1,j  ) + 
                      // args[8]*IN_2D(i+1,j+1) ;
    }
    if(j+4 < m + halo && i < n + halo)
    {
        OUT_2D(j+4,i) = sum1; //args[0]*IN_2D(i-1,j-1) + 
                      // args[1]*IN_2D(i-1,j  ) + 
                      // args[2]*IN_2D(i-1,j+1) +
                      // args[3]*IN_2D(i  ,j-1) + 
                      // args[4]*IN_2D(i  ,j  ) + 
                      // args[5]*IN_2D(i  ,j+1) +
                      // args[6]*IN_2D(i+1,j-1) + 
                      // args[7]*IN_2D(i+1,j  ) + 
                      // args[8]*IN_2D(i+1,j+1) ;
    }
}

// div 4 and plus lane_id>>3 (determined by the warp dimension, eg. warp dim is 4 by 8)
__global__ void Stencil_Cuda_Shfl4(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int halo) 
{
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<4) + (lane_id>>3)  + halo;

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3;
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<4) + (lane_id>>3))>>2;
    int new_i = (warp_id_x<<3) + lane_id%10;
    int new_j = (warp_id_y<<2) + lane_id/10;
    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5;
    threadInput0 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+2)%10;
    new_j = (warp_id_y<<2) + 3 + (lane_id+2)/10;
    threadInput1 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+4)%10;
    new_j = (warp_id_y<<2) + 6 + (lane_id+4)/10;
    threadInput2 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+6)%10;
    new_j = (warp_id_y<<2) + 9 + (lane_id+6)/10;
    threadInput3 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+8)%10;
    new_j = (warp_id_y<<2) + 12 + (lane_id+8)/10;
    threadInput4 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+0)%10;
    new_j = (warp_id_y<<2) + 16 + (lane_id+0)/10;
    if(new_i < n+2*halo && new_j < m+2*halo)
        threadInput5 = IN_2D(new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    int friend_id0, friend_id1;
    int friend_id2, friend_id3;
    friend_id0 = (lane_id+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+8+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    DATA_TYPE tx0, ty0, tx1, ty1, tz1, tx2, ty2, tz2, tx3, ty3, tz3;
    tx0 = args[0]*__shfl(threadInput0, friend_id0);
    ty0 = args[0]*__shfl(threadInput1, friend_id0);
    tx1 = args[0]*__shfl(threadInput1, friend_id1);
    ty1 = args[0]*__shfl(threadInput2, friend_id1);
    tx2 = args[0]*__shfl(threadInput2, friend_id2);
    ty2 = args[0]*__shfl(threadInput3, friend_id2);
    tx3 = args[0]*__shfl(threadInput3, friend_id3);
    ty3 = args[0]*__shfl(threadInput4, friend_id3);
    sum0 += (lane_id < 26)? tx0: ty0;
    sum1 += (lane_id < 20)? tx1: ty1;
    sum2 += (lane_id < 14)? tx2: ty2;
    sum3 += (lane_id < 8 )? tx3: ty3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = ((friend_id2+1)&(warpSize-1));
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[1]*__shfl(threadInput0, friend_id0);
    ty0 = args[1]*__shfl(threadInput1, friend_id0);
    tx1 = args[1]*__shfl(threadInput1, friend_id1);
    ty1 = args[1]*__shfl(threadInput2, friend_id1);
    tx2 = args[1]*__shfl(threadInput2, friend_id2);
    ty2 = args[1]*__shfl(threadInput3, friend_id2);
    tx3 = args[1]*__shfl(threadInput3, friend_id3);
    ty3 = args[1]*__shfl(threadInput4, friend_id3);
    sum0 += (lane_id < 25)? tx0: ty0;
    sum1 += (lane_id < 19)? tx1: ty1;
    sum2 += (lane_id < 13)? tx2: ty2;
    sum3 += (lane_id < 7 )? tx3: ty3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = ((friend_id2+1)&(warpSize-1));
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[2]*__shfl(threadInput0, friend_id0);
    ty0 = args[2]*__shfl(threadInput1, friend_id0);
    tx1 = args[2]*__shfl(threadInput1, friend_id1);
    ty1 = args[2]*__shfl(threadInput2, friend_id1);
    tx2 = args[2]*__shfl(threadInput2, friend_id2);
    ty2 = args[2]*__shfl(threadInput3, friend_id2);
    tx3 = args[2]*__shfl(threadInput3, friend_id3);
    ty3 = args[2]*__shfl(threadInput4, friend_id3);
    sum0 += (lane_id < 24)? tx0: ty0;
    sum1 += (lane_id < 18)? tx1: ty1;
    sum2 += (lane_id < 12)? tx2: ty2;
    sum3 += (lane_id < 6 )? tx3: ty3;

    friend_id0 = ((friend_id0+8)&(warpSize-1));
    friend_id1 = ((friend_id1+8)&(warpSize-1));
    friend_id2 = ((friend_id2+8)&(warpSize-1));
    friend_id3 = ((friend_id3+8)&(warpSize-1));
    tx0 = args[3]*__shfl(threadInput0, friend_id0);
    ty0 = args[3]*__shfl(threadInput1, friend_id0);
    tx1 = args[3]*__shfl(threadInput1, friend_id1);
    ty1 = args[3]*__shfl(threadInput2, friend_id1);
    tx2 = args[3]*__shfl(threadInput2, friend_id2);
    ty2 = args[3]*__shfl(threadInput3, friend_id2);
    // tx3 = args[3]*__shfl(threadInput3, friend_id3);
    ty3 = args[3]*__shfl(threadInput4, friend_id3);
    tz3 = args[3]*__shfl(threadInput5, friend_id3);
    sum0 += (lane_id < 18)? tx0: ty0;
    sum1 += (lane_id < 12)? tx1: ty1;
    sum2 += (lane_id < 6 )? tx2: ty2;
    sum3 += (lane_id < 24)? ty3: tz3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = ((friend_id2+1)&(warpSize-1));
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[4]*__shfl(threadInput0, friend_id0);
    ty0 = args[4]*__shfl(threadInput1, friend_id0);
    tx1 = args[4]*__shfl(threadInput1, friend_id1);
    ty1 = args[4]*__shfl(threadInput2, friend_id1);
    tx2 = args[4]*__shfl(threadInput2, friend_id2);
    ty2 = args[4]*__shfl(threadInput3, friend_id2);
    tz2 = args[4]*__shfl(threadInput4, friend_id2);
    ty3 = args[4]*__shfl(threadInput4, friend_id3);
    tz3 = args[4]*__shfl(threadInput5, friend_id3);
    sum0 += (lane_id < 17)? tx0: ty0;
    sum1 += (lane_id < 11)? tx1: ty1;
    sum2 += (lane_id < 5)? tx2: ((lane_id < 31)? ty2: tz2);
    sum3 += (lane_id < 24)? ty3: tz3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = ((friend_id2+1)&(warpSize-1));
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[5]*__shfl(threadInput0, friend_id0);
    ty0 = args[5]*__shfl(threadInput1, friend_id0);
    tx1 = args[5]*__shfl(threadInput1, friend_id1);
    ty1 = args[5]*__shfl(threadInput2, friend_id1);
    tx2 = args[5]*__shfl(threadInput2, friend_id2);
    ty2 = args[5]*__shfl(threadInput3, friend_id2);
    tz2 = args[5]*__shfl(threadInput4, friend_id2);
    ty3 = args[5]*__shfl(threadInput4, friend_id3);
    tz3 = args[5]*__shfl(threadInput5, friend_id3);
    sum0 += (lane_id < 16)? tx0: ty0;
    sum1 += (lane_id < 10)? tx1: ty1;
    sum2 += (lane_id < 4)? tx2: ((lane_id < 30)? ty2: tz2);
    sum3 += (lane_id < 24)? ty3: tz3;

    friend_id0 = ((friend_id0+8)&(warpSize-1));
    friend_id1 = ((friend_id1+8)&(warpSize-1));
    friend_id2 = ((friend_id2+8)&(warpSize-1));
    friend_id3 = ((friend_id3+8)&(warpSize-1));
    tx0 = args[6]*__shfl(threadInput0, friend_id0);
    ty0 = args[6]*__shfl(threadInput1, friend_id0);
    tx1 = args[6]*__shfl(threadInput1, friend_id1);
    ty1 = args[6]*__shfl(threadInput2, friend_id1);
    tz1 = args[6]*__shfl(threadInput3, friend_id1);
    ty2 = args[6]*__shfl(threadInput3, friend_id2);
    tz2 = args[6]*__shfl(threadInput4, friend_id2);
    ty3 = args[6]*__shfl(threadInput4, friend_id3);
    tz3 = args[6]*__shfl(threadInput5, friend_id3);
    sum0 += (lane_id < 10)? tx0: ty0;
    sum1 += (lane_id < 4)? tx1: ((lane_id < 30)? ty1: tz1);
    sum2 += (lane_id < 24)? ty2: tz2;
    sum3 += (lane_id < 16)? ty3: tz3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = ((friend_id2+1)&(warpSize-1));
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[7]*__shfl(threadInput0, friend_id0);
    ty0 = args[7]*__shfl(threadInput1, friend_id0);
    tx1 = args[7]*__shfl(threadInput1, friend_id1);
    ty1 = args[7]*__shfl(threadInput2, friend_id1);
    tz1 = args[7]*__shfl(threadInput3, friend_id1);
    ty2 = args[7]*__shfl(threadInput3, friend_id2);
    tz2 = args[7]*__shfl(threadInput4, friend_id2);
    ty3 = args[7]*__shfl(threadInput4, friend_id3);
    tz3 = args[7]*__shfl(threadInput5, friend_id3);
    sum0 += (lane_id < 9)? tx0: ty0;
    sum1 += (lane_id < 3)? tx1: ((lane_id < 29)? ty1: tz1);
    sum2 += (lane_id < 23)? ty2: tz2;
    sum3 += (lane_id < 16)? ty3: tz3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = ((friend_id2+1)&(warpSize-1));
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[8]*__shfl(threadInput0, friend_id0);
    ty0 = args[8]*__shfl(threadInput1, friend_id0);
    tx1 = args[8]*__shfl(threadInput1, friend_id1);
    ty1 = args[8]*__shfl(threadInput2, friend_id1);
    tz1 = args[8]*__shfl(threadInput3, friend_id1);
    ty2 = args[8]*__shfl(threadInput3, friend_id2);
    tz2 = args[8]*__shfl(threadInput4, friend_id2);
    ty3 = args[8]*__shfl(threadInput4, friend_id3);
    tz3 = args[8]*__shfl(threadInput5, friend_id3);
    sum0 += (lane_id < 8)? tx0: ty0;
    sum1 += (lane_id < 2)? tx1: ((lane_id < 28)? ty1: tz1);
    sum2 += (lane_id < 22)? ty2: tz2;
    sum3 += (lane_id < 16)? ty3: tz3;
    /*
    // */


    if(j < m + halo && i < n + halo)
    {
        OUT_2D(j,i) = sum0; 
    }
    if(j+4 < m + halo && i < n + halo)
    {
        OUT_2D(j+4,i) = sum1; 
    }
    if(j+8 < m + halo && i < n + halo)
    {
        OUT_2D(j+8,i) = sum2; 
    }
    if(j+12 < m + halo && i < n + halo)
    {
        OUT_2D(j+12,i) = sum3; 
    }
}

__global__ void Stencil_Cuda_Shfl4_2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int halo) 
{
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;
    int i = (((threadIdx.x + blockIdx.x * blockDim.x)>>3)<<4) + (lane_id&7)   + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<3) + (lane_id>>3)  + halo;

    int warp_id_x = ((((threadIdx.x + blockIdx.x * blockDim.x)>>3)<<4) + (lane_id&7) )>>3;
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<3) + (lane_id>>3))>>2;
    int new_i = (warp_id_x<<3) + lane_id%18;
    int new_j = (warp_id_y<<2) + lane_id/18;
    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5;
    threadInput0 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+14)%18;
    new_j = (warp_id_y<<2) + 1 + (lane_id+14)/18;
    threadInput1 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+10)%18;
    new_j = (warp_id_y<<2) + 3 + (lane_id+10)/18;
    threadInput2 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+6)%18;
    new_j = (warp_id_y<<2) + 5 + (lane_id+6)/18;
    threadInput3 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+2)%18;
    new_j = (warp_id_y<<2) + 7 + (lane_id+2)/18;
    threadInput4 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+16)%18;
    new_j = (warp_id_y<<2) + 8 + (lane_id+16)/18;
    if(new_i < n+2*halo && new_j < m+2*halo)
        threadInput5 = IN_2D(new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    int friend_id0, friend_id1;
    int friend_id2, friend_id3;
    friend_id0 = (lane_id+((lane_id>>3)*10))&(warpSize-1);
    friend_id1 = (lane_id+8+((lane_id>>3)*10))&(warpSize-1);
    friend_id2 = friend_id1;
    friend_id3 = (lane_id+16+((lane_id>>3)*10))&(warpSize-1);
    DATA_TYPE tx0, ty0, tz0, tx1, ty1, tz1, tx2, ty2, tz2, tx3, ty3, tz3;
    tx0 = args[0]*__shfl(threadInput0, friend_id0);
    ty0 = args[0]*__shfl(threadInput1, friend_id0);
    tx1 = args[0]*__shfl(threadInput0, friend_id1);
    ty1 = args[0]*__shfl(threadInput1, friend_id1);
    tz1 = args[0]*__shfl(threadInput2, friend_id1);
    tx2 = args[0]*__shfl(threadInput2, friend_id2);
    ty2 = args[0]*__shfl(threadInput3, friend_id2);
    tz2 = args[0]*__shfl(threadInput4, friend_id2);
    tx3 = args[0]*__shfl(threadInput2, friend_id3);
    ty3 = args[0]*__shfl(threadInput3, friend_id3);
    tz3 = args[0]*__shfl(threadInput4, friend_id3);
    sum0 += (lane_id < 16)? tx0: ty0;
    sum1 += (lane_id < 14)? tx1: ((lane_id < 26)? ty1: tz1);
    sum2 += (lane_id < 14)? tx2: ((lane_id < 26)? ty2: tz2);
    sum3 += (lane_id < 8 )? tx3: ((lane_id < 24)? ty3: tz3);

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = friend_id1;
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[1]*__shfl(threadInput0, friend_id0);
    ty0 = args[1]*__shfl(threadInput1, friend_id0);
    tx1 = args[1]*__shfl(threadInput0, friend_id1);
    ty1 = args[1]*__shfl(threadInput1, friend_id1);
    tz1 = args[1]*__shfl(threadInput2, friend_id1);
    tx2 = args[1]*__shfl(threadInput2, friend_id2);
    ty2 = args[1]*__shfl(threadInput3, friend_id2);
    tz2 = args[1]*__shfl(threadInput4, friend_id2);
    tx3 = args[1]*__shfl(threadInput2, friend_id3);
    ty3 = args[1]*__shfl(threadInput3, friend_id3);
    tz3 = args[1]*__shfl(threadInput4, friend_id3);
    sum0 += (lane_id < 16)? tx0: ty0;
    sum1 += (lane_id < 13)? tx1: ((lane_id < 25)? ty1: tz1);
    sum2 += (lane_id < 13)? tx2: ((lane_id < 25)? ty2: tz2);
    sum3 += (lane_id < 8 )? tx3: ((lane_id < 24)? ty3: tz3);

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = friend_id1;
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[2]*__shfl(threadInput0, friend_id0);
    ty0 = args[2]*__shfl(threadInput1, friend_id0);

    tx1 = args[2]*__shfl(threadInput0, friend_id1);
    ty1 = args[2]*__shfl(threadInput1, friend_id1);
    tz1 = args[2]*__shfl(threadInput2, friend_id1);

    tx2 = args[2]*__shfl(threadInput2, friend_id2);
    ty2 = args[2]*__shfl(threadInput3, friend_id2);
    tz2 = args[2]*__shfl(threadInput4, friend_id2);

    tx3 = args[2]*__shfl(threadInput2, friend_id3);
    ty3 = args[2]*__shfl(threadInput3, friend_id3);
    tz3 = args[2]*__shfl(threadInput4, friend_id3);
    sum0 += (lane_id < 16)? tx0: ty0;
    sum1 += (lane_id < 12)? tx1: ((lane_id < 24)? ty1: tz1);
    sum2 += (lane_id < 12)? tx2: ((lane_id < 24)? ty2: tz2);
    sum3 += (lane_id < 8 )? tx3: ((lane_id < 24)? ty3: tz3);

    friend_id0 = ((friend_id0+16)&(warpSize-1));
    friend_id1 = ((friend_id1+16)&(warpSize-1));
    friend_id2 = friend_id1;
    friend_id3 = ((friend_id3+16)&(warpSize-1));
    tx0 = args[3]*__shfl(threadInput0, friend_id0);
    ty0 = args[3]*__shfl(threadInput1, friend_id0);
    tz0 = args[3]*__shfl(threadInput2, friend_id0);

    tx1 = args[3]*__shfl(threadInput0, friend_id1);
    ty1 = args[3]*__shfl(threadInput1, friend_id1);
    tz1 = args[3]*__shfl(threadInput2, friend_id1);

    tx2 = args[3]*__shfl(threadInput2, friend_id2);
    ty2 = args[3]*__shfl(threadInput3, friend_id2);
    tz2 = args[3]*__shfl(threadInput4, friend_id2);

    tx3 = args[3]*__shfl(threadInput3, friend_id3);
    ty3 = args[3]*__shfl(threadInput4, friend_id3);
    sum0 += (lane_id < 8 )? tx0: ((lane_id < 24)? ty0: tz0);
    sum1 += (lane_id < 6 )? tx1: ((lane_id < 18)? ty1: tz1);
    sum2 += (lane_id < 6 )? tx2: ((lane_id < 18)? ty2: tz2);
    sum3 += (lane_id < 16)? tx3: ty3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = friend_id1;
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[4]*__shfl(threadInput0, friend_id0);
    ty0 = args[4]*__shfl(threadInput1, friend_id0);
    tz0 = args[4]*__shfl(threadInput2, friend_id0);

    tx1 = args[4]*__shfl(threadInput0, friend_id1);
    ty1 = args[4]*__shfl(threadInput1, friend_id1);
    tz1 = args[4]*__shfl(threadInput2, friend_id1);

    tx2 = args[4]*__shfl(threadInput2, friend_id2);
    ty2 = args[4]*__shfl(threadInput3, friend_id2);
    tz2 = args[4]*__shfl(threadInput4, friend_id2);

    tx3 = args[4]*__shfl(threadInput3, friend_id3);
    ty3 = args[4]*__shfl(threadInput4, friend_id3);
    tz3 = args[4]*__shfl(threadInput5, friend_id3);
    sum0 += (lane_id < 8 )? tx0: ((lane_id < 24)? ty0: tz0);
    sum1 += (lane_id < 5 )? tx1: ((lane_id < 17)? ty1: tz1);
    sum2 += (lane_id < 5 )? tx2: ((lane_id < 17)? ty2: tz2);
    sum3 += (lane_id < 16)? tx3: ((lane_id < 31)? ty3: tz3);

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = friend_id1;
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[5]*__shfl(threadInput0, friend_id0);
    ty0 = args[5]*__shfl(threadInput1, friend_id0);
    tz0 = args[5]*__shfl(threadInput2, friend_id0);

    tx1 = args[5]*__shfl(threadInput0, friend_id1);
    ty1 = args[5]*__shfl(threadInput1, friend_id1);
    tz1 = args[5]*__shfl(threadInput2, friend_id1);

    tx2 = args[5]*__shfl(threadInput2, friend_id2);
    ty2 = args[5]*__shfl(threadInput3, friend_id2);
    tz2 = args[5]*__shfl(threadInput4, friend_id2);

    tx3 = args[5]*__shfl(threadInput3, friend_id3);
    ty3 = args[5]*__shfl(threadInput4, friend_id3);
    tz3 = args[5]*__shfl(threadInput5, friend_id3);
    sum0 += (lane_id < 8 )? tx0: ((lane_id < 24)? ty0: tz0);
    sum1 += (lane_id < 4 )? tx1: ((lane_id < 16)? ty1: tz1);
    sum2 += (lane_id < 4 )? tx2: ((lane_id < 16)? ty2: tz2);
    sum3 += (lane_id < 16)? tx3: ((lane_id < 30)? ty3: tz3);

    friend_id0 = ((friend_id0+16)&(warpSize-1));
    friend_id1 = ((friend_id1+16)&(warpSize-1));
    friend_id2 = friend_id1;
    friend_id3 = ((friend_id3+16)&(warpSize-1));
    tx0 = args[6]*__shfl(threadInput1, friend_id0);
    ty0 = args[6]*__shfl(threadInput2, friend_id0);
    tz0 = args[6]*__shfl(threadInput3, friend_id0);

    tx1 = args[6]*__shfl(threadInput1, friend_id1);
    ty1 = args[6]*__shfl(threadInput2, friend_id1);
    tz1 = args[6]*__shfl(threadInput3, friend_id1);

    tx2 = args[6]*__shfl(threadInput3, friend_id2);
    ty2 = args[6]*__shfl(threadInput4, friend_id2);
    tz2 = args[6]*__shfl(threadInput5, friend_id2);

    tx3 = args[6]*__shfl(threadInput3, friend_id3);
    ty3 = args[6]*__shfl(threadInput4, friend_id3);
    tz3 = args[6]*__shfl(threadInput5, friend_id3);
    sum0 += (lane_id < 16)? tx0: ((lane_id < 30)? ty0: tz0);
    sum1 += (lane_id < 10)? tx1: ((lane_id < 24)? ty1: tz1);
    sum2 += (lane_id < 10)? tx2: ((lane_id < 24)? ty2: tz2);
    sum3 += (lane_id < 8 )? tx3: ((lane_id < 24)? ty3: tz3);

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = friend_id1;
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[7]*__shfl(threadInput1, friend_id0);
    ty0 = args[7]*__shfl(threadInput2, friend_id0);
    tz0 = args[7]*__shfl(threadInput3, friend_id0);
    tx1 = args[7]*__shfl(threadInput1, friend_id1);
    ty1 = args[7]*__shfl(threadInput2, friend_id1);
    tz1 = args[7]*__shfl(threadInput3, friend_id1);
    tx2 = args[7]*__shfl(threadInput3, friend_id2);
    ty2 = args[7]*__shfl(threadInput4, friend_id2);
    tz2 = args[7]*__shfl(threadInput5, friend_id2);
    tx3 = args[7]*__shfl(threadInput3, friend_id3);
    ty3 = args[7]*__shfl(threadInput4, friend_id3);
    tz3 = args[7]*__shfl(threadInput5, friend_id3);
    sum0 += (lane_id < 16)? tx0: ((lane_id < 29)? ty0: tz0);
    sum1 += (lane_id < 9 )? tx1: ((lane_id < 24)? ty1: tz1);
    sum2 += (lane_id < 9 )? tx2: ((lane_id < 24)? ty2: tz2);
    sum3 += (lane_id < 8 )? tx3: ((lane_id < 23)? ty3: tz3);

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = friend_id1;
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[8]*__shfl(threadInput1, friend_id0);
    ty0 = args[8]*__shfl(threadInput2, friend_id0);
    tz0 = args[8]*__shfl(threadInput3, friend_id0);
    tx1 = args[8]*__shfl(threadInput1, friend_id1);
    ty1 = args[8]*__shfl(threadInput2, friend_id1);
    tz1 = args[8]*__shfl(threadInput3, friend_id1);
    tx2 = args[8]*__shfl(threadInput3, friend_id2);
    ty2 = args[8]*__shfl(threadInput4, friend_id2);
    tz2 = args[8]*__shfl(threadInput5, friend_id2);
    tx3 = args[8]*__shfl(threadInput3, friend_id3);
    ty3 = args[8]*__shfl(threadInput4, friend_id3);
    tz3 = args[8]*__shfl(threadInput5, friend_id3);
    sum0 += (lane_id < 16)? tx0: ((lane_id < 28)? ty0: tz0);
    sum1 += (lane_id < 8 )? tx1: ((lane_id < 24)? ty1: tz1);
    sum2 += (lane_id < 8 )? tx2: ((lane_id < 24)? ty2: tz2);
    sum3 += (lane_id < 8 )? tx3: ((lane_id < 22)? ty3: tz3);
    /*
    // */


    if(j < m + halo && i < n + halo)
    {
        OUT_2D(j,i) = sum0; 
    }
    if(j < m + halo && i+8 < n + halo)
    {
        OUT_2D(j,i+8) = sum1; 
    }
    if(j+4 < m + halo && i < n + halo)
    {
        OUT_2D(j+4,i) = sum2; 
    }
    if(j+4< m + halo && i+8 < n + halo)
    {
        OUT_2D(j+4,i+8) = sum3; 
    }
}

__global__ void Stencil_Cuda_Shfl8_2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int halo) 
{
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;
    int i = (((threadIdx.x + blockIdx.x * blockDim.x)>>3)<<4) + (lane_id&7)   + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<4) + (lane_id>>3)  + halo;

    int warp_id_x = ((((threadIdx.x + blockIdx.x * blockDim.x)>>3)<<4) + (lane_id&7) )>>3;
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<4) + (lane_id>>3))>>2;
    int new_i = (warp_id_x<<3) + lane_id%18;
    int new_j = (warp_id_y<<2) + lane_id/18;
    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5;
    DATA_TYPE threadInput6, threadInput7, threadInput8, threadInput9, threadInput10;
    threadInput0 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+14)%18;
    new_j = (warp_id_y<<2) + 1 + (lane_id+14)/18;
    threadInput1 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+10)%18;
    new_j = (warp_id_y<<2) + 3 + (lane_id+10)/18;
    threadInput2 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+6)%18;
    new_j = (warp_id_y<<2) + 5 + (lane_id+6)/18;
    threadInput3 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+2)%18;
    new_j = (warp_id_y<<2) + 7 + (lane_id+2)/18;
    threadInput4 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+16)%18;
    new_j = (warp_id_y<<2) + 8 + (lane_id+16)/18;
    threadInput5 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+12)%18;
    new_j = (warp_id_y<<2) + 10 + (lane_id+12)/18;
    threadInput6 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+8)%18;
    new_j = (warp_id_y<<2) + 12 + (lane_id+8)/18;
    threadInput7 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+4)%18;
    new_j = (warp_id_y<<2) + 14 + (lane_id+4)/18;
    threadInput8 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+0)%18;
    new_j = (warp_id_y<<2) + 16 + (lane_id+0)/18;
    threadInput9 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+14)%18;
    new_j = (warp_id_y<<2) + 17 + (lane_id+14)/18;
    if(new_i < n+2*halo && new_j < m+2*halo)
        threadInput10 = IN_2D(new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    DATA_TYPE sum4 = 0.0;
    DATA_TYPE sum5 = 0.0;
    DATA_TYPE sum6 = 0.0;
    DATA_TYPE sum7 = 0.0;
    int friend_id0, friend_id1, friend_id2, friend_id3;
    int friend_id4, friend_id5, friend_id6, friend_id7;
    friend_id0 = (lane_id+((lane_id>>3)*10))&(warpSize-1);
    friend_id1 = (lane_id+8+((lane_id>>3)*10))&(warpSize-1);
    friend_id2 = friend_id1;
    friend_id3 = (lane_id+16+((lane_id>>3)*10))&(warpSize-1);
    friend_id4 = friend_id3;
    friend_id5 = (lane_id+24+((lane_id>>3)*10))&(warpSize-1);
    friend_id6 = friend_id5;
    friend_id7 = friend_id0;
    DATA_TYPE tx0, ty0, tz0, tx1, ty1, tz1, tx2, ty2, tz2, tx3, ty3, tz3;
    DATA_TYPE rx0, ry0, rz0, rx1, ry1, rz1, rx2, ry2, rz2, rx3, ry3, rz3;
    tx0 = args[0]*__shfl(threadInput0, friend_id0);
    ty0 = args[0]*__shfl(threadInput1, friend_id0);
    tx1 = args[0]*__shfl(threadInput0, friend_id1);
    ty1 = args[0]*__shfl(threadInput1, friend_id1);
    tz1 = args[0]*__shfl(threadInput2, friend_id1);
    tx2 = args[0]*__shfl(threadInput2, friend_id2);
    ty2 = args[0]*__shfl(threadInput3, friend_id2);
    tz2 = args[0]*__shfl(threadInput4, friend_id2);
    tx3 = args[0]*__shfl(threadInput2, friend_id3);
    ty3 = args[0]*__shfl(threadInput3, friend_id3);
    tz3 = args[0]*__shfl(threadInput4, friend_id3);

    rx0 = args[0]*__shfl(threadInput4, friend_id4);
    ry0 = args[0]*__shfl(threadInput5, friend_id4);
    rz0 = args[0]*__shfl(threadInput6, friend_id4);
    rx1 = args[0]*__shfl(threadInput4, friend_id5);
    ry1 = args[0]*__shfl(threadInput5, friend_id5);
    rz1 = args[0]*__shfl(threadInput6, friend_id5);
    rx2 = args[0]*__shfl(threadInput6, friend_id6);
    ry2 = args[0]*__shfl(threadInput7, friend_id6);
    rz2 = args[0]*__shfl(threadInput8, friend_id6);
    rx3 = args[0]*__shfl(threadInput7, friend_id7);
    ry3 = args[0]*__shfl(threadInput8, friend_id7);

    sum0 += (lane_id < 16)? tx0: ty0;
    sum1 += (lane_id < 14)? tx1: ((lane_id < 26)? ty1: tz1);
    sum2 += (lane_id < 14)? tx2: ((lane_id < 26)? ty2: tz2);
    sum3 += (lane_id < 8 )? tx3: ((lane_id < 24)? ty3: tz3);

    sum4 += (lane_id < 8 )? rx0: ((lane_id < 24)? ry0: rz0);
    sum5 += (lane_id < 8 )? rx1: ((lane_id < 20)? ry1: rz1);
    sum6 += (lane_id < 8 )? rx2: ((lane_id < 20)? ry2: rz2);
    sum7 += (lane_id < 16)? rx3: ry3;


    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = friend_id1;
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    friend_id4 = friend_id3;
    friend_id5 = ((friend_id5+1)&(warpSize-1));
    friend_id6 = friend_id5;
    friend_id7 = friend_id0;
    tx0 = args[1]*__shfl(threadInput0, friend_id0);
    ty0 = args[1]*__shfl(threadInput1, friend_id0);
    tx1 = args[1]*__shfl(threadInput0, friend_id1);
    ty1 = args[1]*__shfl(threadInput1, friend_id1);
    tz1 = args[1]*__shfl(threadInput2, friend_id1);
    tx2 = args[1]*__shfl(threadInput2, friend_id2);
    ty2 = args[1]*__shfl(threadInput3, friend_id2);
    tz2 = args[1]*__shfl(threadInput4, friend_id2);
    tx3 = args[1]*__shfl(threadInput2, friend_id3);
    ty3 = args[1]*__shfl(threadInput3, friend_id3);
    tz3 = args[1]*__shfl(threadInput4, friend_id3);

    rx0 = args[1]*__shfl(threadInput4, friend_id4);
    ry0 = args[1]*__shfl(threadInput5, friend_id4);
    rz0 = args[1]*__shfl(threadInput6, friend_id4);
    rx1 = args[1]*__shfl(threadInput4, friend_id5);
    ry1 = args[1]*__shfl(threadInput5, friend_id5);
    rz1 = args[1]*__shfl(threadInput6, friend_id5);
    rx2 = args[1]*__shfl(threadInput6, friend_id6);
    ry2 = args[1]*__shfl(threadInput7, friend_id6);
    rz2 = args[1]*__shfl(threadInput8, friend_id6);
    rx3 = args[1]*__shfl(threadInput7, friend_id7);
    ry3 = args[1]*__shfl(threadInput8, friend_id7);

    sum0 += (lane_id < 16)? tx0: ty0;
    sum1 += (lane_id < 13)? tx1: ((lane_id < 25)? ty1: tz1);
    sum2 += (lane_id < 13)? tx2: ((lane_id < 25)? ty2: tz2);
    sum3 += (lane_id < 8 )? tx3: ((lane_id < 24)? ty3: tz3);

    sum4 += (lane_id < 8 )? rx0: ((lane_id < 24)? ry0: rz0);
    sum5 += (lane_id < 7 )? rx1: ((lane_id < 19)? ry1: rz1);
    sum6 += (lane_id < 7 )? rx2: ((lane_id < 19)? ry2: rz2);
    sum7 += (lane_id < 16)? rx3: ry3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = friend_id1;
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    friend_id4 = friend_id3;
    friend_id5 = ((friend_id5+1)&(warpSize-1));
    friend_id6 = friend_id5;
    friend_id7 = friend_id0;
    tx0 = args[2]*__shfl(threadInput0, friend_id0);
    ty0 = args[2]*__shfl(threadInput1, friend_id0);
    tx1 = args[2]*__shfl(threadInput0, friend_id1);
    ty1 = args[2]*__shfl(threadInput1, friend_id1);
    tz1 = args[2]*__shfl(threadInput2, friend_id1);
    tx2 = args[2]*__shfl(threadInput2, friend_id2);
    ty2 = args[2]*__shfl(threadInput3, friend_id2);
    tz2 = args[2]*__shfl(threadInput4, friend_id2);
    tx3 = args[2]*__shfl(threadInput2, friend_id3);
    ty3 = args[2]*__shfl(threadInput3, friend_id3);
    tz3 = args[2]*__shfl(threadInput4, friend_id3);

    rx0 = args[2]*__shfl(threadInput4, friend_id4);
    ry0 = args[2]*__shfl(threadInput5, friend_id4);
    rz0 = args[2]*__shfl(threadInput6, friend_id4);
    rx1 = args[2]*__shfl(threadInput4, friend_id5);
    ry1 = args[2]*__shfl(threadInput5, friend_id5);
    rz1 = args[2]*__shfl(threadInput6, friend_id5);
    rx2 = args[2]*__shfl(threadInput6, friend_id6);
    ry2 = args[2]*__shfl(threadInput7, friend_id6);
    rz2 = args[2]*__shfl(threadInput8, friend_id6);
    rx3 = args[2]*__shfl(threadInput7, friend_id7);
    ry3 = args[2]*__shfl(threadInput8, friend_id7);

    sum0 += (lane_id < 16)? tx0: ty0;
    sum1 += (lane_id < 12)? tx1: ((lane_id < 24)? ty1: tz1);
    sum2 += (lane_id < 12)? tx2: ((lane_id < 24)? ty2: tz2);
    sum3 += (lane_id < 8 )? tx3: ((lane_id < 24)? ty3: tz3);
    
    sum4 += (lane_id < 8 )? rx0: ((lane_id < 24)? ry0: rz0);
    sum5 += (lane_id < 6 )? rx1: ((lane_id < 18)? ry1: rz1);
    sum6 += (lane_id < 6 )? rx2: ((lane_id < 18)? ry2: rz2);
    sum7 += (lane_id < 16)? rx3: ry3;

    friend_id0 = ((friend_id0+16)&(warpSize-1));
    friend_id1 = ((friend_id1+16)&(warpSize-1));
    friend_id2 = friend_id1;
    friend_id3 = ((friend_id3+16)&(warpSize-1));
    friend_id4 = friend_id3;
    friend_id5 = ((friend_id5+16)&(warpSize-1));
    friend_id6 = friend_id5;
    friend_id7 = friend_id0;
    tx0 = args[3]*__shfl(threadInput0, friend_id0);
    ty0 = args[3]*__shfl(threadInput1, friend_id0);
    tz0 = args[3]*__shfl(threadInput2, friend_id0);
    tx1 = args[3]*__shfl(threadInput0, friend_id1);
    ty1 = args[3]*__shfl(threadInput1, friend_id1);
    tz1 = args[3]*__shfl(threadInput2, friend_id1);
    tx2 = args[3]*__shfl(threadInput2, friend_id2);
    ty2 = args[3]*__shfl(threadInput3, friend_id2);
    tz2 = args[3]*__shfl(threadInput4, friend_id2);
    tx3 = args[3]*__shfl(threadInput3, friend_id3);
    ty3 = args[3]*__shfl(threadInput4, friend_id3);
    
    rx0 = args[3]*__shfl(threadInput5, friend_id4);
    ry0 = args[3]*__shfl(threadInput6, friend_id4);
    rx1 = args[3]*__shfl(threadInput5, friend_id5);
    ry1 = args[3]*__shfl(threadInput6, friend_id5);
    rz1 = args[3]*__shfl(threadInput7, friend_id5);
    rx2 = args[3]*__shfl(threadInput7, friend_id6);
    ry2 = args[3]*__shfl(threadInput8, friend_id6);
    rz2 = args[3]*__shfl(threadInput9, friend_id6);
    rx3 = args[3]*__shfl(threadInput7, friend_id7);
    ry3 = args[3]*__shfl(threadInput8, friend_id7);
    rz3 = args[3]*__shfl(threadInput9, friend_id7);

    sum0 += (lane_id < 8 )? tx0: ((lane_id < 24)? ty0: tz0);
    sum1 += (lane_id < 6 )? tx1: ((lane_id < 18)? ty1: tz1);
    sum2 += (lane_id < 6 )? tx2: ((lane_id < 18)? ty2: tz2);
    sum3 += (lane_id < 16)? tx3: ty3;

    sum4 += (lane_id < 16)? rx0: ry0;
    sum5 += (lane_id < 12)? rx1: ((lane_id < 24)? ry1: rz1);
    sum6 += (lane_id < 12)? rx2: ((lane_id < 24)? ry2: rz2);
    sum7 += (lane_id < 8 )? rx3: ((lane_id < 24)? ry3: rz3);

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = friend_id1;
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    friend_id4 = friend_id3;
    friend_id5 = ((friend_id5+1)&(warpSize-1));
    friend_id6 = friend_id5;
    friend_id7 = friend_id0;
    tx0 = args[4]*__shfl(threadInput0, friend_id0);
    ty0 = args[4]*__shfl(threadInput1, friend_id0);
    tz0 = args[4]*__shfl(threadInput2, friend_id0);
    tx1 = args[4]*__shfl(threadInput0, friend_id1);
    ty1 = args[4]*__shfl(threadInput1, friend_id1);
    tz1 = args[4]*__shfl(threadInput2, friend_id1);
    tx2 = args[4]*__shfl(threadInput2, friend_id2);
    ty2 = args[4]*__shfl(threadInput3, friend_id2);
    tz2 = args[4]*__shfl(threadInput4, friend_id2);
    tx3 = args[4]*__shfl(threadInput3, friend_id3);
    ty3 = args[4]*__shfl(threadInput4, friend_id3);
    tz3 = args[4]*__shfl(threadInput5, friend_id3);

    rx0 = args[4]*__shfl(threadInput5, friend_id4);
    ry0 = args[4]*__shfl(threadInput6, friend_id4);
    rz0 = args[4]*__shfl(threadInput7, friend_id4);
    rx1 = args[4]*__shfl(threadInput5, friend_id5);
    ry1 = args[4]*__shfl(threadInput6, friend_id5);
    rz1 = args[4]*__shfl(threadInput7, friend_id5);
    rx2 = args[4]*__shfl(threadInput7, friend_id6);
    ry2 = args[4]*__shfl(threadInput8, friend_id6);
    rz2 = args[4]*__shfl(threadInput9, friend_id6);
    rx3 = args[4]*__shfl(threadInput7, friend_id7);
    ry3 = args[4]*__shfl(threadInput8, friend_id7);
    rz3 = args[4]*__shfl(threadInput9, friend_id7);


    sum0 += (lane_id < 8 )? tx0: ((lane_id < 24)? ty0: tz0);
    sum1 += (lane_id < 5 )? tx1: ((lane_id < 17)? ty1: tz1);
    sum2 += (lane_id < 5 )? tx2: ((lane_id < 17)? ty2: tz2);
    sum3 += (lane_id < 16)? tx3: ((lane_id < 31)? ty3: tz3);

    sum4 += (lane_id < 16)? rx0: ((lane_id < 31)? ry0: rz0);
    sum5 += (lane_id < 11)? rx1: ((lane_id < 24)? ry1: rz1);
    sum6 += (lane_id < 11)? rx2: ((lane_id < 24)? ry2: rz2);
    sum7 += (lane_id < 8 )? rx3: ((lane_id < 24)? ry3: rz3);

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = friend_id1;
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    friend_id4 = friend_id3;
    friend_id5 = ((friend_id5+1)&(warpSize-1));
    friend_id6 = friend_id5;
    friend_id7 = friend_id0;
    tx0 = args[5]*__shfl(threadInput0, friend_id0);
    ty0 = args[5]*__shfl(threadInput1, friend_id0);
    tz0 = args[5]*__shfl(threadInput2, friend_id0);
    tx1 = args[5]*__shfl(threadInput0, friend_id1);
    ty1 = args[5]*__shfl(threadInput1, friend_id1);
    tz1 = args[5]*__shfl(threadInput2, friend_id1);
    tx2 = args[5]*__shfl(threadInput2, friend_id2);
    ty2 = args[5]*__shfl(threadInput3, friend_id2);
    tz2 = args[5]*__shfl(threadInput4, friend_id2);
    tx3 = args[5]*__shfl(threadInput3, friend_id3);
    ty3 = args[5]*__shfl(threadInput4, friend_id3);
    tz3 = args[5]*__shfl(threadInput5, friend_id3);

    rx0 = args[5]*__shfl(threadInput5, friend_id4);
    ry0 = args[5]*__shfl(threadInput6, friend_id4);
    rz0 = args[5]*__shfl(threadInput7, friend_id4);
    rx1 = args[5]*__shfl(threadInput5, friend_id5);
    ry1 = args[5]*__shfl(threadInput6, friend_id5);
    rz1 = args[5]*__shfl(threadInput7, friend_id5);
    rx2 = args[5]*__shfl(threadInput7, friend_id6);
    ry2 = args[5]*__shfl(threadInput8, friend_id6);
    rz2 = args[5]*__shfl(threadInput9, friend_id6);
    rx3 = args[5]*__shfl(threadInput7, friend_id7);
    ry3 = args[5]*__shfl(threadInput8, friend_id7);
    rz3 = args[5]*__shfl(threadInput9, friend_id7);

    sum0 += (lane_id < 8 )? tx0: ((lane_id < 24)? ty0: tz0);
    sum1 += (lane_id < 4 )? tx1: ((lane_id < 16)? ty1: tz1);
    sum2 += (lane_id < 4 )? tx2: ((lane_id < 16)? ty2: tz2);
    sum3 += (lane_id < 16)? tx3: ((lane_id < 30)? ty3: tz3);

    sum4 += (lane_id < 16)? rx0: ((lane_id < 30)? ry0: rz0);
    sum5 += (lane_id < 10)? rx1: ((lane_id < 24)? ry1: rz1);
    sum6 += (lane_id < 10)? rx2: ((lane_id < 24)? ry2: rz2);
    sum7 += (lane_id < 8 )? rx3: ((lane_id < 24)? ry3: rz3);

    friend_id0 = ((friend_id0+16)&(warpSize-1));
    friend_id1 = ((friend_id1+16)&(warpSize-1));
    friend_id2 = friend_id1;
    friend_id3 = ((friend_id3+16)&(warpSize-1));
    friend_id4 = friend_id3;
    friend_id5 = ((friend_id5+16)&(warpSize-1));
    friend_id6 = friend_id5;
    friend_id7 = friend_id0;
    tx0 = args[6]*__shfl(threadInput1, friend_id0);
    ty0 = args[6]*__shfl(threadInput2, friend_id0);
    tz0 = args[6]*__shfl(threadInput3, friend_id0);
    tx1 = args[6]*__shfl(threadInput1, friend_id1);
    ty1 = args[6]*__shfl(threadInput2, friend_id1);
    tz1 = args[6]*__shfl(threadInput3, friend_id1);
    tx2 = args[6]*__shfl(threadInput3, friend_id2);
    ty2 = args[6]*__shfl(threadInput4, friend_id2);
    tz2 = args[6]*__shfl(threadInput5, friend_id2);
    tx3 = args[6]*__shfl(threadInput3, friend_id3);
    ty3 = args[6]*__shfl(threadInput4, friend_id3);
    tz3 = args[6]*__shfl(threadInput5, friend_id3);

    rx0 = args[6]*__shfl(threadInput5, friend_id4);
    ry0 = args[6]*__shfl(threadInput6, friend_id4);
    rz0 = args[6]*__shfl(threadInput7, friend_id4);
    rx1 = args[6]*__shfl(threadInput5, friend_id5);
    ry1 = args[6]*__shfl(threadInput6, friend_id5);
    rz1 = args[6]*__shfl(threadInput7, friend_id5);
    rx2 = args[6]*__shfl(threadInput7, friend_id6);
    ry2 = args[6]*__shfl(threadInput8, friend_id6);
    rz2 = args[6]*__shfl(threadInput9, friend_id6);
    rx3 = args[6]*__shfl(threadInput8, friend_id7);
    ry3 = args[6]*__shfl(threadInput9, friend_id7);
    rz3 = args[6]*__shfl(threadInput10, friend_id7);

    sum0 += (lane_id < 16)? tx0: ((lane_id < 30)? ty0: tz0);
    sum1 += (lane_id < 10)? tx1: ((lane_id < 24)? ty1: tz1);
    sum2 += (lane_id < 10)? tx2: ((lane_id < 24)? ty2: tz2);
    sum3 += (lane_id < 8 )? tx3: ((lane_id < 24)? ty3: tz3);

    sum4 += (lane_id < 8 )? rx0: ((lane_id < 24)? ry0: rz0);
    sum5 += (lane_id < 4 )? rx1: ((lane_id < 16)? ry1: rz1);
    sum6 += (lane_id < 4 )? rx2: ((lane_id < 16)? ry2: rz2);
    sum7 += (lane_id < 16)? rx3: ((lane_id < 30)? ry3: rz3);

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = friend_id1;
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    friend_id4 = friend_id3;
    friend_id5 = ((friend_id5+1)&(warpSize-1));
    friend_id6 = friend_id5;
    friend_id7 = friend_id0;
    tx0 = args[7]*__shfl(threadInput1, friend_id0);
    ty0 = args[7]*__shfl(threadInput2, friend_id0);
    tz0 = args[7]*__shfl(threadInput3, friend_id0);
    tx1 = args[7]*__shfl(threadInput1, friend_id1);
    ty1 = args[7]*__shfl(threadInput2, friend_id1);
    tz1 = args[7]*__shfl(threadInput3, friend_id1);
    tx2 = args[7]*__shfl(threadInput3, friend_id2);
    ty2 = args[7]*__shfl(threadInput4, friend_id2);
    tz2 = args[7]*__shfl(threadInput5, friend_id2);
    tx3 = args[7]*__shfl(threadInput3, friend_id3);
    ty3 = args[7]*__shfl(threadInput4, friend_id3);
    tz3 = args[7]*__shfl(threadInput5, friend_id3);

    rx0 = args[7]*__shfl(threadInput5, friend_id4);
    ry0 = args[7]*__shfl(threadInput6, friend_id4);
    rz0 = args[7]*__shfl(threadInput7, friend_id4);
    rx1 = args[7]*__shfl(threadInput5, friend_id5);
    ry1 = args[7]*__shfl(threadInput6, friend_id5);
    rz1 = args[7]*__shfl(threadInput7, friend_id5);
    rx2 = args[7]*__shfl(threadInput7, friend_id6);
    ry2 = args[7]*__shfl(threadInput8, friend_id6);
    rz2 = args[7]*__shfl(threadInput9, friend_id6);
    rx3 = args[7]*__shfl(threadInput8, friend_id7);
    ry3 = args[7]*__shfl(threadInput9, friend_id7);
    rz3 = args[7]*__shfl(threadInput10, friend_id7);

    sum0 += (lane_id < 16)? tx0: ((lane_id < 29)? ty0: tz0);
    sum1 += (lane_id < 9 )? tx1: ((lane_id < 24)? ty1: tz1);
    sum2 += (lane_id < 9 )? tx2: ((lane_id < 24)? ty2: tz2);
    sum3 += (lane_id < 8 )? tx3: ((lane_id < 23)? ty3: tz3);

    sum4 += (lane_id < 8 )? rx0: ((lane_id < 23)? ry0: rz0);
    sum5 += (lane_id < 3 )? rx1: ((lane_id < 16)? ry1: rz1);
    sum6 += (lane_id < 3 )? rx2: ((lane_id < 16)? ry2: rz2);
    sum7 += (lane_id < 16)? rx3: ((lane_id < 29)? ry3: rz3);

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = friend_id1;
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    friend_id4 = friend_id3;
    friend_id5 = ((friend_id5+1)&(warpSize-1));
    friend_id6 = friend_id5;
    friend_id7 = friend_id0;
    tx0 = args[8]*__shfl(threadInput1, friend_id0);
    ty0 = args[8]*__shfl(threadInput2, friend_id0);
    tz0 = args[8]*__shfl(threadInput3, friend_id0);
    tx1 = args[8]*__shfl(threadInput1, friend_id1);
    ty1 = args[8]*__shfl(threadInput2, friend_id1);
    tz1 = args[8]*__shfl(threadInput3, friend_id1);
    tx2 = args[8]*__shfl(threadInput3, friend_id2);
    ty2 = args[8]*__shfl(threadInput4, friend_id2);
    tz2 = args[8]*__shfl(threadInput5, friend_id2);
    tx3 = args[8]*__shfl(threadInput3, friend_id3);
    ty3 = args[8]*__shfl(threadInput4, friend_id3);
    tz3 = args[8]*__shfl(threadInput5, friend_id3);

    rx0 = args[8]*__shfl(threadInput5, friend_id4);
    ry0 = args[8]*__shfl(threadInput6, friend_id4);
    rz0 = args[8]*__shfl(threadInput7, friend_id4);
    rx1 = args[8]*__shfl(threadInput5, friend_id5);
    ry1 = args[8]*__shfl(threadInput6, friend_id5);
    rz1 = args[8]*__shfl(threadInput7, friend_id5);
    rx2 = args[8]*__shfl(threadInput7, friend_id6);
    ry2 = args[8]*__shfl(threadInput8, friend_id6);
    rz2 = args[8]*__shfl(threadInput9, friend_id6);
    rx3 = args[8]*__shfl(threadInput8, friend_id7);
    ry3 = args[8]*__shfl(threadInput9, friend_id7);
    rz3 = args[8]*__shfl(threadInput10, friend_id7);

    sum0 += (lane_id < 16)? tx0: ((lane_id < 28)? ty0: tz0);
    sum1 += (lane_id < 8 )? tx1: ((lane_id < 24)? ty1: tz1);
    sum2 += (lane_id < 8 )? tx2: ((lane_id < 24)? ty2: tz2);
    sum3 += (lane_id < 8 )? tx3: ((lane_id < 22)? ty3: tz3);

    sum4 += (lane_id < 8 )? rx0: ((lane_id < 22)? ry0: rz0);
    sum5 += (lane_id < 2 )? rx1: ((lane_id < 16)? ry1: rz1);
    sum6 += (lane_id < 2 )? rx2: ((lane_id < 16)? ry2: rz2);
    sum7 += (lane_id < 16)? rx3: ((lane_id < 28)? ry3: rz3);
    /*
    // */


    if(j < m + halo && i < n + halo)
    {
        OUT_2D(j,i) = sum0; 
    }
    if(j < m + halo && i+8 < n + halo)
    {
        OUT_2D(j,i+8) = sum1; 
    }
    if(j+4 < m + halo && i < n + halo)
    {
        OUT_2D(j+4,i) = sum2; 
    }
    if(j+4< m + halo && i+8 < n + halo)
    {
        OUT_2D(j+4,i+8) = sum3; 
    }
    if(j+8< m + halo && i < n + halo)
    {
        OUT_2D(j+8,i) = sum4; 
    }
    if(j+8< m + halo && i+8 < n + halo)
    {
        OUT_2D(j+8,i+8) = sum5; 
    }
    if(j+12< m + halo && i < n + halo)
    {
        OUT_2D(j+12,i) = sum6; 
    }
    if(j+12< m + halo && i+8 < n + halo)
    {
        OUT_2D(j+12,i+8) = sum7; 
    }
}

__global__ void Stencil_Cuda_Shfl8(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int halo) 
{
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<5) + (lane_id>>3)  + halo;

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3;
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<5) + (lane_id>>3))>>2;
    int new_i = (warp_id_x<<3) + lane_id%10;
    int new_j = (warp_id_y<<2) + lane_id/10;
    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5;
    DATA_TYPE threadInput6, threadInput7, threadInput8, threadInput9, threadInput10;
    threadInput0 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+2)%10;
    new_j = (warp_id_y<<2) + 3 + (lane_id+2)/10;
    threadInput1 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+4)%10;
    new_j = (warp_id_y<<2) + 6 + (lane_id+4)/10;
    threadInput2 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+6)%10;
    new_j = (warp_id_y<<2) + 9 + (lane_id+6)/10;
    threadInput3 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+8)%10;
    new_j = (warp_id_y<<2) + 12 + (lane_id+8)/10;
    threadInput4 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+0)%10;
    new_j = (warp_id_y<<2) + 16 + (lane_id+0)/10;
    threadInput5 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+2)%10;
    new_j = (warp_id_y<<2) + 19 + (lane_id+2)/10;
    threadInput6 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+4)%10;
    new_j = (warp_id_y<<2) + 22 + (lane_id+4)/10;
    threadInput7 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+6)%10;
    new_j = (warp_id_y<<2) + 25 + (lane_id+6)/10;
    threadInput8 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+8)%10;
    new_j = (warp_id_y<<2) + 28 + (lane_id+8)/10;
    threadInput9 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+0)%10;
    new_j = (warp_id_y<<2) + 32 + (lane_id+0)/10;
    if(new_i < n+2*halo && new_j < m+2*halo)
        threadInput10 = IN_2D(new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    DATA_TYPE sum4 = 0.0;
    DATA_TYPE sum5 = 0.0;
    DATA_TYPE sum6 = 0.0;
    DATA_TYPE sum7 = 0.0;
    int friend_id0, friend_id1, friend_id2, friend_id3;
    // int friend_id4, friend_id5, friend_id6, friend_id7;
    friend_id0 = (lane_id+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+8+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    DATA_TYPE tx0, ty0, tx1, ty1, tz1, tx2, ty2, tz2, tx3, ty3, tz3;
    DATA_TYPE rx0, ry0, rx1, ry1, rz1, rx2, ry2, rz2, rx3, ry3, rz3;
    tx0 = args[0]*__shfl(threadInput0, friend_id0);
    ty0 = args[0]*__shfl(threadInput1, friend_id0);
    tx1 = args[0]*__shfl(threadInput1, friend_id1);
    ty1 = args[0]*__shfl(threadInput2, friend_id1);
    tx2 = args[0]*__shfl(threadInput2, friend_id2);
    ty2 = args[0]*__shfl(threadInput3, friend_id2);
    tx3 = args[0]*__shfl(threadInput3, friend_id3);
    ty3 = args[0]*__shfl(threadInput4, friend_id3);
    rx0 = args[0]*__shfl(threadInput5, friend_id0);
    ry0 = args[0]*__shfl(threadInput6, friend_id0);
    rx1 = args[0]*__shfl(threadInput6, friend_id1);
    ry1 = args[0]*__shfl(threadInput7, friend_id1);
    rx2 = args[0]*__shfl(threadInput7, friend_id2);
    ry2 = args[0]*__shfl(threadInput8, friend_id2);
    rx3 = args[0]*__shfl(threadInput8, friend_id3);
    ry3 = args[0]*__shfl(threadInput9, friend_id3);
    sum0 += (lane_id < 26)? tx0: ty0;
    sum1 += (lane_id < 20)? tx1: ty1;
    sum2 += (lane_id < 14)? tx2: ty2;
    sum3 += (lane_id < 8 )? tx3: ty3;
    sum4 += (lane_id < 26)? rx0: ry0;
    sum5 += (lane_id < 20)? rx1: ry1;
    sum6 += (lane_id < 14)? rx2: ry2;
    sum7 += (lane_id < 8 )? rx3: ry3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = ((friend_id2+1)&(warpSize-1));
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[1]*__shfl(threadInput0, friend_id0);
    ty0 = args[1]*__shfl(threadInput1, friend_id0);
    tx1 = args[1]*__shfl(threadInput1, friend_id1);
    ty1 = args[1]*__shfl(threadInput2, friend_id1);
    tx2 = args[1]*__shfl(threadInput2, friend_id2);
    ty2 = args[1]*__shfl(threadInput3, friend_id2);
    tx3 = args[1]*__shfl(threadInput3, friend_id3);
    ty3 = args[1]*__shfl(threadInput4, friend_id3);
    rx0 = args[1]*__shfl(threadInput5, friend_id0);
    ry0 = args[1]*__shfl(threadInput6, friend_id0);
    rx1 = args[1]*__shfl(threadInput6, friend_id1);
    ry1 = args[1]*__shfl(threadInput7, friend_id1);
    rx2 = args[1]*__shfl(threadInput7, friend_id2);
    ry2 = args[1]*__shfl(threadInput8, friend_id2);
    rx3 = args[1]*__shfl(threadInput8, friend_id3);
    ry3 = args[1]*__shfl(threadInput9, friend_id3);
    sum0 += (lane_id < 25)? tx0: ty0;
    sum1 += (lane_id < 19)? tx1: ty1;
    sum2 += (lane_id < 13)? tx2: ty2;
    sum3 += (lane_id < 7 )? tx3: ty3;
    sum4 += (lane_id < 25)? rx0: ry0;
    sum5 += (lane_id < 19)? rx1: ry1;
    sum6 += (lane_id < 13)? rx2: ry2;
    sum7 += (lane_id < 7 )? rx3: ry3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = ((friend_id2+1)&(warpSize-1));
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[2]*__shfl(threadInput0, friend_id0);
    ty0 = args[2]*__shfl(threadInput1, friend_id0);
    tx1 = args[2]*__shfl(threadInput1, friend_id1);
    ty1 = args[2]*__shfl(threadInput2, friend_id1);
    tx2 = args[2]*__shfl(threadInput2, friend_id2);
    ty2 = args[2]*__shfl(threadInput3, friend_id2);
    tx3 = args[2]*__shfl(threadInput3, friend_id3);
    ty3 = args[2]*__shfl(threadInput4, friend_id3);
    rx0 = args[2]*__shfl(threadInput5, friend_id0);
    ry0 = args[2]*__shfl(threadInput6, friend_id0);
    rx1 = args[2]*__shfl(threadInput6, friend_id1);
    ry1 = args[2]*__shfl(threadInput7, friend_id1);
    rx2 = args[2]*__shfl(threadInput7, friend_id2);
    ry2 = args[2]*__shfl(threadInput8, friend_id2);
    rx3 = args[2]*__shfl(threadInput8, friend_id3);
    ry3 = args[2]*__shfl(threadInput9, friend_id3);
    sum0 += (lane_id < 24)? tx0: ty0;
    sum1 += (lane_id < 18)? tx1: ty1;
    sum2 += (lane_id < 12)? tx2: ty2;
    sum3 += (lane_id < 6 )? tx3: ty3;
    sum4 += (lane_id < 24)? rx0: ry0;
    sum5 += (lane_id < 18)? rx1: ry1;
    sum6 += (lane_id < 12)? rx2: ry2;
    sum7 += (lane_id < 6 )? rx3: ry3;

    friend_id0 = ((friend_id0+8)&(warpSize-1));
    friend_id1 = ((friend_id1+8)&(warpSize-1));
    friend_id2 = ((friend_id2+8)&(warpSize-1));
    friend_id3 = ((friend_id3+8)&(warpSize-1));
    tx0 = args[3]*__shfl(threadInput0, friend_id0);
    ty0 = args[3]*__shfl(threadInput1, friend_id0);
    tx1 = args[3]*__shfl(threadInput1, friend_id1);
    ty1 = args[3]*__shfl(threadInput2, friend_id1);
    tx2 = args[3]*__shfl(threadInput2, friend_id2);
    ty2 = args[3]*__shfl(threadInput3, friend_id2);
    ty3 = args[3]*__shfl(threadInput4, friend_id3);
    tz3 = args[3]*__shfl(threadInput5, friend_id3);
    rx0 = args[3]*__shfl(threadInput5, friend_id0);
    ry0 = args[3]*__shfl(threadInput6, friend_id0);
    rx1 = args[3]*__shfl(threadInput6, friend_id1);
    ry1 = args[3]*__shfl(threadInput7, friend_id1);
    rx2 = args[3]*__shfl(threadInput7, friend_id2);
    ry2 = args[3]*__shfl(threadInput8, friend_id2);
    ry3 = args[3]*__shfl(threadInput9, friend_id3);
    rz3 = args[3]*__shfl(threadInput10, friend_id3);
    sum0 += (lane_id < 18)? tx0: ty0;
    sum1 += (lane_id < 12)? tx1: ty1;
    sum2 += (lane_id < 6 )? tx2: ty2;
    sum3 += (lane_id < 24)? ty3: tz3;
    sum4 += (lane_id < 18)? rx0: ry0;
    sum5 += (lane_id < 12)? rx1: ry1;
    sum6 += (lane_id < 6 )? rx2: ry2;
    sum7 += (lane_id < 24)? ry3: rz3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = ((friend_id2+1)&(warpSize-1));
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[4]*__shfl(threadInput0, friend_id0);
    ty0 = args[4]*__shfl(threadInput1, friend_id0);
    tx1 = args[4]*__shfl(threadInput1, friend_id1);
    ty1 = args[4]*__shfl(threadInput2, friend_id1);
    tx2 = args[4]*__shfl(threadInput2, friend_id2);
    ty2 = args[4]*__shfl(threadInput3, friend_id2);
    tz2 = args[4]*__shfl(threadInput4, friend_id2);
    ty3 = args[4]*__shfl(threadInput4, friend_id3);
    tz3 = args[4]*__shfl(threadInput5, friend_id3);
    rx0 = args[4]*__shfl(threadInput5, friend_id0);
    ry0 = args[4]*__shfl(threadInput6, friend_id0);
    rx1 = args[4]*__shfl(threadInput6, friend_id1);
    ry1 = args[4]*__shfl(threadInput7, friend_id1);
    rx2 = args[4]*__shfl(threadInput7, friend_id2);
    ry2 = args[4]*__shfl(threadInput8, friend_id2);
    rz2 = args[4]*__shfl(threadInput9, friend_id2);
    ry3 = args[4]*__shfl(threadInput9, friend_id3);
    rz3 = args[4]*__shfl(threadInput10, friend_id3);
    sum0 += (lane_id < 17)? tx0: ty0;
    sum1 += (lane_id < 11)? tx1: ty1;
    sum2 += (lane_id < 5)? tx2: ((lane_id < 31)? ty2: tz2);
    sum3 += (lane_id < 24)? ty3: tz3;
    sum4 += (lane_id < 17)? rx0: ry0;
    sum5 += (lane_id < 11)? rx1: ry1;
    sum6 += (lane_id < 5)?  rx2: ((lane_id < 31)? ry2: rz2);
    sum7 += (lane_id < 24)? ry3: rz3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = ((friend_id2+1)&(warpSize-1));
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[5]*__shfl(threadInput0, friend_id0);
    ty0 = args[5]*__shfl(threadInput1, friend_id0);
    tx1 = args[5]*__shfl(threadInput1, friend_id1);
    ty1 = args[5]*__shfl(threadInput2, friend_id1);
    tx2 = args[5]*__shfl(threadInput2, friend_id2);
    ty2 = args[5]*__shfl(threadInput3, friend_id2);
    tz2 = args[5]*__shfl(threadInput4, friend_id2);
    ty3 = args[5]*__shfl(threadInput4, friend_id3);
    tz3 = args[5]*__shfl(threadInput5, friend_id3);
    rx0 = args[5]*__shfl(threadInput5, friend_id0);
    ry0 = args[5]*__shfl(threadInput6, friend_id0);
    rx1 = args[5]*__shfl(threadInput6, friend_id1);
    ry1 = args[5]*__shfl(threadInput7, friend_id1);
    rx2 = args[5]*__shfl(threadInput7, friend_id2);
    ry2 = args[5]*__shfl(threadInput8, friend_id2);
    rz2 = args[5]*__shfl(threadInput9, friend_id2);
    ry3 = args[5]*__shfl(threadInput9, friend_id3);
    rz3 = args[5]*__shfl(threadInput10, friend_id3);
    sum0 += (lane_id < 16)? tx0: ty0;
    sum1 += (lane_id < 10)? tx1: ty1;
    sum2 += (lane_id < 4)? tx2: ((lane_id < 30)? ty2: tz2);
    sum3 += (lane_id < 24)? ty3: tz3;
    sum4 += (lane_id < 16)? rx0: ry0;
    sum5 += (lane_id < 10)? rx1: ry1;
    sum6 += (lane_id < 4)?  rx2: ((lane_id < 30)? ry2: rz2);
    sum7 += (lane_id < 24)? ry3: rz3;

    friend_id0 = ((friend_id0+8)&(warpSize-1));
    friend_id1 = ((friend_id1+8)&(warpSize-1));
    friend_id2 = ((friend_id2+8)&(warpSize-1));
    friend_id3 = ((friend_id3+8)&(warpSize-1));
    tx0 = args[6]*__shfl(threadInput0, friend_id0);
    ty0 = args[6]*__shfl(threadInput1, friend_id0);
    tx1 = args[6]*__shfl(threadInput1, friend_id1);
    ty1 = args[6]*__shfl(threadInput2, friend_id1);
    tz1 = args[6]*__shfl(threadInput3, friend_id1);
    ty2 = args[6]*__shfl(threadInput3, friend_id2);
    tz2 = args[6]*__shfl(threadInput4, friend_id2);
    ty3 = args[6]*__shfl(threadInput4, friend_id3);
    tz3 = args[6]*__shfl(threadInput5, friend_id3);
    rx0 = args[6]*__shfl(threadInput5, friend_id0);
    ry0 = args[6]*__shfl(threadInput6, friend_id0);
    rx1 = args[6]*__shfl(threadInput6, friend_id1);
    ry1 = args[6]*__shfl(threadInput7, friend_id1);
    rz1 = args[6]*__shfl(threadInput8, friend_id1);
    ry2 = args[6]*__shfl(threadInput8, friend_id2);
    rz2 = args[6]*__shfl(threadInput9, friend_id2);
    ry3 = args[6]*__shfl(threadInput9, friend_id3);
    rz3 = args[6]*__shfl(threadInput10, friend_id3);
    sum0 += (lane_id < 10)? tx0: ty0;
    sum1 += (lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1);
    sum2 += (lane_id < 24)? ty2: tz2;
    sum3 += (lane_id < 16)? ty3: tz3;
    sum4 += (lane_id < 10)? rx0: ry0;
    sum5 += (lane_id < 4 )? rx1: ((lane_id < 30)? ry1: rz1);
    sum6 += (lane_id < 24)? ry2: rz2;
    sum7 += (lane_id < 16)? ry3: rz3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = ((friend_id2+1)&(warpSize-1));
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[7]*__shfl(threadInput0, friend_id0);
    ty0 = args[7]*__shfl(threadInput1, friend_id0);
    tx1 = args[7]*__shfl(threadInput1, friend_id1);
    ty1 = args[7]*__shfl(threadInput2, friend_id1);
    tz1 = args[7]*__shfl(threadInput3, friend_id1);
    ty2 = args[7]*__shfl(threadInput3, friend_id2);
    tz2 = args[7]*__shfl(threadInput4, friend_id2);
    ty3 = args[7]*__shfl(threadInput4, friend_id3);
    tz3 = args[7]*__shfl(threadInput5, friend_id3);
    rx0 = args[7]*__shfl(threadInput5, friend_id0);
    ry0 = args[7]*__shfl(threadInput6, friend_id0);
    rx1 = args[7]*__shfl(threadInput6, friend_id1);
    ry1 = args[7]*__shfl(threadInput7, friend_id1);
    rz1 = args[7]*__shfl(threadInput8, friend_id1);
    ry2 = args[7]*__shfl(threadInput8, friend_id2);
    rz2 = args[7]*__shfl(threadInput9, friend_id2);
    ry3 = args[7]*__shfl(threadInput9, friend_id3);
    rz3 = args[7]*__shfl(threadInput10, friend_id3);
    sum0 += (lane_id < 9 )? tx0: ty0;
    sum1 += (lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1);
    sum2 += (lane_id < 23)? ty2: tz2;
    sum3 += (lane_id < 16)? ty3: tz3;
    sum4 += (lane_id < 9 )? rx0: ry0;
    sum5 += (lane_id < 3 )? rx1: ((lane_id < 29)? ry1: rz1);
    sum6 += (lane_id < 23)? ry2: rz2;
    sum7 += (lane_id < 16)? ry3: rz3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = ((friend_id2+1)&(warpSize-1));
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[8]*__shfl(threadInput0, friend_id0);
    ty0 = args[8]*__shfl(threadInput1, friend_id0);
    tx1 = args[8]*__shfl(threadInput1, friend_id1);
    ty1 = args[8]*__shfl(threadInput2, friend_id1);
    tz1 = args[8]*__shfl(threadInput3, friend_id1);
    ty2 = args[8]*__shfl(threadInput3, friend_id2);
    tz2 = args[8]*__shfl(threadInput4, friend_id2);
    ty3 = args[8]*__shfl(threadInput4, friend_id3);
    tz3 = args[8]*__shfl(threadInput5, friend_id3);
    rx0 = args[8]*__shfl(threadInput5, friend_id0);
    ry0 = args[8]*__shfl(threadInput6, friend_id0);
    rx1 = args[8]*__shfl(threadInput6, friend_id1);
    ry1 = args[8]*__shfl(threadInput7, friend_id1);
    rz1 = args[8]*__shfl(threadInput8, friend_id1);
    ry2 = args[8]*__shfl(threadInput8, friend_id2);
    rz2 = args[8]*__shfl(threadInput9, friend_id2);
    ry3 = args[8]*__shfl(threadInput9, friend_id3);
    rz3 = args[8]*__shfl(threadInput10, friend_id3);
    sum0 += (lane_id < 8 )? tx0: ty0;
    sum1 += (lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1);
    sum2 += (lane_id < 22)? ty2: tz2;
    sum3 += (lane_id < 16)? ty3: tz3;
    sum4 += (lane_id < 8 )? rx0: ry0;
    sum5 += (lane_id < 2 )? rx1: ((lane_id < 28)? ry1: rz1);
    sum6 += (lane_id < 22)? ry2: rz2;
    sum7 += (lane_id < 16)? ry3: rz3;
    /*
    // */


    if(j < m + halo && i < n + halo)
        OUT_2D(j,i) = sum0; 
    if(j+4 < m + halo && i < n + halo)
        OUT_2D(j+4,i) = sum1; 
    if(j+8 < m + halo && i < n + halo)
        OUT_2D(j+8,i) = sum2; 
    if(j+12 < m + halo && i < n + halo)
        OUT_2D(j+12,i) = sum3; 
    if(j+16 < m + halo && i < n + halo)
        OUT_2D(j+16,i) = sum4; 
    if(j+20 < m + halo && i < n + halo)
        OUT_2D(j+20,i) = sum5; 
    if(j+24 < m + halo && i < n + halo)
        OUT_2D(j+24,i) = sum6; 
    if(j+28 < m + halo && i < n + halo)
        OUT_2D(j+28,i) = sum7; 
}

__global__ void Stencil_Cuda_Shfl16(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int halo) 
{
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int lane_id = tid % warpSize;
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<6) + (lane_id>>3)  + halo;

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3;
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<6) + (lane_id>>3))>>2;
    int new_i = (warp_id_x<<3) + lane_id%10;
    int new_j = (warp_id_y<<2) + lane_id/10;
    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5;
    DATA_TYPE threadInput6, threadInput7, threadInput8, threadInput9, threadInput10;
    DATA_TYPE threadInput11, threadInput12, threadInput13, threadInput14, threadInput15;
    DATA_TYPE threadInput16, threadInput17, threadInput18, threadInput19, threadInput20;
    threadInput0 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+2)%10;
    new_j = (warp_id_y<<2) + 3 + (lane_id+2)/10;
    threadInput1 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+4)%10;
    new_j = (warp_id_y<<2) + 6 + (lane_id+4)/10;
    threadInput2 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+6)%10;
    new_j = (warp_id_y<<2) + 9 + (lane_id+6)/10;
    threadInput3 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+8)%10;
    new_j = (warp_id_y<<2) + 12 + (lane_id+8)/10;
    threadInput4 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+0)%10;
    new_j = (warp_id_y<<2) + 16 + (lane_id+0)/10;
    threadInput5 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+2)%10;
    new_j = (warp_id_y<<2) + 19 + (lane_id+2)/10;
    threadInput6 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+4)%10;
    new_j = (warp_id_y<<2) + 22 + (lane_id+4)/10;
    threadInput7 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+6)%10;
    new_j = (warp_id_y<<2) + 25 + (lane_id+6)/10;
    threadInput8 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+8)%10;
    new_j = (warp_id_y<<2) + 28 + (lane_id+8)/10;
    threadInput9 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+0)%10;
    new_j = (warp_id_y<<2) + 32 + (lane_id+0)/10;
    threadInput10 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+2)%10;
    new_j = (warp_id_y<<2) + 35 + (lane_id+2)/10;
    threadInput11 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+4)%10;
    new_j = (warp_id_y<<2) + 38 + (lane_id+4)/10;
    threadInput12 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+6)%10;
    new_j = (warp_id_y<<2) + 41 + (lane_id+6)/10;
    threadInput13 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+8)%10;
    new_j = (warp_id_y<<2) + 44 + (lane_id+8)/10;
    threadInput14 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+0)%10;
    new_j = (warp_id_y<<2) + 48 + (lane_id+0)/10;
    threadInput15 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+2)%10;
    new_j = (warp_id_y<<2) + 51 + (lane_id+2)/10;
    threadInput16 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+4)%10;
    new_j = (warp_id_y<<2) + 54 + (lane_id+4)/10;
    threadInput17 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+6)%10;
    new_j = (warp_id_y<<2) + 57 + (lane_id+6)/10;
    threadInput18 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+8)%10;
    new_j = (warp_id_y<<2) + 60 + (lane_id+8)/10;
    threadInput19 = IN_2D(new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+0)%10;
    new_j = (warp_id_y<<2) + 64 + (lane_id+0)/10;
    if(new_i < n+2*halo && new_j < m+2*halo)
        threadInput20 = IN_2D(new_j, new_i);

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
    int friend_id0, friend_id1, friend_id2, friend_id3;
    // int friend_id4, friend_id5, friend_id6, friend_id7;
    friend_id0 = (lane_id+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+8+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    DATA_TYPE tx0, ty0, tx1, ty1, tz1, tx2, ty2, tz2, tx3, ty3, tz3;
    DATA_TYPE rx0, ry0, rx1, ry1, rz1, rx2, ry2, rz2, rx3, ry3, rz3;
    DATA_TYPE mx0, my0, mx1, my1, mz1, mx2, my2, mz2, mx3, my3, mz3;
    DATA_TYPE nx0, ny0, nx1, ny1, nz1, nx2, ny2, nz2, nx3, ny3, nz3;
    tx0 = args[0]*__shfl(threadInput0, friend_id0);
    ty0 = args[0]*__shfl(threadInput1, friend_id0);
    tx1 = args[0]*__shfl(threadInput1, friend_id1);
    ty1 = args[0]*__shfl(threadInput2, friend_id1);
    tx2 = args[0]*__shfl(threadInput2, friend_id2);
    ty2 = args[0]*__shfl(threadInput3, friend_id2);
    tx3 = args[0]*__shfl(threadInput3, friend_id3);
    ty3 = args[0]*__shfl(threadInput4, friend_id3);
    rx0 = args[0]*__shfl(threadInput5, friend_id0);
    ry0 = args[0]*__shfl(threadInput6, friend_id0);
    rx1 = args[0]*__shfl(threadInput6, friend_id1);
    ry1 = args[0]*__shfl(threadInput7, friend_id1);
    rx2 = args[0]*__shfl(threadInput7, friend_id2);
    ry2 = args[0]*__shfl(threadInput8, friend_id2);
    rx3 = args[0]*__shfl(threadInput8, friend_id3);
    ry3 = args[0]*__shfl(threadInput9, friend_id3);
    mx0 = args[0]*__shfl(threadInput10, friend_id0);
    my0 = args[0]*__shfl(threadInput11, friend_id0);
    mx1 = args[0]*__shfl(threadInput11, friend_id1);
    my1 = args[0]*__shfl(threadInput12, friend_id1);
    mx2 = args[0]*__shfl(threadInput12, friend_id2);
    my2 = args[0]*__shfl(threadInput13, friend_id2);
    mx3 = args[0]*__shfl(threadInput13, friend_id3);
    my3 = args[0]*__shfl(threadInput14, friend_id3);
    nx0 = args[0]*__shfl(threadInput15, friend_id0);
    ny0 = args[0]*__shfl(threadInput16, friend_id0);
    nx1 = args[0]*__shfl(threadInput16, friend_id1);
    ny1 = args[0]*__shfl(threadInput17, friend_id1);
    nx2 = args[0]*__shfl(threadInput17, friend_id2);
    ny2 = args[0]*__shfl(threadInput18, friend_id2);
    nx3 = args[0]*__shfl(threadInput18, friend_id3);
    ny3 = args[0]*__shfl(threadInput19, friend_id3);
    sum0 += (lane_id < 26)? tx0: ty0;
    sum1 += (lane_id < 20)? tx1: ty1;
    sum2 += (lane_id < 14)? tx2: ty2;
    sum3 += (lane_id < 8 )? tx3: ty3;
    sum4 += (lane_id < 26)? rx0: ry0;
    sum5 += (lane_id < 20)? rx1: ry1;
    sum6 += (lane_id < 14)? rx2: ry2;
    sum7 += (lane_id < 8 )? rx3: ry3;
    sum8  += (lane_id < 26)? mx0: my0;
    sum9  += (lane_id < 20)? mx1: my1;
    sum10 += (lane_id < 14)? mx2: my2;
    sum11 += (lane_id < 8 )? mx3: my3;
    sum12 += (lane_id < 26)? nx0: ny0;
    sum13 += (lane_id < 20)? nx1: ny1;
    sum14 += (lane_id < 14)? nx2: ny2;
    sum15 += (lane_id < 8 )? nx3: ny3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = ((friend_id2+1)&(warpSize-1));
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[1]*__shfl(threadInput0, friend_id0);
    ty0 = args[1]*__shfl(threadInput1, friend_id0);
    tx1 = args[1]*__shfl(threadInput1, friend_id1);
    ty1 = args[1]*__shfl(threadInput2, friend_id1);
    tx2 = args[1]*__shfl(threadInput2, friend_id2);
    ty2 = args[1]*__shfl(threadInput3, friend_id2);
    tx3 = args[1]*__shfl(threadInput3, friend_id3);
    ty3 = args[1]*__shfl(threadInput4, friend_id3);
    rx0 = args[1]*__shfl(threadInput5, friend_id0);
    ry0 = args[1]*__shfl(threadInput6, friend_id0);
    rx1 = args[1]*__shfl(threadInput6, friend_id1);
    ry1 = args[1]*__shfl(threadInput7, friend_id1);
    rx2 = args[1]*__shfl(threadInput7, friend_id2);
    ry2 = args[1]*__shfl(threadInput8, friend_id2);
    rx3 = args[1]*__shfl(threadInput8, friend_id3);
    ry3 = args[1]*__shfl(threadInput9, friend_id3);
    mx0 = args[1]*__shfl(threadInput10, friend_id0);
    my0 = args[1]*__shfl(threadInput11, friend_id0);
    mx1 = args[1]*__shfl(threadInput11, friend_id1);
    my1 = args[1]*__shfl(threadInput12, friend_id1);
    mx2 = args[1]*__shfl(threadInput12, friend_id2);
    my2 = args[1]*__shfl(threadInput13, friend_id2);
    mx3 = args[1]*__shfl(threadInput13, friend_id3);
    my3 = args[1]*__shfl(threadInput14, friend_id3);
    nx0 = args[1]*__shfl(threadInput15, friend_id0);
    ny0 = args[1]*__shfl(threadInput16, friend_id0);
    nx1 = args[1]*__shfl(threadInput16, friend_id1);
    ny1 = args[1]*__shfl(threadInput17, friend_id1);
    nx2 = args[1]*__shfl(threadInput17, friend_id2);
    ny2 = args[1]*__shfl(threadInput18, friend_id2);
    nx3 = args[1]*__shfl(threadInput18, friend_id3);
    ny3 = args[1]*__shfl(threadInput19, friend_id3);
    sum0 += (lane_id < 25)? tx0: ty0;
    sum1 += (lane_id < 19)? tx1: ty1;
    sum2 += (lane_id < 13)? tx2: ty2;
    sum3 += (lane_id < 7 )? tx3: ty3;
    sum4 += (lane_id < 25)? rx0: ry0;
    sum5 += (lane_id < 19)? rx1: ry1;
    sum6 += (lane_id < 13)? rx2: ry2;
    sum7 += (lane_id < 7 )? rx3: ry3;
    sum8  += (lane_id < 25)? mx0: my0;
    sum9  += (lane_id < 19)? mx1: my1;
    sum10 += (lane_id < 13)? mx2: my2;
    sum11 += (lane_id < 7 )? mx3: my3;
    sum12 += (lane_id < 25)? nx0: ny0;
    sum13 += (lane_id < 19)? nx1: ny1;
    sum14 += (lane_id < 13)? nx2: ny2;
    sum15 += (lane_id < 7 )? nx3: ny3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = ((friend_id2+1)&(warpSize-1));
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[2]*__shfl(threadInput0, friend_id0);
    ty0 = args[2]*__shfl(threadInput1, friend_id0);
    tx1 = args[2]*__shfl(threadInput1, friend_id1);
    ty1 = args[2]*__shfl(threadInput2, friend_id1);
    tx2 = args[2]*__shfl(threadInput2, friend_id2);
    ty2 = args[2]*__shfl(threadInput3, friend_id2);
    tx3 = args[2]*__shfl(threadInput3, friend_id3);
    ty3 = args[2]*__shfl(threadInput4, friend_id3);
    rx0 = args[2]*__shfl(threadInput5, friend_id0);
    ry0 = args[2]*__shfl(threadInput6, friend_id0);
    rx1 = args[2]*__shfl(threadInput6, friend_id1);
    ry1 = args[2]*__shfl(threadInput7, friend_id1);
    rx2 = args[2]*__shfl(threadInput7, friend_id2);
    ry2 = args[2]*__shfl(threadInput8, friend_id2);
    rx3 = args[2]*__shfl(threadInput8, friend_id3);
    ry3 = args[2]*__shfl(threadInput9, friend_id3);
    mx0 = args[2]*__shfl(threadInput10, friend_id0);
    my0 = args[2]*__shfl(threadInput11, friend_id0);
    mx1 = args[2]*__shfl(threadInput11, friend_id1);
    my1 = args[2]*__shfl(threadInput12, friend_id1);
    mx2 = args[2]*__shfl(threadInput12, friend_id2);
    my2 = args[2]*__shfl(threadInput13, friend_id2);
    mx3 = args[2]*__shfl(threadInput13, friend_id3);
    my3 = args[2]*__shfl(threadInput14, friend_id3);
    nx0 = args[2]*__shfl(threadInput15, friend_id0);
    ny0 = args[2]*__shfl(threadInput16, friend_id0);
    nx1 = args[2]*__shfl(threadInput16, friend_id1);
    ny1 = args[2]*__shfl(threadInput17, friend_id1);
    nx2 = args[2]*__shfl(threadInput17, friend_id2);
    ny2 = args[2]*__shfl(threadInput18, friend_id2);
    nx3 = args[2]*__shfl(threadInput18, friend_id3);
    ny3 = args[2]*__shfl(threadInput19, friend_id3);
    sum0 += (lane_id < 24)? tx0: ty0;
    sum1 += (lane_id < 18)? tx1: ty1;
    sum2 += (lane_id < 12)? tx2: ty2;
    sum3 += (lane_id < 6 )? tx3: ty3;
    sum4 += (lane_id < 24)? rx0: ry0;
    sum5 += (lane_id < 18)? rx1: ry1;
    sum6 += (lane_id < 12)? rx2: ry2;
    sum7 += (lane_id < 6 )? rx3: ry3;
    sum8  += (lane_id < 24)? mx0: my0;
    sum9  += (lane_id < 18)? mx1: my1;
    sum10 += (lane_id < 12)? mx2: my2;
    sum11 += (lane_id < 6 )? mx3: my3;
    sum12 += (lane_id < 24)? nx0: ny0;
    sum13 += (lane_id < 18)? nx1: ny1;
    sum14 += (lane_id < 12)? nx2: ny2;
    sum15 += (lane_id < 6 )? nx3: ny3;

    friend_id0 = ((friend_id0+8)&(warpSize-1));
    friend_id1 = ((friend_id1+8)&(warpSize-1));
    friend_id2 = ((friend_id2+8)&(warpSize-1));
    friend_id3 = ((friend_id3+8)&(warpSize-1));
    tx0 = args[3]*__shfl(threadInput0, friend_id0);
    ty0 = args[3]*__shfl(threadInput1, friend_id0);
    tx1 = args[3]*__shfl(threadInput1, friend_id1);
    ty1 = args[3]*__shfl(threadInput2, friend_id1);
    tx2 = args[3]*__shfl(threadInput2, friend_id2);
    ty2 = args[3]*__shfl(threadInput3, friend_id2);
    ty3 = args[3]*__shfl(threadInput4, friend_id3);
    tz3 = args[3]*__shfl(threadInput5, friend_id3);
    rx0 = args[3]*__shfl(threadInput5, friend_id0);
    ry0 = args[3]*__shfl(threadInput6, friend_id0);
    rx1 = args[3]*__shfl(threadInput6, friend_id1);
    ry1 = args[3]*__shfl(threadInput7, friend_id1);
    rx2 = args[3]*__shfl(threadInput7, friend_id2);
    ry2 = args[3]*__shfl(threadInput8, friend_id2);
    ry3 = args[3]*__shfl(threadInput9, friend_id3);
    rz3 = args[3]*__shfl(threadInput10, friend_id3);
    mx0 = args[3]*__shfl(threadInput10, friend_id0);
    my0 = args[3]*__shfl(threadInput11, friend_id0);
    mx1 = args[3]*__shfl(threadInput11, friend_id1);
    my1 = args[3]*__shfl(threadInput12, friend_id1);
    mx2 = args[3]*__shfl(threadInput12, friend_id2);
    my2 = args[3]*__shfl(threadInput13, friend_id2);
    my3 = args[3]*__shfl(threadInput14, friend_id3);
    mz3 = args[3]*__shfl(threadInput15, friend_id3);
    nx0 = args[3]*__shfl(threadInput15, friend_id0);
    ny0 = args[3]*__shfl(threadInput16, friend_id0);
    nx1 = args[3]*__shfl(threadInput16, friend_id1);
    ny1 = args[3]*__shfl(threadInput17, friend_id1);
    nx2 = args[3]*__shfl(threadInput17, friend_id2);
    ny2 = args[3]*__shfl(threadInput18, friend_id2);
    ny3 = args[3]*__shfl(threadInput19, friend_id3);
    nz3 = args[3]*__shfl(threadInput20, friend_id3);
    sum0 += (lane_id < 18)? tx0: ty0;
    sum1 += (lane_id < 12)? tx1: ty1;
    sum2 += (lane_id < 6 )? tx2: ty2;
    sum3 += (lane_id < 24)? ty3: tz3;
    sum4 += (lane_id < 18)? rx0: ry0;
    sum5 += (lane_id < 12)? rx1: ry1;
    sum6 += (lane_id < 6 )? rx2: ry2;
    sum7 += (lane_id < 24)? ry3: rz3;
    sum8  += (lane_id < 18)? mx0: my0;
    sum9  += (lane_id < 12)? mx1: my1;
    sum10 += (lane_id < 6 )? mx2: my2;
    sum11 += (lane_id < 24)? my3: mz3;
    sum12 += (lane_id < 18)? nx0: ny0;
    sum13 += (lane_id < 12)? nx1: ny1;
    sum14 += (lane_id < 6 )? nx2: ny2;
    sum15 += (lane_id < 24)? ny3: nz3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = ((friend_id2+1)&(warpSize-1));
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[4]*__shfl(threadInput0, friend_id0);
    ty0 = args[4]*__shfl(threadInput1, friend_id0);
    tx1 = args[4]*__shfl(threadInput1, friend_id1);
    ty1 = args[4]*__shfl(threadInput2, friend_id1);
    tx2 = args[4]*__shfl(threadInput2, friend_id2);
    ty2 = args[4]*__shfl(threadInput3, friend_id2);
    tz2 = args[4]*__shfl(threadInput4, friend_id2);
    ty3 = args[4]*__shfl(threadInput4, friend_id3);
    tz3 = args[4]*__shfl(threadInput5, friend_id3);
    rx0 = args[4]*__shfl(threadInput5, friend_id0);
    ry0 = args[4]*__shfl(threadInput6, friend_id0);
    rx1 = args[4]*__shfl(threadInput6, friend_id1);
    ry1 = args[4]*__shfl(threadInput7, friend_id1);
    rx2 = args[4]*__shfl(threadInput7, friend_id2);
    ry2 = args[4]*__shfl(threadInput8, friend_id2);
    rz2 = args[4]*__shfl(threadInput9, friend_id2);
    ry3 = args[4]*__shfl(threadInput9, friend_id3);
    rz3 = args[4]*__shfl(threadInput10, friend_id3);
    mx0 = args[4]*__shfl(threadInput10, friend_id0);
    my0 = args[4]*__shfl(threadInput11, friend_id0);
    mx1 = args[4]*__shfl(threadInput11, friend_id1);
    my1 = args[4]*__shfl(threadInput12, friend_id1);
    mx2 = args[4]*__shfl(threadInput12, friend_id2);
    my2 = args[4]*__shfl(threadInput13, friend_id2);
    mz2 = args[4]*__shfl(threadInput14, friend_id2);
    my3 = args[4]*__shfl(threadInput14, friend_id3);
    mz3 = args[4]*__shfl(threadInput15, friend_id3);
    nx0 = args[4]*__shfl(threadInput15, friend_id0);
    ny0 = args[4]*__shfl(threadInput16, friend_id0);
    nx1 = args[4]*__shfl(threadInput16, friend_id1);
    ny1 = args[4]*__shfl(threadInput17, friend_id1);
    nx2 = args[4]*__shfl(threadInput17, friend_id2);
    ny2 = args[4]*__shfl(threadInput18, friend_id2);
    nz2 = args[4]*__shfl(threadInput19, friend_id2);
    ny3 = args[4]*__shfl(threadInput19, friend_id3);
    nz3 = args[4]*__shfl(threadInput20, friend_id3);
    sum0 += (lane_id < 17)? tx0: ty0;
    sum1 += (lane_id < 11)? tx1: ty1;
    sum2 += (lane_id < 5)? tx2: ((lane_id < 31)? ty2: tz2);
    sum3 += (lane_id < 24)? ty3: tz3;
    sum4 += (lane_id < 17)? rx0: ry0;
    sum5 += (lane_id < 11)? rx1: ry1;
    sum6 += (lane_id < 5)?  rx2: ((lane_id < 31)? ry2: rz2);
    sum7 += (lane_id < 24)? ry3: rz3;
    sum8  += (lane_id < 17)? mx0: my0;
    sum9  += (lane_id < 11)? mx1: my1;
    sum10 += (lane_id < 5 )? mx2: ((lane_id < 31)? my2: mz2);
    sum11 += (lane_id < 24)? my3: mz3;
    sum12 += (lane_id < 17)? nx0: ny0;
    sum13 += (lane_id < 11)? nx1: ny1;
    sum14 += (lane_id < 5 )? nx2: ((lane_id < 31)? ny2: nz2);
    sum15 += (lane_id < 24)? ny3: nz3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = ((friend_id2+1)&(warpSize-1));
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[5]*__shfl(threadInput0, friend_id0);
    ty0 = args[5]*__shfl(threadInput1, friend_id0);
    tx1 = args[5]*__shfl(threadInput1, friend_id1);
    ty1 = args[5]*__shfl(threadInput2, friend_id1);
    tx2 = args[5]*__shfl(threadInput2, friend_id2);
    ty2 = args[5]*__shfl(threadInput3, friend_id2);
    tz2 = args[5]*__shfl(threadInput4, friend_id2);
    ty3 = args[5]*__shfl(threadInput4, friend_id3);
    tz3 = args[5]*__shfl(threadInput5, friend_id3);
    rx0 = args[5]*__shfl(threadInput5, friend_id0);
    ry0 = args[5]*__shfl(threadInput6, friend_id0);
    rx1 = args[5]*__shfl(threadInput6, friend_id1);
    ry1 = args[5]*__shfl(threadInput7, friend_id1);
    rx2 = args[5]*__shfl(threadInput7, friend_id2);
    ry2 = args[5]*__shfl(threadInput8, friend_id2);
    rz2 = args[5]*__shfl(threadInput9, friend_id2);
    ry3 = args[5]*__shfl(threadInput9, friend_id3);
    rz3 = args[5]*__shfl(threadInput10, friend_id3);
    mx0 = args[5]*__shfl(threadInput10, friend_id0);
    my0 = args[5]*__shfl(threadInput11, friend_id0);
    mx1 = args[5]*__shfl(threadInput11, friend_id1);
    my1 = args[5]*__shfl(threadInput12, friend_id1);
    mx2 = args[5]*__shfl(threadInput12, friend_id2);
    my2 = args[5]*__shfl(threadInput13, friend_id2);
    mz2 = args[5]*__shfl(threadInput14, friend_id2);
    my3 = args[5]*__shfl(threadInput14, friend_id3);
    mz3 = args[5]*__shfl(threadInput15, friend_id3);
    nx0 = args[5]*__shfl(threadInput15, friend_id0);
    ny0 = args[5]*__shfl(threadInput16, friend_id0);
    nx1 = args[5]*__shfl(threadInput16, friend_id1);
    ny1 = args[5]*__shfl(threadInput17, friend_id1);
    nx2 = args[5]*__shfl(threadInput17, friend_id2);
    ny2 = args[5]*__shfl(threadInput18, friend_id2);
    nz2 = args[5]*__shfl(threadInput19, friend_id2);
    ny3 = args[5]*__shfl(threadInput19, friend_id3);
    nz3 = args[5]*__shfl(threadInput20, friend_id3);
    sum0 += (lane_id < 16)? tx0: ty0;
    sum1 += (lane_id < 10)? tx1: ty1;
    sum2 += (lane_id < 4)? tx2: ((lane_id < 30)? ty2: tz2);
    sum3 += (lane_id < 24)? ty3: tz3;
    sum4 += (lane_id < 16)? rx0: ry0;
    sum5 += (lane_id < 10)? rx1: ry1;
    sum6 += (lane_id < 4)?  rx2: ((lane_id < 30)? ry2: rz2);
    sum7 += (lane_id < 24)? ry3: rz3;
    sum8  += (lane_id < 16)? mx0: my0;
    sum9  += (lane_id < 10)? mx1: my1;
    sum10 += (lane_id < 4 )? mx2: ((lane_id < 30)? my2: mz2);
    sum11 += (lane_id < 24)? my3: mz3;
    sum12 += (lane_id < 16)? nx0: ny0;
    sum13 += (lane_id < 10)? nx1: ny1;
    sum14 += (lane_id < 4 )? nx2: ((lane_id < 30)? ny2: nz2);
    sum15 += (lane_id < 24)? ny3: nz3;

    friend_id0 = ((friend_id0+8)&(warpSize-1));
    friend_id1 = ((friend_id1+8)&(warpSize-1));
    friend_id2 = ((friend_id2+8)&(warpSize-1));
    friend_id3 = ((friend_id3+8)&(warpSize-1));
    tx0 = args[6]*__shfl(threadInput0, friend_id0);
    ty0 = args[6]*__shfl(threadInput1, friend_id0);
    tx1 = args[6]*__shfl(threadInput1, friend_id1);
    ty1 = args[6]*__shfl(threadInput2, friend_id1);
    tz1 = args[6]*__shfl(threadInput3, friend_id1);
    ty2 = args[6]*__shfl(threadInput3, friend_id2);
    tz2 = args[6]*__shfl(threadInput4, friend_id2);
    ty3 = args[6]*__shfl(threadInput4, friend_id3);
    tz3 = args[6]*__shfl(threadInput5, friend_id3);
    rx0 = args[6]*__shfl(threadInput5, friend_id0);
    ry0 = args[6]*__shfl(threadInput6, friend_id0);
    rx1 = args[6]*__shfl(threadInput6, friend_id1);
    ry1 = args[6]*__shfl(threadInput7, friend_id1);
    rz1 = args[6]*__shfl(threadInput8, friend_id1);
    ry2 = args[6]*__shfl(threadInput8, friend_id2);
    rz2 = args[6]*__shfl(threadInput9, friend_id2);
    ry3 = args[6]*__shfl(threadInput9, friend_id3);
    rz3 = args[6]*__shfl(threadInput10, friend_id3);
    mx0 = args[6]*__shfl(threadInput10, friend_id0);
    my0 = args[6]*__shfl(threadInput11, friend_id0);
    mx1 = args[6]*__shfl(threadInput11, friend_id1);
    my1 = args[6]*__shfl(threadInput12, friend_id1);
    mz1 = args[6]*__shfl(threadInput13, friend_id1);
    my2 = args[6]*__shfl(threadInput13, friend_id2);
    mz2 = args[6]*__shfl(threadInput14, friend_id2);
    my3 = args[6]*__shfl(threadInput14, friend_id3);
    mz3 = args[6]*__shfl(threadInput15, friend_id3);
    nx0 = args[6]*__shfl(threadInput15, friend_id0);
    ny0 = args[6]*__shfl(threadInput16, friend_id0);
    nx1 = args[6]*__shfl(threadInput16, friend_id1);
    ny1 = args[6]*__shfl(threadInput17, friend_id1);
    nz1 = args[6]*__shfl(threadInput18, friend_id1);
    ny2 = args[6]*__shfl(threadInput18, friend_id2);
    nz2 = args[6]*__shfl(threadInput19, friend_id2);
    ny3 = args[6]*__shfl(threadInput19, friend_id3);
    nz3 = args[6]*__shfl(threadInput20, friend_id3);
    sum0 += (lane_id < 10)? tx0: ty0;
    sum1 += (lane_id < 4 )? tx1: ((lane_id < 30)? ty1: tz1);
    sum2 += (lane_id < 24)? ty2: tz2;
    sum3 += (lane_id < 16)? ty3: tz3;
    sum4 += (lane_id < 10)? rx0: ry0;
    sum5 += (lane_id < 4 )? rx1: ((lane_id < 30)? ry1: rz1);
    sum6 += (lane_id < 24)? ry2: rz2;
    sum7 += (lane_id < 16)? ry3: rz3;
    sum8  += (lane_id < 10)? mx0: my0;
    sum9  += (lane_id < 4 )? mx1: ((lane_id < 30)? my1: mz1);
    sum10 += (lane_id < 24)? my2: mz2;
    sum11 += (lane_id < 16)? my3: mz3;
    sum12 += (lane_id < 10)? nx0: ny0;
    sum13 += (lane_id < 4 )? nx1: ((lane_id < 30)? ny1: nz1);
    sum14 += (lane_id < 24)? ny2: nz2;
    sum15 += (lane_id < 16)? ny3: nz3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = ((friend_id2+1)&(warpSize-1));
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[7]*__shfl(threadInput0, friend_id0);
    ty0 = args[7]*__shfl(threadInput1, friend_id0);
    tx1 = args[7]*__shfl(threadInput1, friend_id1);
    ty1 = args[7]*__shfl(threadInput2, friend_id1);
    tz1 = args[7]*__shfl(threadInput3, friend_id1);
    ty2 = args[7]*__shfl(threadInput3, friend_id2);
    tz2 = args[7]*__shfl(threadInput4, friend_id2);
    ty3 = args[7]*__shfl(threadInput4, friend_id3);
    tz3 = args[7]*__shfl(threadInput5, friend_id3);
    rx0 = args[7]*__shfl(threadInput5, friend_id0);
    ry0 = args[7]*__shfl(threadInput6, friend_id0);
    rx1 = args[7]*__shfl(threadInput6, friend_id1);
    ry1 = args[7]*__shfl(threadInput7, friend_id1);
    rz1 = args[7]*__shfl(threadInput8, friend_id1);
    ry2 = args[7]*__shfl(threadInput8, friend_id2);
    rz2 = args[7]*__shfl(threadInput9, friend_id2);
    ry3 = args[7]*__shfl(threadInput9, friend_id3);
    rz3 = args[7]*__shfl(threadInput10, friend_id3);
    mx0 = args[7]*__shfl(threadInput10, friend_id0);
    my0 = args[7]*__shfl(threadInput11, friend_id0);
    mx1 = args[7]*__shfl(threadInput11, friend_id1);
    my1 = args[7]*__shfl(threadInput12, friend_id1);
    mz1 = args[7]*__shfl(threadInput13, friend_id1);
    my2 = args[7]*__shfl(threadInput13, friend_id2);
    mz2 = args[7]*__shfl(threadInput14, friend_id2);
    my3 = args[7]*__shfl(threadInput14, friend_id3);
    mz3 = args[7]*__shfl(threadInput15, friend_id3);
    nx0 = args[7]*__shfl(threadInput15, friend_id0);
    ny0 = args[7]*__shfl(threadInput16, friend_id0);
    nx1 = args[7]*__shfl(threadInput16, friend_id1);
    ny1 = args[7]*__shfl(threadInput17, friend_id1);
    nz1 = args[7]*__shfl(threadInput18, friend_id1);
    ny2 = args[7]*__shfl(threadInput18, friend_id2);
    nz2 = args[7]*__shfl(threadInput19, friend_id2);
    ny3 = args[7]*__shfl(threadInput19, friend_id3);
    nz3 = args[7]*__shfl(threadInput20, friend_id3);
    sum0 += (lane_id < 9 )? tx0: ty0;
    sum1 += (lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1);
    sum2 += (lane_id < 23)? ty2: tz2;
    sum3 += (lane_id < 16)? ty3: tz3;
    sum4 += (lane_id < 9 )? rx0: ry0;
    sum5 += (lane_id < 3 )? rx1: ((lane_id < 29)? ry1: rz1);
    sum6 += (lane_id < 23)? ry2: rz2;
    sum7 += (lane_id < 16)? ry3: rz3;
    sum8  += (lane_id < 9 )? mx0: my0;
    sum9  += (lane_id < 3 )? mx1: ((lane_id < 29)? my1: mz1);
    sum10 += (lane_id < 23)? my2: mz2;
    sum11 += (lane_id < 16)? my3: mz3;
    sum12 += (lane_id < 9 )? nx0: ny0;
    sum13 += (lane_id < 3 )? nx1: ((lane_id < 29)? ny1: nz1);
    sum14 += (lane_id < 23)? ny2: nz2;
    sum15 += (lane_id < 16)? ny3: nz3;

    friend_id0 = ((friend_id0+1)&(warpSize-1));
    friend_id1 = ((friend_id1+1)&(warpSize-1));
    friend_id2 = ((friend_id2+1)&(warpSize-1));
    friend_id3 = ((friend_id3+1)&(warpSize-1));
    tx0 = args[8]*__shfl(threadInput0, friend_id0);
    ty0 = args[8]*__shfl(threadInput1, friend_id0);
    tx1 = args[8]*__shfl(threadInput1, friend_id1);
    ty1 = args[8]*__shfl(threadInput2, friend_id1);
    tz1 = args[8]*__shfl(threadInput3, friend_id1);
    ty2 = args[8]*__shfl(threadInput3, friend_id2);
    tz2 = args[8]*__shfl(threadInput4, friend_id2);
    ty3 = args[8]*__shfl(threadInput4, friend_id3);
    tz3 = args[8]*__shfl(threadInput5, friend_id3);
    rx0 = args[8]*__shfl(threadInput5, friend_id0);
    ry0 = args[8]*__shfl(threadInput6, friend_id0);
    rx1 = args[8]*__shfl(threadInput6, friend_id1);
    ry1 = args[8]*__shfl(threadInput7, friend_id1);
    rz1 = args[8]*__shfl(threadInput8, friend_id1);
    ry2 = args[8]*__shfl(threadInput8, friend_id2);
    rz2 = args[8]*__shfl(threadInput9, friend_id2);
    ry3 = args[8]*__shfl(threadInput9, friend_id3);
    rz3 = args[8]*__shfl(threadInput10, friend_id3);
    mx0 = args[8]*__shfl(threadInput10, friend_id0);
    my0 = args[8]*__shfl(threadInput11, friend_id0);
    mx1 = args[8]*__shfl(threadInput11, friend_id1);
    my1 = args[8]*__shfl(threadInput12, friend_id1);
    mz1 = args[8]*__shfl(threadInput13, friend_id1);
    my2 = args[8]*__shfl(threadInput13, friend_id2);
    mz2 = args[8]*__shfl(threadInput14, friend_id2);
    my3 = args[8]*__shfl(threadInput14, friend_id3);
    mz3 = args[8]*__shfl(threadInput15, friend_id3);
    nx0 = args[8]*__shfl(threadInput15, friend_id0);
    ny0 = args[8]*__shfl(threadInput16, friend_id0);
    nx1 = args[8]*__shfl(threadInput16, friend_id1);
    ny1 = args[8]*__shfl(threadInput17, friend_id1);
    nz1 = args[8]*__shfl(threadInput18, friend_id1);
    ny2 = args[8]*__shfl(threadInput18, friend_id2);
    nz2 = args[8]*__shfl(threadInput19, friend_id2);
    ny3 = args[8]*__shfl(threadInput19, friend_id3);
    nz3 = args[8]*__shfl(threadInput20, friend_id3);
    sum0 += (lane_id < 8 )? tx0: ty0;
    sum1 += (lane_id < 2 )? tx1: ((lane_id < 28)? ty1: tz1);
    sum2 += (lane_id < 22)? ty2: tz2;
    sum3 += (lane_id < 16)? ty3: tz3;
    sum4 += (lane_id < 8 )? rx0: ry0;
    sum5 += (lane_id < 2 )? rx1: ((lane_id < 28)? ry1: rz1);
    sum6 += (lane_id < 22)? ry2: rz2;
    sum7 += (lane_id < 16)? ry3: rz3;
    sum8  += (lane_id < 8 )? mx0: my0;
    sum9  += (lane_id < 2 )? mx1: ((lane_id < 28)? my1: mz1);
    sum10 += (lane_id < 22)? my2: mz2;
    sum11 += (lane_id < 16)? my3: mz3;
    sum12 += (lane_id < 8 )? nx0: ny0;
    sum13 += (lane_id < 2 )? nx1: ((lane_id < 28)? ny1: nz1);
    sum14 += (lane_id < 22)? ny2: nz2;
    sum15 += (lane_id < 16)? ny3: nz3;
    /*
    // */


    if(j < m + halo && i < n + halo)
        OUT_2D(j,i) = sum0; 
    if(j+4 < m + halo && i < n + halo)
        OUT_2D(j+4,i) = sum1; 
    if(j+8 < m + halo && i < n + halo)
        OUT_2D(j+8,i) = sum2; 
    if(j+12 < m + halo && i < n + halo)
        OUT_2D(j+12,i) = sum3; 
    if(j+16 < m + halo && i < n + halo)
        OUT_2D(j+16,i) = sum4; 
    if(j+20 < m + halo && i < n + halo)
        OUT_2D(j+20,i) = sum5; 
    if(j+24 < m + halo && i < n + halo)
        OUT_2D(j+24,i) = sum6; 
    if(j+28 < m + halo && i < n + halo)
        OUT_2D(j+28,i) = sum7; 
    if(j+32 < m + halo && i < n + halo)
        OUT_2D(j+32,i) = sum8; 
    if(j+36 < m + halo && i < n + halo)
        OUT_2D(j+36,i) = sum9; 
    if(j+40 < m + halo && i < n + halo)
        OUT_2D(j+40,i) = sum10; 
    if(j+44 < m + halo && i < n + halo)
        OUT_2D(j+44,i) = sum11; 
    if(j+48 < m + halo && i < n + halo)
        OUT_2D(j+48,i) = sum12; 
    if(j+52 < m + halo && i < n + halo)
        OUT_2D(j+52,i) = sum13; 
    if(j+56 < m + halo && i < n + halo)
        OUT_2D(j+56,i) = sum14; 
    if(j+60 < m + halo && i < n + halo)
        OUT_2D(j+60,i) = sum15; 
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
            std::cout << "difference: " << fabs(test[i]-ref[i])-precision << std::endl;
            std::cout << "wrong at " << i << " test:" << test[i] << " (ref: " << ref[i] << ")";
            std::cout << std::endl;
            flag = false;
            break;
        }
    }
    return flag;
}

__global__ void Stencil_Cuda_Sm(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int m, int n, int halo) 
{
    extern __shared__ DATA_TYPE local[];
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo; 
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo; 
    int li = threadIdx.x + halo;
    int lj = threadIdx.y + halo;

    LOC_2D(lj,li) = IN_2D(j,i);

    if(li == halo) LOC_2D(lj,li-1) = IN_2D(j,i-1);

    if(li == 16)   LOC_2D(lj,li+1) = IN_2D(j,i+1);
            
    if(lj == halo) LOC_2D(lj-1,li) = IN_2D(j-1,i);
                                                
    if(lj == 16)   LOC_2D(lj+1,li) = IN_2D(j+1,i);

    if(li == halo && lj == halo) LOC_2D(lj-1,li-1) = IN_2D(j-1,i-1);
                                                                  
    if(li == halo && lj == 16)   LOC_2D(lj+1,li-1) = IN_2D(j+1,i-1);
                                                                  
    if(li == 16 && lj == halo)   LOC_2D(lj-1,li+1) = IN_2D(j-1,i+1);
                                                                  
    if(li == 16 && lj == 16)     LOC_2D(lj+1,li+1) = IN_2D(j+1,i+1);

    __syncthreads();

    if(j < m+halo && i < n+halo)
    {
        OUT_2D(j,i) = args[0 ]*LOC_2D(lj-1,li-1) + 
                      args[1 ]*LOC_2D(lj  ,li-1) + 
                      args[2 ]*LOC_2D(lj+1,li-1) +
                      args[3 ]*LOC_2D(lj-1,li  ) + 
                      args[4 ]*LOC_2D(lj  ,li  ) + 
                      args[5 ]*LOC_2D(lj+1,li  ) +
                      args[6 ]*LOC_2D(lj-1,li+1) + 
                      args[7 ]*LOC_2D(lj  ,li+1) + 
                      args[8 ]*LOC_2D(lj+1,li+1) ;
    }
}

int main(int argc, char **argv)
{
    int m = 10008; // multiple of 16 (shfl4_2)
    int n = 10008; // multiple of 16 (shfl4_2)
    // int m = 32;
    // int n = 32;
    int halo = 1; 
    int total = (m+2*halo)*(n+2*halo);
    const int K = 9;
    DATA_TYPE args[K] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    DATA_TYPE *in = new DATA_TYPE[total];
    DATA_TYPE *out_ref = new DATA_TYPE[total];
    Init_Input_2D(in, m, n, halo);

    // Show_Me(in, m, n, halo, "Input:");
    Stencil_Seq(in, out_ref, args, m, n, halo);
    // Show_Me(out_ref, m, n, halo, "Output:");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    DATA_TYPE *in_d;
    DATA_TYPE *args_d;
    DATA_TYPE *out_d;
    DATA_TYPE *out = new DATA_TYPE[total];
    cudaMalloc((void**)&in_d, total*sizeof(DATA_TYPE));
    cudaMalloc((void**)&out_d, total*sizeof(DATA_TYPE));
    cudaMalloc((void**)&args_d, (K)*sizeof(DATA_TYPE));
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(args_d, args, (K)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid((n+15)/16, (m+15)/16, 1);
    dim3 dimBlock(16, 16, 1);
    Stencil_Cuda<<<dimGrid, dimBlock>>>(in_d, out_d, args_d, m, n, halo); // warmup

    Init_Input_2D(out, m, n, halo);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    Stencil_Cuda<<<dimGrid, dimBlock>>>(in_d, out_d, args_d, m, n, halo);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_2D(out, m, n, halo);

    // Show_Me(out, m, n, halo,  "Output(Device):");
    cout << "Verify Cuda: " << boolalpha << Verify(out, out_ref, total) << endl;

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda Time: " << milliseconds << endl;

    Init_Input_2D(out, m, n, halo);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid2((n+7)/8, (m+31)/32, 1);
    dim3 dimBlock2(8, 32, 1);
    cudaEventRecord(start);
    Stencil_Cuda_Shfl<<<dimGrid2, dimBlock2>>>(in_d, out_d, args_d, m, n, halo);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_2D(out, m, n, halo);

    // Show_Me(out, m, n, halo,  "Output(Device):");
    cout << "Verify Cuda_Shfl: " << boolalpha << Verify(out, out_ref, total) << endl;

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Shfl Time: " << milliseconds << endl;

    Init_Input_2D(out, m, n, halo);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid3((n+7)/8, ((m+31)/32+1)/2, 1);
    dim3 dimBlock3(8, 32, 1);
    cudaEventRecord(start);
    Stencil_Cuda_Shfl2<<<dimGrid3, dimBlock3>>>(in_d, out_d, args_d, m, n, halo);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_2D(out, m, n, halo);

    // Show_Me(out, m, n, halo,  "Output(Device):");
    cout << "Verify Cuda_Shfl2: " << boolalpha << Verify(out, out_ref, total) << endl;

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Shfl2 Time: " << milliseconds << endl;

    Init_Input_2D(out, m, n, halo);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid4((n+7)/8, ((m+31)/32+3)/4, 1);
    dim3 dimBlock4(8, 32, 1);
    cudaEventRecord(start);
    Stencil_Cuda_Shfl4<<<dimGrid4, dimBlock4>>>(in_d, out_d, args_d, m, n, halo);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_2D(out, m, n, halo);

    // Show_Me(out, m, n, halo,  "Output(Device):");
    cout << "Verify Cuda_Shfl4: " << boolalpha << Verify(out, out_ref, total) << endl;

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Shfl4 Time: " << milliseconds << endl;

    Init_Input_2D(out, m, n, halo);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid6((n+7)/8, ((m+31)/32+7)/8, 1);
    dim3 dimBlock6(8, 32, 1);
    cudaEventRecord(start);
    Stencil_Cuda_Shfl8<<<dimGrid6, dimBlock6>>>(in_d, out_d, args_d, m, n, halo);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_2D(out, m, n, halo);

    // Show_Me(out, m, n, halo,  "Output(Device):");
    cout << "Verify Cuda_Shfl8: " << boolalpha << Verify(out, out_ref, total) << endl;

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Shfl8 Time: " << milliseconds << endl;

    Init_Input_2D(out, m, n, halo);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid7((n+7)/8, ((m+31)/32+15)/16, 1);
    dim3 dimBlock7(8, 32, 1);
    cudaEventRecord(start);
    Stencil_Cuda_Shfl16<<<dimGrid7, dimBlock7>>>(in_d, out_d, args_d, m, n, halo);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_2D(out, m, n, halo);

    // Show_Me(out, m, n, halo,  "Output(Device):");
    cout << "Verify Cuda_Shfl16: " << boolalpha << Verify(out, out_ref, total) << endl;

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Shfl16 Time: " << milliseconds << endl;

    Init_Input_2D(out, m, n, halo);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid8(((n+7)/8+1)/2, ((m+31)/32+1)/2, 1);
    dim3 dimBlock8(8, 32, 1);
    cudaEventRecord(start);
    Stencil_Cuda_Shfl4_2<<<dimGrid8, dimBlock8>>>(in_d, out_d, args_d, m, n, halo);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_2D(out, m, n, halo);

    // Show_Me(out, m, n, halo,  "Output(Device):");
    cout << "Verify Cuda_Shfl4_2: " << boolalpha << Verify(out, out_ref, total) << endl;

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Shfl4_2 Time: " << milliseconds << endl;

    Init_Input_2D(out, m, n, halo);
    cudaMemcpy(out_d, out, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid9(((n+7)/8+1)/2, ((m+31)/32+3)/4, 1);
    dim3 dimBlock9(8, 32, 1);
    cudaEventRecord(start);
    Stencil_Cuda_Shfl8_2<<<dimGrid9, dimBlock9>>>(in_d, out_d, args_d, m, n, halo);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_2D(out, m, n, halo);

    // Show_Me(out, m, n, halo,  "Output(Device):");
    cout << "Verify Cuda_Shfl8_2: " << boolalpha << Verify(out, out_ref, total) << endl;

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Shfl8_2 Time: " << milliseconds << endl;

    Init_Input_2D(out, m, n, halo);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    dim3 dimGrid5((n+15)/16, (m+15)/16, 1);
    dim3 dimBlock5(16, 16, 1);
    Stencil_Cuda_Sm<<<dimGrid5, dimBlock5, ((16+2*halo)*(16+2*halo))*sizeof(DATA_TYPE)>>>(in_d, out_d, args_d, m, n, halo);
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_2D(out, m, n, halo);
    // Show_Me(out, m, n, halo, "Output(Device):");
    cout << "Verify Cuda_Sm: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Sm Time: " << milliseconds << endl;
    cudaFree(in_d);
    cudaFree(out_d);

    delete[] in;
    delete[] out;
    delete[] out_ref;
}


