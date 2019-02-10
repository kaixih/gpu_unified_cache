__global__ void Stencil_Cuda(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int z, int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = threadIdx.z + blockIdx.z * blockDim.z + halo;
    if(k < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k,j,i) = 0 * IN_3D(k-1,j  ,i  ) +
                        1 * IN_3D(k  ,j-1,i  ) +
                        2 * IN_3D(k  ,j  ,i-1) +
                        3 * IN_3D(k  ,j  ,i  ) +
                        4 * IN_3D(k  ,j  ,i+1) +
                        5 * IN_3D(k  ,j+1,i  ) +
                        6 * IN_3D(k+1,j  ,i  ) ;
    }
}

__global__ void Stencil_Cuda_Sweep(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int z, int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    // int k = threadIdx.z + blockIdx.z * blockDim.z + halo;
    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;

    if(j < m + halo && i < n + halo)
    {
#pragma unroll // it seems the loop-unroll is useless to performance
        for(; k < k_end; ++k)
        {
            OUT_3D(k,j,i) = 0 * IN_3D(k-1,j  ,i  ) +
                            1 * IN_3D(k  ,j-1,i  ) +
                            2 * IN_3D(k  ,j  ,i-1) +
                            3 * IN_3D(k  ,j  ,i  ) +
                            4 * IN_3D(k  ,j  ,i+1) +
                            5 * IN_3D(k  ,j+1,i  ) +
                            6 * IN_3D(k+1,j  ,i  ) ;
        }
    }
}

__global__ void Stencil_Cuda_Sweep_Sm(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int z, int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    // int k = threadIdx.z + blockIdx.z * blockDim.z + halo;
    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;

#define SM_2D_M 4 
#define SM_2D_N 64 
    extern __shared__ DATA_TYPE local[];
    int li = threadIdx.x + halo;
    int lj = threadIdx.y + halo;

    if(j < m + halo && i < n + halo)
    {
        DATA_TYPE t1, t2, t3;
        DATA_TYPE r1, r2, r3, r4, r5;
        DATA_TYPE sum = 0.0;
        t3 = IN_3D(k  , j, i); // current layer
        if(li == halo)           r1 = IN_3D(k,j,i-1);
        if(li == SM_2D_N+halo-1) r2 = IN_3D(k,j,i+1);
        if(lj == halo)           r3 = IN_3D(k,j-1,i);
        if(lj == SM_2D_M+halo-1) r4 = IN_3D(k,j+1,i);

        t2 = IN_3D(k-1, j, i);

#pragma unroll // it seems the loop-unroll is useless to performance
        for(; k < k_end; ++k)
        {
            sum = 0.0;
            t1 = t2;
            t2 = t3;
            LOC_2D(lj,li) = t2;
            if(li == halo)           LOC_2D(lj,li-1) = r1; 
            if(li == SM_2D_N+halo-1) LOC_2D(lj,li+1) = r2; 
            if(lj == halo)           LOC_2D(lj-1,li) = r3; 
            if(lj == SM_2D_M+halo-1) LOC_2D(lj+1,li) = r4; 

            t3 = IN_3D(k+1, j, i); // next layer
            if(li == halo)           r1 = IN_3D(k+1,j,i-1);
            if(li == SM_2D_N+halo-1) r2 = IN_3D(k+1,j,i+1);
            if(lj == halo)           r3 = IN_3D(k+1,j-1,i);
            if(lj == SM_2D_M+halo-1) r4 = IN_3D(k+1,j+1,i);

            sum += args[0] * t1 + args[3] * t2 + args[5] *t3;
            __syncthreads();
            sum += args[1] * LOC_2D(lj-1,li  );
            sum += args[2] * LOC_2D(lj  ,li-1);
            sum += args[4] * LOC_2D(lj  ,li+1);
            sum += args[6] * LOC_2D(lj+1,li  );

            OUT_3D(k,j,i) = sum;
            __syncthreads();
        }
    }
}

__global__ void Stencil_Cuda_Sweep_Temp_Sm(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int z, int m, int n, int halo) 
{
    // each block logically processes (blockDim.x-2)*(blockDim.y-2) elements
    int i = threadIdx.x + blockIdx.x * (blockDim.x-2) + halo - 1;
    int j = threadIdx.y + blockIdx.y * (blockDim.y-2) + halo - 1;
    // 2 layers of out-of-bound elements 
    int i2 = threadIdx.x + blockIdx.x * (blockDim.x-2) + halo;
    int j2 = threadIdx.y + blockIdx.y * (blockDim.y-2) + halo; 
    // int k = threadIdx.z + blockIdx.z * blockDim.z + halo;
    // const int block_z = z / gridDim.z;
    // int k = block_z * blockIdx.z + halo;
    // const int k_end = k + block_z;

#define SM_2D_M 4 
#define SM_2D_N 64 
    extern __shared__ DATA_TYPE local[];
    int li = threadIdx.x;
    int lj = threadIdx.y;

    int li2 = threadIdx.x + halo;
    int lj2 = threadIdx.y + halo;

    int k = 0;
    int t1=0, t2=1, t3=2;
    DATA_TYPE v = 0.0;        
    LOC_L_2D(t3,lj,li) = v;
    k++;
    bool out_2d = (i==0 || j == 0 || i == n+2*halo-1 || j == m+2*halo-1);
    bool write_2d = (j2 < m + halo && i2 < n + halo && lj2 < blockDim.y+halo-2 && li2 < blockDim.x+halo-2);
    v = (out_2d)? 0: (args[0] * IN_3D(k-1,j  ,i  ) +
                      args[1] * IN_3D(k  ,j-1,i  ) +
                      args[2] * IN_3D(k  ,j  ,i-1) +
                      args[3] * IN_3D(k  ,j  ,i  ) +
                      args[4] * IN_3D(k  ,j  ,i+1) +
                      args[5] * IN_3D(k  ,j+1,i  ) +
                      args[6] * IN_3D(k+1,j  ,i  ) );
    LOC_L_2D(t2,lj,li) = v;
    k++;
    for(; k < z+halo; k++)
    {
        v = (out_2d)? 0: (args[0] * IN_3D(k-1,j  ,i  ) +
                          args[1] * IN_3D(k  ,j-1,i  ) +
                          args[2] * IN_3D(k  ,j  ,i-1) +
                          args[3] * IN_3D(k  ,j  ,i  ) +
                          args[4] * IN_3D(k  ,j  ,i+1) +
                          args[5] * IN_3D(k  ,j+1,i  ) +
                          args[6] * IN_3D(k+1,j  ,i  ) );

        LOC_L_2D(t1,lj,li) = v;   

        __syncthreads();
            
        if(write_2d) 
            OUT_3D(k-1,j2,i2) = args[0] * LOC_L_2D(t3 ,lj2  ,li2  ) +
                                args[1] * LOC_L_2D(t2 ,lj2-1,li2  ) +
                                args[2] * LOC_L_2D(t2 ,lj2  ,li2-1) +
                                args[3] * LOC_L_2D(t2 ,lj2  ,li2  ) +
                                args[4] * LOC_L_2D(t2 ,lj2  ,li2+1) +
                                args[5] * LOC_L_2D(t2 ,lj2+1,li2  ) +
                                args[6] * LOC_L_2D(t1 ,lj2  ,li2  ) ;
        __syncthreads();
        int tmp;
        tmp = t3;
        t3 = t2;
        t2 = t1;
        t1 = tmp;
    }
    LOC_L_2D(t1,lj,li) = 0;   
    __syncthreads();
    if(write_2d) 
        OUT_3D(k-1,j2,i2) = args[0] * LOC_L_2D(t3 ,lj2  ,li2  ) +
                            args[1] * LOC_L_2D(t2 ,lj2-1,li2  ) +
                            args[2] * LOC_L_2D(t2 ,lj2  ,li2-1) +
                            args[3] * LOC_L_2D(t2 ,lj2  ,li2  ) +
                            args[4] * LOC_L_2D(t2 ,lj2  ,li2+1) +
                            args[5] * LOC_L_2D(t2 ,lj2+1,li2  ) +
                            args[6] * LOC_L_2D(t1 ,lj2  ,li2  ) ;
    
        
}

__global__ void Stencil_Cuda_Sweep_Shfl(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int z, int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    // int k = threadIdx.z + blockIdx.z * blockDim.z + halo;
    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % warpSize;
    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
    DATA_TYPE tx, ty;
    int friend_id;
    int new_i, new_j;
    DATA_TYPE t3_threadInput0, t3_threadInput1;
    DATA_TYPE t2_threadInput0, t2_threadInput1;
    DATA_TYPE t1_threadInput0, t1_threadInput1;

#define SM_2D_M2 32 
#define SM_2D_N2 8 
    // extern __shared__ DATA_TYPE local[];
    int li = threadIdx.x + halo;
    int lj = threadIdx.y + halo;

    if(j < m + halo && i < n + halo)
    {
        DATA_TYPE sum = 0.0;

        // t3 is current layer; t2 is previous layer
        new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
        new_j = (warp_id_y<<2) + lane_id/10;     
        t3_threadInput0 = IN_3D(k  , new_j, new_i);
        t2_threadInput0 = IN_3D(k-1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+32)%10;
        new_j = (warp_id_y<<2) + (lane_id+32)/10;
        if(new_i < n+2*halo && new_j < m+2*halo) 
        {
            t3_threadInput1 = IN_3D(k  , new_j, new_i);
            t2_threadInput1 = IN_3D(k-1, new_j, new_i);
        }

#pragma unroll // it seems the loop-unroll is useless to performance
        for(; k < k_end; ++k)
        {
            sum = 0.0;
            // move the current storage down 
            t1_threadInput0 = t2_threadInput0;
            t1_threadInput1 = t2_threadInput1;
            t2_threadInput0 = t3_threadInput0;
            t2_threadInput1 = t3_threadInput1;

            new_i = (warp_id_x<<3) + lane_id%10;  
            new_j = (warp_id_y<<2) + lane_id/10;     
            t3_threadInput0 = IN_3D(k+1, new_j, new_i);
            new_i = (warp_id_x<<3) + (lane_id+32)%10;
            new_j = (warp_id_y<<2) + (lane_id+32)/10;
            if(new_i < n+2*halo && new_j < m+2*halo) 
            {
                t3_threadInput1 = IN_3D(k+1, new_j, new_i);
            }

            friend_id = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
            tx = args[0]*__shfl(t1_threadInput0, friend_id);
            ty = args[0]*__shfl(t1_threadInput1, friend_id);
            sum += (lane_id < 17)? tx: ty;
            tx = args[3]*__shfl(t2_threadInput0, friend_id);
            ty = args[3]*__shfl(t2_threadInput1, friend_id);
            sum += (lane_id < 17)? tx: ty;
            tx = args[5]*__shfl(t3_threadInput0, friend_id);
            ty = args[5]*__shfl(t3_threadInput1, friend_id);
            sum += (lane_id < 17)? tx: ty;
            friend_id = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
            tx = args[1]*__shfl(t2_threadInput0, friend_id);
            ty = args[1]*__shfl(t2_threadInput1, friend_id);
            sum += (lane_id < 25)? tx: ty;
            friend_id = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
            tx = args[2]*__shfl(t2_threadInput0, friend_id);
            ty = args[2]*__shfl(t2_threadInput1, friend_id);
            sum += (lane_id < 18)? tx: ty;
            friend_id = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
            tx = args[4]*__shfl(t2_threadInput0, friend_id);
            ty = args[4]*__shfl(t2_threadInput1, friend_id);
            sum += (lane_id < 16)? tx: ty;
            friend_id = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
            tx = args[6]*__shfl(t2_threadInput0, friend_id);
            ty = args[6]*__shfl(t2_threadInput1, friend_id);
            sum += (lane_id < 9 )? tx: ty;

            OUT_3D(k,j,i) = sum;
        }
    }
}

__global__ void Stencil_Cuda_Sweep_Shfl2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int z, int m, int n, int halo) 
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = (((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<3) + (lane_id>>3) + halo;
    // int k = threadIdx.z + blockIdx.z * blockDim.z + halo;
    const int block_z = z / gridDim.z;
    int k = block_z * blockIdx.z + halo;
    const int k_end = k + block_z;
    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>2)<<3) + (lane_id>>3))>>2; // 1x4x8, warp_ids are division of 
    DATA_TYPE tx0, ty0;
    DATA_TYPE tx1, ty1, tz1;
    int friend_id0, friend_id1;
    int new_i, new_j;
    DATA_TYPE t3_threadInput0, t3_threadInput1, t3_threadInput2, t3_threadInput3;
    DATA_TYPE t2_threadInput0, t2_threadInput1, t2_threadInput2, t2_threadInput3;
    DATA_TYPE t1_threadInput0, t1_threadInput1, t1_threadInput2, t1_threadInput3;

#define SM_2D_M2 32 
#define SM_2D_N2 8 
    // extern __shared__ DATA_TYPE local[];
    int li = threadIdx.x + halo;
    int lj = threadIdx.y + halo;

    if(j < m + halo && i < n + halo)
    {
        DATA_TYPE sum0 = 0.0;
        DATA_TYPE sum1 = 0.0;

        // t3 is current layer; t2 is previous layer
        new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
        new_j = (warp_id_y<<2) + lane_id/10;     
        t3_threadInput0 = IN_3D(k  , new_j, new_i);
        t2_threadInput0 = IN_3D(k-1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+32)%10;
        new_j = (warp_id_y<<2) + (lane_id+32)/10;
        t3_threadInput1 = IN_3D(k  , new_j, new_i);
        t2_threadInput1 = IN_3D(k-1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+64)%10;
        new_j = (warp_id_y<<2) + (lane_id+64)/10;
        t3_threadInput2 = IN_3D(k  , new_j, new_i);
        t2_threadInput2 = IN_3D(k-1, new_j, new_i);
        new_i = (warp_id_x<<3) + (lane_id+96)%10;
        new_j = (warp_id_y<<2) + (lane_id+96)/10;
        if(new_i < n+2*halo && new_j < m+2*halo) 
        {
            t3_threadInput3 = IN_3D(k  , new_j, new_i);
            t2_threadInput3 = IN_3D(k-1, new_j, new_i);
        }

#pragma unroll // it seems the loop-unroll is useless to performance
        for(; k < k_end; ++k)
        {
            sum0 = 0.0;
            sum1 = 0.0;
            // move the current storage down 
            t1_threadInput0 = t2_threadInput0;
            t1_threadInput1 = t2_threadInput1;
            t1_threadInput2 = t2_threadInput2;
            t1_threadInput3 = t2_threadInput3;
            t2_threadInput0 = t3_threadInput0;
            t2_threadInput1 = t3_threadInput1;
            t2_threadInput2 = t3_threadInput2;
            t2_threadInput3 = t3_threadInput3;

            new_i = (warp_id_x<<3) + lane_id%10;  
            new_j = (warp_id_y<<2) + lane_id/10;     
            t3_threadInput0 = IN_3D(k+1, new_j, new_i);
            new_i = (warp_id_x<<3) + (lane_id+32)%10;
            new_j = (warp_id_y<<2) + (lane_id+32)/10;
            t3_threadInput1 = IN_3D(k+1, new_j, new_i);
            new_i = (warp_id_x<<3) + (lane_id+64)%10;
            new_j = (warp_id_y<<2) + (lane_id+64)/10;
            t3_threadInput2 = IN_3D(k+1, new_j, new_i);
            new_i = (warp_id_x<<3) + (lane_id+96)%10;
            new_j = (warp_id_y<<2) + (lane_id+96)/10;
            if(new_i < n+2*halo && new_j < m+2*halo) 
            {
                t3_threadInput3 = IN_3D(k+1, new_j, new_i);
            }

            friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
            friend_id1 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
            tx0 = args[0]*__shfl(t1_threadInput0, friend_id0);
            ty0 = args[0]*__shfl(t1_threadInput1, friend_id0);
            tx1 = args[0]*__shfl(t1_threadInput1, friend_id1);
            ty1 = args[0]*__shfl(t1_threadInput2, friend_id1);
            sum0 += (lane_id < 17)? tx0: ty0;
            sum1 += (lane_id < 11)? tx1: ty1;
            tx0 = args[3]*__shfl(t2_threadInput0, friend_id0);
            ty0 = args[3]*__shfl(t2_threadInput1, friend_id0);
            tx1 = args[3]*__shfl(t2_threadInput1, friend_id1);
            ty1 = args[3]*__shfl(t2_threadInput2, friend_id1);
            sum0 += (lane_id < 17)? tx0: ty0;
            sum1 += (lane_id < 11)? tx1: ty1;
            tx0 = args[5]*__shfl(t3_threadInput0, friend_id0);
            ty0 = args[5]*__shfl(t3_threadInput1, friend_id0);
            tx1 = args[5]*__shfl(t3_threadInput1, friend_id1);
            ty1 = args[5]*__shfl(t3_threadInput2, friend_id1);
            sum0 += (lane_id < 17)? tx0: ty0;
            sum1 += (lane_id < 11)? tx1: ty1;

            friend_id0 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
            friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
            tx0 = args[1]*__shfl(t2_threadInput0, friend_id0);
            ty0 = args[1]*__shfl(t2_threadInput1, friend_id0);
            tx1 = args[1]*__shfl(t2_threadInput1, friend_id1);
            ty1 = args[1]*__shfl(t2_threadInput2, friend_id1);
            sum0 += (lane_id < 25)? tx0: ty0;
            sum1 += (lane_id < 19)? tx1: ty1;
            friend_id0 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
            friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
            tx0 = args[2]*__shfl(t2_threadInput0, friend_id0);
            ty0 = args[2]*__shfl(t2_threadInput1, friend_id0);
            tx1 = args[2]*__shfl(t2_threadInput1, friend_id1);
            ty1 = args[2]*__shfl(t2_threadInput2, friend_id1);
            sum0 += (lane_id < 18)? tx0: ty0;
            sum1 += (lane_id < 12)? tx1: ty1;
            friend_id0 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
            friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
            tx0 = args[4]*__shfl(t2_threadInput0, friend_id0);
            ty0 = args[4]*__shfl(t2_threadInput1, friend_id0);
            tx1 = args[4]*__shfl(t2_threadInput1, friend_id1);
            ty1 = args[4]*__shfl(t2_threadInput2, friend_id1);
            sum0 += (lane_id < 16)? tx0: ty0;
            sum1 += (lane_id < 10)? tx1: ty1;
            friend_id0 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
            friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
            tx0 = args[5]*__shfl(t2_threadInput0, friend_id0);
            ty0 = args[5]*__shfl(t2_threadInput1, friend_id0);
            tx1 = args[5]*__shfl(t2_threadInput1, friend_id1);
            ty1 = args[5]*__shfl(t2_threadInput2, friend_id1);
            tz1 = args[5]*__shfl(t2_threadInput3, friend_id1);
            sum0 += (lane_id < 9 )? tx0: ty0;
            sum1 += (lane_id < 3 )? tx1: ((lane_id < 29)? ty1: tz1);

            OUT_3D(k,j  ,i) = sum0;
            OUT_3D(k,j+4,i) = sum1;
        }
    }
}


__global__ void Stencil_Cuda_Sm(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int z, int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = threadIdx.z + blockIdx.z * blockDim.z + halo;

#define SM_Z 8
#define SM_M 4 
#define SM_N 8
    extern __shared__ DATA_TYPE local[];
    int li = threadIdx.x + halo;
    int lj = threadIdx.y + halo;
    int lk = threadIdx.z + halo;
    LOC_3D(lk,lj,li) = IN_3D(k,j,i);

    if(li == halo)        LOC_3D(lk,lj,li-1) = IN_3D(k,j,i-1);
    if(li == SM_N+halo-1) LOC_3D(lk,lj,li+1) = IN_3D(k,j,i+1);
    if(lj == halo)        LOC_3D(lk,lj-1,li) = IN_3D(k,j-1,i);
    if(lj == SM_M+halo-1) LOC_3D(lk,lj+1,li) = IN_3D(k,j+1,i);
    if(lk == halo)        LOC_3D(lk-1,lj,li) = IN_3D(k-1,j,i);
    if(lk == SM_Z+halo-1) LOC_3D(lk+1,lj,li) = IN_3D(k+1,j,i);
    __syncthreads();

    if(k < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k,j,i) = args[0] * LOC_3D(lk-1,lj  ,li  ) +
                        args[1] * LOC_3D(lk  ,lj-1,li  ) +
                        args[2] * LOC_3D(lk  ,lj  ,li-1) +
                        args[3] * LOC_3D(lk  ,lj  ,li  ) +
                        args[4] * LOC_3D(lk  ,lj  ,li+1) +
                        args[5] * LOC_3D(lk  ,lj+1,li  ) +
                        args[6] * LOC_3D(lk+1,lj  ,li  ) ;
    }
}

__global__ void Stencil_Cuda_Sm8(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int z, int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = (((threadIdx.z + blockIdx.z * blockDim.z)>>3)<<6) + threadIdx.z + halo;
    // int k = threadIdx.z + blockIdx.z * blockDim.z + halo;

#define SM_Z 8
#define SM_M 4 
#define SM_N 8
    extern __shared__ DATA_TYPE local[];
    int li = threadIdx.x + halo;
    int lj = threadIdx.y + halo;
    int lk = threadIdx.z + halo;
    LOC_3D(lk,lj,li) = IN_3D(k,j,i);

    if(li == halo)        LOC_3D(lk,lj,li-1) = IN_3D(k,j,i-1);
    if(li == SM_N+halo-1) LOC_3D(lk,lj,li+1) = IN_3D(k,j,i+1);
    if(lj == halo)        LOC_3D(lk,lj-1,li) = IN_3D(k,j-1,i);
    if(lj == SM_M+halo-1) LOC_3D(lk,lj+1,li) = IN_3D(k,j+1,i);
    if(lk == halo)        LOC_3D(lk-1,lj,li) = IN_3D(k-1,j,i);
    if(lk == SM_Z+halo-1) LOC_3D(lk+1,lj,li) = IN_3D(k+1,j,i);
    __syncthreads();

    if(k < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k,j,i) = args[0] * LOC_3D(lk-1,lj  ,li  ) +
                        args[1] * LOC_3D(lk  ,lj-1,li  ) +
                        args[2] * LOC_3D(lk  ,lj  ,li-1) +
                        args[3] * LOC_3D(lk  ,lj  ,li  ) +
                        args[4] * LOC_3D(lk  ,lj  ,li+1) +
                        args[5] * LOC_3D(lk  ,lj+1,li  ) +
                        args[6] * LOC_3D(lk+1,lj  ,li  ) ;
    }
    __syncthreads();

    k += SM_Z;
    LOC_3D(lk,lj,li) = IN_3D(k,j,i);

    if(li == halo)        LOC_3D(lk,lj,li-1) = IN_3D(k,j,i-1);
    if(li == SM_N+halo-1) LOC_3D(lk,lj,li+1) = IN_3D(k,j,i+1);
    if(lj == halo)        LOC_3D(lk,lj-1,li) = IN_3D(k,j-1,i);
    if(lj == SM_M+halo-1) LOC_3D(lk,lj+1,li) = IN_3D(k,j+1,i);
    if(lk == halo)        LOC_3D(lk-1,lj,li) = IN_3D(k-1,j,i);
    if(lk == SM_Z+halo-1) LOC_3D(lk+1,lj,li) = IN_3D(k+1,j,i);
    __syncthreads();

    if(k < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k,j,i) = args[0] * LOC_3D(lk-1,lj  ,li  ) +
                        args[1] * LOC_3D(lk  ,lj-1,li  ) +
                        args[2] * LOC_3D(lk  ,lj  ,li-1) +
                        args[3] * LOC_3D(lk  ,lj  ,li  ) +
                        args[4] * LOC_3D(lk  ,lj  ,li+1) +
                        args[5] * LOC_3D(lk  ,lj+1,li  ) +
                        args[6] * LOC_3D(lk+1,lj  ,li  ) ;
    }
    __syncthreads();

    k += SM_Z;
    LOC_3D(lk,lj,li) = IN_3D(k,j,i);

    if(li == halo)        LOC_3D(lk,lj,li-1) = IN_3D(k,j,i-1);
    if(li == SM_N+halo-1) LOC_3D(lk,lj,li+1) = IN_3D(k,j,i+1);
    if(lj == halo)        LOC_3D(lk,lj-1,li) = IN_3D(k,j-1,i);
    if(lj == SM_M+halo-1) LOC_3D(lk,lj+1,li) = IN_3D(k,j+1,i);
    if(lk == halo)        LOC_3D(lk-1,lj,li) = IN_3D(k-1,j,i);
    if(lk == SM_Z+halo-1) LOC_3D(lk+1,lj,li) = IN_3D(k+1,j,i);
    __syncthreads();

    if(k < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k,j,i) = args[0] * LOC_3D(lk-1,lj  ,li  ) +
                        args[1] * LOC_3D(lk  ,lj-1,li  ) +
                        args[2] * LOC_3D(lk  ,lj  ,li-1) +
                        args[3] * LOC_3D(lk  ,lj  ,li  ) +
                        args[4] * LOC_3D(lk  ,lj  ,li+1) +
                        args[5] * LOC_3D(lk  ,lj+1,li  ) +
                        args[6] * LOC_3D(lk+1,lj  ,li  ) ;
    }
    __syncthreads();
    k += SM_Z;
    LOC_3D(lk,lj,li) = IN_3D(k,j,i);

    if(li == halo)        LOC_3D(lk,lj,li-1) = IN_3D(k,j,i-1);
    if(li == SM_N+halo-1) LOC_3D(lk,lj,li+1) = IN_3D(k,j,i+1);
    if(lj == halo)        LOC_3D(lk,lj-1,li) = IN_3D(k,j-1,i);
    if(lj == SM_M+halo-1) LOC_3D(lk,lj+1,li) = IN_3D(k,j+1,i);
    if(lk == halo)        LOC_3D(lk-1,lj,li) = IN_3D(k-1,j,i);
    if(lk == SM_Z+halo-1) LOC_3D(lk+1,lj,li) = IN_3D(k+1,j,i);
    __syncthreads();

    if(k < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k,j,i) = args[0] * LOC_3D(lk-1,lj  ,li  ) +
                        args[1] * LOC_3D(lk  ,lj-1,li  ) +
                        args[2] * LOC_3D(lk  ,lj  ,li-1) +
                        args[3] * LOC_3D(lk  ,lj  ,li  ) +
                        args[4] * LOC_3D(lk  ,lj  ,li+1) +
                        args[5] * LOC_3D(lk  ,lj+1,li  ) +
                        args[6] * LOC_3D(lk+1,lj  ,li  ) ;
    }

    __syncthreads();
    k += SM_Z;
    LOC_3D(lk,lj,li) = IN_3D(k,j,i);

    if(li == halo)        LOC_3D(lk,lj,li-1) = IN_3D(k,j,i-1);
    if(li == SM_N+halo-1) LOC_3D(lk,lj,li+1) = IN_3D(k,j,i+1);
    if(lj == halo)        LOC_3D(lk,lj-1,li) = IN_3D(k,j-1,i);
    if(lj == SM_M+halo-1) LOC_3D(lk,lj+1,li) = IN_3D(k,j+1,i);
    if(lk == halo)        LOC_3D(lk-1,lj,li) = IN_3D(k-1,j,i);
    if(lk == SM_Z+halo-1) LOC_3D(lk+1,lj,li) = IN_3D(k+1,j,i);
    __syncthreads();

    if(k < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k,j,i) = args[0] * LOC_3D(lk-1,lj  ,li  ) +
                        args[1] * LOC_3D(lk  ,lj-1,li  ) +
                        args[2] * LOC_3D(lk  ,lj  ,li-1) +
                        args[3] * LOC_3D(lk  ,lj  ,li  ) +
                        args[4] * LOC_3D(lk  ,lj  ,li+1) +
                        args[5] * LOC_3D(lk  ,lj+1,li  ) +
                        args[6] * LOC_3D(lk+1,lj  ,li  ) ;
    }
    __syncthreads();
    k += SM_Z;
    LOC_3D(lk,lj,li) = IN_3D(k,j,i);

    if(li == halo)        LOC_3D(lk,lj,li-1) = IN_3D(k,j,i-1);
    if(li == SM_N+halo-1) LOC_3D(lk,lj,li+1) = IN_3D(k,j,i+1);
    if(lj == halo)        LOC_3D(lk,lj-1,li) = IN_3D(k,j-1,i);
    if(lj == SM_M+halo-1) LOC_3D(lk,lj+1,li) = IN_3D(k,j+1,i);
    if(lk == halo)        LOC_3D(lk-1,lj,li) = IN_3D(k-1,j,i);
    if(lk == SM_Z+halo-1) LOC_3D(lk+1,lj,li) = IN_3D(k+1,j,i);
    __syncthreads();

    if(k < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k,j,i) = args[0] * LOC_3D(lk-1,lj  ,li  ) +
                        args[1] * LOC_3D(lk  ,lj-1,li  ) +
                        args[2] * LOC_3D(lk  ,lj  ,li-1) +
                        args[3] * LOC_3D(lk  ,lj  ,li  ) +
                        args[4] * LOC_3D(lk  ,lj  ,li+1) +
                        args[5] * LOC_3D(lk  ,lj+1,li  ) +
                        args[6] * LOC_3D(lk+1,lj  ,li  ) ;
    }
    __syncthreads();
    k += SM_Z;
    LOC_3D(lk,lj,li) = IN_3D(k,j,i);

    if(li == halo)        LOC_3D(lk,lj,li-1) = IN_3D(k,j,i-1);
    if(li == SM_N+halo-1) LOC_3D(lk,lj,li+1) = IN_3D(k,j,i+1);
    if(lj == halo)        LOC_3D(lk,lj-1,li) = IN_3D(k,j-1,i);
    if(lj == SM_M+halo-1) LOC_3D(lk,lj+1,li) = IN_3D(k,j+1,i);
    if(lk == halo)        LOC_3D(lk-1,lj,li) = IN_3D(k-1,j,i);
    if(lk == SM_Z+halo-1) LOC_3D(lk+1,lj,li) = IN_3D(k+1,j,i);
    __syncthreads();

    if(k < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k,j,i) = args[0] * LOC_3D(lk-1,lj  ,li  ) +
                        args[1] * LOC_3D(lk  ,lj-1,li  ) +
                        args[2] * LOC_3D(lk  ,lj  ,li-1) +
                        args[3] * LOC_3D(lk  ,lj  ,li  ) +
                        args[4] * LOC_3D(lk  ,lj  ,li+1) +
                        args[5] * LOC_3D(lk  ,lj+1,li  ) +
                        args[6] * LOC_3D(lk+1,lj  ,li  ) ;
    }
    __syncthreads();
    k += SM_Z;
    LOC_3D(lk,lj,li) = IN_3D(k,j,i);

    if(li == halo)        LOC_3D(lk,lj,li-1) = IN_3D(k,j,i-1);
    if(li == SM_N+halo-1) LOC_3D(lk,lj,li+1) = IN_3D(k,j,i+1);
    if(lj == halo)        LOC_3D(lk,lj-1,li) = IN_3D(k,j-1,i);
    if(lj == SM_M+halo-1) LOC_3D(lk,lj+1,li) = IN_3D(k,j+1,i);
    if(lk == halo)        LOC_3D(lk-1,lj,li) = IN_3D(k-1,j,i);
    if(lk == SM_Z+halo-1) LOC_3D(lk+1,lj,li) = IN_3D(k+1,j,i);
    __syncthreads();

    if(k < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k,j,i) = args[0] * LOC_3D(lk-1,lj  ,li  ) +
                        args[1] * LOC_3D(lk  ,lj-1,li  ) +
                        args[2] * LOC_3D(lk  ,lj  ,li-1) +
                        args[3] * LOC_3D(lk  ,lj  ,li  ) +
                        args[4] * LOC_3D(lk  ,lj  ,li+1) +
                        args[5] * LOC_3D(lk  ,lj+1,li  ) +
                        args[6] * LOC_3D(lk+1,lj  ,li  ) ;
    }
    /*
    */
}

__global__ void Stencil_Cuda_Shfl(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int z, int m, int n, int halo) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = threadIdx.z + blockIdx.z * blockDim.z + halo;
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    int lane_id = tid % warpSize;
    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
    int warp_id_z = (threadIdx.z + blockIdx.z * blockDim.z)>>0; // there numbers
    int new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
    int new_j = (warp_id_y<<2) + (lane_id/10)%6; // 6  is extended dimension of j 
    int new_k = (warp_id_z<<0) + lane_id/60;     // 60 is extended area of ixj = 10x6
    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5;
    threadInput0 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+32)/60;
    threadInput1 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10;
    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+64)/60;
    threadInput2 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10;
    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+96)/60;
    threadInput3 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10;
    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+128)/60;
    threadInput4 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10;
    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+160)/60;
    if(new_i < n+2*halo && new_j < m+2*halo) // assume new_k is within the range
        threadInput5 = IN_3D(new_k, new_j, new_i);

    DATA_TYPE sum = 0.0;
    int friend_id;
    DATA_TYPE tx, ty, tz;
    friend_id = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    tx = args[0]*__shfl(threadInput0, friend_id);
    ty = args[0]*__shfl(threadInput1, friend_id);
    sum += (lane_id < 17)? tx: ty;

    friend_id = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    tx = args[1]*__shfl(threadInput1, friend_id);
    ty = args[1]*__shfl(threadInput2, friend_id);
    tz = args[1]*__shfl(threadInput3, friend_id);
    sum += (lane_id < 3 )? tx: ((lane_id < 29)? ty: tz);

    friend_id = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    tx = args[2]*__shfl(threadInput2, friend_id);
    ty = args[2]*__shfl(threadInput3, friend_id);
    sum += (lane_id < 22)? tx: ty;

    friend_id = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    tx = args[3]*__shfl(threadInput2, friend_id);
    ty = args[3]*__shfl(threadInput3, friend_id);
    sum += (lane_id < 21)? tx: ty;

    friend_id = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    tx = args[4]*__shfl(threadInput2, friend_id);
    ty = args[4]*__shfl(threadInput3, friend_id);
    sum += (lane_id < 20)? tx: ty;

    friend_id = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    tx = args[5]*__shfl(threadInput2, friend_id);
    ty = args[5]*__shfl(threadInput3, friend_id);
    sum += (lane_id < 13)? tx: ty;

    friend_id = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    tx = args[6]*__shfl(threadInput4, friend_id);
    ty = args[6]*__shfl(threadInput5, friend_id);
    sum += (lane_id < 24)? tx: ty;

    if(k < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k,j,i) = sum;
    }

}

__global__ void Stencil_Cuda_Shfl2(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int z, int m, int n, int halo) 
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<1) + (lane_id>>5) + halo; 
    // thread coarsening: related to warp dimensions 1x4x8. 
    // We coarsen from dimension z from 2^0 to 2^1, also need to know there are how many values in dimension z

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
    int warp_id_z = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<1) + (lane_id>>5); // these numbers
    int new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
    int new_j = (warp_id_y<<2) + (lane_id/10)%6; // 6  is extended dimension of j 
    int new_k = (warp_id_z<<0) + lane_id/60;     // 60 is extended area of ixj = 10x6
    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5,
              threadInput6, threadInput7;
    threadInput0 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+32)/60;
    threadInput1 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10;
    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+64)/60;
    threadInput2 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10;
    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+96)/60;
    threadInput3 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10;
    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+128)/60;
    threadInput4 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10;
    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+160)/60;
    threadInput5 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+192)%10;
    new_j = (warp_id_y<<2) + ((lane_id+192)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+192)/60;
    threadInput6 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+224)%10;
    new_j = (warp_id_y<<2) + ((lane_id+224)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+224)/60;
    if(new_i < n+2*halo && new_j < m+2*halo) // assume new_k is within the range
        threadInput7 = IN_3D(new_k, new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    int friend_id0, friend_id1;
    DATA_TYPE tx0, ty0, tz0, tx1, ty1, tz1;
    friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[0]*__shfl(threadInput0, friend_id0);
    ty0 = args[0]*__shfl(threadInput1, friend_id0);
    tx1 = args[0]*__shfl(threadInput2, friend_id1);
    ty1 = args[0]*__shfl(threadInput3, friend_id1);
    sum0 += (lane_id < 17)? tx0: ty0;
    sum1 += (lane_id < 21)? tx1: ty1;

    friend_id0 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[1]*__shfl(threadInput1, friend_id0);
    ty0 = args[1]*__shfl(threadInput2, friend_id0);
    tz0 = args[1]*__shfl(threadInput3, friend_id0);
    tx1 = args[1]*__shfl(threadInput3, friend_id1);
    ty1 = args[1]*__shfl(threadInput4, friend_id1);
    sum0 += (lane_id < 3 )? tx0: ((lane_id < 29)? ty0: tz0);
    sum1 += (lane_id < 7 )? tx1: ty1;

    friend_id0 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[2]*__shfl(threadInput2, friend_id0);
    ty0 = args[2]*__shfl(threadInput3, friend_id0);
    tx1 = args[2]*__shfl(threadInput4, friend_id1);
    ty1 = args[2]*__shfl(threadInput5, friend_id1);
    sum0 += (lane_id < 22)? tx0: ty0;
    sum1 += (lane_id < 24)? tx1: ty1;

    friend_id0 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[3]*__shfl(threadInput2, friend_id0);
    ty0 = args[3]*__shfl(threadInput3, friend_id0);
    tx1 = args[3]*__shfl(threadInput4, friend_id1);
    ty1 = args[3]*__shfl(threadInput5, friend_id1);
    sum0 += (lane_id < 21)? tx0: ty0;
    sum1 += (lane_id < 24)? tx1: ty1;

    friend_id0 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[4]*__shfl(threadInput2, friend_id0);
    ty0 = args[4]*__shfl(threadInput3, friend_id0);
    tx1 = args[4]*__shfl(threadInput4, friend_id1);
    ty1 = args[4]*__shfl(threadInput5, friend_id1);
    sum0 += (lane_id < 20)? tx0: ty0;
    sum1 += (lane_id < 24)? tx1: ty1;

    friend_id0 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[5]*__shfl(threadInput2, friend_id0);
    ty0 = args[5]*__shfl(threadInput3, friend_id0);
    tx1 = args[5]*__shfl(threadInput4, friend_id1);
    ty1 = args[5]*__shfl(threadInput5, friend_id1);
    sum0 += (lane_id < 13)? tx0: ty0;
    sum1 += (lane_id < 16)? tx1: ty1;

    friend_id0 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[6]*__shfl(threadInput4, friend_id0);
    ty0 = args[6]*__shfl(threadInput5, friend_id0);
    tx1 = args[6]*__shfl(threadInput5, friend_id1);
    ty1 = args[6]*__shfl(threadInput6, friend_id1);
    tz1 = args[6]*__shfl(threadInput7, friend_id1);
    sum0 += (lane_id < 24)? tx0: ty0;
    sum1 += (lane_id < 1 )? tx1: ((lane_id < 27)? ty1: tz1);
    /*
    */

    if(k < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k,j,i) = sum0;
    }
    if(k+1 < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k+1,j,i) = sum1;
    }

}

__global__ void Stencil_Cuda_Shfl4(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int z, int m, int n, int halo) 
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<2) + (lane_id>>5) + halo; 
    // Thread coarsening: related to warp dimensions 1x4x8. 
    // We coarsen from dimension z from 2^0 to 2^2, also need to know there are how many values in dimension z,
    // which is (lane_id>>5) 

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
    int warp_id_z = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<2) + (lane_id>>5); // these numbers
    int new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
    int new_j = (warp_id_y<<2) + (lane_id/10)%6; // 6  is extended dimension of j 
    int new_k = (warp_id_z<<0) + lane_id/60;     // 60 is extended area of ixj = 10x6
    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5,
              threadInput6, threadInput7, threadInput8, threadInput9, threadInput10, threadInput11;
    threadInput0 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+32)/60;
    threadInput1 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10;
    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+64)/60;
    threadInput2 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10;
    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+96)/60;
    threadInput3 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10;
    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+128)/60;
    threadInput4 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10;
    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+160)/60;
    threadInput5 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+192)%10;
    new_j = (warp_id_y<<2) + ((lane_id+192)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+192)/60;
    threadInput6 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+224)%10;
    new_j = (warp_id_y<<2) + ((lane_id+224)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+224)/60;
    threadInput7 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+256)%10;
    new_j = (warp_id_y<<2) + ((lane_id+256)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+256)/60;
    threadInput8 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+288)%10;
    new_j = (warp_id_y<<2) + ((lane_id+288)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+288)/60;
    threadInput9 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+320)%10;
    new_j = (warp_id_y<<2) + ((lane_id+320)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+320)/60;
    threadInput10 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+352)%10;
    new_j = (warp_id_y<<2) + ((lane_id+352)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+352)/60;
    if(new_i < n+2*halo && new_j < m+2*halo) // assume new_k is within the range
        threadInput11 = IN_3D(new_k, new_j, new_i);

    DATA_TYPE sum0 = 0.0;
    DATA_TYPE sum1 = 0.0;
    DATA_TYPE sum2 = 0.0;
    DATA_TYPE sum3 = 0.0;
    int friend_id0, friend_id1, friend_id2, friend_id3;
    DATA_TYPE tx0, ty0, tz0, tx1, ty1, tz1, tx2, ty2, tz2, tx3, ty3, tz3;
    friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[0]*__shfl(threadInput0, friend_id0);
    ty0 = args[0]*__shfl(threadInput1, friend_id0);
    tx1 = args[0]*__shfl(threadInput2, friend_id1);
    ty1 = args[0]*__shfl(threadInput3, friend_id1);
    tx2 = args[0]*__shfl(threadInput4, friend_id2);
    ty2 = args[0]*__shfl(threadInput5, friend_id2);
    tx3 = args[0]*__shfl(threadInput5, friend_id3);
    ty3 = args[0]*__shfl(threadInput6, friend_id3);
    tz3 = args[0]*__shfl(threadInput7, friend_id3);
    sum0 += (lane_id < 17)? tx0: ty0;
    sum1 += (lane_id < 21)? tx1: ty1;
    sum2 += (lane_id < 24)? tx2: ty2;
    sum3 += (lane_id < 1 )? tx3: ((lane_id < 27)? ty3: tz3);

    friend_id0 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[1]*__shfl(threadInput1, friend_id0);
    ty0 = args[1]*__shfl(threadInput2, friend_id0);
    tz0 = args[1]*__shfl(threadInput3, friend_id0);
    tx1 = args[1]*__shfl(threadInput3, friend_id1);
    ty1 = args[1]*__shfl(threadInput4, friend_id1);
    tx2 = args[1]*__shfl(threadInput5, friend_id2);
    ty2 = args[1]*__shfl(threadInput6, friend_id2);
    tx3 = args[1]*__shfl(threadInput7, friend_id3);
    ty3 = args[1]*__shfl(threadInput8, friend_id3);
    sum0 += (lane_id < 3 )? tx0: ((lane_id < 29)? ty0: tz0);
    sum1 += (lane_id < 7 )? tx1: ty1;
    sum2 += (lane_id < 9 )? tx2: ty2;
    sum3 += (lane_id < 13)? tx3: ty3;

    friend_id0 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[2]*__shfl(threadInput2, friend_id0);
    ty0 = args[2]*__shfl(threadInput3, friend_id0);
    tx1 = args[2]*__shfl(threadInput4, friend_id1);
    ty1 = args[2]*__shfl(threadInput5, friend_id1);
    tx2 = args[2]*__shfl(threadInput5, friend_id2);
    ty2 = args[2]*__shfl(threadInput6, friend_id2);
    tz2 = args[2]*__shfl(threadInput7, friend_id2);
    tx3 = args[2]*__shfl(threadInput7, friend_id3);
    ty3 = args[2]*__shfl(threadInput8, friend_id3);
    sum0 += (lane_id < 22)? tx0: ty0;
    sum1 += (lane_id < 24)? tx1: ty1;
    sum2 += (lane_id < 2 )? tx2: ((lane_id < 28)? ty2: tz2);
    sum3 += (lane_id < 6 )? tx3: ty3;

    friend_id0 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[3]*__shfl(threadInput2, friend_id0);
    ty0 = args[3]*__shfl(threadInput3, friend_id0);
    tx1 = args[3]*__shfl(threadInput4, friend_id1);
    ty1 = args[3]*__shfl(threadInput5, friend_id1);
    tx2 = args[3]*__shfl(threadInput5, friend_id2);
    ty2 = args[3]*__shfl(threadInput6, friend_id2);
    tz2 = args[3]*__shfl(threadInput7, friend_id2);
    tx3 = args[3]*__shfl(threadInput7, friend_id3);
    ty3 = args[3]*__shfl(threadInput8, friend_id3);
    tz3 = args[3]*__shfl(threadInput9, friend_id3);
    sum0 += (lane_id < 21)? tx0: ty0;
    sum1 += (lane_id < 24)? tx1: ty1;
    sum2 += (lane_id < 1 )? tx2: ((lane_id < 27)? ty2: tz2);
    sum3 += (lane_id < 5 )? tx3: ((lane_id < 31)? ty3: tz3);

    friend_id0 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[4]*__shfl(threadInput2, friend_id0);
    ty0 = args[4]*__shfl(threadInput3, friend_id0);
    tx1 = args[4]*__shfl(threadInput4, friend_id1);
    ty1 = args[4]*__shfl(threadInput5, friend_id1);
    tx2 = args[4]*__shfl(threadInput6, friend_id2);
    ty2 = args[4]*__shfl(threadInput7, friend_id2);
    tx3 = args[4]*__shfl(threadInput7, friend_id3);
    ty3 = args[4]*__shfl(threadInput8, friend_id3);
    tz3 = args[4]*__shfl(threadInput9, friend_id3);
    sum0 += (lane_id < 20)? tx0: ty0;
    sum1 += (lane_id < 24)? tx1: ty1;
    sum2 += (lane_id < 26)? tx2: ty2;
    sum3 += (lane_id < 4 )? tx3: ((lane_id < 30)? ty3: tz3);

    friend_id0 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[5]*__shfl(threadInput2, friend_id0);
    ty0 = args[5]*__shfl(threadInput3, friend_id0);
    tx1 = args[5]*__shfl(threadInput4, friend_id1);
    ty1 = args[5]*__shfl(threadInput5, friend_id1);
    tx2 = args[5]*__shfl(threadInput6, friend_id2);
    ty2 = args[5]*__shfl(threadInput7, friend_id2);
    tx3 = args[5]*__shfl(threadInput8, friend_id3);
    ty3 = args[5]*__shfl(threadInput9, friend_id3);
    sum0 += (lane_id < 13)? tx0: ty0;
    sum1 += (lane_id < 16)? tx1: ty1;
    sum2 += (lane_id < 19)? tx2: ty2;
    sum3 += (lane_id < 23)? tx3: ty3;

    friend_id0 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+23+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[6]*__shfl(threadInput4 , friend_id0);
    ty0 = args[6]*__shfl(threadInput5 , friend_id0);
    tx1 = args[6]*__shfl(threadInput5 , friend_id1);
    ty1 = args[6]*__shfl(threadInput6 , friend_id1);
    tz1 = args[6]*__shfl(threadInput7 , friend_id1);
    tx2 = args[6]*__shfl(threadInput7 , friend_id2);
    ty2 = args[6]*__shfl(threadInput8 , friend_id2);
    tz2 = args[6]*__shfl(threadInput9 , friend_id2);
    tx3 = args[6]*__shfl(threadInput9 , friend_id3);
    ty3 = args[6]*__shfl(threadInput10, friend_id3);
    sum0 += (lane_id < 24)? tx0: ty0;
    sum1 += (lane_id < 1 )? tx1: ((lane_id < 27)? ty1: tz1);
    sum2 += (lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2);
    sum3 += (lane_id < 8 )? tx3: ty3;
    /*
    */

    if(k < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k,j,i) = sum0;
    }
    if(k+1 < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k+1,j,i) = sum1;
    }
    if(k+2 < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k+2,j,i) = sum2;
    }
    if(k+3 < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k+3,j,i) = sum3;
    }
}

__global__ void Stencil_Cuda_Shfl8(DATA_TYPE *in, DATA_TYPE *out, DATA_TYPE *args, int z, int m, int n, int halo) 
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    int lane_id = tid % warpSize;

    int i = threadIdx.x + blockIdx.x * blockDim.x + halo;
    int j = threadIdx.y + blockIdx.y * blockDim.y + halo;
    int k = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<3) + (lane_id>>5) + halo; 
    // Thread coarsening: related to warp dimensions 1x4x8. 
    // We coarsen from dimension z from 2^0 to 2^3, also need to know there are how many values in dimension z,
    // which is (lane_id>>5) 

    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
    int warp_id_z = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<3) + (lane_id>>5); // these numbers
    int new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
    int new_j = (warp_id_y<<2) + (lane_id/10)%6; // 6  is extended dimension of j 
    int new_k = (warp_id_z<<0) + lane_id/60;     // 60 is extended area of ixj = 10x6
    DATA_TYPE threadInput0, threadInput1, threadInput2, threadInput3, threadInput4, threadInput5,
              threadInput6, threadInput7, threadInput8, threadInput9, threadInput10, threadInput11,
              threadInput12, threadInput13, threadInput14, threadInput15, threadInput16, threadInput17,
              threadInput18;
    threadInput0 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+32)%10;
    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+32)/60;
    threadInput1 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+64)%10;
    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+64)/60;
    threadInput2 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+96)%10;
    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+96)/60;
    threadInput3 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+128)%10;
    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+128)/60;
    threadInput4 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+160)%10;
    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+160)/60;
    threadInput5 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+192)%10;
    new_j = (warp_id_y<<2) + ((lane_id+192)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+192)/60;
    threadInput6 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+224)%10;
    new_j = (warp_id_y<<2) + ((lane_id+224)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+224)/60;
    threadInput7 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+256)%10;
    new_j = (warp_id_y<<2) + ((lane_id+256)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+256)/60;
    threadInput8 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+288)%10;
    new_j = (warp_id_y<<2) + ((lane_id+288)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+288)/60;
    threadInput9 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+320)%10;
    new_j = (warp_id_y<<2) + ((lane_id+320)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+320)/60;
    threadInput10 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+352)%10;
    new_j = (warp_id_y<<2) + ((lane_id+352)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+352)/60;
    threadInput11 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+384)%10;
    new_j = (warp_id_y<<2) + ((lane_id+384)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+384)/60;
    threadInput12 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+416)%10;
    new_j = (warp_id_y<<2) + ((lane_id+416)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+416)/60;
    threadInput13 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+448)%10;
    new_j = (warp_id_y<<2) + ((lane_id+448)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+448)/60;
    threadInput14 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+480)%10;
    new_j = (warp_id_y<<2) + ((lane_id+480)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+480)/60;
    threadInput15 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+512)%10;
    new_j = (warp_id_y<<2) + ((lane_id+512)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+512)/60;
    threadInput16 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+544)%10;
    new_j = (warp_id_y<<2) + ((lane_id+544)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+544)/60;
    threadInput17 = IN_3D(new_k, new_j, new_i);
    new_i = (warp_id_x<<3) + (lane_id+576)%10;
    new_j = (warp_id_y<<2) + ((lane_id+576)/10)%6;
    new_k = (warp_id_z<<0) + (lane_id+576)/60;
    if(new_i < n+2*halo && new_j < m+2*halo) // assume new_k is within the range
        threadInput18 = IN_3D(new_k, new_j, new_i);

    /*
    rx0 = args[]*__shfl(threadInput, friend_id4);
    ry0 = args[]*__shfl(threadInput, friend_id4);
    rz0 = args[]*__shfl(threadInput, friend_id4);
    rx1 = args[]*__shfl(threadInput, friend_id5);
    ry1 = args[]*__shfl(threadInput, friend_id5);
    rz1 = args[]*__shfl(threadInput, friend_id5);
    rx2 = args[]*__shfl(threadInput, friend_id6);
    ry2 = args[]*__shfl(threadInput, friend_id6);
    rz2 = args[]*__shfl(threadInput, friend_id6);
    rx3 = args[]*__shfl(threadInput, friend_id7);
    ry3 = args[]*__shfl(threadInput, friend_id7);
    rz3 = args[]*__shfl(threadInput, friend_id7);
    sum4 += (lane_id < )? rx0: ((lane_id < )? ry0: rz0);
    sum5 += (lane_id < )? rx1: ((lane_id < )? ry1: rz1);
    sum6 += (lane_id < )? rx2: ((lane_id < )? ry2: rz2);
    sum7 += (lane_id < )? rx3: ((lane_id < )? ry3: rz3);
    */
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
    DATA_TYPE tx0, ty0, tz0, tx1, ty1, tz1, tx2, ty2, tz2, tx3, ty3, tz3;
    DATA_TYPE rx0, ry0, rz0, rx1, ry1, rz1, rx2, ry2, rz2, rx3, ry3, rz3;
    friend_id0 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+23+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+15+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[0]*__shfl(threadInput0, friend_id0);
    ty0 = args[0]*__shfl(threadInput1, friend_id0);
    tx1 = args[0]*__shfl(threadInput2, friend_id1);
    ty1 = args[0]*__shfl(threadInput3, friend_id1);
    tx2 = args[0]*__shfl(threadInput4, friend_id2);
    ty2 = args[0]*__shfl(threadInput5, friend_id2);
    tx3 = args[0]*__shfl(threadInput5, friend_id3);
    ty3 = args[0]*__shfl(threadInput6, friend_id3);
    tz3 = args[0]*__shfl(threadInput7, friend_id3);
    rx0 = args[0]*__shfl(threadInput7, friend_id4);
    ry0 = args[0]*__shfl(threadInput8, friend_id4);
    rz0 = args[0]*__shfl(threadInput9, friend_id4);
    rx1 = args[0]*__shfl(threadInput9, friend_id5);
    ry1 = args[0]*__shfl(threadInput10, friend_id5);
    rx2 = args[0]*__shfl(threadInput11, friend_id6);
    ry2 = args[0]*__shfl(threadInput12, friend_id6);
    rx3 = args[0]*__shfl(threadInput13, friend_id7);
    ry3 = args[0]*__shfl(threadInput14, friend_id7);

    sum0 += (lane_id < 17)? tx0: ty0;
    sum1 += (lane_id < 21)? tx1: ty1;
    sum2 += (lane_id < 24)? tx2: ty2;
    sum3 += (lane_id < 1 )? tx3: ((lane_id < 27)? ty3: tz3);
    sum4 += (lane_id < 5 )? rx0: ((lane_id < 31)? ry0: rz0);
    sum5 += (lane_id < 8 )? rx1: ry1;
    sum6 += (lane_id < 11)? rx2: ry2;
    sum7 += (lane_id < 15)? rx3: ry3;

    friend_id0 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[1]*__shfl(threadInput1, friend_id0);
    ty0 = args[1]*__shfl(threadInput2, friend_id0);
    tz0 = args[1]*__shfl(threadInput3, friend_id0);
    tx1 = args[1]*__shfl(threadInput3, friend_id1);
    ty1 = args[1]*__shfl(threadInput4, friend_id1);
    tx2 = args[1]*__shfl(threadInput5, friend_id2);
    ty2 = args[1]*__shfl(threadInput6, friend_id2);
    tx3 = args[1]*__shfl(threadInput7, friend_id3);
    ty3 = args[1]*__shfl(threadInput8, friend_id3);
    rx0 = args[1]*__shfl(threadInput9, friend_id4);
    ry0 = args[1]*__shfl(threadInput10, friend_id4);
    rx1 = args[1]*__shfl(threadInput11, friend_id5);
    ry1 = args[1]*__shfl(threadInput12, friend_id5);
    rx2 = args[1]*__shfl(threadInput13, friend_id6);
    ry2 = args[1]*__shfl(threadInput14, friend_id6);
    rx3 = args[1]*__shfl(threadInput15, friend_id7);
    ry3 = args[1]*__shfl(threadInput16, friend_id7);
    sum0 += (lane_id < 3 )? tx0: ((lane_id < 29)? ty0: tz0);
    sum1 += (lane_id < 7 )? tx1: ty1;
    sum2 += (lane_id < 9 )? tx2: ty2;
    sum3 += (lane_id < 13)? tx3: ty3;
    sum4 += (lane_id < 16)? rx0: ry0;
    sum5 += (lane_id < 19)? rx1: ry1;
    sum6 += (lane_id < 23)? rx2: ry2;
    sum7 += (lane_id < 25)? rx3: ry3;

    friend_id0 = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+22+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+18+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+14+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+10+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[2]*__shfl(threadInput2, friend_id0);
    ty0 = args[2]*__shfl(threadInput3, friend_id0);
    tx1 = args[2]*__shfl(threadInput4, friend_id1);
    ty1 = args[2]*__shfl(threadInput5, friend_id1);
    tx2 = args[2]*__shfl(threadInput5, friend_id2);
    ty2 = args[2]*__shfl(threadInput6, friend_id2);
    tz2 = args[2]*__shfl(threadInput7, friend_id2);
    tx3 = args[2]*__shfl(threadInput7, friend_id3);
    ty3 = args[2]*__shfl(threadInput8, friend_id3);
    rx0 = args[2]*__shfl(threadInput9, friend_id4);
    ry0 = args[2]*__shfl(threadInput10, friend_id4);
    rx1 = args[2]*__shfl(threadInput11, friend_id5);
    ry1 = args[2]*__shfl(threadInput12, friend_id5);
    rx2 = args[2]*__shfl(threadInput13, friend_id6);
    ry2 = args[2]*__shfl(threadInput14, friend_id6);
    rx3 = args[2]*__shfl(threadInput15, friend_id7);
    ry3 = args[2]*__shfl(threadInput16, friend_id7);
    sum0 += (lane_id < 22)? tx0: ty0;
    sum1 += (lane_id < 24)? tx1: ty1;
    sum2 += (lane_id < 2 )? tx2: ((lane_id < 28)? ty2: tz2);
    sum3 += (lane_id < 6 )? tx3: ty3;
    sum4 += (lane_id < 8 )? rx0: ry0;
    sum5 += (lane_id < 12)? rx1: ry1;
    sum6 += (lane_id < 16)? rx2: ry2;
    sum7 += (lane_id < 18)? rx3: ry3;

    friend_id0 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+23+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+15+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[3]*__shfl(threadInput2, friend_id0);
    ty0 = args[3]*__shfl(threadInput3, friend_id0);
    tx1 = args[3]*__shfl(threadInput4, friend_id1);
    ty1 = args[3]*__shfl(threadInput5, friend_id1);
    tx2 = args[3]*__shfl(threadInput5, friend_id2);
    ty2 = args[3]*__shfl(threadInput6, friend_id2);
    tz2 = args[3]*__shfl(threadInput7, friend_id2);
    tx3 = args[3]*__shfl(threadInput7, friend_id3);
    ty3 = args[3]*__shfl(threadInput8, friend_id3);
    tz3 = args[3]*__shfl(threadInput9, friend_id3);
    rx0 = args[3]*__shfl(threadInput9, friend_id4);
    ry0 = args[3]*__shfl(threadInput10, friend_id4);
    rx1 = args[3]*__shfl(threadInput11, friend_id5);
    ry1 = args[3]*__shfl(threadInput12, friend_id5);
    rx2 = args[3]*__shfl(threadInput13, friend_id6);
    ry2 = args[3]*__shfl(threadInput14, friend_id6);
    rx3 = args[3]*__shfl(threadInput15, friend_id7);
    ry3 = args[3]*__shfl(threadInput16, friend_id7);
    sum0 += (lane_id < 21)? tx0: ty0;
    sum1 += (lane_id < 24)? tx1: ty1;
    sum2 += (lane_id < 1 )? tx2: ((lane_id < 27)? ty2: tz2);
    sum3 += (lane_id < 5 )? tx3: ((lane_id < 31)? ty3: tz3);
    sum4 += (lane_id < 8 )? rx0: ry0;
    sum5 += (lane_id < 11)? rx1: ry1;
    sum6 += (lane_id < 15)? rx2: ry2;
    sum7 += (lane_id < 17)? rx3: ry3;

    friend_id0 = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+24+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+20+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+16+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+12+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[4]*__shfl(threadInput2, friend_id0);
    ty0 = args[4]*__shfl(threadInput3, friend_id0);
    tx1 = args[4]*__shfl(threadInput4, friend_id1);
    ty1 = args[4]*__shfl(threadInput5, friend_id1);
    tx2 = args[4]*__shfl(threadInput6, friend_id2);
    ty2 = args[4]*__shfl(threadInput7, friend_id2);
    tx3 = args[4]*__shfl(threadInput7, friend_id3);
    ty3 = args[4]*__shfl(threadInput8, friend_id3);
    tz3 = args[4]*__shfl(threadInput9, friend_id3);
    rx0 = args[4]*__shfl(threadInput9, friend_id4);
    ry0 = args[4]*__shfl(threadInput10, friend_id4);
    rx1 = args[4]*__shfl(threadInput11, friend_id5);
    ry1 = args[4]*__shfl(threadInput12, friend_id5);
    rx2 = args[4]*__shfl(threadInput13, friend_id6);
    ry2 = args[4]*__shfl(threadInput14, friend_id6);
    rx3 = args[4]*__shfl(threadInput15, friend_id7);
    ry3 = args[4]*__shfl(threadInput16, friend_id7);
    sum0 += (lane_id < 20)? tx0: ty0;
    sum1 += (lane_id < 24)? tx1: ty1;
    sum2 += (lane_id < 26)? tx2: ty2;
    sum3 += (lane_id < 4 )? tx3: ((lane_id < 30)? ty3: tz3);
    sum4 += (lane_id < 8 )? rx0: ry0;
    sum5 += (lane_id < 10)? rx1: ry1;
    sum6 += (lane_id < 14)? rx2: ry2;
    sum7 += (lane_id < 16)? rx3: ry3;

    friend_id0 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+1 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[5]*__shfl(threadInput2, friend_id0);
    ty0 = args[5]*__shfl(threadInput3, friend_id0);
    tx1 = args[5]*__shfl(threadInput4, friend_id1);
    ty1 = args[5]*__shfl(threadInput5, friend_id1);
    tx2 = args[5]*__shfl(threadInput6, friend_id2);
    ty2 = args[5]*__shfl(threadInput7, friend_id2);
    tx3 = args[5]*__shfl(threadInput8, friend_id3);
    ty3 = args[5]*__shfl(threadInput9, friend_id3);
    rx0 = args[5]*__shfl(threadInput10, friend_id4);
    ry0 = args[5]*__shfl(threadInput11, friend_id4);
    rx1 = args[5]*__shfl(threadInput11, friend_id5);
    ry1 = args[5]*__shfl(threadInput12, friend_id5);
    rz1 = args[5]*__shfl(threadInput13, friend_id5);
    rx2 = args[5]*__shfl(threadInput13, friend_id6);
    ry2 = args[5]*__shfl(threadInput14, friend_id6);
    rx3 = args[5]*__shfl(threadInput15, friend_id7);
    ry3 = args[5]*__shfl(threadInput16, friend_id7);
    sum0 += (lane_id < 13)? tx0: ty0;
    sum1 += (lane_id < 16)? tx1: ty1;
    sum2 += (lane_id < 19)? tx2: ty2;
    sum3 += (lane_id < 23)? tx3: ty3;
    sum4 += (lane_id < 25)? rx0: ry0;
    sum5 += (lane_id < 3 )? rx1: ((lane_id < 29)? ry1: rz1);
    sum6 += (lane_id < 7 )? rx2: ry2;
    sum7 += (lane_id < 9 )? rx3: ry3;

    friend_id0 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
    friend_id1 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
    friend_id2 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
    friend_id3 = (lane_id+23+((lane_id>>3)<<1))&(warpSize-1);
    friend_id4 = (lane_id+19+((lane_id>>3)<<1))&(warpSize-1);
    friend_id5 = (lane_id+15+((lane_id>>3)<<1))&(warpSize-1);
    friend_id6 = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
    friend_id7 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
    tx0 = args[6]*__shfl(threadInput4 , friend_id0);
    ty0 = args[6]*__shfl(threadInput5 , friend_id0);
    tx1 = args[6]*__shfl(threadInput5 , friend_id1);
    ty1 = args[6]*__shfl(threadInput6 , friend_id1);
    tz1 = args[6]*__shfl(threadInput7 , friend_id1);
    tx2 = args[6]*__shfl(threadInput7 , friend_id2);
    ty2 = args[6]*__shfl(threadInput8 , friend_id2);
    tz2 = args[6]*__shfl(threadInput9 , friend_id2);
    tx3 = args[6]*__shfl(threadInput9 , friend_id3);
    ty3 = args[6]*__shfl(threadInput10, friend_id3);
    rx0 = args[6]*__shfl(threadInput11, friend_id4);
    ry0 = args[6]*__shfl(threadInput12, friend_id4);
    rx1 = args[6]*__shfl(threadInput13, friend_id5);
    ry1 = args[6]*__shfl(threadInput14, friend_id5);
    rx2 = args[6]*__shfl(threadInput15, friend_id6);
    ry2 = args[6]*__shfl(threadInput16, friend_id6);
    rx3 = args[6]*__shfl(threadInput17, friend_id7);
    ry3 = args[6]*__shfl(threadInput18, friend_id7);
    sum0 += (lane_id < 24)? tx0: ty0;
    sum1 += (lane_id < 1 )? tx1: ((lane_id < 27)? ty1: tz1);
    sum2 += (lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2);
    sum3 += (lane_id < 8 )? tx3: ty3;
    sum4 += (lane_id < 11)? rx0: ry0;
    sum5 += (lane_id < 15)? rx1: ry1;
    sum6 += (lane_id < 17)? rx2: ry2;
    sum7 += (lane_id < 21)? rx3: ry3;
    /*
    */

    if(k < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k,j,i) = sum0;
    }
    if(k+1 < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k+1,j,i) = sum1;
    }
    if(k+2 < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k+2,j,i) = sum2;
    }
    if(k+3 < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k+3,j,i) = sum3;
    }
    if(k+4 < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k+4,j,i) = sum4;
    }
    if(k+5 < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k+5,j,i) = sum5;
    }
    if(k+6 < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k+6,j,i) = sum6;
    }
    if(k+7 < z + halo && j < m + halo && i < n + halo)
    {
        OUT_3D(k+7,j,i) = sum7;
    }
}

int main(int argc, char **argv)
{

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    float time_wo_pci;

    DATA_TYPE *in_d;
    DATA_TYPE *args_d;
    DATA_TYPE *out_d;
    DATA_TYPE *out = new DATA_TYPE[total];
    cudaMalloc((void**)&in_d, total*sizeof(DATA_TYPE));
    cudaMalloc((void**)&out_d, total*sizeof(DATA_TYPE));
    cudaMalloc((void**)&args_d, (K)*sizeof(DATA_TYPE));
    cudaMemcpy(in_d, in, total*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(args_d, args, (K)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid((n+7)/8, (m+3)/4, (z+7)/8);
    dim3 dimBlock(8, 4, 8);
    cudaEventRecord(start);
#ifdef SPAC1
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda<<<dimGrid, dimBlock>>>(in_d, out_d, args_d, z, m, n, halo); 
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
#endif
    cudaEventRecord(stop);

    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_3D(out, z, m, n, halo);
    // Show_Me(out, z, m, n, halo, "Output(Cuda):");

    cout << "Verify Cuda: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, 13, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(z, m, n, ITER, time_wo_pci));

    Init_Input_3D(out, z, m, n, halo);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid1((n+7)/8, (m+3)/4, (z+7)/8);
    dim3 dimBlock1(8, 4, 8);
    cudaEventRecord(start);
#ifdef SPAC1
    for(int i =0; i< ITER; i++)
    {
        Stencil_Cuda_Sm<<<dimGrid1, dimBlock1, ((8+2*halo)*(4+2*halo)*(8+2*halo))*sizeof(DATA_TYPE)>>>(
            in_d, out_d, args_d, z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
#endif
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_3D(out, z, m, n, halo);
    cout << "Verify Cuda_Sm: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Sm Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, 13, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(z, m, n, ITER, time_wo_pci));

    Init_Input_3D(out, z, m, n, halo);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid2((n+7)/8, (m+3)/4, (z+7)/8);
    dim3 dimBlock2(8, 4, 8);
    cudaEventRecord(start);
#ifdef SPAC
    Stencil_Cuda_Shfl<<<dimGrid2, dimBlock2>>>(in_d, out_d, args_d, z, m, n, halo);
#endif
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_3D(out, z, m, n, halo);
    cout << "Verify Cuda_Shfl: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Shfl Time: " << milliseconds << endl;
    // Show_Me(out, z, m, n, halo, "Output(Shfl):");

    Init_Input_3D(out, z, m, n, halo);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid3((n+7)/8, (m+3)/4, (((z+1)/2)+7)/8);
    dim3 dimBlock3(8, 4, 8);
    cudaEventRecord(start);
#ifdef SPAC
    Stencil_Cuda_Shfl2<<<dimGrid3, dimBlock3>>>(in_d, out_d, args_d, z, m, n, halo);
#endif
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_3D(out, z, m, n, halo);
    cout << "Verify Cuda_Shfl2: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Shfl2 Time: " << milliseconds << endl;
    // Show_Me(out, z, m, n, halo, "Output(Shfl2):");

    Init_Input_3D(out, z, m, n, halo);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid4((n+7)/8, (m+3)/4, (((z+3)/4)+7)/8);
    dim3 dimBlock4(8, 4, 8);
    cudaEventRecord(start);
#ifdef SPAC
    Stencil_Cuda_Shfl4<<<dimGrid4, dimBlock4>>>(in_d, out_d, args_d, z, m, n, halo);
#endif
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_3D(out, z, m, n, halo);
    cout << "Verify Cuda_Shfl4: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Shfl4 Time: " << milliseconds << endl;
    // Show_Me(out, z, m, n, halo, "Output(Shfl4):");

    Init_Input_3D(out, z, m, n, halo);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid5((n+7)/8, (m+3)/4, (((z+7)/8)+7)/8);
    dim3 dimBlock5(8, 4, 8);
    cudaEventRecord(start);
#ifdef SPAC
    Stencil_Cuda_Shfl8<<<dimGrid5, dimBlock5>>>(in_d, out_d, args_d, z, m, n, halo);
#endif
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_3D(out, z, m, n, halo);
    cout << "Verify Cuda_Shfl8: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Shfl8 Time: " << milliseconds << endl;
    // time_wo_pci = milliseconds * 1.0e-03;
    // printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, 1, 13, time_wo_pci));
    // printf("Throughput   : %.3f (GB/s)\n", GetThroughput(z, m, n, 1, time_wo_pci));
    // Show_Me(out, z, m, n, halo, "Output(Shfl8):");

    Init_Input_3D(out, z, m, n, halo);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid6((n+63)/64, (m+3)/4, 4);
    dim3 dimBlock6(64, 4, 1);
    // dim3 dimGrid6((n+15)/16, (m+15)/16, 4);
    // dim3 dimBlock6(16, 16, 1);
    cudaEventRecord(start);
#ifdef SPAC1
    for(int i =0; i< ITER; i++)
    {
    Stencil_Cuda_Sweep<<<dimGrid6, dimBlock6>>>(in_d, out_d, args_d, z, m, n, halo);
        swap(in_d, out_d);
    }
    swap(in_d, out_d);
#endif
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_3D(out, z, m, n, halo);
    cout << "Verify Cuda_Sweep: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Sweep Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, ITER, 13, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(z, m, n, ITER, time_wo_pci));
    // Show_Me(out, z, m, n, halo, "Output(Sweep):");

    Init_Input_3D(out, z, m, n, halo);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid7((n+7)/8, (m+3)/4, (((z+7)/8)+7)/8);
    // cout << "zblock = " << (((z+7)/8)+7)/8 << endl;
    dim3 dimBlock7(8, 4, 8);
    cudaEventRecord(start);
#ifdef SPAC
    Stencil_Cuda_Sm8<<<dimGrid7, dimBlock7, ((8+2*halo)*(4+2*halo)*(8+2*halo)*sizeof(DATA_TYPE))>>>(in_d, out_d, args_d, z, m, n, halo);
#endif
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_3D(out, z, m, n, halo);
    cout << "Verify Cuda_Sm8: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Sm8 Time: " << milliseconds << endl;
    // Show_Me(out, z, m, n, halo, "Output(Sm8):");

    Init_Input_3D(out, z, m, n, halo);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid8((n+63)/64, (m+3)/4, 4);
    dim3 dimBlock8(64, 4, 1);
    cudaEventRecord(start);
#ifdef SPAC
    Stencil_Cuda_Sweep_Sm<<<dimGrid8, dimBlock8, ((64+2*halo)*(4+2*halo)*sizeof(DATA_TYPE))>>>(in_d, out_d, args_d, z, m, n, halo);
#endif
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_3D(out, z, m, n, halo);
    cout << "Verify Cuda_Sweep_Sm: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Sweep_Sm Time: " << milliseconds << endl;
    // Show_Me(out, z, m, n, halo, "Output(Sweep_Sm):");

    Init_Input_3D(out, z, m, n, halo);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid9((n+7)/8, (m+31)/32, 4);
    dim3 dimBlock9(8, 32, 1);
    cudaEventRecord(start);
#ifdef SPAC
    Stencil_Cuda_Sweep_Shfl<<<dimGrid9, dimBlock9>>>(in_d, out_d, args_d, z, m, n, halo);
#endif
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_3D(out, z, m, n, halo);
    cout << "Verify Cuda_Sweep_Shfl: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Sweep_Shfl Time: " << milliseconds << endl;
    // time_wo_pci = milliseconds * 1.0e-03;
    // printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, 1, 13, time_wo_pci));
    // printf("Throughput   : %.3f (GB/s)\n", GetThroughput(z, m, n, 1, time_wo_pci));
    // Show_Me(out, z, m, n, halo, "Output(Sweep_Shfl):");
 
    Init_Input_3D(out, z, m, n, halo);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid10((n+7)/8, ((m+1)/2+31)/32, 4);
    dim3 dimBlock10(8, 32, 1);
    cudaEventRecord(start);
#ifdef SPAC
    Stencil_Cuda_Sweep_Shfl2<<<dimGrid10, dimBlock10>>>(in_d, out_d, args_d, z, m, n, halo);
#endif
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_3D(out, z, m, n, halo);
    cout << "Verify Cuda_Sweep_Shfl2: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Sweep_Shfl2 Time: " << milliseconds << endl;
    // Show_Me(out, z, m, n, halo, "Output(Sweep_Shfl2):");
    // New codes add here:
    // Iteration codes:
    
    Init_Input_3D(in, z, m, n, halo);
    // Show_Me(in, z, m, n, halo, "Input:");
    for(int i=0; i < ITER; i++)
    {
        Stencil_Seq(in, out_ref, args, z, m, n, halo);
        swap(in, out_ref);
        // Show_Me(in, z, m, n, halo, "Output:");
    }
    swap(in, out_ref);
    // Show_Me(out_ref, z, m, n, halo, "Output:");

    Init_Input_3D(out, z, m, n, halo);
    cudaMemcpy(out_d, out, (total)*sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 dimGrid11((n+(64-2-1))/(64-2), (m+(4-2-1))/(4-2), 1);
    dim3 dimBlock11(64, 4, 1);
    // cout << "gridx = " << (n+(64-2-1))/(64-2) << endl;
    // cout << "gridy = " << (m+(4-2-1))/(4-2) << endl;
    cudaEventRecord(start);
    // do the computation twice in temporal manner
#ifdef TEMP
    Stencil_Cuda_Sweep_Temp_Sm<<<dimGrid11, dimBlock11, ((64)*(4)*(3)*sizeof(DATA_TYPE))>>>(in_d, out_d, args_d, z, m, n, halo);
#endif
    cudaEventRecord(stop);
    cudaMemcpy(out, out_d, (total)*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    Fill_Halo_3D(out, z, m, n, halo);
    cout << "Verify Cuda_Sweep_Temp_Sm: " << boolalpha << Verify(out, out_ref, total) << endl;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Cuda_Sweep_Temp_Sm Time: " << milliseconds << endl;
    time_wo_pci = milliseconds * 1.0e-03;
    printf("FLOPS        : %.3f (GFLOPS)\n", GetGFLOPS(z, m, n, 2, 13, time_wo_pci));
    printf("Throughput   : %.3f (GB/s)\n", GetThroughput(z, m, n, 2, time_wo_pci));
    // Show_Me(out, z, m, n, halo, "Output(Sweep_Temp_Sm):");

    cudaFree(in_d);
    cudaFree(out_d);


    delete[] in;
    delete[] out;
    delete[] out_ref;
}
