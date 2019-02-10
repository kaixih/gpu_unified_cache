#ifndef _CUDA_UBUF_3D7
#define _CUDA_UBUF_3D7

template<class T>
class StenBuffer
{
public:
    __device__ virtual T load(int idx, int idy, int idz)=0;
    __device__ virtual void store(T v, int idx, int idy, int idz)=0;
    __device__ virtual void glb2buf(T* in, int type=NONE)=0;
};

template<class T>
class L1Buffer: public StenBuffer<T>
{
private:
    T *l1m;
    int z, m, n;
    int halo;
public:
    int gidx, gidy, gidz;
    // int lidx, lidy, lidz;
    // int ldimx, ldimy, ldimz;
    // int bidx, bidy, bidz;
public:
    __device__ T load(int idx, int idy, int idz)
    {
        return ACC_3D(l1m, idx, idy, idz);
    }
    __device__ void store(T v, int idx, int idy, int idz)
    {
        ACC_3D(l1m, idx, idy, idz) = v;
    }
    __device__ L1Buffer(int k, int j, int i, int h): z(k), m(j), n(i), halo(h){
        gidx = threadIdx.x + blockIdx.x * blockDim.x + halo;
        gidy = threadIdx.y + blockIdx.y * blockDim.y + halo;
        gidz = threadIdx.z + blockIdx.z * blockDim.z + halo;
    }
    __device__ void glb2buf(T* in, int type=NONE)
    {
        l1m = in;
    }
    // __device__ T Sten_L1M_Fetch(int pz, int py, int px, int tc=0)
    // {
        // return load(gidz+pz, gidy+py, gidx+px);
    // }
};

// will slow down the performance? why? (Temporarily move the function to original constructor)
template<class T>
__device__ inline void Sten_L1M_glb2buf(L1Buffer<T> *buf, T* in, int type=NONE)
{
    buf->set(in);
    return;
}

template<class T>
__device__ inline T Sten_L1M_Fetch(L1Buffer<T> buf, int pz, int py, int px, int tc=0)
{
    return buf.load(buf.gidz+pz, buf.gidy+py, buf.gidx+px);
}

template<class T>
class LDSBuffer: public StenBuffer<T>
{
private:
    T *local; // lds memory pointer
    int z, m, n;
    int halo;
public:
    int gidx, gidy, gidz;
    int lidx, lidy, lidz;
    // int ldimx, ldimy, ldimz;
    // int bidx, bidy, bidz;
public:
    __device__ T load(int idx, int idy, int idz)
    {
        return LOC_3D(idx, idy, idz);
    }
    __device__ void store(T v, int idx, int idy, int idz)
    {
        LOC_3D(idx, idy, idz) = v;
    }
    __device__ LDSBuffer(int k, int j, int i, int h): z(k), m(j), n(i), halo(h){
        __shared__ T sm[10*6*10];
        local = sm;
        gidx = threadIdx.x + blockIdx.x * blockDim.x + halo;
        gidy = threadIdx.y + blockIdx.y * blockDim.y + halo;
        gidz = threadIdx.z + blockIdx.z * blockDim.z + halo;
        lidx = threadIdx.x + halo;
        lidy = threadIdx.y + halo;
        lidz = threadIdx.z + halo;
    }
    __device__ void glb2buf(T* in, int type=NONE)
    {
        if(type == BRANCH)
        {
            LOC_3D(lidz,lidy,lidx) = ACC_3D(in,gidz,gidy,gidx);

            if(lidx == halo) LOC_3D(lidz  ,lidy  ,lidx-1) = ACC_3D(in,gidz  ,gidy  ,gidx-1);
            if(lidx == 8   ) LOC_3D(lidz  ,lidy  ,lidx+1) = ACC_3D(in,gidz  ,gidy  ,gidx+1);
            if(lidy == halo) LOC_3D(lidz  ,lidy-1,lidx  ) = ACC_3D(in,gidz  ,gidy-1,gidx  );
            if(lidy == 4   ) LOC_3D(lidz  ,lidy+1,lidx  ) = ACC_3D(in,gidz  ,gidy+1,gidx  );
            if(lidz == halo) LOC_3D(lidz-1,lidy  ,lidx  ) = ACC_3D(in,gidz-1,gidy  ,gidx  );
            if(lidz == 8   ) LOC_3D(lidz+1,lidy  ,lidx  ) = ACC_3D(in,gidz+1,gidy  ,gidx  );
        } else if(type == CYCLIC)
        {
            int lane_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
            int blk_id_x = blockIdx.x;
            int blk_id_y = blockIdx.y;
            int blk_id_z = blockIdx.z;
            int new_i, new_j, new_k, new_li, new_lj, new_lk;
            new_i  = (blk_id_x<<3) + lane_id%10 ;     
            new_j  = (blk_id_y<<2) + (lane_id/10)%6 ; 
            new_k  = (blk_id_z<<3) + lane_id/60 ;     
            new_li = lane_id%10;
            new_lj = (lane_id/10)%6;
            new_lk = lane_id/60;
            LOC_3D(new_lk,new_lj,new_li) = ACC_3D(in,new_k,new_j,new_i);
            new_i  = (blk_id_x<<3) + (lane_id+256)%10 ;
            new_j  = (blk_id_y<<2) + ((lane_id+256)/10)%6 ;
            new_k  = (blk_id_z<<3) + (lane_id+256)/60 ;
            new_li = (lane_id+256)%10;
            new_lj = ((lane_id+256)/10)%6;
            new_lk = (lane_id+256)/60; 
            LOC_3D(new_lk,new_lj,new_li) = ACC_3D(in,new_k,new_j,new_i);
            new_i  = (blk_id_x<<3) + (lane_id+512)%10 ;
            new_j  = (blk_id_y<<2) + ((lane_id+512)/10)%6 ;
            new_k  = (blk_id_z<<3) + (lane_id+512)/60 ;
            new_li = (lane_id+512)%10;
            new_lj = ((lane_id+512)/10)%6;
            new_lk = (lane_id+512)/60; 
            if(new_li < 10 &&  new_lj < 6 && new_lk < 10 )
                LOC_3D(new_lk,new_lj,new_li) = ACC_3D(in,new_k,new_j,new_i);

        
        }
        __syncthreads();
    }
    __device__ T Sten_LDS_Fetch(int pz, int py, int px, int tc=0)
    {
        return load(lidz+pz, lidy+py, lidx+px);
    }
};

template<class T>
__device__ inline T Sten_LDS_Fetch(LDSBuffer<T> buf, int pz, int py, int px, int tc=0)
{
    return buf.load(buf.lidz+pz, buf.lidy+py, buf.lidx+px);
}

template<class T>
class REGBuffer: public StenBuffer<T>
{
private:
    int z, m, n;
    int halo;
public:
    T reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10;
    int gidx, gidy, gidz;
    int lane_id;
    int tc;
    // int lidx, lidy, lidz;
    // int ldimx, ldimy, ldimz;
    // int bidx, bidy, bidz;
public:
    __device__ T load(int idx, int idy, int idz) // deprecated
    {
        return 0;
    }
    __device__ void store(T v, int idx, int idy, int idz) // deprecated
    {}

    __device__ REGBuffer(int k, int j, int i, int h, int c=1): 
        z(k), m(j), n(i), halo(h), tc(c)
    {
        const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
        lane_id = tid % warpSize;
        switch(tc)
        {
            case(1):
            {
                gidx = threadIdx.x + blockIdx.x * blockDim.x + halo;
                gidy = threadIdx.y + blockIdx.y * blockDim.y + halo;
                gidz = threadIdx.z + blockIdx.z * blockDim.z + halo;
                break;
            }
            case(2):
            {
                gidx = threadIdx.x + blockIdx.x * blockDim.x + halo;
                gidy = threadIdx.y + blockIdx.y * blockDim.y + halo;
                gidz = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<1) + (lane_id>>5) + halo; 
                break;
            }
            case(4):
            {
                gidx = threadIdx.x + blockIdx.x * blockDim.x + halo;
                gidy = threadIdx.y + blockIdx.y * blockDim.y + halo;
                gidz = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<2) + (lane_id>>5) + halo; 
                break;
            }
        }
    }

    __device__ void glb2buf(T* in, int type=CYCLIC)
    {
        if(type == CYCLIC)
        {
            switch (tc)
            {
                case 1:
                {
                    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
                    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
                    int warp_id_z = (threadIdx.z + blockIdx.z * blockDim.z)>>0; // there numbers
                    int new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
                    int new_j = (warp_id_y<<2) + (lane_id/10)%6; // 6  is extended dimension of j 
                    int new_k = (warp_id_z<<0) + lane_id/60;     // 60 is extended area of ixj = 10x6
                    reg0 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+32)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+32)/60;
                    reg1 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+64)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+64)/60;
                    reg2 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+96)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+96)/60;
                    reg3 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+128)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+128)/60;
                    reg4 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+160)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+160)/60;
                    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
                    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
                    new_k = (new_k < z+2*halo)? new_k: z+2*halo-1;
                    reg5 = ACC_3D(in,new_k, new_j, new_i);
                    return ;
                }
                case 2:
                {
                    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
                    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
                    int warp_id_z = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<1) + (lane_id>>5); // these numbers
                    int new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
                    int new_j = (warp_id_y<<2) + (lane_id/10)%6; // 6  is extended dimension of j 
                    int new_k = (warp_id_z<<0) + lane_id/60;     // 60 is extended area of ixj = 10x6
                    reg0 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+32)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+32)/60;
                    reg1 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+64)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+64)/60;
                    reg2 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+96)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+96)/60;
                    reg3 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+128)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+128)/60;
                    reg4 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+160)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+160)/60;
                    reg5 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+192)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+192)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+192)/60;
                    reg6 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+224)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+224)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+224)/60;
                    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
                    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
                    new_k = (new_k < z+2*halo)? new_k: z+2*halo-1;
                    reg7 = ACC_3D(in,new_k, new_j, new_i);
                    return;
                }
                case 4:
                {
                    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>3; // because the warp dimensions are 
                    int warp_id_y = (threadIdx.y + blockIdx.y * blockDim.y)>>2; // 1x4x8, warp_ids are division of 
                    int warp_id_z = (((threadIdx.z + blockIdx.z * blockDim.z)>>0)<<2) + (lane_id>>5); // these numbers
                    int new_i = (warp_id_x<<3) + lane_id%10;     // 10 is extended dimension of i
                    int new_j = (warp_id_y<<2) + (lane_id/10)%6; // 6  is extended dimension of j 
                    int new_k = (warp_id_z<<0) + lane_id/60;     // 60 is extended area of ixj = 10x6
                    reg0 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+32)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+32)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+32)/60;
                    reg1 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+64)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+64)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+64)/60;
                    reg2 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+96)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+96)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+96)/60;
                    reg3 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+128)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+128)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+128)/60;
                    reg4 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+160)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+160)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+160)/60;
                    reg5 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+192)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+192)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+192)/60;
                    reg6 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+224)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+224)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+224)/60;
                    reg7 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+256)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+256)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+256)/60;
                    reg8 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+288)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+288)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+288)/60;
                    reg9 = ACC_3D(in,new_k, new_j, new_i);
                    new_i = (warp_id_x<<3) + (lane_id+320)%10;
                    new_j = (warp_id_y<<2) + ((lane_id+320)/10)%6;
                    new_k = (warp_id_z<<0) + (lane_id+320)/60;
                    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
                    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
                    new_k = (new_k < z+2*halo)? new_k: z+2*halo-1;
                    reg10 = ACC_3D(in,new_k, new_j, new_i);
                }
            }
        }
    }
    __device__ T Sten_REG_Fetch(int pz, int py, int px, int tc=0)
    {
        int friend_id;
        int friend_id1;
        int friend_id2;
        int friend_id3;
        T tx, ty, tz;
        T tx1, ty1, tz1;
        T tx2, ty2, tz2;
        T tx3, ty3, tz3;
        if(pz == -1 && py == 0 && px == 0)
        {
            switch (tc)
            {
                case 0:
                    friend_id = (lane_id+11+((lane_id>>3)<<1))&(warpSize-1);
                    tx = __shfl(reg0, friend_id);
                    ty = __shfl(reg1, friend_id);
                    return ((lane_id < 17)? tx: ty);
                case 1:
                    friend_id1 = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
                    tx1 = __shfl(reg2, friend_id1);
                    ty1 = __shfl(reg3, friend_id1);
                    return ((lane_id < 21)? tx1: ty1);
                case 2:
                    friend_id2 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
                    tx2 = __shfl(reg4, friend_id2);
                    ty2 = __shfl(reg5, friend_id2);
                    return ((lane_id < 24)? tx2: ty2);
                case 3:
                    friend_id3 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
                    tx3 = __shfl(reg5, friend_id3);
                    ty3 = __shfl(reg6, friend_id3);
                    tz3 = __shfl(reg7, friend_id3);
                    return ((lane_id < 1 )? tx3: ((lane_id < 27)? ty3: tz3));
            }
        } 
        if(pz == 0 && py == -1 && px == 0)
        {
            switch (tc)
            {
                case 0:
                    friend_id = (lane_id+29+((lane_id>>3)<<1))&(warpSize-1);
                    tx = __shfl(reg1, friend_id);
                    ty = __shfl(reg2, friend_id);
                    tz = __shfl(reg3, friend_id);
                    return ((lane_id < 3 )? tx: ((lane_id < 29)? ty: tz));
                case 1:
                    friend_id1 = (lane_id+25+((lane_id>>3)<<1))&(warpSize-1);
                    tx1 = __shfl(reg3, friend_id1);
                    ty1 = __shfl(reg4, friend_id1);
                    return ((lane_id < 7 )? tx1: ty1);
                case 2:
                    friend_id2 = (lane_id+21+((lane_id>>3)<<1))&(warpSize-1);
                    tx2 = __shfl(reg5, friend_id2);
                    ty2 = __shfl(reg6, friend_id2);
                    return ((lane_id < 9 )? tx2: ty2);
                case 3:
                    friend_id3 = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
                    tx3 = __shfl(reg7, friend_id3);
                    ty3 = __shfl(reg8, friend_id3);
                    return ((lane_id < 13)? tx3: ty3);
            }
        }
        if(pz == 0 && py == 0 && px == -1)
        {
            switch (tc)
            {
                case 0:
                    friend_id = (lane_id+6 +((lane_id>>3)<<1))&(warpSize-1);
                    tx = __shfl(reg2, friend_id);
                    ty = __shfl(reg3, friend_id);
                    return ((lane_id < 22)? tx: ty);
                case 1:
                    friend_id1 = (lane_id+2 +((lane_id>>3)<<1))&(warpSize-1);
                    tx1 = __shfl(reg4, friend_id1);
                    ty1 = __shfl(reg5, friend_id1);
                    return ((lane_id < 24)? tx1: ty1);
                case 2:
                    friend_id2 = (lane_id+30+((lane_id>>3)<<1))&(warpSize-1);
                    tx2 = __shfl(reg5, friend_id2);
                    ty2 = __shfl(reg6, friend_id2);
                    tz2 = __shfl(reg7, friend_id2);
                    return ((lane_id < 2 )? tx2: ((lane_id < 28)? ty2: tz2));
                case 3:
                    friend_id3 = (lane_id+26+((lane_id>>3)<<1))&(warpSize-1);
                    tx3 = __shfl(reg7, friend_id3);
                    ty3 = __shfl(reg8, friend_id3);
                    return ((lane_id < 6 )? tx3: ty3);
            }
        }
        if(pz == 0 && py == 0 && px == 0)
        {
            switch (tc)
            {
                case 0:
                    friend_id = (lane_id+7 +((lane_id>>3)<<1))&(warpSize-1);
                    tx = __shfl(reg2, friend_id);
                    ty = __shfl(reg3, friend_id);
                    return ((lane_id < 21)? tx: ty);
                case 1:
                    friend_id1 = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
                    tx1 = __shfl(reg4, friend_id1);
                    ty1 = __shfl(reg5, friend_id1);
                    return ((lane_id < 24)? tx1: ty1);
                case 2:
                    friend_id2 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
                    tx2 = __shfl(reg5, friend_id2);
                    ty2 = __shfl(reg6, friend_id2);
                    tz2 = __shfl(reg7, friend_id2);
                    return ((lane_id < 1 )? tx2: ((lane_id < 27)? ty2: tz2));
                case 3:
                    friend_id3 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
                    tx3 = __shfl(reg7, friend_id3);
                    ty3 = __shfl(reg8, friend_id3);
                    tz3 = __shfl(reg9, friend_id3);
                    return ((lane_id < 5 )? tx3: ((lane_id < 31)? ty3: tz3));
            }
        }
        if(pz == 0 && py == 0 && px == 1)
        {
            switch (tc)
            {
                case 0:
                    friend_id = (lane_id+8 +((lane_id>>3)<<1))&(warpSize-1);
                    tx = __shfl(reg2, friend_id);
                    ty = __shfl(reg3, friend_id);
                    return ((lane_id < 20)? tx: ty);
                case 1:
                    friend_id1 = (lane_id+4 +((lane_id>>3)<<1))&(warpSize-1);
                    tx1 = __shfl(reg4, friend_id1);
                    ty1 = __shfl(reg5, friend_id1);
                    return ((lane_id < 24)? tx1: ty1);
                case 2:
                    friend_id2 = (lane_id+0 +((lane_id>>3)<<1))&(warpSize-1);
                    tx2 = __shfl(reg6, friend_id2);
                    ty2 = __shfl(reg7, friend_id2);
                    return ((lane_id < 26)? tx2: ty2);
                case 3:
                    friend_id3 = (lane_id+28+((lane_id>>3)<<1))&(warpSize-1);
                    tx3 = __shfl(reg7, friend_id3);
                    ty3 = __shfl(reg8, friend_id3);
                    tz3 = __shfl(reg9, friend_id3);
                    return ((lane_id < 4 )? tx3: ((lane_id < 30)? ty3: tz3));
            }
        }
        if(pz == 0 && py == 1 && px == 0)
        {
            switch (tc)
            {
                case 0:
                    friend_id = (lane_id+17+((lane_id>>3)<<1))&(warpSize-1);
                    tx = __shfl(reg2, friend_id);
                    ty = __shfl(reg3, friend_id);
                    return ((lane_id < 13)? tx: ty);
                case 1:
                    friend_id1 = (lane_id+13+((lane_id>>3)<<1))&(warpSize-1);
                    tx1 = __shfl(reg4, friend_id1);
                    ty1 = __shfl(reg5, friend_id1);
                    return ((lane_id < 16)? tx1: ty1);
                case 2:
                    friend_id2 = (lane_id+9 +((lane_id>>3)<<1))&(warpSize-1);
                    tx2 = __shfl(reg6, friend_id2);
                    ty2 = __shfl(reg7, friend_id2);
                    return ((lane_id < 19)? tx2: ty2);
                case 3:
                    friend_id3 = (lane_id+5 +((lane_id>>3)<<1))&(warpSize-1);
                    tx3 = __shfl(reg8, friend_id3);
                    ty3 = __shfl(reg9, friend_id3);
                    return ((lane_id < 23)? tx3: ty3);
            }
        }
        if(pz == 1 && py == 0 && px == 0)
        {
            switch (tc)
            {
                case 0:
                    friend_id = (lane_id+3 +((lane_id>>3)<<1))&(warpSize-1);
                    tx = __shfl(reg4, friend_id);
                    ty = __shfl(reg5, friend_id);
                    return ((lane_id < 24)? tx: ty);
                case 1:
                    friend_id1 = (lane_id+31+((lane_id>>3)<<1))&(warpSize-1);
                    tx1 = __shfl(reg5, friend_id1);
                    ty1 = __shfl(reg6, friend_id1);
                    tz1 = __shfl(reg7, friend_id1);
                    return ((lane_id < 1 )? tx1: ((lane_id < 27)? ty1: tz1));
                case 2:
                    friend_id2 = (lane_id+27+((lane_id>>3)<<1))&(warpSize-1);
                    tx2 = __shfl(reg7 , friend_id2);
                    ty2 = __shfl(reg8 , friend_id2);
                    tz2 = __shfl(reg9 , friend_id2);
                    return ((lane_id < 5 )? tx2: ((lane_id < 31)? ty2: tz2));
                case 3:
                    friend_id3 = (lane_id+23+((lane_id>>3)<<1))&(warpSize-1);
                    tx3 = __shfl(reg9 , friend_id3);
                    ty3 = __shfl(reg10, friend_id3);
                    return ((lane_id < 8 )? tx3: ty3);
            }
        }
        return 0;

        
    }
};

template<class T>
__device__ inline T Sten_REG_Fetch(REGBuffer<T> buf, int pz, int py, int px, int tc=0)
{
    int friend_id;
    int friend_id1;
    int friend_id2;
    int friend_id3;
    T tx, ty, tz;
    T tx1, ty1, tz1;
    T tx2, ty2, tz2;
    T tx3, ty3, tz3;
    if(pz == -1 && py == 0 && px == 0)
    {
        switch (tc)
        {
            case 0:
                friend_id = (buf.lane_id+11+((buf.lane_id>>3)<<1))&(warpSize-1);
                tx = __shfl(buf.reg0, friend_id);
                ty = __shfl(buf.reg1, friend_id);
                return ((buf.lane_id < 17)? tx: ty);
            case 1:
                friend_id1 = (buf.lane_id+7 +((buf.lane_id>>3)<<1))&(warpSize-1);
                tx1 = __shfl(buf.reg2, friend_id1);
                ty1 = __shfl(buf.reg3, friend_id1);
                return ((buf.lane_id < 21)? tx1: ty1);
            case 2:
                friend_id2 = (buf.lane_id+3 +((buf.lane_id>>3)<<1))&(warpSize-1);
                tx2 = __shfl(buf.reg4, friend_id2);
                ty2 = __shfl(buf.reg5, friend_id2);
                return ((buf.lane_id < 24)? tx2: ty2);
            case 3:
                friend_id3 = (buf.lane_id+31+((buf.lane_id>>3)<<1))&(warpSize-1);
                tx3 = __shfl(buf.reg5, friend_id3);
                ty3 = __shfl(buf.reg6, friend_id3);
                tz3 = __shfl(buf.reg7, friend_id3);
                return ((buf.lane_id < 1 )? tx3: ((buf.lane_id < 27)? ty3: tz3));
        }
    } 
    if(pz == 0 && py == -1 && px == 0)
    {
        switch (tc)
        {
            case 0:
                friend_id = (buf.lane_id+29+((buf.lane_id>>3)<<1))&(warpSize-1);
                tx = __shfl(buf.reg1, friend_id);
                ty = __shfl(buf.reg2, friend_id);
                tz = __shfl(buf.reg3, friend_id);
                return ((buf.lane_id < 3 )? tx: ((buf.lane_id < 29)? ty: tz));
            case 1:
                friend_id1 = (buf.lane_id+25+((buf.lane_id>>3)<<1))&(warpSize-1);
                tx1 = __shfl(buf.reg3, friend_id1);
                ty1 = __shfl(buf.reg4, friend_id1);
                return ((buf.lane_id < 7 )? tx1: ty1);
            case 2:
                friend_id2 = (buf.lane_id+21+((buf.lane_id>>3)<<1))&(warpSize-1);
                tx2 = __shfl(buf.reg5, friend_id2);
                ty2 = __shfl(buf.reg6, friend_id2);
                return ((buf.lane_id < 9 )? tx2: ty2);
            case 3:
                friend_id3 = (buf.lane_id+17+((buf.lane_id>>3)<<1))&(warpSize-1);
                tx3 = __shfl(buf.reg7, friend_id3);
                ty3 = __shfl(buf.reg8, friend_id3);
                return ((buf.lane_id < 13)? tx3: ty3);
        }
    }
    if(pz == 0 && py == 0 && px == -1)
    {
        switch (tc)
        {
            case 0:
                friend_id = (buf.lane_id+6 +((buf.lane_id>>3)<<1))&(warpSize-1);
                tx = __shfl(buf.reg2, friend_id);
                ty = __shfl(buf.reg3, friend_id);
                return ((buf.lane_id < 22)? tx: ty);
            case 1:
                friend_id1 = (buf.lane_id+2 +((buf.lane_id>>3)<<1))&(warpSize-1);
                tx1 = __shfl(buf.reg4, friend_id1);
                ty1 = __shfl(buf.reg5, friend_id1);
                return ((buf.lane_id < 24)? tx1: ty1);
            case 2:
                friend_id2 = (buf.lane_id+30+((buf.lane_id>>3)<<1))&(warpSize-1);
                tx2 = __shfl(buf.reg5, friend_id2);
                ty2 = __shfl(buf.reg6, friend_id2);
                tz2 = __shfl(buf.reg7, friend_id2);
                return ((buf.lane_id < 2 )? tx2: ((buf.lane_id < 28)? ty2: tz2));
            case 3:
                friend_id3 = (buf.lane_id+26+((buf.lane_id>>3)<<1))&(warpSize-1);
                tx3 = __shfl(buf.reg7, friend_id3);
                ty3 = __shfl(buf.reg8, friend_id3);
                return ((buf.lane_id < 6 )? tx3: ty3);
        }
    }
    if(pz == 0 && py == 0 && px == 0)
    {
        switch (tc)
        {
            case 0:
                friend_id = (buf.lane_id+7 +((buf.lane_id>>3)<<1))&(warpSize-1);
                tx = __shfl(buf.reg2, friend_id);
                ty = __shfl(buf.reg3, friend_id);
                return ((buf.lane_id < 21)? tx: ty);
            case 1:
                friend_id1 = (buf.lane_id+3 +((buf.lane_id>>3)<<1))&(warpSize-1);
                tx1 = __shfl(buf.reg4, friend_id1);
                ty1 = __shfl(buf.reg5, friend_id1);
                return ((buf.lane_id < 24)? tx1: ty1);
            case 2:
                friend_id2 = (buf.lane_id+31+((buf.lane_id>>3)<<1))&(warpSize-1);
                tx2 = __shfl(buf.reg5, friend_id2);
                ty2 = __shfl(buf.reg6, friend_id2);
                tz2 = __shfl(buf.reg7, friend_id2);
                return ((buf.lane_id < 1 )? tx2: ((buf.lane_id < 27)? ty2: tz2));
            case 3:
                friend_id3 = (buf.lane_id+27+((buf.lane_id>>3)<<1))&(warpSize-1);
                tx3 = __shfl(buf.reg7, friend_id3);
                ty3 = __shfl(buf.reg8, friend_id3);
                tz3 = __shfl(buf.reg9, friend_id3);
                return ((buf.lane_id < 5 )? tx3: ((buf.lane_id < 31)? ty3: tz3));
        }
    }
    if(pz == 0 && py == 0 && px == 1)
    {
        switch (tc)
        {
            case 0:
                friend_id = (buf.lane_id+8 +((buf.lane_id>>3)<<1))&(warpSize-1);
                tx = __shfl(buf.reg2, friend_id);
                ty = __shfl(buf.reg3, friend_id);
                return ((buf.lane_id < 20)? tx: ty);
            case 1:
                friend_id1 = (buf.lane_id+4 +((buf.lane_id>>3)<<1))&(warpSize-1);
                tx1 = __shfl(buf.reg4, friend_id1);
                ty1 = __shfl(buf.reg5, friend_id1);
                return ((buf.lane_id < 24)? tx1: ty1);
            case 2:
                friend_id2 = (buf.lane_id+0 +((buf.lane_id>>3)<<1))&(warpSize-1);
                tx2 = __shfl(buf.reg6, friend_id2);
                ty2 = __shfl(buf.reg7, friend_id2);
                return ((buf.lane_id < 26)? tx2: ty2);
            case 3:
                friend_id3 = (buf.lane_id+28+((buf.lane_id>>3)<<1))&(warpSize-1);
                tx3 = __shfl(buf.reg7, friend_id3);
                ty3 = __shfl(buf.reg8, friend_id3);
                tz3 = __shfl(buf.reg9, friend_id3);
                return ((buf.lane_id < 4 )? tx3: ((buf.lane_id < 30)? ty3: tz3));
        }
    }
    if(pz == 0 && py == 1 && px == 0)
    {
        switch (tc)
        {
            case 0:
                friend_id = (buf.lane_id+17+((buf.lane_id>>3)<<1))&(warpSize-1);
                tx = __shfl(buf.reg2, friend_id);
                ty = __shfl(buf.reg3, friend_id);
                return ((buf.lane_id < 13)? tx: ty);
            case 1:
                friend_id1 = (buf.lane_id+13+((buf.lane_id>>3)<<1))&(warpSize-1);
                tx1 = __shfl(buf.reg4, friend_id1);
                ty1 = __shfl(buf.reg5, friend_id1);
                return ((buf.lane_id < 16)? tx1: ty1);
            case 2:
                friend_id2 = (buf.lane_id+9 +((buf.lane_id>>3)<<1))&(warpSize-1);
                tx2 = __shfl(buf.reg6, friend_id2);
                ty2 = __shfl(buf.reg7, friend_id2);
                return ((buf.lane_id < 19)? tx2: ty2);
            case 3:
                friend_id3 = (buf.lane_id+5 +((buf.lane_id>>3)<<1))&(warpSize-1);
                tx3 = __shfl(buf.reg8, friend_id3);
                ty3 = __shfl(buf.reg9, friend_id3);
                return ((buf.lane_id < 23)? tx3: ty3);
        }
    }
    if(pz == 1 && py == 0 && px == 0)
    {
        switch (tc)
        {
            case 0:
                friend_id = (buf.lane_id+3 +((buf.lane_id>>3)<<1))&(warpSize-1);
                tx = __shfl(buf.reg4, friend_id);
                ty = __shfl(buf.reg5, friend_id);
                return ((buf.lane_id < 24)? tx: ty);
            case 1:
                friend_id1 = (buf.lane_id+31+((buf.lane_id>>3)<<1))&(warpSize-1);
                tx1 = __shfl(buf.reg5, friend_id1);
                ty1 = __shfl(buf.reg6, friend_id1);
                tz1 = __shfl(buf.reg7, friend_id1);
                return ((buf.lane_id < 1 )? tx1: ((buf.lane_id < 27)? ty1: tz1));
            case 2:
                friend_id2 = (buf.lane_id+27+((buf.lane_id>>3)<<1))&(warpSize-1);
                tx2 = __shfl(buf.reg7 , friend_id2);
                ty2 = __shfl(buf.reg8 , friend_id2);
                tz2 = __shfl(buf.reg9 , friend_id2);
                return ((buf.lane_id < 5 )? tx2: ((buf.lane_id < 31)? ty2: tz2));
            case 3:
                friend_id3 = (buf.lane_id+23+((buf.lane_id>>3)<<1))&(warpSize-1);
                tx3 = __shfl(buf.reg9 , friend_id3);
                ty3 = __shfl(buf.reg10, friend_id3);
                return ((buf.lane_id < 8 )? tx3: ty3);
        }
    }
    return 0;
}

#endif
