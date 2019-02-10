#ifndef _CUDA_UBUF_3D7_2D
#define _CUDA_UBUF_3D7_2D

#include "sten_macro.h"

template<class T>
class StenBuffer2D
{
public:
    __device__ virtual T load(int idy, int idz)=0;
    __device__ virtual void store(T v, int idy, int idz)=0;
    __device__ virtual void glb2buf(T *in, int off, int type=NONE)=0;
};

template<class T>
class L1Buffer2D: public StenBuffer2D<T>
{
private:
    T *l1m;
    int off;
    int m, n;
    int halo;
public:
    int gidx, gidy;
public:
    __device__ T load(int idx, int idy)
    {
        return ACC_2D(l1m, idx, idy);
    }
    __device__ void store(T v, int idx, int idy)
    {
        ACC_2D(l1m, idx, idy) = v;
    }
    __device__ L1Buffer2D(int j, int i, int h): m(j), n(i), halo(h)
    {
        gidx = threadIdx.x + blockIdx.x * blockDim.x + halo;
        gidy = threadIdx.y + blockIdx.y * blockDim.y + halo;

        off = 0;
    }
    __device__ void glb2buf(T *in, int o, int type=NONE)
    {
        l1m = in;
        off = o;
    }
};

template<class T>
__device__ inline T Sten_L1M_Fetch2D(L1Buffer2D<T> buf, int py, int px, int tc=0)
{
    return buf.load(buf.gidy+py, buf.gidx+px);
}

template<class T>
class LDSBuffer2D: public StenBuffer2D<T>
{
private:
    T *local; // lds memory pointer
    int m, n;
    int halo;
public:
    int off;
    int loff;
    int gidx, gidy;
    int lidx, lidy;
public:
    __device__ T load(int idy, int idx)
    {
        return LOC_2D(idy, idx);
    }
    __device__ void store(T v, int idy, int idx)
    {
        LOC_2D(idy, idx) = v;
    }
    __device__ LDSBuffer2D(int j, int i, int h, int f): m(j), n(i), halo(h)
    {
        __shared__ T sm[3*34*10];
        local = sm;
        gidx = threadIdx.x + blockIdx.x * blockDim.x + halo;
        gidy = threadIdx.y + blockIdx.y * blockDim.y + halo;
        lidx = threadIdx.x + halo;
        lidy = threadIdx.y + halo;
    }
    __device__ void setLayer(int l)
    {
        loff = l*34*10;
    }
    __device__ LDSBuffer2D()
    {} 
    __device__ void glb2buf(T *in, int o, int type=NONE)
    {
        off = o;
        if(type == BRANCH)
        {
            LOC_2D(lidy,lidx) = ACC_2D(in,gidy,gidx);

            if(lidx == halo) LOC_2D(lidy  ,lidx-1) = ACC_2D(in,gidy  ,gidx-1);
            if(lidx == 32  ) LOC_2D(lidy  ,lidx+1) = ACC_2D(in,gidy  ,gidx+1);
            if(lidy == halo) LOC_2D(lidy-1,lidx  ) = ACC_2D(in,gidy-1,gidx  );
            if(lidy == 8   ) LOC_2D(lidy+1,lidx  ) = ACC_2D(in,gidy+1,gidx  );
            __syncthreads();

        } else if(type == CYCLIC)
        {
            unsigned int lane_id = threadIdx.x + threadIdx.y * blockDim.x;
            int blk_id_x = blockIdx.x;
            int blk_id_y = blockIdx.y;
            int new_i, new_j, new_li, new_lj;
            new_i  = (blk_id_x<<5) + lane_id%34;
            new_j  = (blk_id_y<<3) + lane_id/34;
            new_li = lane_id%34;
            new_lj = lane_id/34;
            LOC_2D(new_lj,new_li) = ACC_2D(in,new_j,new_i);
            new_i  = (blk_id_x<<5) + (lane_id+256)%34;
            new_j  = (blk_id_y<<3) + (lane_id+256)/34;
            new_li = (lane_id+256)%34;
            new_lj = (lane_id+256)/34;
            new_i  = (new_i < n+2*halo)? new_i: n+2*halo-1;
            new_j  = (new_j < m+2*halo)? new_j: m+2*halo-1;
            if(new_li < 34 &&  new_lj < 10)
            {
                LOC_2D(new_lj,new_li) = ACC_2D(in,new_j,new_i);
            }
            __syncthreads();
        }
    }
};

template<class T>
__device__ inline T Sten_LDS_Fetch2D(LDSBuffer2D<T> buf, int py, int px, int tc=0)
{
    buf.setLayer(tc);
    return buf.load(buf.lidy+py, buf.lidx+px);
}

template<class T>
class REGBuffer2D: public StenBuffer2D<T>
{
private:
    int m, n;
    int halo;
public:
    T t1_reg0, t1_reg1, t1_reg2, t1_reg3, t1_reg4, t1_reg5, t1_reg6;
    T t2_reg0, t2_reg1, t2_reg2, t2_reg3, t2_reg4, t2_reg5, t2_reg6;
    T t3_reg0, t3_reg1, t3_reg2, t3_reg3, t3_reg4, t3_reg5, t3_reg6;
    int off;
    int layer;
    int gidx, gidy;
    int lane_id;
    int tc;
public:
    __device__ T load(int idy, int idx)
    {
        return 0;
    }
    __device__ void store(T v, int idy, int idx)
    {}
    __device__ REGBuffer2D(int j, int i, int h, int f, int c=1): m(j), n(i), halo(h), tc(c)
    {
        const int tid = threadIdx.x + threadIdx.y * blockDim.x;
        lane_id = tid % 32;
        switch(tc)
        {
            case(1):
            {
                gidx = threadIdx.x + blockIdx.x * blockDim.x + halo;
                gidy = threadIdx.y + blockIdx.y * blockDim.y + halo;
                break;
            }
            case(2):
            {
                gidx = threadIdx.x + blockIdx.x * blockDim.x + halo;
                gidy = (((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<1) + halo;
                break;
            }
            case(4):
            {
                gidx = threadIdx.x + blockIdx.x * blockDim.x + halo;
                gidy = (((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<2) + halo;
                break;
            }
        }
    }
    __device__ void setLayer(int l)
    {
        layer = l;
    }
    __device__ void glb2buf(T *in, int o, int type=NONE)
    {
        off = o;
        if(type == CYCLIC)
        {
            switch (tc)
            {
                case 1:
                {
                    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>5; // because the warp dimensions are 
                    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<0))>>0; // 1x1x32, warp_ids are division of 
                    int new_i, new_j;
                    switch(layer)
                    {
                        case 0:
                        {
                            new_i = (warp_id_x<<5) + lane_id%34;
                            new_j = (warp_id_y<<0) + lane_id/34;     
                            t1_reg0 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+32)%34;
                            new_j = (warp_id_y<<0) + (lane_id+32)/34;
                            t1_reg1 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+64)%34;
                            new_j = (warp_id_y<<0) + (lane_id+64)/34;
                            t1_reg2 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+96)%34;
                            new_j = (warp_id_y<<0) + (lane_id+96)/34;
                            new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
                            new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
                            t1_reg3 = ACC_2D(in, new_j, new_i);
    
                            return;
                        }
                        case 1:
                        {
                            new_i = (warp_id_x<<5) + lane_id%34;
                            new_j = (warp_id_y<<0) + lane_id/34;     
                            t2_reg0 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+32)%34;
                            new_j = (warp_id_y<<0) + (lane_id+32)/34;
                            t2_reg1 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+64)%34;
                            new_j = (warp_id_y<<0) + (lane_id+64)/34;
                            t2_reg2 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+96)%34;
                            new_j = (warp_id_y<<0) + (lane_id+96)/34;
                            new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
                            new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
                            t2_reg3 = ACC_2D(in, new_j, new_i);
    
                            return;
                        }
                        case 2:
                        {
                            new_i = (warp_id_x<<5) + lane_id%34;
                            new_j = (warp_id_y<<0) + lane_id/34;     
                            t3_reg0 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+32)%34;
                            new_j = (warp_id_y<<0) + (lane_id+32)/34;
                            t3_reg1 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+64)%34;
                            new_j = (warp_id_y<<0) + (lane_id+64)/34;
                            t3_reg2 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+96)%34;
                            new_j = (warp_id_y<<0) + (lane_id+96)/34;
                            new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
                            new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
                            t3_reg3 = ACC_2D(in, new_j, new_i);
    
                            return;
                        }
                    
                    }
                }
                case 2:
                {
                    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>5;
                    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<1))>>0;
                    int new_i, new_j;
                    switch(layer)
                    {
                        case 0:
                        {
                            new_i = (warp_id_x<<5) + lane_id%34;
                            new_j = (warp_id_y<<0) + lane_id/34;     
                            t1_reg0 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+32)%34;
                            new_j = (warp_id_y<<0) + (lane_id+32)/34;
                            t1_reg1 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+64)%34;
                            new_j = (warp_id_y<<0) + (lane_id+64)/34;
                            t1_reg2 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+96)%34;
                            new_j = (warp_id_y<<0) + (lane_id+96)/34;
                            t1_reg3 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+128)%34;
                            new_j = (warp_id_y<<0) + (lane_id+128)/34;
                            new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
                            new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
                            t1_reg4 = ACC_2D(in, new_j, new_i);
    
                            return;
                        }
                        case 1:
                        {
                            new_i = (warp_id_x<<5) + lane_id%34;
                            new_j = (warp_id_y<<0) + lane_id/34;     
                            t2_reg0 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+32)%34;
                            new_j = (warp_id_y<<0) + (lane_id+32)/34;
                            t2_reg1 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+64)%34;
                            new_j = (warp_id_y<<0) + (lane_id+64)/34;
                            t2_reg2 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+96)%34;
                            new_j = (warp_id_y<<0) + (lane_id+96)/34;
                            t2_reg3 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+128)%34;
                            new_j = (warp_id_y<<0) + (lane_id+128)/34;
                            new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
                            new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
                            t2_reg4 = ACC_2D(in, new_j, new_i);
    
                            return;
                        }
                        case 2:
                        {
                            new_i = (warp_id_x<<5) + lane_id%34;
                            new_j = (warp_id_y<<0) + lane_id/34;     
                            t3_reg0 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+32)%34;
                            new_j = (warp_id_y<<0) + (lane_id+32)/34;
                            t3_reg1 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+64)%34;
                            new_j = (warp_id_y<<0) + (lane_id+64)/34;
                            t3_reg2 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+96)%34;
                            new_j = (warp_id_y<<0) + (lane_id+96)/34;
                            t3_reg3 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+128)%34;
                            new_j = (warp_id_y<<0) + (lane_id+128)/34;
                            new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
                            new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
                            t3_reg4 = ACC_2D(in, new_j, new_i);
    
                            return;
                        }
                    
                    }
                    return;
                }
                case 4:
                {
                    int warp_id_x = (threadIdx.x + blockIdx.x * blockDim.x)>>5;
                    int warp_id_y = ((((threadIdx.y + blockIdx.y * blockDim.y)>>0)<<2))>>0;
                    int new_i, new_j;
                    switch(layer)
                    {
                        case 0:
                        {
                            new_i = (warp_id_x<<5) + lane_id%34;
                            new_j = (warp_id_y<<0) + lane_id/34;     
                            t1_reg0 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+32)%34;
                            new_j = (warp_id_y<<0) + (lane_id+32)/34;
                            t1_reg1 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+64)%34;
                            new_j = (warp_id_y<<0) + (lane_id+64)/34;
                            t1_reg2 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+96)%34;
                            new_j = (warp_id_y<<0) + (lane_id+96)/34;
                            t1_reg3 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+128)%34;
                            new_j = (warp_id_y<<0) + (lane_id+128)/34;
                            t1_reg4 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+160)%34;
                            new_j = (warp_id_y<<0) + (lane_id+160)/34;
                            t1_reg5 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+192)%34;
                            new_j = (warp_id_y<<0) + (lane_id+192)/34;
                            new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
                            new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
                            t1_reg6 = ACC_2D(in, new_j, new_i);

    
                            return;
                        }
                        case 1:
                        {
                            new_i = (warp_id_x<<5) + lane_id%34;
                            new_j = (warp_id_y<<0) + lane_id/34;     
                            t2_reg0 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+32)%34;
                            new_j = (warp_id_y<<0) + (lane_id+32)/34;
                            t2_reg1 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+64)%34;
                            new_j = (warp_id_y<<0) + (lane_id+64)/34;
                            t2_reg2 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+96)%34;
                            new_j = (warp_id_y<<0) + (lane_id+96)/34;
                            t2_reg3 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+128)%34;
                            new_j = (warp_id_y<<0) + (lane_id+128)/34;
                            t2_reg4 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+160)%34;
                            new_j = (warp_id_y<<0) + (lane_id+160)/34;
                            t2_reg5 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+192)%34;
                            new_j = (warp_id_y<<0) + (lane_id+192)/34;
                            new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
                            new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
                            t2_reg6 = ACC_2D(in, new_j, new_i);

                            return;
                        }
                        case 2:
                        {
                            new_i = (warp_id_x<<5) + lane_id%34;
                            new_j = (warp_id_y<<0) + lane_id/34;     
                            t3_reg0 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+32)%34;
                            new_j = (warp_id_y<<0) + (lane_id+32)/34;
                            t3_reg1 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+64)%34;
                            new_j = (warp_id_y<<0) + (lane_id+64)/34;
                            t3_reg2 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+96)%34;
                            new_j = (warp_id_y<<0) + (lane_id+96)/34;
                            t3_reg3 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+128)%34;
                            new_j = (warp_id_y<<0) + (lane_id+128)/34;
                            t3_reg4 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+160)%34;
                            new_j = (warp_id_y<<0) + (lane_id+160)/34;
                            t3_reg5 = ACC_2D(in, new_j, new_i);
                            new_i = (warp_id_x<<5) + (lane_id+192)%34;
                            new_j = (warp_id_y<<0) + (lane_id+192)/34;
                            new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
                            new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
                            t3_reg6 = ACC_2D(in, new_j, new_i);

                            return;
                        }
                    
                    }
                    return;
                }
            }
            
        }
    }
    __device__ inline T Sten_REG_Fetch2D(int py, int px, int l, int tc=0)
    {
    int friend_id0;
    int friend_id1;
    int friend_id2;
    int friend_id3;
    T tx0, ty0;//, tz0;
    T tx1, ty1;//, tz1;
    T tx2, ty2;//, tz2;
    T tx3, ty3;//, tz3;
    if(py == 0 && px == 0)
    {
        switch (l)
        {
            case 0:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (lane_id+3 )&31;
                        tx0 = __shfl(t1_reg1, friend_id0);
                        ty0 = __shfl(t1_reg2, friend_id0);
                        return ((lane_id < 29)? tx0: ty0);
                    case 1:
                        friend_id1 = (lane_id+5 )&31;
                        tx1 = __shfl(t1_reg2, friend_id1);
                        ty1 = __shfl(t1_reg3, friend_id1);
                        return ((lane_id < 27)? tx1: ty1);
                    case 2:
                        friend_id2 = (lane_id+7 )&31;
                        tx2 = __shfl(t1_reg3, friend_id2);
                        ty2 = __shfl(t1_reg4, friend_id2);
                        return ((lane_id < 25)? tx2: ty2);
                    case 3:
                        friend_id3 = (lane_id+9 )&31;
                        tx3 = __shfl(t1_reg4, friend_id3);
                        ty3 = __shfl(t1_reg5, friend_id3);
                        return ((lane_id < 23)? tx3: ty3);
                }
            case 1:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (lane_id+3 )&31;
                        tx0 = __shfl(t2_reg1, friend_id0);
                        ty0 = __shfl(t2_reg2, friend_id0);
                        return ((lane_id < 29)? tx0: ty0);
                    case 1:
                        friend_id1 = (lane_id+5 )&31;
                        tx1 = __shfl(t2_reg2, friend_id1);
                        ty1 = __shfl(t2_reg3, friend_id1);
                        return ((lane_id < 27)? tx1: ty1);
                    case 2:
                        friend_id2 = (lane_id+7 )&31;
                        tx2 = __shfl(t2_reg3, friend_id2);
                        ty2 = __shfl(t2_reg4, friend_id2);
                        return ((lane_id < 25)? tx2: ty2);
                    case 3:
                        friend_id3 = (lane_id+9 )&31;
                        tx3 = __shfl(t2_reg4, friend_id3);
                        ty3 = __shfl(t2_reg5, friend_id3);
                        return ((lane_id < 23)? tx3: ty3);
                }
            case 2:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (lane_id+3 )&31;
                        tx0 = __shfl(t3_reg1, friend_id0);
                        ty0 = __shfl(t3_reg2, friend_id0);
                        return ((lane_id < 29)? tx0: ty0);
                    case 1:
                        friend_id1 = (lane_id+5 )&31;
                        tx1 = __shfl(t3_reg2, friend_id1);
                        ty1 = __shfl(t3_reg3, friend_id1);
                        return ((lane_id < 27)? tx1: ty1);
                    case 2:
                        friend_id2 = (lane_id+7 )&31;
                        tx2 = __shfl(t3_reg3, friend_id2);
                        ty2 = __shfl(t3_reg4, friend_id2);
                        return ((lane_id < 25)? tx2: ty2);
                    case 3:
                        friend_id3 = (lane_id+9 )&31;
                        tx3 = __shfl(t3_reg4, friend_id3);
                        ty3 = __shfl(t3_reg5, friend_id3);
                        return ((lane_id < 23)? tx3: ty3);
                }
        }
    }
    if(py == -1 && px == 0)
    {
        switch (l)
        {
            case 0:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (lane_id+1 )&31;
                        tx0 = __shfl(t1_reg0, friend_id0);
                        ty0 = __shfl(t1_reg1, friend_id0);
                        return ((lane_id < 31)? tx0: ty0);
                    case 1:
                        friend_id1 = (lane_id+3 )&31;
                        tx1 = __shfl(t1_reg1, friend_id1);
                        ty1 = __shfl(t1_reg2, friend_id1);
                        return ((lane_id < 29)? tx1: ty1);
                    case 2:
                        friend_id2 = (lane_id+5 )&31;
                        tx2 = __shfl(t1_reg2, friend_id2);
                        ty2 = __shfl(t1_reg3, friend_id2);
                        return ((lane_id < 27)? tx2: ty2);
                    case 3:
                        friend_id3 = (lane_id+7 )&31;
                        tx3 = __shfl(t1_reg3, friend_id3);
                        ty3 = __shfl(t1_reg4, friend_id3);
                        return ((lane_id < 25)? tx3: ty3);

                }
            case 1:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (lane_id+1 )&31;
                        tx0 = __shfl(t2_reg0, friend_id0);
                        ty0 = __shfl(t2_reg1, friend_id0);
                        return ((lane_id < 31)? tx0: ty0);
                    case 1:
                        friend_id1 = (lane_id+3 )&31;
                        tx1 = __shfl(t2_reg1, friend_id1);
                        ty1 = __shfl(t2_reg2, friend_id1);
                        return ((lane_id < 29)? tx1: ty1);
                    case 2:
                        friend_id2 = (lane_id+5 )&31;
                        tx2 = __shfl(t2_reg2, friend_id2);
                        ty2 = __shfl(t2_reg3, friend_id2);
                        return ((lane_id < 27)? tx2: ty2);
                    case 3:
                        friend_id3 = (lane_id+7 )&31;
                        tx3 = __shfl(t2_reg3, friend_id3);
                        ty3 = __shfl(t2_reg4, friend_id3);
                        return ((lane_id < 25)? tx3: ty3);
                }
            case 2:
                switch (tc)
                {
                   case 0:
                        friend_id0 = (lane_id+1 )&31;
                        tx0 = __shfl(t3_reg0, friend_id0);
                        ty0 = __shfl(t3_reg1, friend_id0);
                        return ((lane_id < 31)? tx0: ty0);
                    case 1:
                        friend_id1 = (lane_id+3 )&31;
                        tx1 = __shfl(t3_reg1, friend_id1);
                        ty1 = __shfl(t3_reg2, friend_id1);
                        return ((lane_id < 29)? tx1: ty1);
                    case 2:
                        friend_id2 = (lane_id+5 )&31;
                        tx2 = __shfl(t3_reg2, friend_id2);
                        ty2 = __shfl(t3_reg3, friend_id2);
                        return ((lane_id < 27)? tx2: ty2);
                    case 3:
                        friend_id3 = (lane_id+7 )&31;
                        tx3 = __shfl(t3_reg3, friend_id3);
                        ty3 = __shfl(t3_reg4, friend_id3);
                        return ((lane_id < 25)? tx3: ty3);

                }
        }
    }

    if(py == 1 && px == 0)
    {
        switch (l)
        {
            case 0:
                switch (tc)
                {     
                    case 0:
                        friend_id0 = (lane_id+5 )&31;
                        tx0 = __shfl(t1_reg2, friend_id0);
                        ty0 = __shfl(t1_reg3, friend_id0);
                        return ((lane_id < 27)? tx0: ty0);
                    case 1:
                        friend_id1 = (lane_id+7 )&31;
                        tx1 = __shfl(t1_reg3, friend_id1);
                        ty1 = __shfl(t1_reg4, friend_id1);
                        return ((lane_id < 25)? tx1: ty1);
                    case 2:
                        friend_id2 = (lane_id+9 )&31;
                        tx2 = __shfl(t1_reg4, friend_id2);
                        ty2 = __shfl(t1_reg5, friend_id2);
                        return ((lane_id < 23)? tx2: ty2);
                    case 3:
                        friend_id3 = (lane_id+11)&31;
                        tx3 = __shfl(t1_reg5, friend_id3);
                        ty3 = __shfl(t1_reg6, friend_id3);
                        return ((lane_id < 21)? tx3: ty3);
                }
            case 1:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (lane_id+5 )&31;
                        tx0 = __shfl(t2_reg2, friend_id0);
                        ty0 = __shfl(t2_reg3, friend_id0);
                        return ((lane_id < 27)? tx0: ty0);
                    case 1:
                        friend_id1 = (lane_id+7 )&31;
                        tx1 = __shfl(t2_reg3, friend_id1);
                        ty1 = __shfl(t2_reg4, friend_id1);
                        return ((lane_id < 25)? tx1: ty1);
                    case 2:
                        friend_id2 = (lane_id+9 )&31;
                        tx2 = __shfl(t2_reg4, friend_id2);
                        ty2 = __shfl(t2_reg5, friend_id2);
                        return ((lane_id < 23)? tx2: ty2);
                    case 3:
                        friend_id3 = (lane_id+11)&31;
                        tx3 = __shfl(t2_reg5, friend_id3);
                        ty3 = __shfl(t2_reg6, friend_id3);
                        return ((lane_id < 21)? tx3: ty3);

                }
            case 2:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (lane_id+5 )&31;
                        tx0 = __shfl(t3_reg2, friend_id0);
                        ty0 = __shfl(t3_reg3, friend_id0);
                        return ((lane_id < 27)? tx0: ty0);
                    case 1:
                        friend_id1 = (lane_id+7 )&31;
                        tx1 = __shfl(t3_reg3, friend_id1);
                        ty1 = __shfl(t3_reg4, friend_id1);
                        return ((lane_id < 25)? tx1: ty1);
                    case 2:
                        friend_id2 = (lane_id+9 )&31;
                        tx2 = __shfl(t3_reg4, friend_id2);
                        ty2 = __shfl(t3_reg5, friend_id2);
                        return ((lane_id < 23)? tx2: ty2);
                    case 3:
                        friend_id3 = (lane_id+11)&31;
                        tx3 = __shfl(t3_reg5, friend_id3);
                        ty3 = __shfl(t3_reg6, friend_id3);
                        return ((lane_id < 21)? tx3: ty3);

                }
        }
    }

    if(py == 0 && px == -1)
    {
        switch (l)
        {
            case 0:
                switch (tc)
                {     
                    case 0:
                        friend_id0 = (lane_id+2 )&31;
                        tx0 = __shfl(t1_reg1, friend_id0);
                        ty0 = __shfl(t1_reg2, friend_id0);
                        return ((lane_id < 30)? tx0: ty0);
                    case 1:
                        friend_id1 = (lane_id+4 )&31;
                        tx1 = __shfl(t1_reg2, friend_id1);
                        ty1 = __shfl(t1_reg3, friend_id1);
                        return ((lane_id < 28)? tx1: ty1);
                    case 2:
                        friend_id2 = (lane_id+6 )&31;
                        tx2 = __shfl(t1_reg3, friend_id2);
                        ty2 = __shfl(t1_reg4, friend_id2);
                        return ((lane_id < 26)? tx2: ty2);
                    case 3:
                        friend_id3 = (lane_id+8 )&31;
                        tx3 = __shfl(t1_reg4, friend_id3);
                        ty3 = __shfl(t1_reg5, friend_id3);
                        return ((lane_id < 24)? tx3: ty3);
                }
            case 1:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (lane_id+2 )&31;
                        tx0 = __shfl(t2_reg1, friend_id0);
                        ty0 = __shfl(t2_reg2, friend_id0);
                        return ((lane_id < 30)? tx0: ty0);
                    case 1:
                        friend_id1 = (lane_id+4 )&31;
                        tx1 = __shfl(t2_reg2, friend_id1);
                        ty1 = __shfl(t2_reg3, friend_id1);
                        return ((lane_id < 28)? tx1: ty1);
                    case 2:
                        friend_id2 = (lane_id+6 )&31;
                        tx2 = __shfl(t2_reg3, friend_id2);
                        ty2 = __shfl(t2_reg4, friend_id2);
                        return ((lane_id < 26)? tx2: ty2);
                    case 3:
                        friend_id3 = (lane_id+8 )&31;
                        tx3 = __shfl(t2_reg4, friend_id3);
                        ty3 = __shfl(t2_reg5, friend_id3);
                        return ((lane_id < 24)? tx3: ty3);   

                }
            case 2:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (lane_id+2 )&31;
                        tx0 = __shfl(t3_reg1, friend_id0);
                        ty0 = __shfl(t3_reg2, friend_id0);
                        return ((lane_id < 30)? tx0: ty0);
                    case 1:
                        friend_id1 = (lane_id+4 )&31;
                        tx1 = __shfl(t3_reg2, friend_id1);
                        ty1 = __shfl(t3_reg3, friend_id1);
                        return ((lane_id < 28)? tx1: ty1);
                    case 2:
                        friend_id2 = (lane_id+6 )&31;
                        tx2 = __shfl(t3_reg3, friend_id2);
                        ty2 = __shfl(t3_reg4, friend_id2);
                        return ((lane_id < 26)? tx2: ty2);
                    case 3:
                        friend_id3 = (lane_id+8 )&31;
                        tx3 = __shfl(t3_reg4, friend_id3);
                        ty3 = __shfl(t3_reg5, friend_id3);
                        return ((lane_id < 24)? tx3: ty3);  

                }
        }
    }

    if(py == 0 && px == 1)
    {
        switch (l)
        {
            case 0:
                switch (tc)
                {     
                    case 0:
                        friend_id0 = (lane_id+4 )&31;
                        tx0 = __shfl(t1_reg1, friend_id0);
                        ty0 = __shfl(t1_reg2, friend_id0);
                        return ((lane_id < 28)? tx0: ty0);
                    case 1:
                        friend_id1 = (lane_id+6 )&31;
                        tx1 = __shfl(t1_reg2, friend_id1);
                        ty1 = __shfl(t1_reg3, friend_id1);
                        return ((lane_id < 26)? tx1: ty1);
                    case 2:
                        friend_id2 = (lane_id+8 )&31;
                        tx2 = __shfl(t1_reg3, friend_id2);
                        ty2 = __shfl(t1_reg4, friend_id2);
                        return ((lane_id < 24)? tx2: ty2);
                    case 3:
                        friend_id3 = (lane_id+10)&31;
                        tx3 = __shfl(t1_reg4, friend_id3);
                        ty3 = __shfl(t1_reg5, friend_id3);
                        return ((lane_id < 22)? tx3: ty3);
   
                }
            case 1:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (lane_id+4 )&31;
                        tx0 = __shfl(t2_reg1, friend_id0);
                        ty0 = __shfl(t2_reg2, friend_id0);
                        return ((lane_id < 28)? tx0: ty0);
                    case 1:
                        friend_id1 = (lane_id+6 )&31;
                        tx1 = __shfl(t2_reg2, friend_id1);
                        ty1 = __shfl(t2_reg3, friend_id1);
                        return ((lane_id < 26)? tx1: ty1);
                    case 2:
                        friend_id2 = (lane_id+8 )&31;
                        tx2 = __shfl(t2_reg3, friend_id2);
                        ty2 = __shfl(t2_reg4, friend_id2);
                        return ((lane_id < 24)? tx2: ty2);
                    case 3:
                        friend_id3 = (lane_id+10)&31;
                        tx3 = __shfl(t2_reg4, friend_id3);
                        ty3 = __shfl(t2_reg5, friend_id3);
                        return ((lane_id < 22)? tx3: ty3);  
                }
            case 2:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (lane_id+4 )&31;
                        tx0 = __shfl(t3_reg1, friend_id0);
                        ty0 = __shfl(t3_reg2, friend_id0);
                        return ((lane_id < 28)? tx0: ty0);
                    case 1:
                        friend_id1 = (lane_id+6 )&31;
                        tx1 = __shfl(t3_reg2, friend_id1);
                        ty1 = __shfl(t3_reg3, friend_id1);
                        return ((lane_id < 26)? tx1: ty1);
                    case 2:
                        friend_id2 = (lane_id+8 )&31;
                        tx2 = __shfl(t3_reg3, friend_id2);
                        ty2 = __shfl(t3_reg4, friend_id2);
                        return ((lane_id < 24)? tx2: ty2);
                    case 3:
                        friend_id3 = (lane_id+10)&31;
                        tx3 = __shfl(t3_reg4, friend_id3);
                        ty3 = __shfl(t3_reg5, friend_id3);
                        return ((lane_id < 22)? tx3: ty3);
                }
        }
    }
    return 0;

}
};

template<class T>
__device__ inline T Sten_REG_Fetch2D(REGBuffer2D<T> buf, int py, int px, int l, int tc=0)
{
    int friend_id0;
    int friend_id1;
    int friend_id2;
    int friend_id3;
    T tx0, ty0;//, tz0;
    T tx1, ty1;//, tz1;
    T tx2, ty2;//, tz2;
    T tx3, ty3;//, tz3;
    if(py == 0 && px == 0)
    {
        switch (l)
        {
            case 0:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (buf.lane_id+3 )&31;
                        tx0 = __shfl(buf.t1_reg1, friend_id0);
                        ty0 = __shfl(buf.t1_reg2, friend_id0);
                        return ((buf.lane_id < 29)? tx0: ty0);
                    case 1:
                        friend_id1 = (buf.lane_id+5 )&31;
                        tx1 = __shfl(buf.t1_reg2, friend_id1);
                        ty1 = __shfl(buf.t1_reg3, friend_id1);
                        return ((buf.lane_id < 27)? tx1: ty1);
                    case 2:
                        friend_id2 = (buf.lane_id+7 )&31;
                        tx2 = __shfl(buf.t1_reg3, friend_id2);
                        ty2 = __shfl(buf.t1_reg4, friend_id2);
                        return ((buf.lane_id < 25)? tx2: ty2);
                    case 3:
                        friend_id3 = (buf.lane_id+9 )&31;
                        tx3 = __shfl(buf.t1_reg4, friend_id3);
                        ty3 = __shfl(buf.t1_reg5, friend_id3);
                        return ((buf.lane_id < 23)? tx3: ty3);
                }
            case 1:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (buf.lane_id+3 )&31;
                        tx0 = __shfl(buf.t2_reg1, friend_id0);
                        ty0 = __shfl(buf.t2_reg2, friend_id0);
                        return ((buf.lane_id < 29)? tx0: ty0);
                    case 1:
                        friend_id1 = (buf.lane_id+5 )&31;
                        tx1 = __shfl(buf.t2_reg2, friend_id1);
                        ty1 = __shfl(buf.t2_reg3, friend_id1);
                        return ((buf.lane_id < 27)? tx1: ty1);
                    case 2:
                        friend_id2 = (buf.lane_id+7 )&31;
                        tx2 = __shfl(buf.t2_reg3, friend_id2);
                        ty2 = __shfl(buf.t2_reg4, friend_id2);
                        return ((buf.lane_id < 25)? tx2: ty2);
                    case 3:
                        friend_id3 = (buf.lane_id+9 )&31;
                        tx3 = __shfl(buf.t2_reg4, friend_id3);
                        ty3 = __shfl(buf.t2_reg5, friend_id3);
                        return ((buf.lane_id < 23)? tx3: ty3);
                }
            case 2:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (buf.lane_id+3 )&31;
                        tx0 = __shfl(buf.t3_reg1, friend_id0);
                        ty0 = __shfl(buf.t3_reg2, friend_id0);
                        return ((buf.lane_id < 29)? tx0: ty0);
                    case 1:
                        friend_id1 = (buf.lane_id+5 )&31;
                        tx1 = __shfl(buf.t3_reg2, friend_id1);
                        ty1 = __shfl(buf.t3_reg3, friend_id1);
                        return ((buf.lane_id < 27)? tx1: ty1);
                    case 2:
                        friend_id2 = (buf.lane_id+7 )&31;
                        tx2 = __shfl(buf.t3_reg3, friend_id2);
                        ty2 = __shfl(buf.t3_reg4, friend_id2);
                        return ((buf.lane_id < 25)? tx2: ty2);
                    case 3:
                        friend_id3 = (buf.lane_id+9 )&31;
                        tx3 = __shfl(buf.t3_reg4, friend_id3);
                        ty3 = __shfl(buf.t3_reg5, friend_id3);
                        return ((buf.lane_id < 23)? tx3: ty3);
                }
        }
    }
    if(py == -1 && px == 0)
    {
        switch (l)
        {
            case 0:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (buf.lane_id+1 )&31;
                        tx0 = __shfl(buf.t1_reg0, friend_id0);
                        ty0 = __shfl(buf.t1_reg1, friend_id0);
                        return ((buf.lane_id < 31)? tx0: ty0);
                    case 1:
                        friend_id1 = (buf.lane_id+3 )&31;
                        tx1 = __shfl(buf.t1_reg1, friend_id1);
                        ty1 = __shfl(buf.t1_reg2, friend_id1);
                        return ((buf.lane_id < 29)? tx1: ty1);
                    case 2:
                        friend_id2 = (buf.lane_id+5 )&31;
                        tx2 = __shfl(buf.t1_reg2, friend_id2);
                        ty2 = __shfl(buf.t1_reg3, friend_id2);
                        return ((buf.lane_id < 27)? tx2: ty2);
                    case 3:
                        friend_id3 = (buf.lane_id+7 )&31;
                        tx3 = __shfl(buf.t1_reg3, friend_id3);
                        ty3 = __shfl(buf.t1_reg4, friend_id3);
                        return ((buf.lane_id < 25)? tx3: ty3);

                }
            case 1:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (buf.lane_id+1 )&31;
                        tx0 = __shfl(buf.t2_reg0, friend_id0);
                        ty0 = __shfl(buf.t2_reg1, friend_id0);
                        return ((buf.lane_id < 31)? tx0: ty0);
                    case 1:
                        friend_id1 = (buf.lane_id+3 )&31;
                        tx1 = __shfl(buf.t2_reg1, friend_id1);
                        ty1 = __shfl(buf.t2_reg2, friend_id1);
                        return ((buf.lane_id < 29)? tx1: ty1);
                    case 2:
                        friend_id2 = (buf.lane_id+5 )&31;
                        tx2 = __shfl(buf.t2_reg2, friend_id2);
                        ty2 = __shfl(buf.t2_reg3, friend_id2);
                        return ((buf.lane_id < 27)? tx2: ty2);
                    case 3:
                        friend_id3 = (buf.lane_id+7 )&31;
                        tx3 = __shfl(buf.t2_reg3, friend_id3);
                        ty3 = __shfl(buf.t2_reg4, friend_id3);
                        return ((buf.lane_id < 25)? tx3: ty3);
                }
            case 2:
                switch (tc)
                {
                   case 0:
                        friend_id0 = (buf.lane_id+1 )&31;
                        tx0 = __shfl(buf.t3_reg0, friend_id0);
                        ty0 = __shfl(buf.t3_reg1, friend_id0);
                        return ((buf.lane_id < 31)? tx0: ty0);
                    case 1:
                        friend_id1 = (buf.lane_id+3 )&31;
                        tx1 = __shfl(buf.t3_reg1, friend_id1);
                        ty1 = __shfl(buf.t3_reg2, friend_id1);
                        return ((buf.lane_id < 29)? tx1: ty1);
                    case 2:
                        friend_id2 = (buf.lane_id+5 )&31;
                        tx2 = __shfl(buf.t3_reg2, friend_id2);
                        ty2 = __shfl(buf.t3_reg3, friend_id2);
                        return ((buf.lane_id < 27)? tx2: ty2);
                    case 3:
                        friend_id3 = (buf.lane_id+7 )&31;
                        tx3 = __shfl(buf.t3_reg3, friend_id3);
                        ty3 = __shfl(buf.t3_reg4, friend_id3);
                        return ((buf.lane_id < 25)? tx3: ty3);

                }
        }
    }

    if(py == 1 && px == 0)
    {
        switch (l)
        {
            case 0:
                switch (tc)
                {     
                    case 0:
                        friend_id0 = (buf.lane_id+5 )&31;
                        tx0 = __shfl(buf.t1_reg2, friend_id0);
                        ty0 = __shfl(buf.t1_reg3, friend_id0);
                        return ((buf.lane_id < 27)? tx0: ty0);
                    case 1:
                        friend_id1 = (buf.lane_id+7 )&31;
                        tx1 = __shfl(buf.t1_reg3, friend_id1);
                        ty1 = __shfl(buf.t1_reg4, friend_id1);
                        return ((buf.lane_id < 25)? tx1: ty1);
                    case 2:
                        friend_id2 = (buf.lane_id+9 )&31;
                        tx2 = __shfl(buf.t1_reg4, friend_id2);
                        ty2 = __shfl(buf.t1_reg5, friend_id2);
                        return ((buf.lane_id < 23)? tx2: ty2);
                    case 3:
                        friend_id3 = (buf.lane_id+11)&31;
                        tx3 = __shfl(buf.t1_reg5, friend_id3);
                        ty3 = __shfl(buf.t1_reg6, friend_id3);
                        return ((buf.lane_id < 21)? tx3: ty3);
                }
            case 1:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (buf.lane_id+5 )&31;
                        tx0 = __shfl(buf.t2_reg2, friend_id0);
                        ty0 = __shfl(buf.t2_reg3, friend_id0);
                        return ((buf.lane_id < 27)? tx0: ty0);
                    case 1:
                        friend_id1 = (buf.lane_id+7 )&31;
                        tx1 = __shfl(buf.t2_reg3, friend_id1);
                        ty1 = __shfl(buf.t2_reg4, friend_id1);
                        return ((buf.lane_id < 25)? tx1: ty1);
                    case 2:
                        friend_id2 = (buf.lane_id+9 )&31;
                        tx2 = __shfl(buf.t2_reg4, friend_id2);
                        ty2 = __shfl(buf.t2_reg5, friend_id2);
                        return ((buf.lane_id < 23)? tx2: ty2);
                    case 3:
                        friend_id3 = (buf.lane_id+11)&31;
                        tx3 = __shfl(buf.t2_reg5, friend_id3);
                        ty3 = __shfl(buf.t2_reg6, friend_id3);
                        return ((buf.lane_id < 21)? tx3: ty3);

                }
            case 2:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (buf.lane_id+5 )&31;
                        tx0 = __shfl(buf.t3_reg2, friend_id0);
                        ty0 = __shfl(buf.t3_reg3, friend_id0);
                        return ((buf.lane_id < 27)? tx0: ty0);
                    case 1:
                        friend_id1 = (buf.lane_id+7 )&31;
                        tx1 = __shfl(buf.t3_reg3, friend_id1);
                        ty1 = __shfl(buf.t3_reg4, friend_id1);
                        return ((buf.lane_id < 25)? tx1: ty1);
                    case 2:
                        friend_id2 = (buf.lane_id+9 )&31;
                        tx2 = __shfl(buf.t3_reg4, friend_id2);
                        ty2 = __shfl(buf.t3_reg5, friend_id2);
                        return ((buf.lane_id < 23)? tx2: ty2);
                    case 3:
                        friend_id3 = (buf.lane_id+11)&31;
                        tx3 = __shfl(buf.t3_reg5, friend_id3);
                        ty3 = __shfl(buf.t3_reg6, friend_id3);
                        return ((buf.lane_id < 21)? tx3: ty3);

                }
        }
    }

    if(py == 0 && px == -1)
    {
        switch (l)
        {
            case 0:
                switch (tc)
                {     
                    case 0:
                        friend_id0 = (buf.lane_id+2 )&31;
                        tx0 = __shfl(buf.t1_reg1, friend_id0);
                        ty0 = __shfl(buf.t1_reg2, friend_id0);
                        return ((buf.lane_id < 30)? tx0: ty0);
                    case 1:
                        friend_id1 = (buf.lane_id+4 )&31;
                        tx1 = __shfl(buf.t1_reg2, friend_id1);
                        ty1 = __shfl(buf.t1_reg3, friend_id1);
                        return ((buf.lane_id < 28)? tx1: ty1);
                    case 2:
                        friend_id2 = (buf.lane_id+6 )&31;
                        tx2 = __shfl(buf.t1_reg3, friend_id2);
                        ty2 = __shfl(buf.t1_reg4, friend_id2);
                        return ((buf.lane_id < 26)? tx2: ty2);
                    case 3:
                        friend_id3 = (buf.lane_id+8 )&31;
                        tx3 = __shfl(buf.t1_reg4, friend_id3);
                        ty3 = __shfl(buf.t1_reg5, friend_id3);
                        return ((buf.lane_id < 24)? tx3: ty3);
                }
            case 1:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (buf.lane_id+2 )&31;
                        tx0 = __shfl(buf.t2_reg1, friend_id0);
                        ty0 = __shfl(buf.t2_reg2, friend_id0);
                        return ((buf.lane_id < 30)? tx0: ty0);
                    case 1:
                        friend_id1 = (buf.lane_id+4 )&31;
                        tx1 = __shfl(buf.t2_reg2, friend_id1);
                        ty1 = __shfl(buf.t2_reg3, friend_id1);
                        return ((buf.lane_id < 28)? tx1: ty1);
                    case 2:
                        friend_id2 = (buf.lane_id+6 )&31;
                        tx2 = __shfl(buf.t2_reg3, friend_id2);
                        ty2 = __shfl(buf.t2_reg4, friend_id2);
                        return ((buf.lane_id < 26)? tx2: ty2);
                    case 3:
                        friend_id3 = (buf.lane_id+8 )&31;
                        tx3 = __shfl(buf.t2_reg4, friend_id3);
                        ty3 = __shfl(buf.t2_reg5, friend_id3);
                        return ((buf.lane_id < 24)? tx3: ty3);   

                }
            case 2:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (buf.lane_id+2 )&31;
                        tx0 = __shfl(buf.t3_reg1, friend_id0);
                        ty0 = __shfl(buf.t3_reg2, friend_id0);
                        return ((buf.lane_id < 30)? tx0: ty0);
                    case 1:
                        friend_id1 = (buf.lane_id+4 )&31;
                        tx1 = __shfl(buf.t3_reg2, friend_id1);
                        ty1 = __shfl(buf.t3_reg3, friend_id1);
                        return ((buf.lane_id < 28)? tx1: ty1);
                    case 2:
                        friend_id2 = (buf.lane_id+6 )&31;
                        tx2 = __shfl(buf.t3_reg3, friend_id2);
                        ty2 = __shfl(buf.t3_reg4, friend_id2);
                        return ((buf.lane_id < 26)? tx2: ty2);
                    case 3:
                        friend_id3 = (buf.lane_id+8 )&31;
                        tx3 = __shfl(buf.t3_reg4, friend_id3);
                        ty3 = __shfl(buf.t3_reg5, friend_id3);
                        return ((buf.lane_id < 24)? tx3: ty3);  

                }
        }
    }

    if(py == 0 && px == 1)
    {
        switch (l)
        {
            case 0:
                switch (tc)
                {     
                    case 0:
                        friend_id0 = (buf.lane_id+4 )&31;
                        tx0 = __shfl(buf.t1_reg1, friend_id0);
                        ty0 = __shfl(buf.t1_reg2, friend_id0);
                        return ((buf.lane_id < 28)? tx0: ty0);
                    case 1:
                        friend_id1 = (buf.lane_id+6 )&31;
                        tx1 = __shfl(buf.t1_reg2, friend_id1);
                        ty1 = __shfl(buf.t1_reg3, friend_id1);
                        return ((buf.lane_id < 26)? tx1: ty1);
                    case 2:
                        friend_id2 = (buf.lane_id+8 )&31;
                        tx2 = __shfl(buf.t1_reg3, friend_id2);
                        ty2 = __shfl(buf.t1_reg4, friend_id2);
                        return ((buf.lane_id < 24)? tx2: ty2);
                    case 3:
                        friend_id3 = (buf.lane_id+10)&31;
                        tx3 = __shfl(buf.t1_reg4, friend_id3);
                        ty3 = __shfl(buf.t1_reg5, friend_id3);
                        return ((buf.lane_id < 22)? tx3: ty3);
   
                }
            case 1:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (buf.lane_id+4 )&31;
                        tx0 = __shfl(buf.t2_reg1, friend_id0);
                        ty0 = __shfl(buf.t2_reg2, friend_id0);
                        return ((buf.lane_id < 28)? tx0: ty0);
                    case 1:
                        friend_id1 = (buf.lane_id+6 )&31;
                        tx1 = __shfl(buf.t2_reg2, friend_id1);
                        ty1 = __shfl(buf.t2_reg3, friend_id1);
                        return ((buf.lane_id < 26)? tx1: ty1);
                    case 2:
                        friend_id2 = (buf.lane_id+8 )&31;
                        tx2 = __shfl(buf.t2_reg3, friend_id2);
                        ty2 = __shfl(buf.t2_reg4, friend_id2);
                        return ((buf.lane_id < 24)? tx2: ty2);
                    case 3:
                        friend_id3 = (buf.lane_id+10)&31;
                        tx3 = __shfl(buf.t2_reg4, friend_id3);
                        ty3 = __shfl(buf.t2_reg5, friend_id3);
                        return ((buf.lane_id < 22)? tx3: ty3);  
                }
            case 2:
                switch (tc)
                {
                    case 0:
                        friend_id0 = (buf.lane_id+4 )&31;
                        tx0 = __shfl(buf.t3_reg1, friend_id0);
                        ty0 = __shfl(buf.t3_reg2, friend_id0);
                        return ((buf.lane_id < 28)? tx0: ty0);
                    case 1:
                        friend_id1 = (buf.lane_id+6 )&31;
                        tx1 = __shfl(buf.t3_reg2, friend_id1);
                        ty1 = __shfl(buf.t3_reg3, friend_id1);
                        return ((buf.lane_id < 26)? tx1: ty1);
                    case 2:
                        friend_id2 = (buf.lane_id+8 )&31;
                        tx2 = __shfl(buf.t3_reg3, friend_id2);
                        ty2 = __shfl(buf.t3_reg4, friend_id2);
                        return ((buf.lane_id < 24)? tx2: ty2);
                    case 3:
                        friend_id3 = (buf.lane_id+10)&31;
                        tx3 = __shfl(buf.t3_reg4, friend_id3);
                        ty3 = __shfl(buf.t3_reg5, friend_id3);
                        return ((buf.lane_id < 22)? tx3: ty3);
                }
        }
    }
    return 0;

}



#endif
