#ifndef _HCC_UBUF_3D7_2D
#define _HCC_UBUF_3D7_2D

#include <hc.hpp>

using namespace hc;

template<class T>
class StenBuffer2D
{
public:
    virtual T load(int idy, int idz)restrict(amp)=0;
    virtual void store(T v, int idy, int idz)restrict(amp)=0;
    virtual void glb2buf(array<T> &in, int off, int type=NONE)restrict(amp)=0;
};

template<class T>
class L1Buffer2D: public StenBuffer2D<T>
{
private:
    array<T> *l1m;
    int off;
    int m, n;
    int halo;
public:
    int gidx, gidy;
public:
    T load(int idx, int idy) restrict(amp)
    {
        return ACC_2D((*l1m), idx, idy);
    }
    void store(T v, int idx, int idy) restrict(amp)
    {
        ACC_2D((*l1m), idx, idy) = v;
    }
    L1Buffer2D(int j, int i, int h, tiled_index<3> &tidx) restrict(amp): m(j), n(i), halo(h)
    {
        gidx = tidx.global[2] + halo;
        gidy = tidx.global[1] + halo;

        off = 0;
    }
    void glb2buf(array<T> &in, int o, int type=NONE) restrict(amp)
    {
        l1m = &in;
        off = o;
    }
    // in hcc, this works
    // T Sten_L1M_Fetch(int pz, int py, int px, int tc=0) restrict(amp)
    // {
        // return load(gidz+pz, gidy+py, gidx+px);
    // }
};

template<class T>
T Sten_L1M_Fetch2D(L1Buffer2D<T> buf, int py, int px, int tc=0) restrict(amp)
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
    tiled_index<3> *tidx;
public:
    int off;
    int gidx, gidy;
    int lidx, lidy;
public:
    T load(int idy, int idx) restrict(amp)
    {
        return LOC_2D(idy, idx);
    }
    void store(T v, int idy, int idx) restrict(amp)
    {
        LOC_2D(idy, idx) = v;
    }
    LDSBuffer2D(int j, int i, int h, tiled_index<3> &t) restrict(amp): m(j), n(i), halo(h)
    {
        tile_static T sm[66*6];
        local = sm;
        gidx = t.global[2] + halo;
        gidy = t.global[1] + halo;
        lidx = t.local[2] + halo;
        lidy = t.local[1] + halo;
        tidx = &t;
    }
    void glb2buf(array<T> &in, int o, int type=NONE) restrict(amp)
    {
        off = o;
        if(type == BRANCH)
        {
            LOC_2D(lidy,lidx) = ACC_2D(in,gidy,gidx);

            if(lidx == halo) LOC_2D(lidy  ,lidx-1) = ACC_2D(in,gidy  ,gidx-1);
            if(lidx == 64  ) LOC_2D(lidy  ,lidx+1) = ACC_2D(in,gidy  ,gidx+1);
            if(lidy == halo) LOC_2D(lidy-1,lidx  ) = ACC_2D(in,gidy-1,gidx  );
            if(lidy == 4   ) LOC_2D(lidy+1,lidx  ) = ACC_2D(in,gidy+1,gidx  );

        } else if(type == CYCLIC)
        {
            /*
            int lane_id = tidx->local[2] + tidx->local[1] * tidx->tile_dim[2] +
                tidx->local[0] * tidx->tile_dim[2] * tidx->tile_dim[1];
            int blk_id_x = tidx->tile[2];
            int blk_id_y = tidx->tile[1];
            int blk_id_z = tidx->tile[0];
            int new_i, new_j, new_k, new_li, new_lj, new_lk;
            new_i  = (blk_id_x<<3) + lane_id%10 ;     
            new_j  = (blk_id_y<<3) + (lane_id/10)%10 ; 
            new_k  = (blk_id_z<<2) + lane_id/100 ;     
            new_li = lane_id%10;
            new_lj = (lane_id/10)%10;
            new_lk = lane_id/100;
            LOC_3D(new_lk,new_lj,new_li) = ACC_3D(in,new_k,new_j,new_i);
            new_i  = (blk_id_x<<3) + (lane_id+256)%10 ;
            new_j  = (blk_id_y<<3) + ((lane_id+256)/10)%10 ;
            new_k  = (blk_id_z<<2) + (lane_id+256)/100 ;
            new_li = (lane_id+256)%10;
            new_lj = ((lane_id+256)/10)%10 ;
            new_lk = (lane_id+256)/100; 
            LOC_3D(new_lk,new_lj,new_li) = ACC_3D(in,new_k,new_j,new_i);
            new_i  = (blk_id_x<<3) + (lane_id+512)%10 ;
            new_j  = (blk_id_y<<3) + ((lane_id+512)/10)%10 ;
            new_k  = (blk_id_z<<2) + (lane_id+512)/100 ;
            new_li = (lane_id+512)%10;
            new_lj = ((lane_id+512)/10)%10;
            new_lk = (lane_id+512)/100; 
            if(new_li < 10 &&  new_lj < 10 && new_lk < 6 )
                LOC_3D(new_lk,new_lj,new_li) = ACC_3D(in,new_k,new_j,new_i);
                */
        
        }
        tidx->barrier.wait();
    }
    
    // T Sten_LDS_Fetch(int pz, int py, int px, int tc=0) restrict(amp)
    // {
        // return load(lidz+pz, lidy+py, lidx+px);
    // }
};

template<class T>
T Sten_LDS_Fetch2D(LDSBuffer2D<T> buf, int py, int px, int tc=0) restrict(amp)
{
    return buf.load(buf.lidy+py, buf.lidx+px);
}

/*
template<class T>
class REGBuffer: public StenBuffer<T>
{
private:
    int z, m, n;
    int halo;
public:
    T reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9;
    int gidx, gidy, gidz;
    int lane_id;
    int tc;
public:
    T load(int idx, int idy, int idz) restrict(amp) // deprecated
    { return 0;}
    void store(T v, int idx, int idy, int idz) restrict(amp) // deprecated
    {}
    REGBuffer(int k, int j, int i, int h, tiled_index<3> &t, int c=1) restrict(amp): 
        z(k), m(j), n(i), halo(h), tc(c)
    {
        lane_id = __lane_id();
        switch(tc)
        {
            case(1):
            {
                gidx = t.global[2] + halo;
                gidy = t.global[1] + halo;
                gidz = t.global[0] + halo;
                break;
            }
            case(2):
            {
                gidx = t.global[2] + halo;
                gidy = t.global[1] + halo;
                gidz = (((t.global[0])>>0)<<1) + halo; 
                break;
            }
            case(4):
            {
                gidx = t.global[2] + halo;
                gidy = t.global[1] + halo;
                gidz = (((t.global[0])>>0)<<2) + halo; 
                break;
            }
        }
    }

    void glb2buf(array<T> &in, int type=CYCLIC) restrict(amp)
    {
        if(type == CYCLIC)
        {
            switch (tc)
            {
                case 1:
                {
                    int lane_id_it = lane_id;
                    int warp_id_x = (gidx-halo)>>3; // because the warp dimensions are 
                    int warp_id_y = (gidy-halo)>>3; // 1x8x8, warp_ids are division of 
                    int warp_id_z = (gidz-halo)>>0; // there numbers
                    int new_i, new_j, new_k;
                    new_i = (warp_id_x<<3) + lane_id_it%10;      // 10 is extended dimension of i
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10; // 10 is extended dimension of j 
                    new_k = (warp_id_z<<0) + lane_id_it/100;     // 100 is extended area of ixj = 10x10
                    reg0 = ACC_3D(in, new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    reg1 = ACC_3D(in, new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    reg2 = ACC_3D(in, new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    reg3 = ACC_3D(in, new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
                    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
                    new_k = (new_k < z+2*halo)? new_k: z+2*halo-1;
                    reg4 = ACC_3D(in, new_k, new_j, new_i);

                    return ;
                }
                case 2:
                {
                    int lane_id_it = lane_id;
                    int warp_id_x = (gidx-halo)>>3; // because the warp dimensions are 
                    int warp_id_y = (gidy-halo)>>3; // 1x8x8, warp_ids are division of 
                    int warp_id_z = (gidz-halo)>>0; // there numbers
                    int new_i, new_j, new_k;
                    new_i = (warp_id_x<<3) + lane_id_it%10;      // 10 is extended dimension of i
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10; // 10 is extended dimension of j 
                    new_k = (warp_id_z<<0) + lane_id_it/100;     // 100 is extended area of ixj = 10x10
                    reg0 = ACC_3D(in,new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    reg1 = ACC_3D(in,new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    reg2 = ACC_3D(in,new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    reg3 = ACC_3D(in,new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    reg4 = ACC_3D(in,new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    reg5 = ACC_3D(in,new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
                    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
                    new_k = (new_k < z+2*halo)? new_k: z+2*halo-1;
                    reg6 = ACC_3D(in,new_k, new_j, new_i);

                    return;
                }
                case 4:
                {
                    int lane_id_it = lane_id;
                    int warp_id_x = (gidx-halo)>>3; // because the warp dimensions are 
                    int warp_id_y = (gidy-halo)>>3; // 1x8x8, warp_ids are division of 
                    int warp_id_z = (gidz-halo)>>0; // there numbers
                    int new_i, new_j, new_k;
                    new_i = (warp_id_x<<3) + lane_id_it%10;      // 10 is extended dimension of i
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10; // 10 is extended dimension of j 
                    new_k = (warp_id_z<<0) + lane_id_it/100;     // 100 is extended area of ixj = 10x10
                    reg0 = ACC_3D(in,new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    reg1 = ACC_3D(in,new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    reg2 = ACC_3D(in,new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    reg3 = ACC_3D(in,new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    reg4 = ACC_3D(in,new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    reg5 = ACC_3D(in,new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    reg6 = ACC_3D(in,new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    reg7 = ACC_3D(in,new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    reg8 = ACC_3D(in,new_k, new_j, new_i);
                    lane_id_it += 64;
                    new_i = (warp_id_x<<3) + lane_id_it%10;
                    new_j = (warp_id_y<<3) + (lane_id_it/10)%10;
                    new_k = (warp_id_z<<0) + lane_id_it/100;
                    new_i = (new_i < n+2*halo)? new_i: n+2*halo-1;
                    new_j = (new_j < m+2*halo)? new_j: m+2*halo-1;
                    new_k = (new_k < z+2*halo)? new_k: z+2*halo-1;
                    reg9 = ACC_3D(in,new_k, new_j, new_i);
                    return;
                }
            }
        }
    }
    T Sten_REG_Fetch(int pz, int py, int px, int tc=0) restrict(amp)
    {
        int friend_id0;
        int friend_id1;
        int friend_id2;
        int friend_id3;
        T tx0, ty0, tz0;
        T tx1, ty1, tz1;
        T tx2, ty2, tz2;
        T tx3, ty3, tz3;
        if(pz == -1 && py == 0 && px == 0)
        {
            switch (tc)
            {
                case 0:
                    friend_id0 = (lane_id+11+((lane_id>>3)<<1))&63;
                    tx0 = __shfl(reg0, friend_id0);
                    ty0 = __shfl(reg1, friend_id0);
                    return ((lane_id < 43)? tx0: ty0);

                case 1:
                    friend_id1 = (lane_id+47+((lane_id>>3)<<1))&63;
                    tx1 = __shfl(reg1, friend_id1);
                    ty1 = __shfl(reg2, friend_id1);
                    return ((lane_id < 15)? tx1: ty1);

                case 2:
                    friend_id2 = (lane_id+19+((lane_id>>3)<<1))&(63);
                    tx2 = __shfl(reg3, friend_id2);
                    ty2 = __shfl(reg4, friend_id2);
                    return ((lane_id < 37)? tx2: ty2);

                case 3:
                    friend_id3 = (lane_id+55+((lane_id>>3)<<1))&(63);
                    tx3 = __shfl(reg4, friend_id3);
                    ty3 = __shfl(reg5, friend_id3);
                    tz3 = __shfl(reg6, friend_id3);
                    return ((lane_id < 8 )? tx3: ((lane_id < 59)? ty3: tz3));
            }
        } 
        if(pz == 0 && py == -1 && px == 0)
        {
            switch (tc)
            {
                case 0:
                    friend_id0 = (lane_id+37+((lane_id>>3)<<1))&63;
                    tx0 = __shfl(reg1, friend_id0);
                    ty0 = __shfl(reg2, friend_id0);
                    return ((lane_id < 23)? tx0: ty0);

                case 1:
                    friend_id1 = (lane_id+9 +((lane_id>>3)<<1))&63;
                    tx1 = __shfl(reg3, friend_id1);
                    ty1 = __shfl(reg4, friend_id1);
                    return ((lane_id < 45)? tx1: ty1);

                case 2:
                   friend_id2 = (lane_id+45+((lane_id>>3)<<1))&(63);
                   tx2 = __shfl(reg4, friend_id2);
                   ty2 = __shfl(reg5, friend_id2);
                   return ((lane_id < 16)? tx2: ty2);

                case 3:
                   friend_id3 = (lane_id+17+((lane_id>>3)<<1))&(63);
                   tx3 = __shfl(reg6, friend_id3);
                   ty3 = __shfl(reg7, friend_id3);
                   return ((lane_id < 39)? tx3: ty3);
            }
        }
        if(pz == 0 && py == 0 && px == -1)
        {
            switch (tc)
            {
                case 0:
                    friend_id0 = (lane_id+46+((lane_id>>3)<<1))&(63);
                    tx0 = __shfl(reg1, friend_id0);
                    ty0 = __shfl(reg2, friend_id0);
                    return ((lane_id < 16)? tx0: ty0);

                case 1:
                    friend_id1 = (lane_id+18+((lane_id>>3)<<1))&(63);
                    tx1 = __shfl(reg3, friend_id1);
                    ty1 = __shfl(reg4, friend_id1);
                    return ((lane_id < 38)? tx1: ty1);

                case 2:
                    friend_id2 = (lane_id+54+((lane_id>>3)<<1))&(63);
                    tx2 = __shfl(reg4, friend_id2);
                    ty2 = __shfl(reg5, friend_id2);
                    tz2 = __shfl(reg6, friend_id2);
                    return ((lane_id < 8 )? tx2: ((lane_id < 60)? ty2: tz2));

                case 3:
                    friend_id3 = (lane_id+26+((lane_id>>3)<<1))&(63);
                    tx3 = __shfl(reg6, friend_id3);
                    ty3 = __shfl(reg7, friend_id3);
                    return ((lane_id < 32)? tx3: ty3);
            }
        }
        if(pz == 0 && py == 0 && px == 0)
        {
            switch (tc)
            {
                case 0:
                    friend_id0 = (lane_id+47+((lane_id>>3)<<1))&(63);
                    tx0 = __shfl(reg1, friend_id0);
                    ty0 = __shfl(reg2, friend_id0);
                    return ((lane_id < 15)? tx0: ty0);

                case 1:
                    friend_id1 = (lane_id+19+((lane_id>>3)<<1))&(63);
                    tx1 = __shfl(reg3, friend_id1);
                    ty1 = __shfl(reg4, friend_id1);
                    return ((lane_id < 37)? tx1: ty1);
                case 2:
                    friend_id2 = (lane_id+55+((lane_id>>3)<<1))&(63);
                    tx2 = __shfl(reg4, friend_id2);
                    ty2 = __shfl(reg5, friend_id2);
                    tz2 = __shfl(reg6, friend_id2);
                    return ((lane_id < 8 )? tx2: ((lane_id < 59)? ty2: tz2));
                case 3:
                    friend_id3 = (lane_id+27+((lane_id>>3)<<1))&(63);
                    tx3 = __shfl(reg6, friend_id3);
                    ty3 = __shfl(reg7, friend_id3);
                    return ((lane_id < 31)? tx3: ty3);

            }
        }
        if(pz == 0 && py == 0 && px == 1)
        {
            switch (tc)
            {
                case 0:
                    friend_id0 = (lane_id+48+((lane_id>>3)<<1))&(63);
                    tx0 = __shfl(reg1, friend_id0);
                    ty0 = __shfl(reg2, friend_id0);
                    return ((lane_id < 14)? tx0: ty0);

                case 1:
                    friend_id1 = (lane_id+20+((lane_id>>3)<<1))&(63);
                    tx1 = __shfl(reg3, friend_id1);
                    ty1 = __shfl(reg4, friend_id1);
                    return ((lane_id < 36)? tx1: ty1);
                case 2:
                    friend_id2 = (lane_id+56+((lane_id>>3)<<1))&(63);
                    tx2 = __shfl(reg4, friend_id2);
                    ty2 = __shfl(reg5, friend_id2);
                    tz2 = __shfl(reg6, friend_id2);
                    return ((lane_id < 8 )? tx2: ((lane_id < 58)? ty2: tz2));
                case 3:
                    friend_id3 = (lane_id+28+((lane_id>>3)<<1))&(63);
                    tx3 = __shfl(reg6, friend_id3);
                    ty3 = __shfl(reg7, friend_id3);
                    return ((lane_id < 30)? tx3: ty3);

            }
        }
        if(pz == 0 && py == 1 && px == 0)
        {
            switch (tc)
            {
                case 0:
                    friend_id0 = (lane_id+57+((lane_id>>3)<<1))&(63);
                    tx0 = __shfl(reg1, friend_id0);
                    ty0 = __shfl(reg2, friend_id0);
                    tz0 = __shfl(reg3, friend_id0);
                    return ((lane_id < 7 )? tx0: ((lane_id < 57)? ty0: tz0));

                case 1:
                    friend_id1 = (lane_id+29+((lane_id>>3)<<1))&(63);
                    tx1 = __shfl(reg3, friend_id1);
                    ty1 = __shfl(reg4, friend_id1);
                    return ((lane_id < 29)? tx1: ty1);
                case 2:
                    friend_id2 = (lane_id+1 +((lane_id>>3)<<1))&(63);
                    tx2 = __shfl(reg5, friend_id2);
                    ty2 = __shfl(reg6, friend_id2);
                    return ((lane_id < 51)? tx2: ty2);
                case 3:
                    friend_id3 = (lane_id+37+((lane_id>>3)<<1))&(63);
                    tx3 = __shfl(reg6, friend_id3);
                    ty3 = __shfl(reg7, friend_id3);
                    return ((lane_id < 23)? tx3: ty3);

            }
        }
        if(pz == 1 && py == 0 && px == 0)
        {
            switch (tc)
            {
                case 0:
                    friend_id0 = (lane_id+19+((lane_id>>3)<<1))&(63);
                    tx0 = __shfl(reg3, friend_id0);
                    ty0 = __shfl(reg4, friend_id0);
                    return ((lane_id < 37)? tx0: ty0);

                case 1:
                    friend_id1 = (lane_id+55+((lane_id>>3)<<1))&(63);
                    tx1 = __shfl(reg4, friend_id1);
                    ty1 = __shfl(reg5, friend_id1);
                    tz1 = __shfl(reg6, friend_id1);
                    return ((lane_id < 8 )? tx1: ((lane_id < 59)? ty1: tz1));

                case 2:
                    friend_id2 = (lane_id+27+((lane_id>>3)<<1))&(63);
                    tx2 = __shfl(reg6, friend_id2);
                    ty2 = __shfl(reg7, friend_id2);
                    return ((lane_id < 31)? tx2: ty2);
                case 3:
                    friend_id3 = (lane_id+63+((lane_id>>3)<<1))&(63);
                    tx3 = __shfl(reg7, friend_id3);
                    ty3 = __shfl(reg8, friend_id3);
                    tz3 = __shfl(reg9, friend_id3);
                    return ((lane_id < 1 )? tx3: ((lane_id < 53)? ty3: tz3));
            }
        }
        return 0;
    }
};

template<class T>
T Sten_REG_Fetch(REGBuffer<T> buf, int pz, int py, int px, int tc=0) restrict(amp)
{
    int friend_id0;
    int friend_id1;
    int friend_id2;
    int friend_id3;
    T tx0, ty0, tz0;
    T tx1, ty1, tz1;
    T tx2, ty2, tz2;
    T tx3, ty3, tz3;
    if(pz == -1 && py == 0 && px == 0)
    {
        switch (tc)
        {
            case 0:
                friend_id0 = (buf.lane_id+11+((buf.lane_id>>3)<<1))&63;
                tx0 = __shfl(buf.reg0, friend_id0);
                ty0 = __shfl(buf.reg1, friend_id0);
                return ((buf.lane_id < 43)? tx0: ty0);

            case 1:
                friend_id1 = (buf.lane_id+47+((buf.lane_id>>3)<<1))&63;
                tx1 = __shfl(buf.reg1, friend_id1);
                ty1 = __shfl(buf.reg2, friend_id1);
                return ((buf.lane_id < 15)? tx1: ty1);

            case 2:
                friend_id2 = (buf.lane_id+19+((buf.lane_id>>3)<<1))&(63);
                tx2 = __shfl(buf.reg3, friend_id2);
                ty2 = __shfl(buf.reg4, friend_id2);
                return ((buf.lane_id < 37)? tx2: ty2);

            case 3:
                friend_id3 = (buf.lane_id+55+((buf.lane_id>>3)<<1))&(63);
                tx3 = __shfl(buf.reg4, friend_id3);
                ty3 = __shfl(buf.reg5, friend_id3);
                tz3 = __shfl(buf.reg6, friend_id3);
                return ((buf.lane_id < 8 )? tx3: ((buf.lane_id < 59)? ty3: tz3));
        }
    } 
    if(pz == 0 && py == -1 && px == 0)
    {
        switch (tc)
        {
            case 0:
                friend_id0 = (buf.lane_id+37+((buf.lane_id>>3)<<1))&63;
                tx0 = __shfl(buf.reg1, friend_id0);
                ty0 = __shfl(buf.reg2, friend_id0);
                return ((buf.lane_id < 23)? tx0: ty0);

            case 1:
                friend_id1 = (buf.lane_id+9 +((buf.lane_id>>3)<<1))&63;
                tx1 = __shfl(buf.reg3, friend_id1);
                ty1 = __shfl(buf.reg4, friend_id1);
                return ((buf.lane_id < 45)? tx1: ty1);

            case 2:
               friend_id2 = (buf.lane_id+45+((buf.lane_id>>3)<<1))&(63);
               tx2 = __shfl(buf.reg4, friend_id2);
               ty2 = __shfl(buf.reg5, friend_id2);
               return ((buf.lane_id < 16)? tx2: ty2);

            case 3:
               friend_id3 = (buf.lane_id+17+((buf.lane_id>>3)<<1))&(63);
               tx3 = __shfl(buf.reg6, friend_id3);
               ty3 = __shfl(buf.reg7, friend_id3);
               return ((buf.lane_id < 39)? tx3: ty3);
        }
    }
    if(pz == 0 && py == 0 && px == -1)
    {
        switch (tc)
        {
            case 0:
                friend_id0 = (buf.lane_id+46+((buf.lane_id>>3)<<1))&(63);
                tx0 = __shfl(buf.reg1, friend_id0);
                ty0 = __shfl(buf.reg2, friend_id0);
                return ((buf.lane_id < 16)? tx0: ty0);

            case 1:
                friend_id1 = (buf.lane_id+18+((buf.lane_id>>3)<<1))&(63);
                tx1 = __shfl(buf.reg3, friend_id1);
                ty1 = __shfl(buf.reg4, friend_id1);
                return ((buf.lane_id < 38)? tx1: ty1);

            case 2:
                friend_id2 = (buf.lane_id+54+((buf.lane_id>>3)<<1))&(63);
                tx2 = __shfl(buf.reg4, friend_id2);
                ty2 = __shfl(buf.reg5, friend_id2);
                tz2 = __shfl(buf.reg6, friend_id2);
                return ((buf.lane_id < 8 )? tx2: ((buf.lane_id < 60)? ty2: tz2));

            case 3:
                friend_id3 = (buf.lane_id+26+((buf.lane_id>>3)<<1))&(63);
                tx3 = __shfl(buf.reg6, friend_id3);
                ty3 = __shfl(buf.reg7, friend_id3);
                return ((buf.lane_id < 32)? tx3: ty3);
        }
    }
    if(pz == 0 && py == 0 && px == 0)
    {
        switch (tc)
        {
            case 0:
                friend_id0 = (buf.lane_id+47+((buf.lane_id>>3)<<1))&(63);
                tx0 = __shfl(buf.reg1, friend_id0);
                ty0 = __shfl(buf.reg2, friend_id0);
                return ((buf.lane_id < 15)? tx0: ty0);

            case 1:
                friend_id1 = (buf.lane_id+19+((buf.lane_id>>3)<<1))&(63);
                tx1 = __shfl(buf.reg3, friend_id1);
                ty1 = __shfl(buf.reg4, friend_id1);
                return ((buf.lane_id < 37)? tx1: ty1);
            case 2:
                friend_id2 = (buf.lane_id+55+((buf.lane_id>>3)<<1))&(63);
                tx2 = __shfl(buf.reg4, friend_id2);
                ty2 = __shfl(buf.reg5, friend_id2);
                tz2 = __shfl(buf.reg6, friend_id2);
                return ((buf.lane_id < 8 )? tx2: ((buf.lane_id < 59)? ty2: tz2));
            case 3:
                friend_id3 = (buf.lane_id+27+((buf.lane_id>>3)<<1))&(63);
                tx3 = __shfl(buf.reg6, friend_id3);
                ty3 = __shfl(buf.reg7, friend_id3);
                return ((buf.lane_id < 31)? tx3: ty3);

        }
    }
    if(pz == 0 && py == 0 && px == 1)
    {
        switch (tc)
        {
            case 0:
                friend_id0 = (buf.lane_id+48+((buf.lane_id>>3)<<1))&(63);
                tx0 = __shfl(buf.reg1, friend_id0);
                ty0 = __shfl(buf.reg2, friend_id0);
                return ((buf.lane_id < 14)? tx0: ty0);

            case 1:
                friend_id1 = (buf.lane_id+20+((buf.lane_id>>3)<<1))&(63);
                tx1 = __shfl(buf.reg3, friend_id1);
                ty1 = __shfl(buf.reg4, friend_id1);
                return ((buf.lane_id < 36)? tx1: ty1);
            case 2:
                friend_id2 = (buf.lane_id+56+((buf.lane_id>>3)<<1))&(63);
                tx2 = __shfl(buf.reg4, friend_id2);
                ty2 = __shfl(buf.reg5, friend_id2);
                tz2 = __shfl(buf.reg6, friend_id2);
                return ((buf.lane_id < 8 )? tx2: ((buf.lane_id < 58)? ty2: tz2));
            case 3:
                friend_id3 = (buf.lane_id+28+((buf.lane_id>>3)<<1))&(63);
                tx3 = __shfl(buf.reg6, friend_id3);
                ty3 = __shfl(buf.reg7, friend_id3);
                return ((buf.lane_id < 30)? tx3: ty3);

        }
    }
    if(pz == 0 && py == 1 && px == 0)
    {
        switch (tc)
        {
            case 0:
                friend_id0 = (buf.lane_id+57+((buf.lane_id>>3)<<1))&(63);
                tx0 = __shfl(buf.reg1, friend_id0);
                ty0 = __shfl(buf.reg2, friend_id0);
                tz0 = __shfl(buf.reg3, friend_id0);
                return ((buf.lane_id < 7 )? tx0: ((buf.lane_id < 57)? ty0: tz0));

            case 1:
                friend_id1 = (buf.lane_id+29+((buf.lane_id>>3)<<1))&(63);
                tx1 = __shfl(buf.reg3, friend_id1);
                ty1 = __shfl(buf.reg4, friend_id1);
                return ((buf.lane_id < 29)? tx1: ty1);
            case 2:
                friend_id2 = (buf.lane_id+1 +((buf.lane_id>>3)<<1))&(63);
                tx2 = __shfl(buf.reg5, friend_id2);
                ty2 = __shfl(buf.reg6, friend_id2);
                return ((buf.lane_id < 51)? tx2: ty2);
            case 3:
                friend_id3 = (buf.lane_id+37+((buf.lane_id>>3)<<1))&(63);
                tx3 = __shfl(buf.reg6, friend_id3);
                ty3 = __shfl(buf.reg7, friend_id3);
                return ((buf.lane_id < 23)? tx3: ty3);

        }
    }
    if(pz == 1 && py == 0 && px == 0)
    {
        switch (tc)
        {
            case 0:
                friend_id0 = (buf.lane_id+19+((buf.lane_id>>3)<<1))&(63);
                tx0 = __shfl(buf.reg3, friend_id0);
                ty0 = __shfl(buf.reg4, friend_id0);
                return ((buf.lane_id < 37)? tx0: ty0);

            case 1:
                friend_id1 = (buf.lane_id+55+((buf.lane_id>>3)<<1))&(63);
                tx1 = __shfl(buf.reg4, friend_id1);
                ty1 = __shfl(buf.reg5, friend_id1);
                tz1 = __shfl(buf.reg6, friend_id1);
                return ((buf.lane_id < 8 )? tx1: ((buf.lane_id < 59)? ty1: tz1));

            case 2:
                friend_id2 = (buf.lane_id+27+((buf.lane_id>>3)<<1))&(63);
                tx2 = __shfl(buf.reg6, friend_id2);
                ty2 = __shfl(buf.reg7, friend_id2);
                return ((buf.lane_id < 31)? tx2: ty2);
            case 3:
                friend_id3 = (buf.lane_id+63+((buf.lane_id>>3)<<1))&(63);
                tx3 = __shfl(buf.reg7, friend_id3);
                ty3 = __shfl(buf.reg8, friend_id3);
                tz3 = __shfl(buf.reg9, friend_id3);
                return ((buf.lane_id < 1 )? tx3: ((buf.lane_id < 53)? ty3: tz3));
        }
    }
    return 0;
}
*/

#endif
