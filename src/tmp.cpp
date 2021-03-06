// num_regs: 6
T reg0 ;
T reg1 ;
T reg2 ;
T reg3 ;
T reg4 ;
T reg5 ;
// load to regs: 
int new_id0 ;
int new_id1 ;
new_id0 = (warp_id0<<3) + lane_id_it%10 ;
new_id1 = (warp_id1<<3) + lane_id_it/10 ;
reg0 = ACC_2D(in, new_id0, new_id1) ;
lane_id_it += 64 ;
new_id0 = (warp_id0<<3) + lane_id_it%10 ;
new_id1 = (warp_id1<<3) + lane_id_it/10 ;
reg1 = ACC_2D(in, new_id0, new_id1) ;
lane_id_it += 64 ;
new_id0 = (warp_id0<<3) + lane_id_it%10 ;
new_id1 = (warp_id1<<3) + lane_id_it/10 ;
reg2 = ACC_2D(in, new_id0, new_id1) ;
lane_id_it += 64 ;
new_id0 = (warp_id0<<3) + lane_id_it%10 ;
new_id1 = (warp_id1<<3) + lane_id_it/10 ;
reg3 = ACC_2D(in, new_id0, new_id1) ;
lane_id_it += 64 ;
new_id0 = (warp_id0<<3) + lane_id_it%10 ;
new_id1 = (warp_id1<<3) + lane_id_it/10 ;
reg4 = ACC_2D(in, new_id0, new_id1) ;
lane_id_it += 64 ;
new_id0 = (warp_id0<<3) + lane_id_it%10 ;
new_id1 = (warp_id1<<3) + lane_id_it/10 ;
new_id0 = (new_id0 < dim0+2)? new_id0 : dim0+1 ;
new_id1 = (new_id1 < dim1+2)? new_id1 : dim1+1 ;
reg5 = ACC_2D(in, new_id0, new_id1) ;
// neighbor list: 1*3*3
// job0:  0  1  2  | 10 11 12  | 20 21 22  | 
// job1: 16 17 18  | 26 27 28  | 36 37 38  | 
// job2: 32 33 34  | 42 43 44  | 52 53 54  | 
// job3: 48 49 50  | 58 59 60  |  4  5  6  | 
// process (0, 0, 0)
friend_id0 = (lane_id+ 0+((lane_id>>3)*2))&63 ;
tx0 = __shfl(reg0, friend_id0);
ty0 = __shfl(reg1, friend_id0);
return (lane_id < 52 )? tx0: ty0;
friend_id1 = (lane_id+16+((lane_id>>3)*2))&63 ;
tx1 = __shfl(reg1, friend_id1);
ty1 = __shfl(reg2, friend_id1);
return (lane_id < 40 )? tx1: ty1;
friend_id2 = (lane_id+32+((lane_id>>3)*2))&63 ;
tx2 = __shfl(reg2, friend_id2);
ty2 = __shfl(reg3, friend_id2);
return (lane_id < 26 )? tx2: ty2;
friend_id3 = (lane_id+48+((lane_id>>3)*2))&63 ;
tx3 = __shfl(reg3, friend_id3);
ty3 = __shfl(reg4, friend_id3);
return (lane_id < 14 )? tx3: ty3;
// process (1, 0, 0)
friend_id0 = (lane_id+ 1+((lane_id>>3)*2))&63 ;
tx0 = __shfl(reg0, friend_id0);
ty0 = __shfl(reg1, friend_id0);
return (lane_id < 51 )? tx0: ty0;
friend_id1 = (lane_id+17+((lane_id>>3)*2))&63 ;
tx1 = __shfl(reg1, friend_id1);
ty1 = __shfl(reg2, friend_id1);
return (lane_id < 39 )? tx1: ty1;
friend_id2 = (lane_id+33+((lane_id>>3)*2))&63 ;
tx2 = __shfl(reg2, friend_id2);
ty2 = __shfl(reg3, friend_id2);
return (lane_id < 25 )? tx2: ty2;
friend_id3 = (lane_id+49+((lane_id>>3)*2))&63 ;
tx3 = __shfl(reg3, friend_id3);
ty3 = __shfl(reg4, friend_id3);
return (lane_id < 13 )? tx3: ty3;
// process (2, 0, 0)
friend_id0 = (lane_id+ 2+((lane_id>>3)*2))&63 ;
tx0 = __shfl(reg0, friend_id0);
ty0 = __shfl(reg1, friend_id0);
return (lane_id < 50 )? tx0: ty0;
friend_id1 = (lane_id+18+((lane_id>>3)*2))&63 ;
tx1 = __shfl(reg1, friend_id1);
ty1 = __shfl(reg2, friend_id1);
return (lane_id < 38 )? tx1: ty1;
friend_id2 = (lane_id+34+((lane_id>>3)*2))&63 ;
tx2 = __shfl(reg2, friend_id2);
ty2 = __shfl(reg3, friend_id2);
return (lane_id < 24 )? tx2: ty2;
friend_id3 = (lane_id+50+((lane_id>>3)*2))&63 ;
tx3 = __shfl(reg3, friend_id3);
ty3 = __shfl(reg4, friend_id3);
return (lane_id < 12 )? tx3: ty3;
// process (0, 1, 0)
friend_id0 = (lane_id+10+((lane_id>>3)*2))&63 ;
tx0 = __shfl(reg0, friend_id0);
ty0 = __shfl(reg1, friend_id0);
return (lane_id < 44 )? tx0: ty0;
friend_id1 = (lane_id+26+((lane_id>>3)*2))&63 ;
tx1 = __shfl(reg1, friend_id1);
ty1 = __shfl(reg2, friend_id1);
return (lane_id < 32 )? tx1: ty1;
friend_id2 = (lane_id+42+((lane_id>>3)*2))&63 ;
tx2 = __shfl(reg2, friend_id2);
ty2 = __shfl(reg3, friend_id2);
return (lane_id < 18 )? tx2: ty2;
friend_id3 = (lane_id+58+((lane_id>>3)*2))&63 ;
tx3 = __shfl(reg3, friend_id3);
ty3 = __shfl(reg4, friend_id3);
tz3 = __shfl(reg5, friend_id3);
return (lane_id < 6 )? tx3: ((lane_id < 56)? ty3: tz3);
// process (1, 1, 0)
friend_id0 = (lane_id+11+((lane_id>>3)*2))&63 ;
tx0 = __shfl(reg0, friend_id0);
ty0 = __shfl(reg1, friend_id0);
return (lane_id < 43 )? tx0: ty0;
friend_id1 = (lane_id+27+((lane_id>>3)*2))&63 ;
tx1 = __shfl(reg1, friend_id1);
ty1 = __shfl(reg2, friend_id1);
return (lane_id < 31 )? tx1: ty1;
friend_id2 = (lane_id+43+((lane_id>>3)*2))&63 ;
tx2 = __shfl(reg2, friend_id2);
ty2 = __shfl(reg3, friend_id2);
return (lane_id < 17 )? tx2: ty2;
friend_id3 = (lane_id+59+((lane_id>>3)*2))&63 ;
tx3 = __shfl(reg3, friend_id3);
ty3 = __shfl(reg4, friend_id3);
tz3 = __shfl(reg5, friend_id3);
return (lane_id < 5 )? tx3: ((lane_id < 56)? ty3: tz3);
// process (2, 1, 0)
friend_id0 = (lane_id+12+((lane_id>>3)*2))&63 ;
tx0 = __shfl(reg0, friend_id0);
ty0 = __shfl(reg1, friend_id0);
return (lane_id < 42 )? tx0: ty0;
friend_id1 = (lane_id+28+((lane_id>>3)*2))&63 ;
tx1 = __shfl(reg1, friend_id1);
ty1 = __shfl(reg2, friend_id1);
return (lane_id < 30 )? tx1: ty1;
friend_id2 = (lane_id+44+((lane_id>>3)*2))&63 ;
tx2 = __shfl(reg2, friend_id2);
ty2 = __shfl(reg3, friend_id2);
return (lane_id < 16 )? tx2: ty2;
friend_id3 = (lane_id+60+((lane_id>>3)*2))&63 ;
tx3 = __shfl(reg3, friend_id3);
ty3 = __shfl(reg4, friend_id3);
tz3 = __shfl(reg5, friend_id3);
return (lane_id < 4 )? tx3: ((lane_id < 56)? ty3: tz3);
// process (0, 2, 0)
friend_id0 = (lane_id+20+((lane_id>>3)*2))&63 ;
tx0 = __shfl(reg0, friend_id0);
ty0 = __shfl(reg1, friend_id0);
return (lane_id < 36 )? tx0: ty0;
friend_id1 = (lane_id+36+((lane_id>>3)*2))&63 ;
tx1 = __shfl(reg1, friend_id1);
ty1 = __shfl(reg2, friend_id1);
return (lane_id < 24 )? tx1: ty1;
friend_id2 = (lane_id+52+((lane_id>>3)*2))&63 ;
tx2 = __shfl(reg2, friend_id2);
ty2 = __shfl(reg3, friend_id2);
tz2 = __shfl(reg4, friend_id2);
return (lane_id < 10 )? tx2: ((lane_id < 62)? ty2: tz2);
friend_id3 = (lane_id+ 4+((lane_id>>3)*2))&63 ;
tx3 = __shfl(reg4, friend_id3);
ty3 = __shfl(reg5, friend_id3);
return (lane_id < 48 )? tx3: ty3;
// process (1, 2, 0)
friend_id0 = (lane_id+21+((lane_id>>3)*2))&63 ;
tx0 = __shfl(reg0, friend_id0);
ty0 = __shfl(reg1, friend_id0);
return (lane_id < 35 )? tx0: ty0;
friend_id1 = (lane_id+37+((lane_id>>3)*2))&63 ;
tx1 = __shfl(reg1, friend_id1);
ty1 = __shfl(reg2, friend_id1);
return (lane_id < 23 )? tx1: ty1;
friend_id2 = (lane_id+53+((lane_id>>3)*2))&63 ;
tx2 = __shfl(reg2, friend_id2);
ty2 = __shfl(reg3, friend_id2);
tz2 = __shfl(reg4, friend_id2);
return (lane_id < 9 )? tx2: ((lane_id < 61)? ty2: tz2);
friend_id3 = (lane_id+ 5+((lane_id>>3)*2))&63 ;
tx3 = __shfl(reg4, friend_id3);
ty3 = __shfl(reg5, friend_id3);
return (lane_id < 48 )? tx3: ty3;
// process (2, 2, 0)
friend_id0 = (lane_id+22+((lane_id>>3)*2))&63 ;
tx0 = __shfl(reg0, friend_id0);
ty0 = __shfl(reg1, friend_id0);
return (lane_id < 34 )? tx0: ty0;
friend_id1 = (lane_id+38+((lane_id>>3)*2))&63 ;
tx1 = __shfl(reg1, friend_id1);
ty1 = __shfl(reg2, friend_id1);
return (lane_id < 22 )? tx1: ty1;
friend_id2 = (lane_id+54+((lane_id>>3)*2))&63 ;
tx2 = __shfl(reg2, friend_id2);
ty2 = __shfl(reg3, friend_id2);
tz2 = __shfl(reg4, friend_id2);
return (lane_id < 8 )? tx2: ((lane_id < 60)? ty2: tz2);
friend_id3 = (lane_id+ 6+((lane_id>>3)*2))&63 ;
tx3 = __shfl(reg4, friend_id3);
ty3 = __shfl(reg5, friend_id3);
return (lane_id < 48 )? tx3: ty3;
