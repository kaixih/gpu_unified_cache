#ifndef _H_STEN_MACRO
#define _H_STEN_MACRO

#define ACC_3D(array,_z,_y,_x) array[(_z)*(m+2*halo)*(n+2*halo)+(_y)*(n+2*halo)+(_x)]
// #define LOC_3D(_z,_y,_x) local[(_z)*(10*6)+(_y)*(10)+(_x)]
#define LOC_3D(_z,_y,_x) local[(_z)*(10*10)+(_y)*(10)+(_x)]

#define ACC_2D(array,_y,_x) array[off+(_y)*(n+2*halo)+(_x)]
// #define LOC_2D(_y,_x) local[loff+(_y)*(34)+(_x)]
#define LOC_2D(_y,_x) local[(_y)*(66)+(_x)]

#define NONE   11
#define BRANCH 22 
#define CYCLIC 33 

#endif
