#ifndef _METRICS_H
#define _METRICS_H

// operations needed in specific stencils
#define OPS_1D3  5
#define OPS_1D7  13 
#define OPS_2D5  9 
#define OPS_2D9  17 
#define OPS_3D7  13 
#define OPS_3D27 53 

#define ARGC_1D3  3
#define ARGC_1D7  7 
#define ARGC_2D5  5 
#define ARGC_2D9  9 
#define ARGC_3D7  7 
#define ARGC_3D27 27 

float GetGFLOPS(int z, int m, int n, int count, int ops, float time) 
{
    float f = (z*m*n)*(float)(ops)*(float)(count)/time * 1.0e-09;
    return f;
}

float GetThroughput(int argc, int z, int m, int n, int count, float time, int data_type_size) 
{
    return (float)(argc + 2.0) * (z*m*n) * data_type_size * ((float)count)
            / time * 1.0e-09;    
}

#define  IN_3D(_z,_y,_x)  in[(_z)*(m+2*halo)*(n+2*halo)+(_y)*(n+2*halo)+(_x)]
#define OUT_3D(_z,_y,_x) out[(_z)*(m+2*halo)*(n+2*halo)+(_y)*(n+2*halo)+(_x)]

#define LOC_3D(_z,_y,_x) local[(_z)][(_y)][(_x)]

#define  IN_2D(_y,_x)  in[(_y)*(n+2*halo)+(_x)]
#define OUT_2D(_y,_x) out[(_y)*(n+2*halo)+(_x)]

#define LOC_2D(_y,_x) local[(_y)][(_x)]

#define  IN_1D(_x)  in[_x]
#define OUT_1D(_x) out[_x]

#endif
