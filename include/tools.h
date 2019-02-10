#ifndef _H_TOOLS
#define _H_TOOLS

#include "sten_macro.h"

void Init_Input(DATA_TYPE *in, int z, int m, int n, int halo, unsigned int seed)
{
#ifdef __DEBUG
    srand(1);
#else
    srand(seed);
#endif

    for(int k = halo; k < z+halo; k++)
        for(int j = halo; j < m+halo; j++)
            for(int i = halo; i < n+halo; i++)
#ifdef __DEBUG
                ACC_3D(in,k,j,i) = 1;//(DATA_TYPE)rand()*100.0 / ((long)RAND_MAX);
#else
                ACC_3D(in,k,j,i) = (DATA_TYPE)rand()*100.0 / ((long)RAND_MAX);
#endif
}

void Show_Me(DATA_TYPE *in, int z, int m, int n, int halo, std::string prompt)
{
    std::cout << prompt << std::endl;
    for(int k = 0; k < z+2*halo; k++)
    {
        for(int j = 0; j < m+2*halo; j++)
        {
            for(int i = 0; i < n+2*halo; i++)
                std::cout << ACC_3D(in,k,j,i) << ",";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
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

#endif
