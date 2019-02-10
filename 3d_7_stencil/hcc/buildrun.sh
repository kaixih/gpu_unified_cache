#!/bin/bash
export LD_LIBRARY_PATH=/opt/rocm/hsa/lib
HCCVER="hcc-lc"

if [ $1 = "hsail" ]
then 
    HCCVER="hcc-hsail" 
else
    HCCVER="hcc-lc" 
fi

if [ -d /opt/rocm/hcc/bin ] ; then 
   export PATH=$PATH:/opt/rocm/$HCCVER/bin
else
   echo "/opt/hcc/bin missing.  Please install the hcc debian package."
   exit 1
fi

APP=glb
CONFOPTS=`hcc-config --install --cxxflags --ldflags`
#HSAOPTS="$CONFOPTS -Xclang -fhsa-ext "
HSAOPTS="$CONFOPTS -I../../include -DDATA_TYPE=float -fopenmp" 
echo hcc $HSAOPTS main_coef_${APP}_halo.cpp -o main_coef_${APP}_halo.out 
hcc $HSAOPTS main_coef_${APP}_halo.cpp -o main_coef_${APP}_halo.out

./main_coef_${APP}_halo.out 2>&1  | tee result.txt
#/opt/hcc/bin/hcc --version >>matmul.out
#/opt/hcc/bin/hcc --version
/opt/rocm/$HCCVER/bin/hcc --version >> result.txt 
/opt/rocm/$HCCVER/bin/hcc --version
