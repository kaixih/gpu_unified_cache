#!/bin/bash
export LD_LIBRARY_PATH=/opt/rocm/hsa/lib
HCCVER="hcc-lc"
if [ $1 = "hsail" ]
then 
    HCCVER="hcc-hsail" 
else
    HCCVER="hcc-lc" 
fi
if [ -d /opt/rocm/$HCCVER/bin ] ; then 
   export PATH=$PATH:/opt/rocm/$HCCVER/bin
else
   echo "/opt/$HCCVER/bin missing.  Please install the hcc debian package."
   exit 1
fi
CONFOPTS=`hcc-config --install --cxxflags --ldflags`
#HSAOPTS="$CONFOPTS -Xclang -fhsa-ext "
HSAOPTS="$CONFOPTS -O3"
echo hcc $HSAOPTS main_hc.cpp -o main_hc.out 
hcc $HSAOPTS main_hc.cpp -o main_hc.out
./main_hc.out 2>&1  | tee result.txt
#/opt/hcc/bin/hcc --version >>matmul.out
#/opt/hcc/bin/hcc --version
/opt/rocm/$HCCVER/bin/hcc --version >> result.txt 
/opt/rocm/$HCCVER/bin/hcc --version
