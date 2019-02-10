#!/bin/bash
export LD_LIBRARY_PATH=/opt/rocm/hsa/lib
APPNAME="main_1d1_hc"
#APPNAME="main_1d7_hc"
#APPNAME="main_2d5_hc"
#APPNAME="main_2d9_hc"
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
   echo "/opt/rocm/$HCCVER/bin missing.  Please install the hcc debian package."
   exit 1
fi
CONFOPTS=`hcc-config --install --cxxflags --ldflags`
#HSAOPTS="$CONFOPTS -Xclang -fhsa-ext "
HSAOPTS="$CONFOPTS -O3 -std=c++11" 
echo hcc $HSAOPTS ${APPNAME}.cpp -o ${APPNAME}.out 
hcc $HSAOPTS ${APPNAME}.cpp -o ${APPNAME}.out
./${APPNAME}.out 2>&1  | tee result.txt
#/opt/hcc/bin/hcc --version >>matmul.out
#/opt/hcc/bin/hcc --version
/opt/rocm/$HCCVER/bin/hcc --version >> result.txt 
/opt/rocm/$HCCVER/bin/hcc --version
