#!/bin/bash
#export LD_LIBRARY_PATH=/opt/hsa/lib
export LD_LIBRARY_PATH=/opt/rocm/hsa/lib
#if [ -d /opt/hcc/bin ] ; then 
if [ -d /opt/rocm/hcc/bin ] ; then 
   #export PATH=$PATH:/opt/hcc/bin
   export PATH=$PATH:/opt/rocm/hcc/bin
else
   echo "/opt/hcc/bin missing.  Please install the hcc debian package."
   exit 1
fi
CONFOPTS=`hcc-config --install --cxxflags --ldflags`
#HSAOPTS="$CONFOPTS -Xclang -fhsa-ext "
HSAOPTS="$CONFOPTS"
echo hcc $HSAOPTS main.cpp -o main.out 
hcc $HSAOPTS main.cpp -o main.out
./main.out 2>&1  | tee result.txt
#/opt/hcc/bin/hcc --version >>matmul.out
#/opt/hcc/bin/hcc --version
/opt/rocm/hcc/bin/hcc --version >> result.txt 
/opt/rocm/hcc/bin/hcc --version
