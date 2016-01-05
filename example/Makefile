CXX=g++
BLAS=-lopenblas -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 
CUDA=-DMSHADOW_USE_CUDA=1

CFLAGS=-I ../include -Wall -O3 -msse3 -funroll-loops -Wno-unused-parameter -Wno-unknown-pragmas -fopenmp 
LDFLAGS=-L ../lib -lmxnet $(BLAS) $(CUDA)

SRC=../src/MxNetCpp.cpp ../src/symbol.cpp ../src/executor.cpp ../src/ndarray.cpp 

mlp: ./mlp.cpp
	$(CXX) -std=c++0x $(CFLAGS) $(LDFLAGS) $(SRC) -o $@ $^

use_ndarray: ./use_ndarray.cpp
	$(CXX) -std=c++0x $(CFLAGS) $(LDFLAGS) -o $@ $^
