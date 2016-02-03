CXX=g++
BLAS=-L /opt/openblas/lib -lopenblas -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 
CUDA=-DMSHADOW_USE_CUDA=1

CFLAGS=-I ../include -Wall -O3 -msse3 -funroll-loops -Wno-unused-parameter -Wno-unknown-pragmas -fopenmp 
LDFLAGS=-L ../lib/linux -lmxnet $(BLAS) $(CUDA)

all: mlp lenet

lenet: ./lenet.cpp
	$(CXX) -c -std=c++0x $(CFLAGS) $^ && $(CXX) $(basename $@).o -o $@ $(LDFLAGS)

mlp: ./mlp.cpp
	$(CXX) -c -std=c++0x $(CFLAGS) $^ && $(CXX) $(basename $@).o -o $@ $(LDFLAGS)
