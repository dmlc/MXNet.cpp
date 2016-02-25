CXX=g++
BLAS=-L /opt/openblas/lib -lopenblas -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 
CUDA=-DMSHADOW_USE_CUDA=1

#COMMFLAGS=-static -static-libgcc -static-libstdc++

CFLAGS=$(COMMFLAGS) -I ../include -Wall -O3 -msse3 -funroll-loops -Wno-unused-parameter -Wno-unknown-pragmas -fopenmp 
LDFLAGS=$(COMMFLAGS) -L ../lib/linux -lmxnet $(BLAS) $(CUDA) -lgomp -pthread

all: mlp lenet

lenet: ./lenet.cpp
	$(CXX) -c -std=c++0x $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS)
	-rm -f $(basename $@).o

mlp: ./mlp.cpp
	$(CXX) -c -std=c++0x $(CFLAGS) $^ 
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS)
	-rm -f $(basename $@).o

# For simplicity, no link here
travis:
	$(CXX) -c -std=c++0x $(CFLAGS) ./mlp.cpp && rm -f mlp.o
	$(CXX) -c -std=c++0x $(CFLAGS) ./lenet.cpp && rm -f lenet.o

clean:
	-rm -f mlp
	-rm -f lenet
