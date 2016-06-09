BLAS=-L /opt/openblas/lib -lopenblas -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 
CUDA=-DMSHADOW_USE_CUDA=1

#COMMFLAGS=-static -static-libgcc -static-libstdc++

CFLAGS=$(COMMFLAGS) -I ../include -Wall -O3 -msse3 -funroll-loops -Wno-unused-parameter -Wno-unknown-pragmas -fopenmp 
LDFLAGS=$(COMMFLAGS) -L ../lib/linux -lmxnet $(BLAS) $(CUDA) -lgomp -pthread

all: mlp lenet lenet_with_mxdataiter

lenet_with_mxdataiter: ./lenet_with_mxdataiter.cpp
	$(CXX) -c -std=c++11 $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS)
	-rm -f $(basename $@).o

lenet: ./lenet.cpp
	$(CXX) -c -std=c++11 $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS)
	-rm -f $(basename $@).o

mlp: ./mlp.cpp
	$(CXX) -c -std=c++11 $(CFLAGS) $^
	$(CXX) $(basename $@).o -o $@ $(LDFLAGS)
	-rm -f $(basename $@).o

# For simplicity, no link here
travis:
	$(CXX) -c -std=c++11 $(CFLAGS) ./mlp.cpp && rm -f mlp.o
	$(CXX) -c -std=c++11 $(CFLAGS) ./lenet.cpp && rm -f lenet.o
	$(CXX) -c -std=c++11 $(CFLAGS) ./lenet_with_mxdataiter.cpp && rm -f lenet_with_mxdataiter.o

clean:
	-rm -f mlp
	-rm -f lenet
	-rm -f lenet_with_mxdataiter
