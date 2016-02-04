#!/bin/bash

echo getting dependency for $1
mkdir lib 2> /dev/null
for i in `ldd $1 | awk '{print $1}'`; do f=`locate -l 1 $i` && cp $f ./lib/; done
