#!/bin/bash

# usage: getdeps.sh binary_file

echo getting dependency for $1
mkdir lib 2> /dev/null
ldd $1 | while read x; do
	src=`echo $x | awk '{print $1}'`
	dst=`echo $x | awk --field-separator='=>' '{if(NF==2) print $2; else print $1}' | awk '{print $1}'`
	echo "${src} => ${dst}"
	cp $dst ./lib/
done
