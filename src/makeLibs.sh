#!/bin/sh
rm *.o
rm *.so

export PKG_LIBS=-lgomp
export PKG_CPPFLAGS=" -msse2  -I/usr/include/eigen3 -std=c++11"
R CMD SHLIB libgputils.cpp 

