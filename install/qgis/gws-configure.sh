#!/usr/bin/env bash
cmake \
	-D BUILD_TESTING:BOOL=OFF \
	-D CMAKE_BUILD_TYPE:STRING=$1 \
	-D ENABLE_TESTS:BOOL=FALSE \
	-D PEDANTIC:BOOL=FALSE \
	-D WITH_DESKTOP:BOOL=FALSE \
	-D WITH_GEOREFERENCER:BOOL=FALSE \
	-D WITH_GRASS7:BOOL=FALSE \
	-D WITH_QSCIAPI:BOOL=FALSE \
	-D WITH_QSPATIALITE:BOOL=TRUE \
	-D WITH_QT5SERIALPORT:BOOL=FALSE \
	-D WITH_SERVER:BOOL=TRUE \
..
