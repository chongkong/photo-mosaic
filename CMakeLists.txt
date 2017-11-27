cmake_minimum_required(VERSION 3.8)
project(photo_mosaic)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        src/main.c
        src/photomosaic.c
        src/photomosaic.h
        src/qdbmp.c
        src/qdbmp.h
        src/timer.c
        src/timer.h)

add_executable(photo_mosaic ${SOURCE_FILES})
