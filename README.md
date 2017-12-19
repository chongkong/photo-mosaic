# PhotoMosaic

2017F SNU Multicore Computing Project

## Prerequisite

Place CIFAR10 dataset at `data/cifar-10.bin`

## Build & Run

The project uses cmake build system. Running `cmake` will generate GNU Makefile.

``` shell
$ cmake .
```

After generating Makefile you can choose which one to build

``` shell
$ make omp  # for single-cpu implementation
$ make opencl  # for single-gpu implementation
$ make opencl2  # for multiple gpu implementation
$ make mpi  # for mpi with multiple gpu implementation
$ make snucl  # for SNUCL implementation
$ make all  # to make all of above
```

and running

``` shell
$ thorq --add ./omp <input.bmp> <output.bmp>
$ thorq --add --mode single --device gpu/7970 ./opencl <input.bmp> <output.bmp>
$ thorq --add --mode single --device gpu/7970 ./opencl2 <input.bmp> <output.bmp>
$ thorq --add --mode mpi --node 4 --device gpu/7970 ./mpi <input.bmp> <output.bmp>
$ thorq --add --mode snucl --node 4 --device gpu/7970 ./snucl <input.bmp> <output.bmp>
```

or you can use my thorq wrapper

``` shell
$ python3 thorq.py --add ./omp <input.bmp> <output.bmp>
$ python3 thorq.py --add --mode single --device gpu/7970 ./opencl <input.bmp> <output.bmp>
$ python3 thorq.py --add --mode single --device gpu/7970 ./opencl2 <input.bmp> <output.bmp>
$ python3 thorq.py --add --mode mpi --node 4 --device gpu/7970 ./mpi <input.bmp> <output.bmp>
$ python3 thorq.py --add --mode snucl --node 4 --device gpu/7970 ./snucl <input.bmp> <output.bmp>
```
