#pragma once

#include <stdbool.h>
#include "clwrapper.h"

#ifndef NUM_GPUS
#define NUM_GPUS 4
#endif

typedef struct {
  cl_platform_id platform;
  cl_device_id devs[NUM_GPUS];
  cl_context ctx;
  cl_command_queue read_queues[NUM_GPUS];
  cl_command_queue kernel_queues[NUM_GPUS];
  cl_command_queue write_queues[NUM_GPUS];
} CLHost;

CLHost create_host(bool print_stats);
void preprocess_image(CLHost *host, unsigned char *image, int width, int height, bool print_stats);
void photomosaic_opencl(CLHost *host, unsigned char *image, const unsigned char *dataset,
                        int *indices, int num_tiles, bool print_stats);