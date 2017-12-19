#include <log/log.h>
#include <photomosaic.h>
#include <util.h>
#include "clwrapper.h"
#include "common.h"

#define W 32
#define H 32
#define C 3
#define TILE_LEN (W * H * C)
#define CIFAR10_SIZE 60000
#define MIN_GPU_QUOTA 16

void photomosaic(unsigned char *image, int width, int height, const unsigned char *dataset,
                 int *indices) {
  log_info("=================================");
  log_info("Photomosaic OpenCL implementation");
  log_info("=================================");

  CLHost host = create_host(true);
  preprocess_image(&host, image, width, height, true);
  photomosaic_opencl(&host, image, dataset, indices, (width / W) * (height / H), true);
}
