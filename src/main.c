#include <log/log.h>
#include <qdbmp/qdbmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "photomosaic.h"
#include "util.h"

#ifdef _MC_MPI
#include <mpi.h>
#endif

void print_cwd() {
  char buf[1024];
  getcwd(buf, sizeof(buf));
  log_debug("Current working directory: %s", buf);
}

void save_nchw_tiling(const char *filename, int width, int height, unsigned char *nchw_images,
                      int *indices) {
  log_debug("Constructing and saving tiled image..");
  int seg_height = height / 32;
  int seg_width = width / 32;
  BMP *bmp = BMP_Create(width, height, 24);
  for (int sh = 0; sh < seg_height; ++sh) {
    for (int sw = 0; sw < seg_width; ++sw) {
      int index = indices[sh * seg_width + sw];
      for (int h = 0; h < 32; ++h) {
        for (int w = 0; w < 32; ++w) {
          unsigned char rgb[3];
          for (int c = 0; c < 3; ++c) {
            rgb[c] = nchw_images[((index * 3 + c) * 32 + h) * 32 + w];
          }
          BMP_SetPixelRGB(bmp, sw * 32 + w, sh * 32 + h, rgb[0], rgb[1], rgb[2]);
        }
      }
    }
  }
  BMP_WriteFile(bmp, filename);
  log_debug("Image saved to %s", filename);
  BMP_Free(bmp);
}

int main(int argc, char **argv) {
  if (argc != 3) {
    log_error("Usage: %s [input.bmp] [output.bmp]", argv[0]);
    exit(EXIT_FAILURE);
  }

#ifdef _MC_MPI
  MPI_Init(&argc, &argv);

  int world_size;
  int world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
#endif

  // Read image

  BMP *bmp = BMP_ReadFile(argv[1]);
  BMP_CHECK_ERROR(stderr, EXIT_FAILURE);

  int width = BMP_GetWidth(bmp);
  int height = BMP_GetHeight(bmp);
  int depth = BMP_GetDepth(bmp);
#ifdef _MC_MPI
  if (world_rank == 0) {
#endif
    log_debug("Input image %s", argv[1]);
    log_debug("  width: %d", width);
    log_debug("  height: %d", height);
    log_debug("  depth: %d", depth);
#ifdef _MC_MPI
  }
#endif
  if (width % 32 != 0 || height % 32 != 0) {
    log_error("width and height should be multiple of 32.");
    exit(EXIT_FAILURE);
  }
  if (depth != 24) {
    log_error("depth should be 24.");
    exit(EXIT_FAILURE);
  }

  unsigned char *img = (unsigned char *)malloc(height * width * 3);
  unsigned char *it = img;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      BMP_GetPixelRGB(bmp, j, i, it, it + 1, it + 2);
      it += 3;
    }
  }
  BMP_Free(bmp);

  // Read dataset

  unsigned char *dataset = (unsigned char *)malloc(60000 * 3 * 32 * 32);
  FILE *fin = fopen("data/cifar-10.bin", "rb");
  if (!fin) {
    log_error("cifar-10.bin not found");
    exit(EXIT_FAILURE);
  }
  fread(dataset, 1, 60000 * 3 * 32 * 32, fin);
  fclose(fin);

  log_debug("dataset read success");

  // Computation

  int seg_width = width / 32;
  int seg_height = height / 32;
  int *indices = (int *)malloc(seg_height * seg_width * sizeof(int));
#ifdef _MC_MPI
  if (world_rank == 0) timer_start();
  photomosaic_mpi(img, width, height, dataset, indices, world_rank, world_size);
  if (world_rank == 0) timer_stop_and_log("total");
#else
  timer_start();
  photomosaic(img, width, height, dataset, indices);
  timer_stop_and_log("Total elapsed");
#endif

#ifdef _MC_MPI
  if (world_rank == 0) save_nchw_tiling(argv[2], width, height, dataset, indices);
#else
  // Write result
  save_nchw_tiling(argv[2], width, height, dataset, indices);
#endif

  // Free resources

  free(img);
  free(dataset);
#ifdef _MC_MPI
  if (world_rank == 0) free(indices);
  MPI_Finalize();
#else
  free(indices);
#endif

  return 0;
}
