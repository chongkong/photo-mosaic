#include "photomosaic.h"
#include <limits.h>
#include <log/log.h>
#include <omp.h>
#include "util.h"

#define CIFAR10_SIZE 60000
#define W 32
#define H 32
#define C 3
#define TILE_LEN (H * W * C)
#define MAX_DIST (TILE_LEN * 255 * 255)

/**
 * Fetch image data of size 32x32 from HWC format to CHW format
 * @param dest destination buffer with size TILE_LEN
 * @param src source buffer
 * @param width width of the source buffer
 */
inline void fetch_chw(unsigned char *dest, const unsigned char *src, int width) {
  for (int h = 0; h < H; h++) {
    for (int w = 0; w < W; w++) {
      for (int c = 0; c < C; c++) {
        dest[(c * H + h) * W + w] = src[(h * width + w) * C + c];
      }
    }
  }
}

/**
 * Compute L2 distance between buffer a and buffer b for length TILE_LEN
 * @param threshold Computation breaks if error goes above threshold
 */
inline int dist(const unsigned char *a, const unsigned char *b, int threshold) {
  int sum = 0;
  for (int i = 0; i < TILE_LEN; i++) {
    int diff = (int)a[i] - (int)b[i];
    sum += diff * diff;
  }
  return sum;
}

void photomosaic(unsigned char *img, int width, int height, const unsigned char *dataset,
                 int *indices) {
  unsigned char img_local[TILE_LEN];
  omp_set_num_threads(32);

  log_info("=================================");
  log_info("Photomosaic OpenMP implementation");
  log_info("=================================");
  log_info("OpenMP uses %d threads", omp_get_max_threads());

#pragma omp parallel for collapse(2) private(img_local) shared(indices) schedule(guided)
  for (int tile_h = 0; tile_h < height; tile_h += H) {
    for (int tile_w = 0; tile_w < width; tile_w += W) {
      fetch_chw(img_local, img + (tile_h * width + tile_w) * C, width);
      int min_dist = MAX_DIST;
      int min_i = 0;
      for (int i = 0; i < CIFAR10_SIZE; ++i) {
        int d = dist(img_local, dataset + (i * TILE_LEN), min_dist);
        if (d < min_dist) {
          min_dist = d;
          min_i = i;
        }
      }
      int tile_i = (tile_h / H) * (width / W) + (tile_w / W);
      indices[tile_i] = min_i;
    }
  }
}
