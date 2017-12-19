#include <log/log.h>
#include <mpi.h>
#include <opencl/clwrapper.h>
#include <opencl/common.h>
#include <photomosaic.h>
#include <util.h>

#define W 32
#define H 32
#define C 3
#define TILE_LEN (W * H * C)
#define CIFAR10_SIZE 60000
#define MIN_GPU_QUOTA 4

void photomosaic_mpi(unsigned char *image, int width, int height, const unsigned char *dataset,
                     int *indices, int world_rank, int world_size) {
  if (world_rank == 0) {
    log_info("=======================================");
    log_info("Photomosaic MPI + OpenCL implementation");
    log_info("=======================================");
    log_info("MPI communication world size: %d", world_size);
  }

  int num_tiles = (width / W) * (height / H);
  if (num_tiles < MIN_GPU_QUOTA * NUM_GPUS) {
    world_size = 1;
    if (world_rank > 0)
      return;
    else
      log_debug("Image is small; discard other nodes except rank 0");
  }

  CLHost host = create_host(world_rank == 0);
  preprocess_image(&host, image, width, height, world_rank == 0);

  if (world_size == 1) {
    photomosaic_opencl(&host, image, dataset, indices, num_tiles, true);
    return;
  }

  int *offsets = (int *)malloc(world_size * sizeof(int));
  int *tiles = (int *)malloc(world_size * sizeof(int));
  for (int i = 0; i < world_size; ++i) {
    int here = i * num_tiles / world_size;
    int next = (i + 1) * num_tiles / world_size;
    offsets[i] = here;
    tiles[i] = next - here;
  }

  photomosaic_opencl(&host, image + offsets[world_rank] * TILE_LEN, dataset, indices,
                     tiles[world_rank], world_rank);

  if (world_rank == 0) timer_start();
  MPI_Gatherv(indices, tiles[world_rank], MPI_INT32_T, indices, tiles, offsets, MPI_INT32_T, 0,
              MPI_COMM_WORLD);
  if (world_rank == 0) timer_stop_and_log("[photomosaic] gather time");
}
