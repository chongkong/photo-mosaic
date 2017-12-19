#include "common.h"
#include <log/log.h>
#include <util.h>

#define W 32
#define H 32
#define C 3
#define TILE_LEN (W * H * C)
#define CIFAR10_SIZE 60000
#define MIN_GPU_QUOTA 4

CLHost create_host(bool print_stats) {
  CLHost host;
  timer_start();
  host.platform = cl_get_platform_id();
  CHECK_ERROR(clGetDeviceIDs(host.platform, CL_DEVICE_TYPE_GPU, NUM_GPUS, host.devs, NULL));
  if (print_stats) log_debug("OpenCL uses %d GPUs", NUM_GPUS);
  host.ctx = cl_create_context(NUM_GPUS, host.devs);
  for (int d = 0; d < NUM_GPUS; d++) {
    host.read_queues[d] = cl_create_command_queue(host.ctx, host.devs[d]);
    host.kernel_queues[d] = cl_create_command_queue(host.ctx, host.devs[d]);
    host.write_queues[d] = cl_create_command_queue(host.ctx, host.devs[d]);
  }
  if (print_stats) timer_stop_and_log("[init] init time");
  return host;
}

void preprocess_image(CLHost *host, unsigned char *image, int width, int height, bool print_stats) {
#define NUM_BUFS 2

  if (print_stats) timer_start();
  cl_program program = cl_build_program("src/opencl/tiling.cl", host->ctx, NUM_GPUS, host->devs);
  cl_kernel kernel = cl_create_kernel(program, "nchw_tiling");
  if (print_stats) timer_stop_and_log("[preprocess] compile time");

  if (print_stats) timer_start();
  int row_size = width * H * C * sizeof(unsigned char);
  cl_mem buf_src[NUM_GPUS][NUM_BUFS];
  cl_mem buf_dest[NUM_GPUS][NUM_BUFS];
  for (int d = 0; d < NUM_GPUS; ++d) {
    for (int k = 0; k < NUM_BUFS; ++k) {
      buf_src[d][k] = cl_create_buffer(host->ctx, CL_MEM_READ_ONLY, row_size);
      buf_dest[d][k] = cl_create_buffer(host->ctx, CL_MEM_WRITE_ONLY, row_size);
    }
  }
  if (print_stats) timer_stop_and_log("[preprocess] buffer allocation time");

  if (print_stats) timer_start();
  int num_rows = height / H;
  int partitions[NUM_GPUS + 1];
  for (int d = 0; d <= NUM_GPUS; ++d) {
    partitions[d] = (num_rows * d) / NUM_GPUS;
  }

  for (int d = 0; d < NUM_GPUS; ++d) {
    for (int k = 0; k < NUM_BUFS; ++k) {
      clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_src[d][k]);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_dest[d][k]);
      clSetKernelArg(kernel, 2, sizeof(int), &width);
    }
  }

  size_t global_size = 256 * (width / 32);
  size_t local_size = 256;
  cl_event write_events[NUM_BUFS];
  cl_event kernel_events[NUM_BUFS];
  cl_event read_events[NUM_BUFS];
  for (int d = 0; d < NUM_GPUS; ++d) {
    for (int row = partitions[d]; row < partitions[d + 1]; ++row) {
      unsigned char *image_pos = image + row_size * row;
      int i = row - partitions[d];
      int k = i % NUM_BUFS;
      if (i < NUM_BUFS) {
        clEnqueueWriteBuffer(host->write_queues[d], buf_src[d][k], CL_FALSE, 0, row_size, image_pos,
                             0, NULL, &write_events[k]);
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_src[d][k]);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_dest[d][k]);
        clSetKernelArg(kernel, 2, sizeof(int), &width);
        clEnqueueNDRangeKernel(host->kernel_queues[d], kernel, 1, NULL, &global_size, &local_size,
                               1, &write_events[k], &kernel_events[k]);
        clEnqueueReadBuffer(host->read_queues[d], buf_dest[d][k], CL_FALSE, 0, row_size, image_pos,
                            1, &kernel_events[k], &read_events[k]);
      } else {
        clEnqueueWriteBuffer(host->write_queues[d], buf_src[d][k], CL_FALSE, 0, row_size, image_pos,
                             1, &kernel_events[k], &write_events[k]);
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_src[d][k]);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_dest[d][k]);
        clSetKernelArg(kernel, 2, sizeof(int), &width);
        cl_event kernel_wait_list[2] = {write_events[k], read_events[k]};
        clEnqueueNDRangeKernel(host->kernel_queues[d], kernel, 1, NULL, &global_size, &local_size,
                               2, kernel_wait_list, &kernel_events[k]);
        clEnqueueReadBuffer(host->read_queues[d], buf_dest[d][k], CL_FALSE, 0, row_size, image_pos,
                            1, &kernel_events[k], &read_events[k]);
      }
    }
  }

  cl_all_finish(host->read_queues, NUM_GPUS);
  cl_release_program(program);
  for (int d = 0; d < NUM_GPUS; ++d) {
    for (int k = 0; k < NUM_BUFS; ++k) {
      cl_release_mem_object(buf_src[d][k]);
      cl_release_mem_object(buf_dest[d][k]);
    }
  }
  cl_release_kernel(kernel);

  if (print_stats) timer_stop_and_log("[preprocess] preprocessing time");
}

void photomosaic_opencl(CLHost *host, unsigned char *image, const unsigned char *dataset,
                        int *indices, int num_tiles, bool print_stats) {
  int num_gpus = NUM_GPUS;
  if (num_tiles < MIN_GPU_QUOTA * NUM_GPUS)
    num_gpus = (num_tiles + MIN_GPU_QUOTA - 1) / MIN_GPU_QUOTA;

  if (print_stats) timer_start();
  cl_program program =
      cl_build_program("src/opencl/photomosaic.cl", host->ctx, num_gpus, host->devs);
  cl_kernel kernel = cl_create_kernel(program, "photomosaic");
  if (print_stats) timer_stop_and_log("[photomosaic] compile time");

  cl_mem *buf_dataset = (cl_mem *)malloc(num_gpus * sizeof(cl_mem));
  for (int dev = 0; dev < num_gpus; ++dev) {
    buf_dataset[dev] = cl_create_buffer(host->ctx, CL_MEM_READ_ONLY, CIFAR10_SIZE * TILE_LEN);
  }

  cl_mem *buf_image = (cl_mem *)malloc(num_gpus * sizeof(cl_mem));
  int *partitions = (int *)malloc((num_gpus + 1) * sizeof(int));
  for (int dev = 0; dev <= num_gpus; ++dev) {
    partitions[dev] = (num_tiles * dev) / num_gpus;
  }
  for (int dev = 0; dev < num_gpus; ++dev) {
    int tiles = partitions[dev + 1] - partitions[dev];
    buf_image[dev] = cl_create_buffer(host->ctx, CL_MEM_READ_ONLY, tiles * TILE_LEN);
  }

  cl_mem buf_indices[num_gpus];
  for (int dev = 0; dev < num_gpus; ++dev) {
    int tiles = partitions[dev + 1] - partitions[dev];
    buf_indices[dev] = cl_create_buffer(host->ctx, CL_MEM_WRITE_ONLY, tiles * sizeof(int));
  }

  if (print_stats) timer_start();
  for (int dev = 0; dev < num_gpus; ++dev) {
    int tiles = partitions[dev + 1] - partitions[dev];
    clEnqueueWriteBuffer(host->write_queues[dev], buf_image[dev], CL_TRUE, 0, tiles * TILE_LEN,
                         image + (partitions[dev] * TILE_LEN), 0, NULL, NULL);
    clEnqueueWriteBuffer(host->write_queues[dev], buf_dataset[dev], CL_TRUE, 0,
                         CIFAR10_SIZE * TILE_LEN, dataset, 0, NULL, NULL);
  }
  if (print_stats) timer_stop_and_log("[photomosaic] write time");

  if (print_stats) timer_start();
  for (int dev = 0; dev < num_gpus; ++dev) {
    int num_images = partitions[dev + 1] - partitions[dev];
    int num_data = CIFAR10_SIZE;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_image[dev]);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_dataset[dev]);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_indices[dev]);
    clSetKernelArg(kernel, 3, sizeof(int), &num_images);
    clSetKernelArg(kernel, 4, sizeof(int), &num_data);

    size_t global_size = (partitions[dev + 1] - partitions[dev]) * 256;
    size_t local_size = 256;
    clEnqueueNDRangeKernel(host->kernel_queues[dev], kernel, 1, NULL, &global_size, &local_size, 0,
                           NULL, NULL);
  }
  cl_all_finish(host->kernel_queues, num_gpus);
  if (print_stats) timer_stop_and_log("[photomosaic] kernel time");

  if (print_stats) timer_start();
  for (int dev = 0; dev < num_gpus; ++dev) {
    int num_bytes = (partitions[dev + 1] - partitions[dev]) * sizeof(int);
    clEnqueueReadBuffer(host->read_queues[dev], buf_indices[dev], CL_TRUE, 0, num_bytes,
                        indices + partitions[dev], 0, NULL, NULL);
  }
  if (print_stats) timer_stop_and_log("[photomosaic] read time");
}
