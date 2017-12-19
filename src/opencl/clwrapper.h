#pragma once

#include <stdio.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CHECK_ERROR(err)                                          \
  if ((err) != CL_SUCCESS) {                                      \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(1);                                                      \
  }

cl_platform_id cl_get_platform_id();
cl_device_id *cl_get_gpu_device_ids(cl_platform_id platform, cl_uint num_devices);
cl_context cl_create_context(cl_uint num_devices, cl_device_id *devices);
cl_command_queue cl_create_command_queue(cl_context ctx, cl_device_id device);
cl_command_queue *cl_create_command_queues(cl_context ctx, cl_uint num_devices,
                                           cl_device_id *devices);
cl_program cl_build_program(const char *source, cl_context ctx, unsigned num_devices,
                            cl_device_id *devices);
cl_kernel cl_create_kernel(cl_program program, const char *kernel_name);
cl_mem cl_create_buffer(cl_context ctx, cl_mem_flags flags, size_t size);
cl_mem *cl_create_buffers(cl_context ctx, cl_mem_flags flags, size_t size, unsigned count);
void cl_enqueue_write_buffer(cl_command_queue queue, cl_mem buffer, size_t size, const void *data);
void cl_enqueue_read_buffer(cl_command_queue queue, cl_mem buffer, size_t size, void *dest);
void cl_all_finish(cl_command_queue *queues, cl_uint size);
void cl_release_mem_object(cl_mem buffer);
void cl_release_mem_objects(cl_mem *buffers, unsigned count);
void cl_release_context(cl_context ctx);
void cl_release_program(cl_program program);
void cl_release_kernel(cl_kernel kernel);
void cl_release_command_queues(cl_command_queue *queues, unsigned count);
