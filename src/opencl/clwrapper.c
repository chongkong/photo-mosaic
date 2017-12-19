#include "clwrapper.h"
#include <log/log.h>

cl_platform_id cl_get_platform_id() {
  cl_platform_id platform;
  CHECK_ERROR(clGetPlatformIDs(1, &platform, NULL));
  return platform;
}

cl_device_id *cl_get_gpu_device_ids(cl_platform_id platform, cl_uint num_devices) {
  cl_device_id *devices = malloc(sizeof(cl_device_id) * num_devices);
  CHECK_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices, NULL));
  return devices;
}

static void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data) {
  log_error(errinfo);
}

cl_context cl_create_context(cl_uint num_devices, cl_device_id *devices) {
  cl_int err;
  cl_context ctx = clCreateContext(NULL, num_devices, devices, &pfn_notify, NULL, &err);
  CHECK_ERROR(err);
  return ctx;
}

cl_command_queue cl_create_command_queue(cl_context ctx, cl_device_id device) {
  cl_int err;
#if defined(__APPLE__) || defined(_MC_SNUCL)
  cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
#else
  cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, 0, &err);
#endif
  CHECK_ERROR(err);
  return queue;
}

cl_command_queue *cl_create_command_queues(cl_context ctx, cl_uint num_devices,
                                           cl_device_id *devices) {
  cl_int err;
  cl_command_queue *queues = malloc(sizeof(cl_command_queue) * num_devices);
  for (int i = 0; i < num_devices; ++i) {
#if defined(__APPLE__) || defined(_MC_SNUCL)
    queues[i] = clCreateCommandQueue(ctx, devices[i], 0, &err);
#else
    queues[i] = clCreateCommandQueueWithProperties(ctx, devices[i], 0, &err);
#endif
    CHECK_ERROR(err);
  }
  return queues;
}

const char *get_source_code(const char *filename, size_t *len) {
  char *source_code;
  size_t length;
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, filename);
    exit(EXIT_FAILURE);
  }

  fseek(file, 0, SEEK_END);
  length = (size_t)ftell(file);
  rewind(file);

  source_code = (char *)malloc(length + 1);
  fread(source_code, length, 1, file);
  source_code[length] = '\0';

  fclose(file);

  *len = length;
  return source_code;
}

cl_program cl_build_program(const char *source, cl_context ctx, unsigned num_devices,
                            cl_device_id *devices) {
  size_t source_size;
  cl_int err;
  const char *source_code = get_source_code(source, &source_size);
  cl_program program = clCreateProgramWithSource(ctx, 1, &source_code, &source_size, &err);
  CHECK_ERROR(err);
  err = clBuildProgram(program, num_devices, devices, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    char *log;
    size_t log_size;
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    log = (char *)malloc(log_size + 1);
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    log[log_size] = '\0';
    fprintf(stderr, "Compile error:\n%s\n", log);
    free(log);
    exit(EXIT_FAILURE);
  }
  free((char *)source_code);
  return program;
}

cl_kernel cl_create_kernel(cl_program program, const char *kernel_name) {
  cl_int err;
  cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
  CHECK_ERROR(err);
  return kernel;
}

cl_mem cl_create_buffer(cl_context ctx, cl_mem_flags flags, size_t size) {
  cl_int err;
  cl_mem buffer = clCreateBuffer(ctx, flags, size, NULL, &err);
  CHECK_ERROR(err);
  return buffer;
}

cl_mem *cl_create_buffers(cl_context ctx, cl_mem_flags flags, size_t size, unsigned count) {
  cl_mem *buffers = malloc(sizeof(cl_mem) * count);
  for (int i = 0; i < count; ++i) {
    buffers[i] = cl_create_buffer(ctx, flags, size);
  }
  return buffers;
}

void cl_enqueue_write_buffer(cl_command_queue queue, cl_mem buffer, size_t size, const void *data) {
  CHECK_ERROR(clEnqueueWriteBuffer(queue, buffer, CL_FALSE, 0, size, data, 0, NULL, NULL));
}

void cl_enqueue_read_buffer(cl_command_queue queue, cl_mem buffer, size_t size, void *dest) {
  CHECK_ERROR(clEnqueueReadBuffer(queue, buffer, CL_FALSE, 0, size, dest, 0, NULL, NULL));
}

void cl_all_finish(cl_command_queue *queues, cl_uint size) {
  for (int i = 0; i < size; ++i) {
    CHECK_ERROR(clFinish(queues[i]));
  }
}

void cl_release_mem_object(cl_mem buffer) { CHECK_ERROR(clReleaseMemObject(buffer)); }

void cl_release_mem_objects(cl_mem *buffers, unsigned count) {
  for (int i = 0; i < count; ++i) {
    cl_release_mem_object(buffers[i]);
  }
}

void cl_release_context(cl_context ctx) { CHECK_ERROR(clReleaseContext(ctx)); }

void cl_release_program(cl_program program) { CHECK_ERROR(clReleaseProgram(program)); }

void cl_release_kernel(cl_kernel kernel) { CHECK_ERROR(clReleaseKernel(kernel)); }

void cl_release_command_queues(cl_command_queue *queues, unsigned count) {
  for (int i = 0; i < count; ++i) {
    CHECK_ERROR(clReleaseCommandQueue(queues[i]));
  }
}
