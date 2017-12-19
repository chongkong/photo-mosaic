#define W 32
#define H 32
#define C 3
#define TILE_LEN ((W * H * C) / 4)
#define WORK_ITEM_SIZE 256
#define TILES 1
#define WORK_LOAD (TILE_LEN / WORK_ITEM_SIZE)
#define MAX_DIST (W * H * C * 255 * 255)

__kernel void 
photomosaic(
  __global uchar4 *image,
  __global uchar4 *dataset,
  __global int *indices,
  int num_images,
  int num_data
) {
  int gid = get_group_id(0);
  int lid = get_local_id(0);
  
  __local int4 image_cache[TILE_LEN];
  __local int reduce_sum[WORK_ITEM_SIZE];
  
  int min_index = 0;
  int min_dist = MAX_DIST;

  #pragma unroll
  for (int k = 0; k < WORK_LOAD; ++k) {
    image_cache[lid*WORK_LOAD + k] 
      = convert_int4(image[gid*TILE_LEN + lid*WORK_LOAD + k]);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = 0; i < num_data; ++i) {

    int4 sum = (int4)(0);
    #pragma unroll
    for (int k = 0; k < WORK_LOAD; ++k) {
      int offset = lid*WORK_LOAD + k;
      int4 d = convert_int4(dataset[i*TILE_LEN + offset]) - image_cache[offset];
      sum += (d * d);
    }
    int2 tmp = sum.xy + sum.zw;
    reduce_sum[lid] = tmp.x + tmp.y;

    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce sum

    #pragma unroll
    for (int p = 2; p <= WORK_ITEM_SIZE; p <<= 1) {
      if ((lid & (p - 1)) == 0) {
        reduce_sum[lid] += reduce_sum[lid + (p >> 1)];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
      int dist = reduce_sum[0];
      if (dist < min_dist) {
        min_dist = dist;
        min_index = i;
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    indices[gid] = min_index;
  }
}
