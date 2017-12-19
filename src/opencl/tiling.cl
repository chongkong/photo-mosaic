#define W 32
#define H 32
#define C 3
#define TILE_LEN (W * H * C)

__kernel void 
nchw_tiling(
  __global uchar *src, 
  __global uchar *dest,
  int width
) {
  int gid = get_group_id(0);
  int lid = get_local_id(0);

  __local uchar tile[H][W][C];
  
  int h = lid >> 3;
  int w = (lid & 0x7)*4;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
      tile[h][w + i][c] = src[(h*width + gid*W + w + i)*C + c];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  #pragma unroll
  for (int c = 0; c < C; ++c) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      dest[TILE_LEN*gid + (TILE_LEN/C)*c + h*W + w + i] = tile[h][w + i][c];
    }
  }
}
