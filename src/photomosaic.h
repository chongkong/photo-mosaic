#pragma once

void photomosaic(unsigned char *image, int width, int height, const unsigned char *dataset,
                 int *indices);

void photomosaic_mpi(unsigned char *image, int width, int height, const unsigned char *dataset,
                     int *indices, int world_rank, int world_size);
