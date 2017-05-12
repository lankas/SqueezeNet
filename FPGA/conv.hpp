#include "layer.hpp"
#include <limits>

#define WIDTH 5
#define HEIGHT 5
#define CHAN_IN 2
#define CHAN_OUT 3
#define K 3
#define STRIDE 2
#define WIDTH_OUT 3
#define HEIGHT_OUT 3
#define NUM_LAYERS 1
#define IMG_OFFSET K*K*CHAN_IN*CHAN_OUT // Offset needed to access image
#define SCRATCH_OFFSET IMG_OFFSET+WIDTH*HEIGHT*CHAN_IN // Offset needed to access scratch

void define_layers();
void conv(data_t* A, layer_t l);
void fire(data_t* in, layer_t l);
void max_pool(data_t* A, layer_t l);
void glob_pool(data_t* A, layer_t l);
data_t pad_img(data_t* A, layer_t l, int i, int j, int c);
int div_ceil(int i, int j);
void compute(data_t* A, layer_t layer);
void conv_pool(data_t*in, layer_t layer);
void conv_pool_helper(data_t* in, layer_t layer, data_t img_buffer[224*224]);
