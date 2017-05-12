#include "xcompute.h"
#include <xscutimer.h>
#include "xil_cache.h"
#include <stdlib.h>
#include <stdio.h>
#include <queue>
#include <math.h>

#define TIMER_LOAD_VALUE      0xFFFFFFFF
#define DRAM_ADDR 0x0012DA00
#define AXILITE_ADDR 0x43C00000
#define NUM_LAYERS 10
#define TOT_NUM_WEIGHTS 1235496
#define L1_IMG_SIZE 150528
#define MAX_NAME_LEN 15
#define OUTPUT_SIZE 13*13*1000

typedef float data_t;
typedef int offset_t;

struct pooling_t {
	int k;
	int stride;

	pooling_t(int k, int stride) : k(k), stride(stride) {};
	pooling_t() : k(0), stride(0) {};
};

struct fire_t {
	int s_k;
	int s_stride;
	int s_chan_out;
	int e1_k;
	int e1_stride;
	int e1_chan_out;
	int e2_k;
	int e2_stride;
	int e2_chan_out;

	fire_t(int s_k, int s_stride, int s_num_filt, int e1_k, int e1_stride, int e1_num_filt, int e2_k, int e2_stride, int e2_num_filt) :
		   s_k(s_k), s_stride(s_stride), s_chan_out(s_num_filt), e1_k(e1_k), e1_stride(e1_stride), e1_chan_out(e1_num_filt),
		   e2_k(e2_k), e2_stride(e2_stride), e2_chan_out(e2_num_filt) {};
	fire_t() : s_k(0), s_stride(0), s_chan_out(0), e1_k(0), e1_stride(0), e1_chan_out(0), e2_k(0), e2_stride(0), e2_chan_out(0) {};
};

struct layer_t {
	int h;
	int w;
	int chan_in;
	int chan_out;
	int k;
	int stride;
	int pad;
	bool relu;
	bool conv;
	bool fire;
	fire_t fire_params;
	bool max_pool;
	pooling_t pool_params;
	bool glob_pool;

	offset_t weight_offset;
	int num_weights;
	offset_t img_offset;
	offset_t output_offset;
	offset_t scratch_offset;

	layer_t(int height, int width, int channel_in, int channel_out, int kernel, int stride,
			int padding, bool relu, bool conv, bool fire, fire_t fire_params, bool max_pool,
			pooling_t pool_params, bool global_pool, offset_t weight_offset, int num_weights,
			offset_t img_offset, offset_t output_offset, offset_t scratch_offset)
		  :
			h(height),
			w(width),
			chan_in(channel_in),
			chan_out(channel_out),
			k(kernel),
			stride(stride),
			pad(padding),
			relu(relu),
			conv(conv),
			fire(fire),
			fire_params(fire_params),
			max_pool(max_pool),
			pool_params(pool_params),
			glob_pool(global_pool),
			weight_offset(weight_offset),
			num_weights(num_weights),
			img_offset(img_offset),
			output_offset(output_offset),
			scratch_offset(scratch_offset) {
	};

	layer_t(int height, int width, int channel_in, int channel_out, int kernel, int stride,
			int padding, bool relu, bool conv, bool fire, fire_t fire_params, bool max_pool,
			pooling_t pool_params, bool global_pool)
		  :
			h(height),
			w(width),
			chan_in(channel_in),
			chan_out(channel_out),
			k(kernel),
			stride(stride),
			pad(padding),
			relu(relu),
			conv(conv),
			fire(fire),
			fire_params(fire_params),
			max_pool(max_pool),
			pool_params(pool_params),
			glob_pool(global_pool) {
	};

	// empty constructor just in case
	layer_t()
		:
			h(0),
			w(0),
			chan_in(0),
			chan_out(0),
			k(0),
			stride(0),
			pad(0),
			relu(0),
			conv(0),
			fire(0),
			fire_params(fire_t()),
			max_pool(0),
			pool_params(pooling_t()),
			glob_pool(0),
			weight_offset(0),
			num_weights(0),
			img_offset(0),
			output_offset(0),
			scratch_offset(0) {};
};

struct network_t {
	int num_layers;
	int num_weights;
	layer_t* layers;

	network_t(int num_layers, int num_weights) : num_layers(num_layers), num_weights(num_weights) {
		layers = new layer_t[num_layers];
	};
};

network_t* create_network(int num_layers, int tot_num_weights);
void load_image(int tot_num_weights);
void load_weights();
void load_layer(layer_t layer);
void write_axilite(u32 addr_offset, u32 val);
int div_ceil(int i, int j);
void conv_test(XScuTimer* Timer, XCompute* compute);

