#include <stdio.h>
#include <cstring>
#include "conv.hpp"

layer_t layers[NUM_LAYERS];

int div_ceil(int i, int j) {
	if (i % j == 0) {
		return i/j;
	}
	return i/j + 1;
}

data_t pad_img(data_t* A, layer_t l, int i, int j, int c) {
	if (i < 0 || i >= l.h) {
		return 0;
	} else if (j < 0 || j >= l.w) {
		return 0;
	} else {
		return A[(c * l.h * l.w) + (i * l.w + j)];
	}
}

void compute(data_t* DRAM, layer_t layer) {
#pragma HLS INTERFACE m_axi depth=1024 port=DRAM
#pragma HLS INTERFACE s_axilite port=return bundle=axilite
#pragma HLS INTERFACE s_axilite port=layer bundle=axilite

	int w_out = div_ceil(layer.w,layer.stride);
	int h_out = div_ceil(layer.h,layer.stride);

	if (layer.fire && layer.max_pool) {
		offset_t output_offset = layer.output_offset;

		// compute img sizes for offsets
		fire_t params = layer.fire_params;
		int s_out_h = div_ceil(layer.h, params.s_stride);
		int s_out_w = div_ceil(layer.w, params.s_stride);
		w_out = div_ceil(s_out_w,params.e1_stride);
		h_out = div_ceil(s_out_h,params.e1_stride);

		// offset for fire output (put it in scratch space)
		layer.output_offset = layer.scratch_offset;
		int e_out_h = div_ceil(s_out_h, params.e1_stride);
		int e_out_w = div_ceil(s_out_w, params.e1_stride);
		int expand_size = e_out_h * e_out_w * (params.e1_chan_out + params.e2_chan_out);
		layer.scratch_offset = layer.output_offset + expand_size;

		fire(DRAM, layer);

		// calculate new offsets for max pool layer
		offset_t img_offset = layer.output_offset;
		offset_t squeeze_size = s_out_h * s_out_w * params.s_chan_out;
		offset_t scratch_offset = layer.scratch_offset + squeeze_size;
		int max_pool_chan_in = params.e1_chan_out + params.e2_chan_out;

		layer_t l_max_pool = layer_t(h_out, w_out, max_pool_chan_in, -1, layer.k, layer.stride, layer.pad, layer.relu, true, false, fire_t(), false,
								 layer.pool_params, layer.glob_pool, layer.weight_offset, layer.num_weights, img_offset, output_offset, scratch_offset);
		max_pool(DRAM, l_max_pool);
	} else if (layer.fire) {
		fire(DRAM, layer);
	} else if (layer.conv && layer.max_pool) {
		conv_pool(DRAM, layer);
	} else if (layer.conv && (layer.max_pool || layer.glob_pool)) {
		// offset for output (put it in scratch space)
		offset_t output_offset = layer.output_offset;

		offset_t conv_img_size = h_out * w_out * layer.chan_out;
		layer.output_offset = layer.scratch_offset;
		layer.scratch_offset = layer.scratch_offset + conv_img_size;

		conv(DRAM, layer);

		// calculate new offsets for max pool layer
		offset_t img_offset = layer.output_offset;
		offset_t scratch_offset = layer.scratch_offset;

		if (layer.glob_pool) {
			layer_t l_glob_pool = layer_t(h_out, w_out, layer.chan_out, -1, layer.k, layer.stride, layer.pad, layer.relu, true, false, fire_t(), false,
													     layer.pool_params, layer.glob_pool, layer.weight_offset, layer.num_weights, img_offset, output_offset, scratch_offset);
			glob_pool(DRAM, l_glob_pool);
		} else {
			layer_t l_max_pool = layer_t(h_out, w_out, layer.chan_out, -1, layer.k, layer.stride, layer.pad, layer.relu, true, false, fire_t(), false,
										     layer.pool_params, layer.glob_pool, layer.weight_offset, layer.num_weights, img_offset, output_offset, scratch_offset);
			max_pool(DRAM, l_max_pool);
		}

	} else if (layer.conv) {
		conv(DRAM, layer);
	}
}

void fire(data_t* in, layer_t l) {
	offset_t output_offset = l.output_offset;

	// set the parameters for the squeeze layer (memory offsets) and its output sizes
	fire_t params = l.fire_params;
	int s_num_weights = params.s_chan_out * params.s_k * params.s_k * l.chan_in + params.s_chan_out;
	int s_out_h = div_ceil(l.h, params.s_stride);
	int s_out_w = div_ceil(l.w, params.s_stride);
	offset_t s_out_size = s_out_h * s_out_w * params.s_chan_out;
	offset_t s_output_offset = l.scratch_offset;
	offset_t s_scratch_offset = l.scratch_offset + s_out_size;

	layer_t l_s = layer_t(l.h, l.w, l.chan_in, params.s_chan_out, params.s_k, params.s_stride, l.pad, l.relu, true, false,
			          fire_t(), false, l.pool_params, l.glob_pool, l.weight_offset, s_num_weights, l.img_offset, s_output_offset, s_scratch_offset);

	// squeeze
	conv(in, l_s);

	// calculate new offsets/params for e1/e2 layers
	int e_out_h = div_ceil(s_out_h, params.e1_stride);
	int e_out_w = div_ceil(s_out_w, params.e1_stride);
	int e_img_size = e_out_h * e_out_w * (params.e1_chan_out + params.e2_chan_out);
	offset_t e_scratch_offset = l_s.scratch_offset;
	offset_t e_img_offset = l_s.output_offset;

	// calc weight offset for e1
	offset_t e1_w_offset = l_s.weight_offset + s_num_weights;
	int e1_num_weights = params.e1_chan_out * params.e1_k * params.e1_k * params.s_chan_out + params.e1_chan_out;
	offset_t e1_output_offset = output_offset;

	layer_t l_e1 = layer_t(s_out_h, s_out_w, params.s_chan_out, params.e1_chan_out, params.e1_k, params.e1_stride, l.pad, l.relu, true, false,
			           fire_t(), false, pooling_t(), false, e1_w_offset, e1_num_weights, e_img_offset, e1_output_offset, e_scratch_offset);

	// calc weight offset for e2
	offset_t e2_w_offset = l_e1.weight_offset + e1_num_weights;
	int e2_num_weights = params.e2_chan_out * params.e2_k * params.e2_k * params.s_chan_out + params.e2_chan_out;
	offset_t e2_output_offset = e1_output_offset + e_out_h * e_out_w * params.e1_chan_out;

	layer_t l_e2 = layer_t(s_out_h, s_out_w, params.s_chan_out, params.e2_chan_out, params.e2_k, params.e2_stride, l.pad, l.relu, true, false,
			           fire_t(), false, pooling_t(), false, e2_w_offset, e2_num_weights, e_img_offset, e2_output_offset, e_scratch_offset);

	// parallel expansions
	conv(in, l_e1);
	conv(in, l_e2);
}

void glob_pool(data_t* in, layer_t l) {
	int i; int j; int c;

	data_t* img = &in[l.img_offset];
	data_t* out = &in[l.output_offset];

	for (c = 0; c < l.chan_in; c++) {
#pragma HLS LOOP_TRIPCOUNT min=1000 max=1000 avg=1000
#pragma HLS PIPELINE II=1
		data_t sum = 0;
		for (i = 0; i < l.h; i++) {
#pragma HLS LOOP_TRIPCOUNT min=14 max=14 avg=14
			for (j = 0; j < l.w; j++) {
#pragma HLS LOOP_TRIPCOUNT min=14 max=14 avg=14
#pragma HLS UNROLL factor=14
				sum += img[(c * l.w * l.h) + i * l.w + j];
			}
		}
		out[c] = sum/(l.h*l.w);
	}
}

void max_pool(data_t* in, layer_t l) {
	int i; int j; int c; int ii; int jj;
	pooling_t params = l.pool_params;

	data_t* img = &in[l.img_offset];
	data_t* out = &in[l.output_offset];

	for (c = 0; c < l.chan_in; c++) {
#pragma HLS LOOP_TRIPCOUNT min=64 max=256 avg=128
		for (i = 0; i < l.h; i+=params.stride) {
#pragma HLS LOOP_TRIPCOUNT min=14 max=224 avg=56
			for (j = 0; j < l.w; j+=params.stride) {
#pragma HLS LOOP_TRIPCOUNT min=14 max=224 avg=56
#pragma HLS PIPELINE II=1
				data_t max = std::numeric_limits<float>::min();
				for (ii = 0; ii < params.k; ii++) {
#pragma HLS LOOP_TRIPCOUNT avg=3
					for (jj = 0; jj < params.k; jj++) {
#pragma HLS LOOP_TRIPCOUNT avg=3
#pragma HLS UNROLL factor=3
						if (i+ii >= l.h || j+jj >= l.w) {
							continue;
						}
						data_t temp = img[(c * l.w * l.h) + (i+ii) * l.w + (j+jj)];
						if (temp > max) {
							max = temp;
						}
					}
				}
				out[(c * div_ceil(l.h, params.stride) * div_ceil(l.w,params.stride)) + (i/params.stride * div_ceil(l.w,params.stride) + j/params.stride)] = max;
			}
		}
	}
}

void conv_pool(data_t* in, layer_t l) {
	int i; int j; int ii; int jj; int filt_num; int c;

	data_t img_buffer[224*224];
	conv_pool_helper(in, l, img_buffer);
}

void conv_pool_helper(data_t* in, layer_t l, data_t img_buffer[224*224]) {
#pragma HLS INLINE
#pragma HLS RESOURCE variable=img_buffer core=RAM_1P_BRAM
#pragma HLS INTERFACE ap_memory port=img_buffer

	int i, j, ii, jj, filt_num, c;
	data_t* w = &in[l.weight_offset];
	data_t* img = &in[l.img_offset];
	data_t* out = &in[l.output_offset];
	int bias_offset = l.weight_offset + l.num_weights - l.chan_out;

	for (filt_num = 0; filt_num < l.chan_out; filt_num++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=1000 avg=256
		data_t bias = in[bias_offset + filt_num];

		data_t w_buffer[9*512*2];
		for (i = 0; i < l.chan_in * l.k * l.k; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4608
			w_buffer[i] = w[(filt_num * l.chan_in * l.k * l.k) + i];
		}

		for (i = 0; i < l.h; i+= l.stride){
#pragma HLS LOOP_TRIPCOUNT min=14 max=224 avg=56
			for (j = 0; j < l.w; j+= l.stride) {
#pragma HLS LOOP_TRIPCOUNT min=14 max=224 avg=56
				data_t temp = 0;
				for (c = 0; c < l.chan_in; c++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=512 avg=256
#pragma HLS PIPELINE II=1

					for (ii = -l.k/2; ii <= l.k/2; ii++) {
#pragma HLS LOOP_TRIPCOUNT avg=2
						for (jj = -l.k/2; jj <= l.k/2; jj++) {
#pragma HLS LOOP_TRIPCOUNT avg=2
							temp += pad_img(img, l, i+ii , j+jj, c) * w_buffer[(c * l.k * l.k) + (jj+l.k/2) * l.k + (ii+l.k/2)];
						}
					}
				}

				temp += bias;

				if (l.relu && temp < 0.0) {
					temp = 0.0;
				}

				img_buffer[(i/l.stride * div_ceil(l.w,l.stride) + j/l.stride)] = temp;
			}
		}

		int conv_h = div_ceil(l.h, l.stride);
		int conv_w = div_ceil(l.w, l.stride);
		pooling_t params = l.pool_params;

		for (i = 0; i < conv_h; i+=params.stride) {
#pragma HLS LOOP_TRIPCOUNT min=14 max=224 avg=56
			for (j = 0; j < conv_w; j+=params.stride) {
#pragma HLS LOOP_TRIPCOUNT min=14 max=224 avg=56
#pragma HLS PIPELINE II=1
				data_t max = std::numeric_limits<float>::min();
				for (ii = 0; ii < params.k; ii++) {
#pragma HLS LOOP_TRIPCOUNT avg=3
					for (jj = 0; jj < params.k; jj++) {
#pragma HLS LOOP_TRIPCOUNT avg=3
#pragma HLS UNROLL factor=3
						if (i+ii >= conv_h || j+jj >= conv_w) {
							continue;
						}
						data_t temp = img_buffer[(i+ii) * conv_w + (j+jj)];
						if (temp > max) {
							max = temp;
						}
					}
				}
				out[(filt_num * div_ceil(conv_h, params.stride) * div_ceil(conv_w,params.stride)) + (i/params.stride * div_ceil(conv_w,params.stride) + j/params.stride)] = max;
			}
		}
	}
}

void conv(data_t* in, layer_t l) {
	int i; int j; int ii; int jj; int filt_num; int c;

	data_t* w = &in[l.weight_offset];
	data_t* img = &in[l.img_offset];
	data_t* out = &in[l.output_offset];
	int bias_offset = l.weight_offset + l.num_weights - l.chan_out;

	for (filt_num = 0; filt_num < l.chan_out; filt_num++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=1000 avg=256
		data_t bias = in[bias_offset + filt_num];

		data_t w_buffer[9*512*2];
		for (i = 0; i < l.chan_in * l.k * l.k; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4608
			w_buffer[i] = w[(filt_num * l.chan_in * l.k * l.k) + i];
		}
		
		for (i = 0; i < l.h; i+= l.stride){
#pragma HLS LOOP_TRIPCOUNT min=14 max=224 avg=56
			for (j = 0; j < l.w; j+= l.stride) {
#pragma HLS LOOP_TRIPCOUNT min=14 max=224 avg=56
				data_t temp = 0;
				for (c = 0; c < l.chan_in; c++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=512 avg=256
#pragma HLS PIPELINE II=1

					for (ii = -l.k/2; ii <= l.k/2; ii++) {
#pragma HLS LOOP_TRIPCOUNT avg=2
						for (jj = -l.k/2; jj <= l.k/2; jj++) {
#pragma HLS LOOP_TRIPCOUNT avg=2
							temp += pad_img(img, l, i+ii , j+jj, c) * w_buffer[(c * l.k * l.k) + (jj+l.k/2) * l.k + (ii+l.k/2)];
						}
					}
				}

				temp += bias;

				if (l.relu && temp < 0.0) {
					temp = 0.0;
				}

				out[(filt_num * div_ceil(l.h, l.stride) * div_ceil(l.w,l.stride)) + (i/l.stride * div_ceil(l.w,l.stride) + j/l.stride)] = temp;
			}
		}
	}
}

