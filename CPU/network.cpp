#include "network.hpp"

volatile data_t* AXILITE = (data_t*)AXILITE_ADDR;

int div_ceil(int i, int j) {
	if (i % j == 0) {
		return i/j;
	}
	return i/j + 1;
}

void write_axilite(u32 addr_offset, u32 val) {
	*(AXILITE+addr_offset/4) = val;
}

void load_layer(layer_t layer) {
	fire_t fire_params = layer.fire_params;
	pooling_t pool_params = layer.pool_params;
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_H_DATA, layer.h);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_W_DATA, layer.w);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_CHAN_IN_DATA, layer.chan_in);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_CHAN_OUT_DATA, layer.chan_out);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_K_DATA, layer.k);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_STRIDE_DATA, layer.stride);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_PAD_DATA, layer.pad);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_RELU_DATA, layer.relu);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_CONV_DATA, layer.conv);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_FIRE_DATA, layer.fire);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_FIRE_PARAMS_S_K_DATA, fire_params.s_k);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_FIRE_PARAMS_S_STRIDE_DATA, fire_params.s_stride);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_FIRE_PARAMS_S_CHAN_OUT_DATA, fire_params.s_chan_out);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_FIRE_PARAMS_E1_K_DATA, fire_params.e1_k);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_FIRE_PARAMS_E1_STRIDE_DATA, fire_params.e1_stride);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_FIRE_PARAMS_E1_CHAN_OUT_DATA, fire_params.e1_chan_out);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_FIRE_PARAMS_E2_K_DATA, fire_params.e2_k);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_FIRE_PARAMS_E2_STRIDE_DATA, fire_params.e2_stride);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_FIRE_PARAMS_E2_CHAN_OUT_DATA, fire_params.e2_chan_out);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_MAX_POOL_DATA, layer.max_pool);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_POOL_PARAMS_K_DATA, pool_params.k);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_POOL_PARAMS_STRIDE_DATA, pool_params.stride);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_GLOB_POOL_DATA, layer.glob_pool);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_WEIGHT_OFFSET_DATA, layer.weight_offset);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_NUM_WEIGHTS_DATA, layer.num_weights);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_IMG_OFFSET_DATA, layer.img_offset);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_OUTPUT_OFFSET_DATA, layer.output_offset);
	write_axilite(XCOMPUTE_AXILITE_ADDR_LAYER_SCRATCH_OFFSET_DATA, layer.scratch_offset);
}

void set_layers(network_t* network) {
	//name, h, w, c_in, c_out, k, str, pad, relu, conv, fire, fire_params, max_pool, pool_params, global_pool, dropout
	fire_t fire_params_one = fire_t();
	pooling_t pool_params_one = pooling_t(3, 2);
	network->layers[0] = layer_t(224, 224, 3, 64, 3, 2, 0, true, true, false, fire_params_one, true, pool_params_one, false);
	fire_t fire_params_two = fire_t(1, 1, 16, 1, 1, 64, 3, 3, 64);
	pooling_t pool_params_two = pooling_t();
	network->layers[1] = layer_t(56, 56, 64, 128, -1, -1, 0, true, false, true, fire_params_two, false, pool_params_two, false);
	fire_t fire_params_three = fire_t(1, 1, 16, 1, 1, 64, 3, 3, 64);
	pooling_t pool_params_three = pooling_t(3, 2);
	network->layers[2] = layer_t(56, 56, 128, 128, -1, -1, 0, true, false, true, fire_params_three, true, pool_params_three, false);
	fire_t fire_params_four = fire_t(1, 1, 32, 1, 1, 128, 3, 3, 128);
	pooling_t pool_params_four = pooling_t();
	network->layers[3] = layer_t(28, 28, 128, 256, -1, -1, 0, true, false, true, fire_params_four, false, pool_params_four, false);
	fire_t fire_params_five = fire_t(1, 1, 32, 1, 1, 128, 3, 3, 128);
	pooling_t pool_params_five = pooling_t(3, 2);
	network->layers[4] = layer_t(28, 28, 256, 256, -1, -1, 0, true, false, true, fire_params_five, true, pool_params_five, false);
	fire_t fire_params_six = fire_t(1, 1, 48, 1, 1, 192, 3, 3, 192);
	pooling_t pool_params_six = pooling_t();
	network->layers[5] = layer_t(14, 14, 256, 384, -1, -1, 0, true, false, true, fire_params_six, false, pool_params_six, false);
	fire_t fire_params_seven = fire_t(1, 1, 48, 1, 1, 192, 3, 3, 192);
	pooling_t pool_params_seven = pooling_t();
	network->layers[6] = layer_t(14, 14, 384, 384, -1, -1, 0, true, false, true, fire_params_seven, false, pool_params_seven, false);
	fire_t fire_params_eight = fire_t(1, 1, 64, 1, 1, 256, 3, 3, 256);
	pooling_t pool_params_eight = pooling_t();
	network->layers[7] = layer_t(14, 14, 384, 512, -1, -1, 0, true, false, true, fire_params_eight, false, pool_params_eight, false);
	fire_t fire_params_nine = fire_t(1, 1, 64, 1, 1, 256, 3, 3, 256);
	pooling_t pool_params_nine = pooling_t();
	network->layers[8] = layer_t(14, 14, 512, 512, -1, -1, 0, true, false, true, fire_params_nine, false, pool_params_nine, false);
	fire_t fire_params_ten = fire_t();
	pooling_t pool_params_ten = pooling_t();
	network->layers[9] = layer_t(14, 14, 512, 1000, 1, 1, 0, true, true, false, fire_params_ten, false, pool_params_ten, true);
}

network_t* create_network(int num_layers, int tot_num_weights) {
	network_t* network = new network_t(num_layers, tot_num_weights);
	set_layers(network);

	layer_t l1 = network->layers[0];
	int l1_img_size = l1.h * l1.w * l1.chan_in;

	layer_t prev_layer = layer_t();
	layer_t curr_layer;
	prev_layer.weight_offset = 0;
	prev_layer.num_weights = 0;
	prev_layer.output_offset = network->num_weights;
	prev_layer.scratch_offset = prev_layer.output_offset + l1_img_size;

	for (int i = 0; i < network->num_layers; i++) {
		curr_layer = network->layers[i];

		// compute weights offset and num_weights
		curr_layer.weight_offset = prev_layer.weight_offset + prev_layer.num_weights;
		if (curr_layer.conv) {
			curr_layer.num_weights = curr_layer.chan_out * curr_layer.k * curr_layer.k * curr_layer.chan_in + curr_layer.chan_out;
		} else if (curr_layer.fire) {
			fire_t fire_params = curr_layer.fire_params;
			int s_num_weights = fire_params.s_chan_out * fire_params.s_k * fire_params.s_k * curr_layer.chan_in + fire_params.s_chan_out;
			int e1_num_weights = fire_params.e1_chan_out * fire_params.e1_k * fire_params.e1_k * fire_params.s_chan_out + fire_params.e1_chan_out;
			int e2_num_weights = fire_params.e2_chan_out * fire_params.e2_k * fire_params.e2_k * fire_params.s_chan_out + fire_params.e2_chan_out;
			int fire_num_weights = s_num_weights + e1_num_weights + e2_num_weights;
			curr_layer.num_weights = fire_num_weights;
		} else {
			printf("invalid layer format\n");
		}

		// compute img/output offset
		curr_layer.img_offset = prev_layer.output_offset;
		curr_layer.output_offset = prev_layer.scratch_offset;

		// compute scratch offset
		int h_int = curr_layer.h;
		int w_int = curr_layer.w;
		int h_out = 0;
		int w_out = 0;

		if (curr_layer.conv) {
			h_int = div_ceil(h_int, curr_layer.stride);
			w_int = div_ceil(w_int, curr_layer.stride);
		} else if (curr_layer.fire) {
			fire_t fire_params = curr_layer.fire_params;
			int s_h = div_ceil(h_int, fire_params.s_stride);
			int s_w = div_ceil(w_int, fire_params.s_stride);
			h_int = div_ceil(s_h, fire_params.e1_stride);
			w_int = div_ceil(s_w, fire_params.e1_stride);
		} else {
			printf("invalid layer format\n");
		}

		if (curr_layer.max_pool) {
			pooling_t pool_params = curr_layer.pool_params;
			h_int = div_ceil(h_int, pool_params.stride);
			w_int = div_ceil(w_int, pool_params.stride);
		}

		if (curr_layer.glob_pool) {
			h_int = 1;
			w_int = 1;
		}

		h_out = h_int;
		w_out = w_int;

		int output_size = h_out * w_out * curr_layer.chan_out;
		curr_layer.scratch_offset = curr_layer.output_offset + output_size;

		prev_layer = curr_layer;
	}

	return network;
}

void load_images(int tot_num_weights) {
	FILE * iFile;
	long iSize;
	data_t * image_buffer;

	iFile = fopen ( "ILSVRC2012_test_00000001.bin" , "rb" );
	if (iFile==NULL) {
		printf("Image file error\n");
		return;
	}

	printf("Loading image...\n");
	fseek (iFile , 0 , SEEK_END);
	iSize = ftell (iFile);
	rewind (iFile);

	image_buffer = (data_t*)DRAM_ADDR + tot_num_weights;
	fread (image_buffer,sizeof(data_t),iSize/4,iFile);

	for (int i = 0; i < 10; i++) {
		printf("%f\n", *(image_buffer + i));
	}

	fclose (iFile);
}

void load_weights() {
	FILE * pFile;
	long lSize;
	data_t * buffer;

	pFile = fopen("C:/Users/Samy/Documents/School/15-618/SqueezeNet_HW/SqueezeNet_HW.sdk/SqueezeNet/src/agg_weights.bin" , "rb");
	if (pFile == NULL) {
		printf("Weight file error\n");
		return;
	}

	printf("Reading in weights...\n");
	fflush(stdout);
	return;
	fseek (pFile , 0 , SEEK_END);
	lSize = ftell (pFile);
	printf("Number of weights: %ld", lSize);
	fflush(stdout);
	rewind (pFile);

	buffer = (data_t*) DRAM_ADDR;
	fread (buffer,sizeof(data_t),lSize/4,pFile);

	printf("Finished reading weights\n");

	fclose (pFile);
}

