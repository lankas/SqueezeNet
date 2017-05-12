#include <stdio.h>
#include "conv.hpp"
#include <cmath>

void fire_test() {
	int i; int j; int c;
	int s_num_weights = 2 * 1 * 1 * 3 + 2;
	int e1_num_weights = 4 * 1 * 1 * 2 + 4;
	int e2_num_weights = 4 * 3 * 3 * 2 + 4;
	int tot_num_weights = s_num_weights + e1_num_weights + e2_num_weights;

	int height = 5;
	int width = 5;
	int c_in = 3;
	int s_stride = 2;
	int s_c_out = 2;

	int s_h = div_ceil(height, s_stride);
	int s_w = div_ceil(width, s_stride);
	int scratch_space = s_h * s_w * s_c_out;
	int img_size = 75;
	int output_size = 72;
	data_t A[tot_num_weights + img_size + output_size + scratch_space];
	data_t weights[20] = {0, 1, 1, 1, 0, 1, 0, 0, 0.5, 0.5, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0};
	data_t image[75] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,
	                25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,
	                1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
	data_t* B = &A[tot_num_weights + img_size];
	// Hard-coded answer
	data_t C[72] = {14,14,14,14,14,14,14,14,14,
					26,24,22,16,14,12,6,4,2,
					2,4,6,12,14,16,22,24,26,
					28,28,28,28,28,28,28,28,28,
					56,84,56,84,126,84,56,84,56,
					32,54,40,78,126,90,72,114,80,
					80,114,72,90,126,78,40,54,32,
	                112,168,112,168,252,168,112,168,112};

	offset_t weight_offset = 0;
	offset_t img_offset = tot_num_weights;
	offset_t output_offset = img_offset + img_size;
	offset_t scratch_offset = output_offset + output_size;
	data_t* w = A;
	data_t* img = &A[img_offset];

	for (i = 0; i < 75; i++) {
		img[i] = image[i];
	}

	for (i=0; i < 72; i++) {
		B[i] = -1;
	}

	for (i = 0; i < s_num_weights + e1_num_weights; i++) {
		w[i] = weights[i];
	}

	for (i = s_num_weights + e1_num_weights; i < s_num_weights + e1_num_weights + 18; i++) {
		w[i] = 0.5;
	}

	for (i = s_num_weights + e1_num_weights + 18; i < s_num_weights + e1_num_weights + 27; i++) {
		w[i] = 0;
	}

	for (i = s_num_weights + e1_num_weights + 27; i < s_num_weights + e1_num_weights + 45; i++) {
		w[i] = 1;
	}

	for (i = s_num_weights + e1_num_weights + 45; i < s_num_weights + e1_num_weights + 54; i++) {
		w[i] = 0;
	}

	for (i = s_num_weights + e1_num_weights + 54; i < tot_num_weights - 4; i++) {
		w[i] = 1;
	}

	// bias weights
	for (i = 0; i < 4; i++) {
		w[tot_num_weights - 4 + i] = 0;
	}

//	printf("weights: \n");
//	for (int i = 0; i < tot_num_weights; i++) {
//		if (i % 10 == 0) {
//			printf("\n");
//		}
//		printf("%f ", w[i]);
//	}
//	printf("\n");

	fire_t fire_params = fire_t(1,2,2,1,1,4,3,1,4);
	layer_t l = layer_t(5, 5, 3, 8, -1, -1, 0, true, false, true, fire_params, false, pooling_t(), false, weight_offset, tot_num_weights, img_offset, output_offset, scratch_offset);

	// Call the hardware function
	compute(A, l);

	//Compare results
	for(i=0; i < 72; i++){
		if (std::abs(B[i] - C[i]) > 0.00001) {
			printf("ERROR value mismatch at: %d found: %f expected: %f\n", i, B[i], C[i]);
			return;
//		} else {
//			printf("SUCCESS value matches at: %d found: %f expected: %f\n", i, B[i], C[i]);
		}
	}
	printf("SUCCESS: fire results match\n");
	printf("\n\n");
	return;
}
