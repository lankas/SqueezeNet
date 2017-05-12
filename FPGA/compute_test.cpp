#include <stdio.h>
#include "conv.hpp"
#include <cmath>

void compute_test() {
	int i; int j; int c;
	int img_size = 50;
	int tot_num_weights = 3 * 3 * 3 * 2;
	int height = 5;
	int width = 5;
	int stride = 2;
	int c_out = 3;
	int w_out = div_ceil(width,stride);
	int h_out = div_ceil(height,stride);
	int scratch_space = w_out * h_out * c_out;
	data_t A[img_size + tot_num_weights + scratch_space];
	data_t B[12];
	// Hard-coded answer
	data_t C[12] = {2.970000, 2.475000, 2.475000, 1.980000,
				  54.000000, 45.000000, 45.000000, 36.000000,
				  0.000000, 0.000000, 0.000000, 0.000000};
	data_t* w = A;
	offset_t img_offset = tot_num_weights;
	data_t* img = &A[img_offset];
	offset_t weight_offset = 0;
	offset_t scratch_offset = img_size + img_offset;

	for(i=img_size + tot_num_weights; i < img_size + tot_num_weights + scratch_space; i++) {
		A[i] = 0;
	}
	//Put data into A
	for (i=0; i < 5; i++){
		for (j=0; j < 5; j++) {
			img[i * 5 + j] = i+1;
		}
	}

	for (i=0; i < 5; i++){
		for (j=0; j < 5; j++) {
			img[25 + i * 5 + j] = j+1;
		}
	}

	for (i=0; i < 12; i++) {
	  B[i] = -1;
	}

	for (j = 0; j < 3; j++) {
		for (i = 0; i < 3 * 3 * 2; i++) {
			if (j == 0) {
				w[i] = 0.055;
			} else if (j == 1) {
				w[j * (3 * 3 * 2) + i] = 1;
			} else {
				w[j * (3 * 3 * 2) + i] = 0;
			}

		}
	}

	compute(A,B);

	// Compare results
	for(i=0; i<12; i++){
		if (std::abs(B[i] - C[i]) > 0.00001) {
			printf("ERROR value mismatch at: %d found: %f expected: %f\n", i, B[i], C[i]);
			return;
//		} else {
//			printf("SUCCESS value matches at: %d found: %f expected: %f\n", i, B[i], C[i]);
		}
	}

	printf("\n\n");
	printf("SUCCESS: compute results match\n");
	printf("\n\n");
	return;
}
