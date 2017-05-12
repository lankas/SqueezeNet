#include <stdio.h>
#include "conv.hpp"
#include <cmath>
#include <ctime>

void conv_test() {
	int i; int j; int c;
	int num_weights = 3 * 3 * 3 * 2 + 3;
	data_t A[3 * 3 * 3 * 2 + 3 + 50 + 27];
	// Hard-coded answer
	data_t C[27] = {0.660000, 1.485000, 1.320000, 1.485000, 2.970000, 2.475000, 1.320000, 2.475000, 1.980000,
				  12.000000, 27.000000, 24.000000, 27.000000, 54.000000, 45.000000, 24.000000, 45.000000, 36.000000,
				  0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000};
	data_t* w = A;
	data_t* img = &A[num_weights];
	data_t* B = &A[num_weights + 50];

	int num_weights_2 = 2 * 1 * 1 * 2 + 2;
	data_t A_2[2 * 1 * 1 * 2 + 2 + 50 + 18] = {2,0,-1,0, 1, 2};
	data_t* img_2 = &A_2[num_weights_2];
	data_t image[50] = {-5,4,3,2,1,1,2,3,4,5,1,2,3,4,5,4,4,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
	data_t* B_2 = &A_2[num_weights_2 + 50];
	// Hard-coded answer
//
	data_t C_2[18] = {0,7,3,3,7,11,3,3,3,7,0,1,1,0,0,1,1,1};

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

	for (i=0; i < 27; i++) {
		B[i] = -1;
	}

	for (j = 0; j < 3; j++) {
		for (i = 0; i < 3 * 3 * 2; i++) {
			if (j == 0) {
				w[j * (3 * 3 * 2) + i] = 0.055;
			} else if (j == 1) {
				w[j * (3 * 3 * 2) + i] = 1;
			} else {
				w[j * (3 * 3 * 2) + i] = 0;
			}

		}
	}

	// bias weights
	for (i = 0; i < 3; i++) {
		w[3 * 3 * 3 * 2 + i] = 0;
	}

	for (i = 0; i < 50; i++) {
		img_2[i] = image[i];
	}

//	for (j = 0; j < 3; j++) {
//		for (i = 0; i < 3 * 3 * 2; i++) {
//			printf("%f ", w[j * (3 * 3 * 2) + i]);
//		}
//		printf("\n");
//	}
//	printf("\n");

	offset_t weight_offset = 0;
	offset_t img_offset = num_weights;
	offset_t img_offset_2 = num_weights_2;
	offset_t output_offset = img_offset + 50;
	offset_t output_offset_2 = img_offset_2 + 50;
	offset_t scratch_offset = output_offset + 27;
	offset_t scratch_offset_2 = output_offset_2 + 18;

	// Call the hardware function
	layer_t l = layer_t(5, 5, 2, 3, 3, 2, 0, true, true, false, fire_t(), false, pooling_t(), false, weight_offset, num_weights, img_offset, output_offset, scratch_offset);

	std::clock_t start;
	long double duration;

	start = std::clock();
	compute(A,l);
	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

	printf("time taken: %Lf\n", duration);
	layer_t l_2 = layer_t(5, 5, 2, 2, 1, 2, 0, true, true, false, fire_t(), false, pooling_t(), false, weight_offset, num_weights_2, img_offset_2, output_offset_2, scratch_offset_2);
	compute(A_2,l_2);

	// Compare results
	for(i=0; i<27; i++){
		if (std::abs(B[i] - C[i]) > 0.00001) {
			printf("ERROR A value mismatch at: %d found: %f expected: %f\n", i, B[i], C[i]);
//			return;
//			} else {
//				printf("SUCCESS A value matches at: %d found: %f expected: %f\n", i, B[i], C[i]);
		}
	}
//	printf("\n\n");
	for(i=0; i<18; i++){
		if (std::abs(B_2[i] - C_2[i]) > 0.00001) {
			printf("ERROR A_2 value mismatch at: %d found: %f expected: %f\n", i, B_2[i], C_2[i]);
//			return;
//		} else {
//			printf("SUCCESS A_2 value matches at: %d found: %f expected: %f\n", i, B_2[i], C_2[i]);
		}
	}
//  printf("\n\n");

  printf("SUCCESS: conv results match\n");
  printf("\n\n");
  return;
}
