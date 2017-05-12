#include <stdio.h>
#include "conv.hpp"
#include <cmath>

void conv_max_pool_test()
{
  int i; int j; int c;
  int img_size = 50;
  int output_size = 12;
  int tot_num_weights = 3 * 3 * 3 * 2 + 3;
  int height = 5;
  int width = 5;
  int stride = 2;
  int c_out = 3;
  int w_out = div_ceil(width,stride);
  int h_out = div_ceil(height,stride);
  int scratch_space = w_out * h_out * c_out;
  data_t A[tot_num_weights + img_size + output_size + scratch_space];
  // Hard-coded answer
  data_t C[12] = {2.970000, 2.475000, 2.475000, 1.980000,
		          54.000000, 45.000000, 45.000000, 36.000000,
				  0.000000, 0.000000, 0.000000, 0.000000};
  data_t* w = A;
  offset_t img_offset = tot_num_weights;
  data_t* img = &A[img_offset];
  data_t* B = &A[tot_num_weights + img_size];
  offset_t weight_offset = 0;
  offset_t output_offset = tot_num_weights + img_size;
  offset_t scratch_offset = img_offset + img_size + output_size;

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

  	  // bias weights
  	for (i = 0; i < 3; i++) {
  		w[3 * 3 * 3 * 2 + i] = 0;
  	}

  layer_t l = layer_t(5, 5, 2, 3, 3, 2, 0, true, true, false, fire_t(), true, pooling_t(2,2), false, weight_offset, tot_num_weights, img_offset, output_offset, scratch_offset);

//  for (i = tot_num_weights; i < img_size + tot_num_weights + scratch_space; i++) {
//	  if (i % 5 == 0) {
//		  printf("\n");
//	  }
//	  printf("%f ", A[i]);
//  }
//  printf("\n\n\n");
//
//  for (i = 0; i < 12; i++) {
//	  printf("%f ", B[i]);
//  }
//  printf("\n\n\n");

  // Call the hardware function
  compute(A, l);

//  for (i = img_size + tot_num_weights; i < img_size + tot_num_weights + scratch_space; i++) {
//  	  if (i % 5 == 0) {
//  		  printf("\n");
//  	  }
//  	  printf("%f ", A[i]);
//    }
//    printf("\n");
//
//    for (i = 0; i < 12; i++) {
//    	  printf("%f ", B[i]);
//      }
//      printf("\n\n\n");

	// Check results
	for(i=0; i<12; i++){
		if (std::abs(B[i] - C[i]) > 0.00001) {
			printf("ERROR value mismatch at: %d found: %f expected: %f\n", i, B[i], C[i]);
			return;
//		} else {
//			printf("SUCCESS value matches at: %d found: %f expected: %f\n", i, B[i], C[i]);
		}
	}
	printf("SUCCESS: combined conv/max pool results match\n");
	printf("\n\n");
	return;
}
