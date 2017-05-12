#include <stdio.h>
#include "conv.hpp"
#include <cmath>

void max_pool_test()
{
  int i; int j; int c;
  data_t A[16 + 4] = {1,2,5,8,3,4,6,7,9,12,15,14,10,11,16,13};
  data_t A_2[50 + 8] = {10,1,0,23,29,9,2,0,24,33,8,3,0,26,52,7,4,0,27,55,6,5,0,28,90,1,2,3,4,5,10,9,8,7,6,23,0,1,2,3,54,1,2,4,8,10,4,3,2,1};
  data_t* B = &A[16];
  data_t* B_2 = &A_2[50];
  // Hard-coded answer
  data_t C[4] = {4, 8, 12, 16};
  data_t C_2[8] = {10,52,7,90,23,7,54,8};
  offset_t weight_offset = 0;
  offset_t img_offset = 0;
  offset_t img_offset_2 = 0;
  offset_t output_offset = 16;
  offset_t output_offset_2 = 50;
  offset_t scratch_offset = 16 + 4;
  offset_t scratch_offset_2 = 50 + 8;

  for (i=0; i < 4; i++) {
	  B[i] = -1;
  }

  for (i=0; i < 8; i++) {
	  B_2[i] = -1;
  }

  // Call the hardware function
  pooling_t pool = pooling_t(2, 2);
  layer_t l = layer_t(4, 4, 1, 1, 2, 2, 0, false, false, false, fire_t(), true, pool, false, weight_offset, 0, img_offset, output_offset, scratch_offset);
  max_pool(A,l);

  pooling_t pool_2 = pooling_t(3, 3);
  layer_t l_2 = layer_t(5, 5, 2, 1, 3, 3, 0, false, false, false, fire_t(), true, pool_2, false, weight_offset, 0, img_offset_2, output_offset_2, scratch_offset_2);
  max_pool(A_2,l_2);

  //Compare results
  for(i=0; i<4; i++){
    if (std::abs(B[i] - C[i]) > 0.00001) {
      printf("ERROR value mismatch at: %d found: %f expected: %f\n", i, B[i], C[i]);
      return;
//    } else {
//		printf("SUCCESS value matches at: %d found: %f expected: %f\n", i, B[i], C[i]);
	}
  }
  for(i=0; i<4; i++){
      if (std::abs(B_2[i] - C_2[i]) > 0.00001) {
        printf("ERROR value mismatch at: %d found: %f expected: %f\n", i, B_2[i], C_2[i]);
        return;
//      } else {
//  		printf("SUCCESS value matches at: %d found: %f expected: %f\n", i, B[i], C[i]);
      }
    }
  printf("SUCCESS: max pool results match\n");
  printf("\n\n");
  return;
}
