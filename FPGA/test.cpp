#include <stdio.h>
#include "conv.hpp"

int conv_test();
int max_pool_test();
int conv_max_pool_test();
int conv_glob_pool_test();
int fire_test();
int fire_max_pool_test();

int main() {
	conv_test();
	max_pool_test();
	conv_max_pool_test();
	fire_test();
	fire_max_pool_test();
	conv_glob_pool_test();

	return 0;
}
