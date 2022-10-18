
#include "2048.h"

int fill_compress_right_lut() {
	for (int i = 0; i < sizeof(compress_right_lut) / sizeof(uint32_t); ++i) {
		uint32_t k = 0;

		for (int j = 0; j < 4; ++j) {
			k |= (k & (3 << (4 * j))) << (4 * j);
		}

		compress_right_lut[i] = k;
	}

	return 0;
}
