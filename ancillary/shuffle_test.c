#include "shuffle.h"

int main() {
	// 0xaa025411fe034102
	uint64_t r1 = shuffle_nibbles(0xfedcba9876543210, 0xaa025411fe034102);
	// 0x55fdabee01fcbefd
	uint64_t r2 = shuffle_nibbles(0x0123456789abcdef, 0xaa025411fe034102);
	// 0xaaaa00002222aaaa
	uint64_t r3 = shuffle_nibbles(0xaa025411fe034102, 0xeeee11110000ffff);

	printf("0x%" PRIx64 "\n", r1);
	printf("0x%" PRIx64 "\n", r2);
	printf("0x%" PRIx64 "\n", r3);
}
