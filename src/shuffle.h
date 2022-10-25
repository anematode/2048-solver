#include "defs.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

/**
 * Implementations for some important basic functions, such as nibble shuffling in scalar and vector
 * cases.
 */

namespace Analysis {
	namespace fallback {
		uint64_t shuffle_nibbles(uint64_t data, uint64_t idx);

		// Shuffle entries in array
		void shuffle_nibbles_arr(uint64_t* result, const uint64_t* data, const uint64_t* idx, int len);
		void shuffle_nibbles_arr_same(uint64_t* result, const uint64_t* data, int len, uint64_t idx);
	}

	// Shuffle a 64-bit chunk of 16 nibbles by indices in idx, choosing the best implementation at run time
	uint64_t shuffle_nibbles(uint64_t data, uint64_t idx);

#if USE_X86_VECTORIZE
	// Shuffle 64-bit chunks of nibbles in parallel
	__m128i shuffle_nibbles(__m128i data, __m128i idx);
	__m256i shuffle_nibbles(__m256i data, __m256i idx);

	// Shuffle 64-bit chunks of nibbles all by the same 64-bit integer index lookup table
	__m128i shuffle_nibbles_same(__m128i data, uint64_t idx);
	__m256i shuffle_nibbles_same(__m256i data, uint64_t idx);
#endif

#ifdef USE_AVX512_VECTORIZE
	__m512i shuffle_nibbles(__m512i data, __m512i idx);
	__m512i shuffle_nibbles_same(__m512i data, uint64_t idx);
#endif
}
