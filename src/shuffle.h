
/**
 * Implementations for some important basic functions, such as nibble shuffling in scalar and vector
 * cases.
 */
#pragma once

#include "defs.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <array>

namespace Analysis {
	namespace fallback {
		uint64_t shuffle_nibbles(uint64_t data, uint64_t idx);

		// Shuffle entries in array
		void shuffle_nibbles_arr(uint64_t* result, const uint64_t* data, const uint64_t* idx, int len);
		void shuffle_nibbles_arr_same(uint64_t* result, const uint64_t* data, int len, uint64_t idx);

		template <int cnt>
		std::array<uint64_t, cnt> shuffle_8x64(std::array<uint64_t, cnt> idxs, const uint64_t values[8]) {
			decltype(idxs) result;
			int a = 0;

			for (uint64_t i : idxs) {
				result[a] = values[i];
				++a;
			}

			return result;
		}
	}

	namespace constants {
#define PERM_64(name) constexpr inline uint64_t name 
		PERM_64(identity) = 0xfedcba9876543210;
		PERM_64(rotate_90) = 0xc840d951ea62fb73;
		PERM_64(rotate_180) = 0x0123456789abcdef;
		PERM_64(rotate_270) = 0x37bf26ae159d048c;
		PERM_64(reflect_h) = 0xcdef89ab45670123;
		PERM_64(reflect_v) = 0x32107654ba98fedc;
		PERM_64(reflect_tl) = 0xfb73ea62d951c840;
		PERM_64(reflect_tr) = 0x048c159d26ae37bf;
#undef PERM_64
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

	// Extract 64-bit values with idxs in idx between 0 and 7
	__m128i shuffle_8x64(__m128i idx, const uint64_t values[8]);
	__m256i shuffle_8x64(__m256i idx, const uint64_t values[8]);
#endif

#ifdef USE_AVX512_VECTORIZE
	__m512i shuffle_nibbles(__m512i data, __m512i idx);
	__m512i shuffle_nibbles_same(__m512i data, uint64_t idx);
	__m512i shuffle_8x64(__m512i idx, const uint64_t values[8]);
#endif

	
}
