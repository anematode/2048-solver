
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

	// Return 4 bits indicating whether there are any 16-bit values duplicated across 2, 4, or 8 elements, in that position

#if USE_X86_VECTORIZE
	int detect_4x16_dup(__m128i data);
	int detect_4x16_dup(__m256i data);
#endif

#ifdef USE_AVX512_VECTORIZE
	int detect_4x16_dup(__m512i data);
#endif
	
	// generate 4-bits one for each zero nibble, and 4-bits zero for each nonzero nibble
	uint64_t mask_zero_nibbles(uint64_t data);	

	// Count number of nonzero/zero nibbles to int
	int count_tiles(uint64_t data);
	int count_empty(uint64_t data);

	uint64_t count_rows(uint64_t data);
	uint8_t nibble_max(uint64_t data);
	
	// sum of the tiles -- as powers of two, not scalar
	uint32_t tile_sum(uint64_t data);

	// Create an array of all empty indices in a position
	void grab_empty_idxs(uint64_t data, uint8_t* idxs, int* count);

	// Remove duplicate positions ON THE ASSUMPTION that any duplicates are necessarily contiguous/consecutive. Write the frequencies
	// of each position to result_freqs
	void dedup_positions_consecutive(const uint64_t* __restrict__ positions, int count, uint64_t* __restrict__ results, int* result_freqs, int* result_count);

	// Whether generated is a valid next-tile position from the base position
	bool is_valid_gen_tile(uint64_t generated, uint64_t base);

#ifdef USE_X86_VECTORIZE
	__m128i mask_zero_nibbles(__m128i, __m128i);
	__m256i mask_zero_nibbles(__m256i, __m256i);

	__m128i count_tiles(__m128i);
	__m256i count_tiles(__m256i);

	__m128i count_empty(__m128i);
	__m256i count_empty(__m256i);
#ifdef USE_AVX512_VECTORIZE

#endif
#endif
}
