#include "shuffle.h"
#include "defs.h"


// Convention: (a & (0xf << (4 * i))) >> (4 * i) is the ith nibble of a (i.e., lowest-significant is 0)
namespace Analysis {

#ifdef USE_NIBBLE_SHUFFLE_VBMI
	// Given a shuffle, convert it to the indices of the high part and of the low part for a vpmultishiftqb lookup
	void split_nibble_shuffle(__m128i shuf, __m128i* hi, __m128i* lo) {
		const __m128i lo_nibble_msk = _mm_set1_epi8(0xf);
		__m128i indices_lo = _mm_and_si128(lo_nibble_msk, shuf);
		__m128i indices_hi = _mm_andnot_si128(lo_nibble_msk, shuf);
		*lo = _mm_slli_epi32(indices_lo, 2);
		*hi = _mm_srli_epi32(indices_hi, 2);
	}

	void split_nibble_shuffle(__m256i shuf, __m256i* hi, __m256i* lo) {
		const __m256i lo_nibble_msk = _mm256_set1_epi8(0xf);
		__m256i indices_lo = _mm256_and_si256(lo_nibble_msk, shuf);
		__m256i indices_hi = _mm256_andnot_si256(lo_nibble_msk, shuf);
		*lo = _mm256_slli_epi32(indices_lo, 2);
		*hi = _mm256_srli_epi32(indices_hi, 2);
	}

	void split_nibble_shuffle(__m512i shuf, __m512i* hi, __m512i* lo) {
		const __m512i lo_nibble_msk = _mm512_set1_epi8(0xf);
		__m512i indices_lo = _mm512_and_si512(lo_nibble_msk, shuf);
		__m512i indices_hi = _mm512_andnot_si512(lo_nibble_msk, shuf);
		*lo = _mm512_slli_epi32(indices_lo, 2);
		*hi = _mm512_srli_epi32(indices_hi, 2);
	}

	// Broadcast shuffle idx to both hi and lo. Probably use this sparingly, though, due to constant propagation
	void split_nibble_shuffle(uint64_t shuf, __m128i* hi, __m128i* lo) {
		uint64_t lo_s = a & LO_NIBBLES;
		uint64_t hi_s = (a - lo_s) >> 2;
		lo_s <<= 2;

		*hi = _mm_set1_epi64(hi_s);
		*lo = _mm_set1_epi64(lo_s);
	}

	void split_nibble_shuffle_scalar(uint64_t shuf, uint64_t* hi, uint64_t* lo) {
		*lo = a & LO_NIBBLES;
		*hi = (a - *lo) >> 2;
		*lo <<= 2;
	}
	
	void split_nibble_shuffle(uint64_t shuf, __m128i* hi, __m128i* lo) {
		uint64_t hi_s, lo_s;
		split_nibble_shuffle_scalar(shuf, &hi_s, &lo_s);
		
		*hi = _mm_set1_epi64x(hi_s);
		*lo = _mm_set1_epi64x(lo_s);
	}

	void split_nibble_shuffle(uint64_t shuf, __m256i* hi, __m256i* lo) {
		uint64_t hi_s, lo_s;
		split_nibble_shuffle_scalar(shuf, &hi_s, &lo_s);
		
		*hi = _mm256_set1_epi64x(hi_s);
		*lo = _mm256_set1_epi64x(lo_s);
	}

	void split_nibble_shuffle(uint64_t shuf, __m512i* hi, __m512i* lo) {
		uint64_t hi_s, lo_s;
		split_nibble_shuffle_scalar(shuf, &hi_s, &lo_s);
		
		*hi = _mm512_set1_epi64(hi_s);
		*lo = _mm512_set1_epi64(lo_s);
	}
#endif

	namespace fallback {
		uint64_t shuffle_nibbles(uint64_t data, uint64_t indices) {
			uint64_t result = 0;
			for (int i = 0; i < 16; ++i) {
				indices = (indices >> 60) + (indices << 4);

				int idx = indices & 0xf;
				result <<= 4;
				result |= (data >> (4 * idx)) & 0xf;
			}

			return result;
		}

		// Shuffle nibbles by array with a given length, with each entry by a different index set
		void shuffle_nibbles_arr(const uint64_t* data, const uint64_t* indices, uint64_t* result, int len) {
			for (int i = 0; i < len; ++i) 
				result[i] = shuffle_nibbles(data[i], indices[i]);
		}

		// Shuffle array where all shuffles are the same
		void shuffle_nibbles_arr_same(const uint64_t* data, uint64_t indices, uint64_t* result, int len) {
			for (int i = 0; i < len; ++i)
				result[i] = shuffle_nibbles(data[i], indices);
		}
	}

#ifdef USE_X86_VECTORIZE

	__m128i shuffle_nibbles(__m128i data, __m128i idx) {
#ifdef USE_NIBBLE_SHUFFLE_VBMI
		__m128i lo_nibble_msk = _mm_set1_epi8(0x0f);

		__m128i shuf_lo, shuf_hi;
		split_nibble_shuffle_128(idx, &shuf_hi, &shuf_lo);

		__m128i shuffled_lo = _mm_multishift_epi64_epi8(shuf_lo, data);
		__m128i shuffled_hi = _mm_multishift_epi64_epi8(shuf_hi, data);

		shuffled_hi = _mm_slli_epi32(shuffled_hi, 4);
		return _mm_ternarylogic_epi32(lo_nibble_msk, shuffled_lo, shuffled_hi, 202);
#else
		// TODO: implement okay fallback
		uint64_t s_data[2], s_idx[2], result[2];
		_mm_storeu_si128((__m128i*) s_data, data);
		_mm_storeu_si128((__m128i*) s_idx, idx);
		shuffle_nibbles_arr(s_data, s_idx, result, 2);

		return _mm_loadu_si128((const __m128i*) result);
#endif
	}

	__m128i shuffle_nibbles_same(__m128i data, uint64_t idx) {
		return shuffle_nibbles(data, _mm_set1_epi64x(idx));
	}

	__m256i shuffle_nibbles(__m256i data, __m256i idx) {
#ifdef USE_NIBBLE_SHUFFLE_VBMI
		__m256i lo_nibble_msk = _mm256_set1_epi8(0x0f);

		__m256i shuf_lo, shuf_hi;
		split_nibble_shuffle_256(idx, &shuf_hi, &shuf_lo);

		__m256i shuffled_lo = _mm256_multishift_epi64_epi8(shuf_lo, data);
		__m256i shuffled_hi = _mm256_multishift_epi64_epi8(shuf_hi, data);

		shuffled_hi = _mm256_slli_epi32(shuffled_hi, 4);
		return _mm256_ternarylogic_epi32(lo_nibble_msk, shuffled_lo, shuffled_hi, 202);
#else
		uint64_t s_data[2], s_idx[2], result[2];
		_mm256_storeu_si256((__m256i*) s_data, data);
		_mm256_storeu_si256((__m256i*) s_idx, idx);
		shuffle_nibbles_arr(s_data, s_idx, result, 2);

		return _mm256_loadu_si256((const __m256i*) result);
#endif
	}

	__m256i shuffle_nibbles_same(__m256i data, uint64_t idx) {
		return shuffle_nibbles(data, _mm256_set1_epi64x(idx));
	}
#endif

#ifdef USE_AVX512_VECTORIZE

	__m512i shuffle_nibbles(__m512i data, __m512i idx) {
#ifdef USE_NIBBLE_SHUFFLE_VBMI
		__m512i lo_nibble_msk = _mm512_set1_epi8(0x0f);

		__m512i shuf_lo, shuf_hi;
		split_nibble_shuffle_512(idx, &shuf_hi, &shuf_lo);

		__m512i shuffled_lo = _mm512_multishift_epi64_epi8(shuf_lo, data);
		__m512i shuffled_hi = _mm512_multishift_epi64_epi8(shuf_hi, data);

		shuffled_hi = _mm512_slli_epi32(shuffled_hi, 4);
		return _mm512_ternarylogic_epi32(lo_nibble_msk, shuffled_lo, shuffled_hi, 202);
#endif
		uint64_t s_data[2], s_idx[2], result[2];
		_mm512_storeu_si512((__m512i*) s_data, data);
		_mm512_storeu_si512((__m512i*) s_idx, idx);
		shuffle_nibbles_arr(s_data, s_idx, result, 2);

		return _mm512_loadu_si512((const __m512i*) result);
	}

	__m512i shuffle_nibbles_same(__m512i data, uint64_t idx) {
		return shuffle_nibbles(data, _mm512_set1_epi64(idx));
	}
#endif  // USE_AVX512_VECTORIZE

	uint64_t shuffle_nibbles(uint64_t a, uint64_t b) {
#ifdef USE_NIBBLE_SHUFFLE_VBMI
		return _mm_cvtsi128_si64(shuffle_nibbles(_mm_cvtsi64_si128(a), _mm_cvtsi64_si128(b)));
#else
		return fallback::shuffle_nibbles(a, b);
#endif
	}


#ifdef USE_NIBBLE_SHUFFLE_VBMI
	// Prefer vpermi2q	
	void load_8x64_split(const uint64_t values[8], __m256i* values1, __m256i* values2) {
		*values1 = _mm256_loadu_si256((const __m256i*) values);	
		*values2 = _mm256_loadu_si256((const __m256i*) (values + 4));	
	}

	__m128i shuffle_8x64(__m128i idx, const uint64_t values[8]) {
		return _mm256_castsi256_si128(
				shuffle_8x64(_mm256_castsi128_si256(idx), values)
				);

	}
	__m256i shuffle_8x64(__m256i idx, const uint64_t values[8]) {
		__m256i values1, values2;
		load_8x64_split(&values1, &values2);

		return _mm256_permutex2var_epi64(values1, idx, values2);
	}
#elif defined(USE_AVX512_VECTORIZE)			
	__m256i shuffle_8x64(__m256i idx, const uint64_t values[8]) {
		// vpermq followed by blend based on higher bit
		__m256i values1, values2;
		load_8x64_split(&values1, &values2);

		__m256 shuffled1 = _mm256_castsi256_pd(_mm256_permutexvar_epi64(idx, values1));
		__m256 shuffled2 = _mm256_castsi256_pd(_mm256_permutexvar_epi64(idx, values2));

		__m256 blendv = _mm256_castsi256_pd(_mm256_slli_epi64(idx, 64));

		return _mm256_castpd_si256(_mm256_blendv_pd(shuffled1, shuffled2, blendv));
	}
#elif defined(USE_AVX2_VECTORIZE)
#error Unimplemented
	__m256i shuffle_8x64(__m256i idx, const uint64_t values[8]) {
		// collapse values into two halves, then 2x vpermd
		__m256i hi_half = _mm256_
	}
#endif

#ifdef USE_AVX512_VECTORIZE
	// Prefer 512-bit vpermq for 512-bit vectors
	__m512i shuffle_8x64(__m512i idx, const uint64_t values[8]) {
		return _mm512_permutexvar_epi64(idx, values);
	}
#endif

#ifdef USE_X86_VECTORIZE
	__m128i shuffle_8x64(__m128i idx, const uint64_t values[8]) {
		return _mm256_castsi256_si128(shuffle_8x64(_mm256_castsi128_si256(idx), values));
	}
#endif

#ifdef USE_X86_VECTORIZE
	int movemask_epi16(__m128i v) {
#ifdef USE_AVX512_VECTORIZE
		return _mm_movepi16_mask(v);
#else
		int m = _mm_movemask_epi8(v);
		return _pdep_u32(m, 0x5555);
#endif
	}

	int movemask_epi16(__m256i v) {
#ifdef USE_AVX512_VECTORIZE
		return _mm256_movepi16_mask(v);
#else
		int m = _mm256_movemask_epi8(v);
		return _pdep_u32(m, 0x5555'5555);
#endif
	}

	int detect_4x16_dup(__m128i data) {
		__m128i xchg = _mm_shuffle_epi32(data, 0b00'01'02'03);
		__m128i cmp = _mm_cmpeq_epi16(data, xchg);

		return movemask_epi16(cmp);
	}

	int detect_4x16_dup(__m256i data) {
		__m256i xchg1 = _mm256_permute4x64_epi64(data, 0b10'01'00'11);
		__m256i xchg2 = _mm256_permute4x64_epi64(data, 0b11'10'01'00);
		__m256i xchg2 = _mm256_permute4x64_epi64(data, 0b00'11'10'01);

		__m256i cmp = _mm256_cmpeq_epi16(data, xchg1);
		__m256i cmp2 = _mm256_cmpeq_epi16(data, xchg2);
		__m256i cmp3 = _mm256_cmpeq_epi16(data, xchg3);

#ifdef USE_AVX512_VECTORIZE
		cmp = _mm256_ternarylogic_epi32(cmp, cmp2, cmp3, 128);  // three-way AND
#else
		cmp = _mm256_and_si256(cmp, _mm256_and_si256(cmp2, cmp3));
#endif

		return movemask_epi16(cmp);
	}

#ifdef USE_AVX512_VECTORIZE
	int detect_4x16_dup(__m512i data) {
		assert(false);
		return 0;
	}
#endif

#endif // USE_X86_VECTORIZE

	uint64_t mask_nonzero_nibbles_to_ones(uint64_t data) {
		// Concept: repeated or to the right

		uint64_t hi_2 = data & 0xcccccccc'cccccccc;
		uint64_t lo_2 = (data - hi_2) | (hi_2 >> 2);
		
		uint64_t hi_1 = lo_2 & 0x22222222'22222222;

		return (lo_2 - hi_1) | (hi_1 >> 1);
	}

	uint64_t mask_zero_nibbles(uint64_t data) {
		return ~(mask_nonzero_nibbles_to_ones(data) * 15);
	}

	int count_tiles(uint64_t data) {
		return __builtin_popcountll(mask_nonzero_nibbles_to_ones(data));
	}

	int count_empty(uint64_t data) {
		return 16 - count_tiles(data);
	}


	// Perhaps useless, but good for testing: Put number of each tile in each row
	uint64_t count_rows(uint64_t data) {
		uint64_t m = ~mask_zero_nibbles(data);

		auto pc = [&] (uint64_t a) { return __builtin_popcountll(a & 0xffff); };

		return (((uint64_t)pc(m >> 48) << 48) +
			((uint64_t)pc(m >> 32) << 32) +
			((uint64_t)pc(m >> 16) << 16) +
			pc(m)) >> 2;
	}

	uint8_t nibble_max(uint64_t data) {
		assert(0);
		return 0;
	}

	uint64_t nibble_row_max(uint64_t data) {
		assert(0);
		return 0;
	}

	// Compute the maximum nibble in each row, and the maximum nibble overall
	void nibble_maxes(uint64_t data, uint64_t* nibble_row_max, uint8_t* nibble_max) {
		assert(0);
	}

	// sum of the tiles -- as powers of two, not scalar. This will monotonically increase by
	// 2 or 4 each time a move is played.
	uint32_t tile_sum(uint64_t data) {
		uint32_t s = 0;
		for (int i = 0; i < 16; ++i) {
			uint32_t t = data & 0xf;

			if (t) s += 1 << t;

			data >>= 4;
		}

		return s;
	}

#ifdef USE_X86_VECTORIZE
	__m128i mask_zero_nibbles(__m128i data) {
		// split into hi half and low half, followed by 2x pcmpeqb and then ternlogd
		const __m128i zero = _mm_setzero_si128();
		const __m128i low_nibbles = _mm_set1_epi8(0xf);

		__m128i lo_half = _mm_and_si128(low_nibbles, data);
		__m128i hi_half = _mm_andnot_si128(low_nibbles, data);

#ifdef USE_AVX2_VECTORIZE
		return _mm_or_si128(_mm_and_si128(lo_half, low_nibbles), _mm_andnot_si128(low_nibbles, hi_half));
#else
		return _mm_ternarylogic_epi32(low_nibbles, hi_half, lo_half, 172); // low_nibbles ? lo_half : hi_half
#endif
	}

	__m256i mask_zero_nibbles(__m256i data) {
		// split into hi half and low half, followed by 2x pcmpeqb and then ternlogd
		const __m256i zero = _mm256_setzero_si128();
		const __m256i low_nibbles = _mm256_set1_epi8(0xf);

		__m256i lo_half = _mm256_and_si128(low_nibbles, data);
		__m256i hi_half = _mm256_andnot_si128(low_nibbles, data);

#ifdef USE_AVX2_VECTORIZE
		return _mm256_or_si128(_mm256_and_si128(lo_half, low_nibbles), _mm256_andnot_si128(low_nibbles, hi_half));
#else
		return _mm256_ternarylogic_epi32(low_nibbles, hi_half, lo_half, 172); // low_nibbles ? lo_half : hi_half
#endif
	}

	__m128i not_si128(__m128i a) {
#ifdef USE_AVX2_VECTORIZE
		return _mm_xor_si128(_mm_set1_epi32(1), a);
#else
		return _mm_ternarylogic_epi32(a, a, a, 0x55);
#endif
	}

	__m256i not_si256(__m256i a) {
#ifdef USE_AVX2_VECTORIZE	
		return _mm256_xor_si256(_mm256_set1_epi32(1), a);
#else
		return _mm256_ternarylogic_epi32(a, a, a, 0x55);
#endif
	}

	// Count, in each 64-bit element, the number of nonzero nibbles
	__m128i count_tiles(__m128i data) {
#if defined(USE_AVX512_VECTORIZE) && defined(__AVX512VPOPCNTDQ__)
		return _mm_srli_epi32(_mm_popcnt_epi64(not_si128(mask_zero_nibbles(data))), 2);
#else // USE_AVX2_VECTORIZE
		// No vectorized popcount, so split into 64-bit elements and enjoy
		uint64_t d[2];
		_mm_storeu_si128((__m128i*) d, data);

		d[0] = count_tiles(d[0]);
		d[1] = count_tiles(d[1]);

		return _mm_loadu_si128((const __m128i*) d);
#endif
	}

	__m256i count_tiles(__m256i data) {
#if defined(USE_AVX512_VECTORIZE) && defined(__AVX512VPOPCNTDQ__)
		return _mm256_srli_epi32(_mm256_popcnt_epi64(not_si256(mask_zero_nibbles(data))), 2); // popcnt, then divide by 4
#else // USE_AVX2_VECTORIZE
		uint64_t d[4];
		_mm256_storeu_si256((__m256i*) d, data);

		for (int i = 0; i < 4; ++i)
			d[i] = count_tiles(d[i]);

		return _mm256_loadu_si256((const __m256i*) d);
#endif
	}

#endif // USE_X86_VECTORIZE
	void grab_empty_idxs(uint64_t data, uint8_t* idxs, int* count) {
		int write_i = 0;

		for (uint8_t idx = 0; idx < 16; ++idx) {
			if (!(data & (0xfULL << (4 * idx)))) {
				idxs[write_i++] = idx;
			}
		}

		*count = write_i;
	}

	void dedup_positions_consecutive(const uint64_t* __restrict__ positions, int count, uint64_t* __restrict__ results, int* result_freqs, int* result_count) {
		if (unlikely(count == 0)) {
			*result_count = 0;
			return;
		}

		uint64_t previous = positions[0];
		results[0] = previous;
		result_freqs[0] = 1;

		int write_i = 0;

		for (int i = 1; i < count; ++i) {
			if (positions[i] != previous) {
				write_i++;
				results[write_i] = positions[i];
				result_freqs[write_i] = 1;
				previous = positions[i];
			} else {
				result_freqs[write_i]++;
			}
		}

		*result_count = write_i + 1;
	}

	bool is_valid_gen_tile(uint64_t generated, uint64_t base) {
		uint64_t kk = generated ^ base;
		if (unlikely(kk == 0)) return false;

		int tz = __builtin_ctzll(kk);
		int tzr = tz / 4 * 4;

		if (kk != 1ULL << tzr && kk != 2ULL << tzr) {
			return false;	
		}

		return !(base & (0xfULL << tzr));
	}
};
