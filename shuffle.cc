#include "shuffle.h"
#include "defs.h"

// Convention: (a & (0xf << (4 * i))) >> (4 * i) is the ith nibble of a (i.e., lowest-significant is 0)
namespace Analysis {
	namespace {
		constexpr uint64_t low_nibbles = 0x0f0f'0f0f'0f0f'0f0f;
	}

#ifdef USE_NIBBLE_SHUFFLE_VBMI
	// Given a shuffle, convert it to the indices of the high part and of the low part for a vpmultishiftqb lookup
	inline void split_nibble_shuffle(__m128i shuf, __m128i* hi, __m128i* lo) {
		const __m128i lo_nibble_msk = _mm_set1_epi8(0xf);
		__m128i indices_lo = _mm_and_si128(lo_nibble_msk, shuf);
		__m128i indices_hi = _mm_andnot_si128(lo_nibble_msk, shuf);
		*lo = _mm_slli_epi32(indices_lo, 2);
		*hi = _mm_srli_epi32(indices_hi, 2);
	}

	inline void split_nibble_shuffle(__m256i shuf, __m256i* hi, __m256i* lo) {
		const __m256i lo_nibble_msk = _mm256_set1_epi8(0xf);
		__m256i indices_lo = _mm256_and_si256(lo_nibble_msk, shuf);
		__m256i indices_hi = _mm256_andnot_si256(lo_nibble_msk, shuf);
		*lo = _mm256_slli_epi32(indices_lo, 2);
		*hi = _mm256_srli_epi32(indices_hi, 2);
	}

	inline void split_nibble_shuffle(__m512i shuf, __m512i* hi, __m512i* lo) {
		const __m512i lo_nibble_msk = _mm512_set1_epi8(0xf);
		__m512i indices_lo = _mm512_and_si512(lo_nibble_msk, shuf);
		__m512i indices_hi = _mm512_andnot_si512(lo_nibble_msk, shuf);
		*lo = _mm512_slli_epi32(indices_lo, 2);
		*hi = _mm512_srli_epi32(indices_hi, 2);
	}

	// Broadcast shuffle idx to both hi and lo. Probably use this sparingly, though, due to constant propagation
	inline void split_nibble_shuffle(uint64_t shuf, __m128i* hi, __m128i* lo) {
		uint64_t lo_s = a & LO_NIBBLES;
		uint64_t hi_s = (a - lo_s) >> 2;
		lo_s <<= 2;

		*hi = _mm_set1_epi64(hi_s);
		*lo = _mm_set1_epi64(lo_s);
	}

	inline void split_nibble_shuffle_scalar(uint64_t shuf, uint64_t* hi, uint64_t* lo) {
		*lo = a & LO_NIBBLES;
		*hi = (a - *lo) >> 2;
		*lo <<= 2;
	}
	
	inline void split_nibble_shuffle(uint64_t shuf, __m128i* hi, __m128i* lo) {
		uint64_t hi_s, lo_s;
		split_nibble_shuffle_scalar(shuf, &hi_s, &lo_s);
		
		*hi = _mm_set1_epi64x(hi_s);
		*lo = _mm_set1_epi64x(lo_s);
	}

	inline void split_nibble_shuffle(uint64_t shuf, __m256i* hi, __m256i* lo) {
		uint64_t hi_s, lo_s;
		split_nibble_shuffle_scalar(shuf, &hi_s, &lo_s);
		
		*hi = _mm256_set1_epi64x(hi_s);
		*lo = _mm256_set1_epi64x(lo_s);
	}

	inline void split_nibble_shuffle(uint64_t shuf, __m512i* hi, __m512i* lo) {
		uint64_t hi_s, lo_s;
		split_nibble_shuffle_scalar(shuf, &hi_s, &lo_s);
		
		*hi = _mm512_set1_epi64(hi_s);
		*lo = _mm512_set1_epi64(lo_s);
	}
#endif

	namespace fallback {
		inline uint64_t shuffle_nibbles(uint64_t data, uint64_t indices) {
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
		inline void shuffle_nibbles_arr(const uint64_t* data, const uint64_t* indices, uint64_t* result, int len) {
			for (int i = 0; i < len; ++i) 
				result[i] = shuffle_nibbles_scalar(data[i], indices[i]);
		}

		// Shuffle array where all shuffles are the same
		inline void shuffle_nibbles_arr_same(const uint64_t* data, uint64_t indices, uint64_t* result, int len) {
			for (int i = 0; i < len; ++i)
				result[i] = shuffle_nibbles_scalar(data[i], indices);
		}
	}

#ifdef USE_X86_VECTORIZE

	inline __m128i shuffle_nibbles(__m128i data, __m128i idx) {
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

	inline __m128i shuffle_nibbles_same(__m128i data, uint64_t idx) {
		return shuffle_nibbles(data, _mm_set1_epi64x(idx));
	}

	inline __m256i shuffle_nibbles(__m256i data, __m256i idx) {
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

	inline __m256i shuffle_nibbles_same(__m256i data, uint64_t idx) {
		return shuffle_nibbles(data, _mm256_set1_epi64x(idx));
	}

#endif

#ifdef USE_AVX512_VECTORIZE

	inline __m512i shuffle_nibbles(__m512i data, __m512i idx) {
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

	inline __m512i shuffle_nibbles_same(__m512i data, uint64_t idx) {
		return shuffle_nibbles(data, _mm512_set1_epi64(idx));
	}
#endif  // USE_AVX512_VECTORIZE
	inline uint64_t shuffle_nibbles(uint64_t a, uint64_t b) {
#ifdef USE_NIBBLE_SHUFFLE_VBMI
		return _mm_cvtsi128_si64(shuffle_nibbles(_mm_cvtsi64_si128(a), _mm_cvtsi64_si128(b)));
#else
		return fallback::shuffle_nibbles(a, b);
#endif
	}

#else   // USE_X86_VECTORIZE
	uint64_t shuffle_nibbles(uint64_t, uint64_t b) {
		return fallback::shuffle_nibbles(a, b);
	}
};