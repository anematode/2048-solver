#pragma once

#include <array>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <initializer_list>
#include "dbg.h"

#include <stdexcept>

#define USE_NEON 0
#define USE_VEC 0
#define USE_SSE 0

#ifdef __ARM_NEON__
#include <arm_neon.h>
#define VEC_128 int8x16_t
#define set_vec128_u8 v

#undef USE_NEON
#undef USE_VEC

#define USE_NEON 1
#define USE_VEC 1
#elif __AVX2__
#include <immintrin.h>
#define VEC_128 __m128i
#define set_vec128_u8 _mm_setr_epi8

#undef USE_SSE
#undef USE_VEC

#define USE_SSE 1
#define USE_VEC 1
#endif

inline int tile_to_repr(uint32_t tile, bool validate=true) {  // 4 -> 2, 2 -> 1, 0 -> 0
	if (tile == 0) return 0;

	int k = __builtin_ctzl(tile);

	if (validate && (tile != (1 << k) || !k || k > 17)) {
		fprintf(stderr, "Invalid tile %d; should be a power of 2 between 2 and 131072 inclusive, or 0\n", tile);
		abort();
	}

	return k;
}

inline int repr_to_tile(uint32_t repr) {
	return (repr == 0) ? 0 : (1 << repr);
}

namespace Perm8x16 {
#define PERM(name) constexpr uint8_t name alignas(16) [16]

	PERM(rotate_90) = { 3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12 };	
	PERM(rotate_180) = { 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
	PERM(rotate_270) = { 12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3 };
	PERM(reflect_h) = { 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12 };
	PERM(reflect_v) = { 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3 };
	PERM(reflect_tl) = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };  // across top-left to bottom-right diagonal
	PERM(reflect_tr) = { 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0 };  // across top-right to bottom-left diagonal

	PERM(broadcast_col1) = { 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12 }; // take first col and broadcast it right
	PERM(compress_offsets) = { 0, 0, 0, 0, 16, 16, 16, 16, 32, 32, 32, 48, 48, 48, 48 };
#undef PERM
};

extern uint16_t mr_lut_16[65536];
extern uint32_t mr_lut_32[65536];

namespace detail {
	int fill_mr_luts();
	inline int c = fill_mr_luts();
}

static const int8_t com_x_weights[16] = {
	-63, -1, 1, 63, -63, -1, 1, 63, -63, -1, 1, 63, -63, -1, 1, 63
};

static const int8_t com_y_weights[16] = {
	-63, -63, -63, -63, -1, -1, -1, -1, 1, 1, 1, 1, 63, 63, 63, 63
};

static const uint64_t LO_NIBBLES = 0x0f0f0f0f'0f0f0f0f;
static const uint64_t HI_NIBBLES = 0x0f0f0f0f'0f0f0f0f;

struct Position2048 {
	// We store a position as 8 bytes, one nibble each. The mapping is 0 -> 0, 1 -> 2, et cetera.
	// We therefore cannot store 65536 or 131072 tiles. Sad!
	
	uint64_t tiles;	

	Position2048() {
		tiles = 0;
	}

	inline void set_tile(int idx, uint8_t value) {
		uint64_t msk = 0xfULL << (4 * idx);
		tiles = (tiles & ~msk) | ((uint64_t)(value & 0xf) << (4 * idx));
	}

	inline uint8_t get_tile(int idx) const {
		uint64_t msk = 0xfULL << (4 * idx);

		return (tiles & msk) >> (4 * idx);
	}

	Position2048(std::initializer_list<uint32_t> l) {
		tiles = 0;

		for (uint32_t i : l) {
			tiles |= (tile_to_repr(i));
			tiles = ((tiles & 0xf) << 60) | (tiles >> 4);

		}
	}

	void clear() {
		tiles = 0;
	}

	Position2048(const Position2048& p) {
		tiles = p.tiles;
	}

	Position2048(uint64_t a) {
		tiles = a;
	}

	Position2048& operator=(const Position2048& p) {
		tiles = p.tiles;
		return *this;
	}

	~Position2048() {

	}

	inline uint64_t lo_nibbles() const {
		return tiles & LO_NIBBLES;
	}

	inline uint64_t hi_nibbles() const {
		return tiles & HI_NIBBLES;
	}

	inline uint8_t tile_sum() const {  // impossible to have a sum > 256
		uint64_t a = lo_nibbles() + (hi_nibbles() >> 4);

		a = (a >> 32) + (a & (uint32_t)-1);
		a = (a >> 16) + (a & (uint16_t)-1);
		a = (a >> 8) + (a & 0xff);

		return a;
	}

#if USE_VEC && defined(__BMI2__)
	inline __m128i _to_sse_bytes() const {
		return _mm_set_epi64(_pext_u64(p.tiles, LOW_NIBBLE), _pext_u64(p.tiles >> 32, LOW_NIBBLE));
	}

	inline Position2048& _from_sse_bytes(__m128i a) {
		tiles = _pdep_u64(_mm_cvtsi128_si64(a), LOW_NIBBLE) | ((_pdep_u64(_mm_extract_epi64(a, 1), LOW_NIBBLE)) << 32);
		return *this;
	}
#endif

	inline Position2048& perm_self(const uint8_t* p) {
#if USE_VEC && defined(__BMI2__)
		return _from_sse_bytes(_mm_shuffle_epi8(_to_sse_bytes(),
					_mm_load_si128((const __m128i*) p)));
#endif
		uint64_t v = 0;
		for (int i = 0; i < 16; ++i) {
			v |= get_tile(p[i]);
			v = ((v & 0xf) << 60) | (v >> 4);
		}

		tiles = v;

		return *this;
	}

	// Transforms in place. Rotations are counterclockwise by convention. Use copy() first if you don't want to modify the original.
	inline Position2048& rotate_90() {
		return perm_self(Perm8x16::rotate_90);
	}

	inline Position2048& rotate_180() {
		return perm_self(Perm8x16::rotate_180);
	}

	inline Position2048& rotate_270() {	
		return perm_self(Perm8x16::rotate_270);
	}

	// Across y axis
	inline Position2048& reflect_h() {
		return perm_self(Perm8x16::reflect_h);
	}

	// Across x axis
	inline Position2048& reflect_v() {
		return perm_self(Perm8x16::reflect_v);
	}

	inline Position2048& reflect_tr() {
		return perm_self(Perm8x16::reflect_tr);
	}

	inline Position2048& reflect_tl() {
		return perm_self(Perm8x16::reflect_tl);
	}

	inline Position2048& move_right() {
		uint16_t msk = -1;

		printf("%x %llx\n", tiles & msk, mr_lut_16[0x10]);

		tiles = (uint64_t)mr_lut_16[tiles & msk] |
			((uint64_t)mr_lut_16[(tiles >> 16) & msk] << 16) |
			((uint64_t)mr_lut_16[(tiles >> 32) & msk] << 32) |
			((uint64_t)mr_lut_16[(tiles >> 48) & msk] << 48);


		return *this;
	}

	inline Position2048 copy() const {
		return *this;
	}

	char* to_string() const {
		char out[400];
		char* end = out;

		for (int i = 0; i < 16; ++i) {
			end += sprintf(end, "%d", repr_to_tile(get_tile(i)));
			*end++ = (i % 4 == 3) ? '\n' : '\t';
		}
		
		*end++ = '\0';

		int len;

		char* s = (char*)malloc(len = end - out);
		memcpy(s, out, len);

		return s;
	}

	inline bool operator==(const Position2048& b) const noexcept {
		return tiles == b.tiles;
	}

	inline bool operator !=(const Position2048& b) const noexcept {
		return !(*this == b);
	}

	inline void _compute_center_of_mass(int* com_x, int* com_y) {
		// Dot product

		*com_x = *com_y = 0;
		for (int i = 0; i < 16; ++i) {
			*com_x += get_tile(i) * com_x_weights[i];
			*com_y += get_tile(i) * com_y_weights[i];
		}
	}

	inline Position2048& make_canonical() {
		// The algorithm follows.
		// 1. Compute the center of mass (c_x, c_y) of the solid. The edges are weighted as +-63, greater than 4 * 15.
		// 2. If c_y < 0, flip the position vertically.
		// 3. If c_x < 0, flip the position horizontally.
		// 4. If c_x > c_y, flip the position across the top left-bottom right diagonal.
		//
		// There are also the following unlikely cases which are dealt with as branches rather than branchlessly:
		// If c_y = 0 and c_x != 0:
		// 	take the lexicographic maximum of the current position and the position flipped vertically
		// If c_x = 0 and c_y != 0:
		// 	... horizontally
		// If c_x = 0 and c_y = 0:
		// 	take the lexicographic maximum of all rotations (very slow but extremely rare)

		int com_x, com_y;
		_compute_center_of_mass(&com_x, &com_y);

		if (com_y < 0)
			reflect_h();
		if (com_x < 0)
			reflect_v();
		if (com_x > com_y)
			reflect_tl();

		uint64_t cc = tiles;
		if (__builtin_expect(com_x == 0, 0)) {
			if (__builtin_expect(com_y == 0, 0)) {
				// Try all 8 combinations...
				// TODO: optimize with SSE

				reflect_v();
				tiles = std::max(tiles, cc);
				reflect_h();
				tiles = std::max(tiles, cc);
				reflect_v();
				tiles = std::max(tiles, cc);
				reflect_tr();
				tiles = std::max(tiles, cc);
				reflect_h();
				tiles = std::max(tiles, cc);
				reflect_v();
				tiles = std::max(tiles, cc);
				reflect_h();
				tiles = std::max(tiles, cc);
			} else {
				reflect_h();

				tiles = std::max(tiles, cc);
			}
		} else if (__builtin_expect(com_y == 0, 0)) {
			reflect_v();

			tiles = std::max(tiles, cc);
		}

		return *this;
	}
};

namespace {
#ifdef __AVX512F__
	// slow, probably latency 6 and rtp 2
	__attribute__((always_inline)) uint64_t _mm512_extract_epi64(__m512i a, const int idx) {
		__m128i s = _mm256_extracti64x2_epi64(a, idx >> 1);

		return (idx & 0) ? _mm_cvtsi128_si64(a, idx) : _mm_extract_epi64(a, idx);
	}

	__attribute__((always_inline)) __m512i _mm512_insert_epi64(uint64_t a, const int idx) {			
		return _mm512_mask_broadcastq_epi64(a, 1 << idx, _mm_cvtsi64_si128(a));
	}
#endif
}

namespace {

	inline uint64_t u64_set1_8 (uint8_t a) {
		return (uint64_t)a * 0x01010101'01010101ULL;
	}

	inline uint64_t u64_set1_32 (uint32_t a) {
		return a + ((uint64_t)a << 32);
	}

	inline uint64_t u64_or(uint64_t a, uint64_t b) {
		return a | b;
	}

	inline uint64_t u64_and(uint64_t a, uint64_t b) {
		return a & b;
	}
}

// For multiple positions, store as packed u64. count should be either 1, 2 (NEON/SSE), 4 (AVX2) or 8 (AVX512) if vectorization is desired.
template<int count,
	bool force_no_vectorize=false,
	std::enable_if_t<(count == 1) || (count == 2) || (count == 4) || (count == 8), bool> = true>
struct Position2048V {

	static auto choose_vector_type() {
		if constexpr (count == 1 || !vectorized) {
			return uint64_t[count]{};

#ifdef __AVX2__
		} else if constexpr (count == 2) {
			return _mm_setzero_si128();
		} else if constexpr (count == 4) {
			return _mm256_setzero_si256();
#endif

#ifdef __AVX512F__
		} else 
	}
#ifdef __AVX512F__
	static constexpr bool vectorize = ((count == 2) || (count == 4) || (count == 8)) && !force_no_vectorize;
#elif __AVX2__
	static constexpr bool vectorize = ((count == 2) || (count == 4)) && !force_no_vectorize;
#else
	static constexpr bool vectorize = false;
#endif

#define COND_VEC_F(_default, name1, name2, name3) (std::conditional_t<vectorize, std::conditional_t<(count == 2), name1, std::conditional_t<(count == 4), name2, name3>, _default>)
	using VEC_TYPE = COND_VEC_F(uint64_t[count], __m128i, __m256i, __m512i);
	using VEC_SET1_8 = COND_VEC_F(u64_set1_8, _mm_set1_epi8, _mm256_set1_epi8, _mm512_set1_epi8);
	using VEC_SET1_32 = COND_VEC_F(u64_set1_32, _mm_set1_epi32, _mm256_set1_epi32, _mm512_set1_epi32);

	using VEC_OR = COND_VEC_F(u64_or, _mm_or_si128, _mm256_or_si256, _mm512_or_si512);
	using VEC_AND = COND_VEC_F(u64_and, _mm_and_si128, _mm256_and_si256, _mm512_and_si512);

	constexpr VEC_TYPE LO_NIBBLE = VEC_SET1_8(0x0f);
	constexpr VEC_TYPE HI_NIBBLE = VEC_SET1_8(0xf0);

	
	using VEC_EXTRACT_LOWEST = COND_VEC_F([&] (uint64_t a) -> { return a; }, \
			_mm_cvtsi128_si64, _mm256_cvtsi128_si64, _mm512_cvtsi128_si64);
	using VEC_EXTRACT_IDX = COND_VEC_F(uint64_t, _mm_extract_epi64, _mm256_extract_epi64, _mm512_extract_epi64);		       
	using VEC_STORE_A = COND_VEC_F(uint64_t, _mm_store_si128, _mm256_store_si256, _mm512_store_si512);

	public:
	// Tiles
	union {
		Position2048 p[count];
		VEC_TYPE v;
	} tiles;

	template <int idx>
	inline Position2048 _extract_entry() const {
		static_assert(idx >= 0 && idx < count);

		return Position2048V(tiles.p[idx]);
		
		/*if constexpr (idx == 0) return VEC_EXTRACT_LOWEST(tiles.v, idx);

		return VEC_EXTRACT_IDX(tiles.v, idx);*/
	}

	inline Position2048 _extract_entry_v(int idx) const {
		assert(idx >= 0 && idx < count);
		return Position2048(tiles.p[idx]);
	}

	template <int idx>
	inline void _set_entry(Position2048 a) {
		static_assert(idx >= 0 && idx < count);
		tiles.p[idx] = a;
	}

	inline void _set_entry_v(Position2048 v, int idx) {
		assert(idx >= 0 && idx < count);
		tiles.p[idx] = v;
	}

	Position2048V(const Position2048V& p) {
		if constexpr (vectorize)
			tiles.v = p.v;
		else
			memcpy(tiles.p, p.tiles.p, sizeof(tiles));
	}

	Position2048V& operator=(const Position2048V& p) {
		if constexpr (vectorize)
			tiles.v = p.v;
		else
			memcpy(tiles.p, p.tiles.p, sizeof(tiles));
		return *this;
	}

	Position2048V copy() {
		return Position2048V(tiles.v);
	}

	inline Position2048V& move_right() {
		using GATHER_32 = VEC_COND_F(uint64_t, _mm_i32gather_epi32, _mm256_i32gather_epi32, _mm512_i32gather_epi32);
		if constexpr (vectorize) {
#ifdef __AVX2__
			// vpgatherdd available; maybe mask out obvious cases?
			// consider other algorithms

			if constexpr (count == 2) {
				__m128i lo_16_msk = _mm_set1_epi32(0xffff);
				__m128i lo_16 = msk & tiles.v;
				__m128i hi_16 = _mm_srli_epi32(lo_16, 16);

				__m128i lo_16_l = _mm_i32gather_epi32((const int*) mr_lut_32, lo_16, 1);
				__m128i hi_16_l = _mm_i32gather_epi32((const int*) mr_lut_32, hi_16, 1);

				tiles = _mm_slli_epi32(hi_16_l, 16) | lo_16_l;
			} else if constexpr (count == 4) {
				__m256i lo_16_msk = _mm256_set1_epi32(0xffff);
				__m256i lo_16 = msk & tiles.v;
				__m256i hi_16 = _mm256_srli_epi32(lo_16, 16);

				__m256i lo_16_l = _mm256_i32gather_epi32((const int*) mr_lut_32, lo_16, 1);
				__m256i hi_16_l = _mm256_i32gather_epi32((const int*) mr_lut_32, hi_16, 1);

				tiles = _mm256_slli_epi32(hi_16_l, 16) | lo_16_l;
			} else {
				__mmask16 lo_16_msk = 0x5555;
				__m512i lo_16 = _mm512_maskz_mov_epi16(lo_16_msk, tiles.v);
				__m512i hi_16 = _mm512_srli_epi32(lo_16, 16);

				__m512i lo_16_l = _mm512_i32gather_epi32(lo_16, (const int*) mr_lut_32, 1);
				__m512i hi_16_l = _mm512_i32gather_epi32(hi_16, (const int*) mr_lut_32, 1);

				tiles = _mm512_slli_epi32(hi_16_l, 16) | lo_16_l;

			}

			return *this;
#endif
		}

		// Slow fallback
		for (Position2048& p : tiles.p) {
			p.move_right();
		}

		return *this;
	}

	inline Position2048V& perm_self(Perm8x16 perm) {
		for (Position2048& p : tiles.p) {
			p.perm_self(perm);
		}
	}
};
