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

#ifdef __ARM_NEON__
#include <arm_neon.h>
#define VEC_128 int8x16_t
#define set_vec128_u8 v
#define USE_NEON
#define USE_VEC
#elif __AVX2__
#include <immintrin.h>
#define VEC_128 __m128i
#define set_vec128_u8 _mm_setr_epi8
#define USE_SSE
#define USE_VEC
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

static uint32_t move_right_lut[65536];

int fill_move_right_lut();
namespace {
	int c = fill_move_right_lut();
}

static const uint8_t com_x_weights[16] = {
	-201, -1, 1, 201, -201, -1, 1, 201, -201, -1, 1, 201, -201, -1, 1, 201
};

static const uint8_t com_y_weights[16] = {
	-201, -201, -201, -201, -1, -1, -1, -1, 1, 1, 1, 1, 201, 201, 201, 201
};

struct alignas(16) Position2048 {

	// We store a position as 16 bytes. The mapping is 0 -> 0, 1 -> 2, et cetera.
	
	union {
		alignas(16) uint8_t b[16];
#ifdef USE_VEC
		VEC_128 v;
#endif
	} tiles;

	Position2048() {

	}

	Position2048(std::initializer_list<uint32_t> l) {
		int k = 0;
		uint8_t* b = tiles.b;

		for (uint32_t i : l) {
			if (k > 15) break;
			b[k++] = tile_to_repr(i);
		}

		memset(b + k, 0, 16 - k);
	}

	void clear() {
#ifdef USE_SSE
		tiles.v = _mm_setzero_si128();
#elif defined(USE_NEON)
		tiles.v = vmovq_n_s8(0);
#else
		memset(tiles.b, 0, 16);
#endif
	}

	Position2048(const Position2048& p) {
#ifdef USE_VEC
		tiles.v = p.tiles.v;
#else
		for (int i = 0; i < 16; ++i) tiles.b[i] = p.tiles.b[i];
#endif
	}

#ifdef USE_VEC
	Position2048(VEC_128 v) {
		tiles.v = v;
	}
#endif

	~Position2048() {

	}

	Position2048 copy() const {
#ifdef USE_VEC
		return Position2048{tiles.v};
#else
		Position2048 p;
		memcpy(&p.tiles.b, tiles.b, 16);

		return p;
#endif
	}

	uint64_t hash() const {
		uint64_t b = 0;

		for (int i = 0; i < 16; ++i) {
			b *= tiles.b[i] * 16;
			b += 1021;
		}

		return b;
	}

	inline uint8_t tile_sum() {  // impossible to have a sum > 256
#ifdef USE_VEC
		VEC_128 v = tiles.v;

		// horizontal byte sum
#ifdef USE_SSE
		// Credit: https://stackoverflow.com/questions/36998538/fastest-way-to-horizontally-sum-sse-unsigned-byte-vector
		__m128i ss = _mm_sad_epu8(v, _mm_setzero_si128());
		return _mm_cvtsi128_si32(ss) + _mm_extract_epi16(ss, 4);
#else
		return (uint8_t)vaddvq_s8(v);
#endif
#else
		uint8_t s = 0;	
		for (int i = 0; i < 16; ++i) {
			s = tiles.b[i];
		}

		return s;
#endif
	}

#ifdef USE_NEON
	inline void _compress_right_neon() {

	}

	inline void _merge_right_neon() {
		// ...
	}

	inline void _move_right_neon() {
		_compress_right_neon();
		_merge_right_neon();
		_compress_right_neon();
	}
#endif

#ifdef USE_SSE
	inline void _move_right_x86() {
		// Strategy: compress right, merge right, compress rest
#ifdef __AVX2__

#else
		_move_right_scalar();
#endif
	}
#endif

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

	inline Position2048& reflect_h() {
		return perm_self(Perm8x16::reflect_h);
	}

	inline Position2048& reflect_v() {
		return perm_self(Perm8x16::reflect_v);
	}

	inline Position2048& reflect_tr() {
		return perm_self(Perm8x16::reflect_tr);
	}

	inline Position2048& reflect_tl() {
		return perm_self(Perm8x16::reflect_tl);
	}

	// Permute in place according to given permutation, return self
	inline Position2048& perm_self(const uint8_t perm[16]) {
#ifdef USE_VEC

#ifdef USE_SSE
		tiles.v = _mm_shuffle_epi8(tiles.v, _mm_load_si128((const __m128i*) perm));
#else
		tiles.v = vqtbl1q_u8(tiles.v, vld1q_u8(perm));
#endif

#else
		uint8_t new_v[16];
		for (int i = 0; i < 16; ++i) {
			new_v[i] = tiles.b[perm[i]];
		}
		memcpy(tiles.b, new_v, 16);
#endif

		return *this;
	}


	inline void _merge_right_scalar() {  // see move_right
		for (int i = 0; i < 4; ++i) {
			uint8_t* row = tiles.b + 4 * i;

			for (int j = 2; j >= 0; --j) {
				if (row[j] && (row[j] == row[j + 1])) {
					row[j + 1]++;
					row[j] = 0;
				}
			}
		}
	}

	inline void _collapse_col_scalar(int col) {	// see move_right
		for (int i = 0; i < 4; ++i) {
			uint8_t* row = tiles.b + 4 * i;

			uint8_t a = row[col];
			uint8_t b = row[col+1];
			
			if (b == 0) {
				row[col] = 0;
				row[col+1] = a;
			}
		}
	}

	inline void _move_right_scalar() {
		_collapse_col_scalar(2);
		_collapse_col_scalar(1);
		_collapse_col_scalar(0);
		_collapse_col_scalar(2);
		_collapse_col_scalar(1);
		_collapse_col_scalar(2);

		_merge_right_scalar();

		_collapse_col_scalar(1);
		_collapse_col_scalar(0);
	}

	inline Position2048& move_right() {

#ifdef USE_SSE
		_move_right_x86();
#elif 0 //defined(USE_NEON)
		_move_right_neon();
#else
		_move_right_scalar();
#endif
		return *this;
	}


	char* to_string() const {
		char out[400];
		char* end = out;

		for (int i = 0; i < 16; ++i) {
			end += sprintf(end, "%d", repr_to_tile(tiles.b[i]));
			*end++ = (i % 4 == 3) ? '\n' : '\t';
		}
		
		*end++ = '\0';

		int len;

		char* s = (char*)malloc(len = end - out);
		memcpy(s, out, len);

		return s;
	}

	inline bool operator==(const Position2048& b) const noexcept {
#ifdef USE_VEC
#ifdef USE_SSE
		return _mm_movemask_epi8(_mm_cmpeq_epi8(tiles.v, b.tiles.v)) == 0xffff;
#else
		return vminvq_u8(vceqq_u8(tiles.v, b.tiles.v)) == 0xff;
#endif
#else
		return memcmp(tiles.b, b.tiles.b, 16) == 0;
#endif
	}

	inline bool operator !=(const Position2048& b) const noexcept {
		return !(*this == b);
	}

	// Compress to 64-bit integer. 65536 tiles will become 0. Lowest significant bit is top-left corner, highest is bottom-right.
	uint64_t compress_u64() const noexcept {
#if defined(USE_SSE) && defined(__BMI2__)
		uint64_t a = _mm_extract_epi64(tiles.v, 1);
		uint64_t b = _mm_castsi128_si64(tiles.v);

		const uint64_t low_nibbles = 0x0f0f0f0f0f0f0f0f;

		return (_pext_u64(a, low_nibbles) << 32) + _pext_u64(b, low_nibbles);
#else
		uint64_t r = 0;

		for (int i = 0; i < 16; ++i) {
			r |= ((uint64_t)tiles.b[i] & 0xf) << (4 * i);
		}

		return r;
#endif
	}

	Position2048& decompress_u64(uint64_t c) noexcept {
#if defined(USE_SSE) && defined(__BMI2__)
		const uint64_t low_nibbles = 0x0f0f0f0f0f0f0f0f;

		uint64_t a = _pdep_u64(c, low_nibbles);
		uint64_t b = _pdep_u64(c >> 32, low_nibbles);

		tiles.v = _mm_set_epi64x(a, b);
#else
		for (int i = 0; i < 16; ++i) {
			tiles.b[i] = c & 0xf;
			c >>= 4;
		}
#endif

		return *this;
	}

	int scalar_com_x() {
		
	}
};

#ifdef __AVX512F__


// For multiple positions, store as packed u64. count should be either 1, 2 (NEON/SSE), 4 (AVX2) or 8 (AVX512)
template<int count, std::enable_if_t<(count == 1) || (count == 2) || (count == 4) || (count == 8), bool> = true>
struct Position2048V {

#ifdef __AVX512F__
	static constexpr bool vectorize = (count == 2) || (count == 4) | (count == 8);

	using VEC_TYPE = (!vectorize) ? uint64_t : ((count == 2) ? __m128i : ((count == 4) ? __m256i : __m512i));

#elif __AVX2__
	static constexpr bool vectorize = (count == 2) || (count == 4);

	using VEC_TYPE = (!vectorize) ? uint64_t : ((count == 2) ? __m128i : __m256i);
#else //__ARM_NEON__, eventually...
	static constexpr bool vectorize = false;

	using VEC_TYPE = uint64_t;
#endif
	union {
		uint64_t b[count];
		VEC_TYPE v;
	} tiles;

	// count == 2, count == 4, count == 8.
#define COND_VEC_F(name1, name2, name3) ((count == 2) ? name1 : ((count == 4) ? name2 : name3))

	private:
	template <int idx, std::enable_if_t<vectorize, bool> = true>
	inline uint64_t _extract_vec_entry() {
		using extract_lowest = COND_VEC_F(_mm_castsi128_si64, _mm256_castsi128_si64, _mm512_castsi128_si64);
		using extract_idx = COND_VEC_F(_mm_extract_epi64, _mm256_extract_epi64, _mm512_extract_epi64);
		       
			(count == 2) ? _mm_extract_epi64 : ((count == 4) ? _mm256_extract_epi64 : ((count == 8) ? _mm512_castsi128_si64));

		if constexpr (idx == 0) {

		}
	}

	public:	

	Position2048V(const Position2048V& p) {
		if constexpr (vectorize)
			tiles.v = p.v;
		else
			memcpy(tiles.b, p.tiles.b, sizeof(tiles));
	}

	Position2048V& operator=(const Position2048V& P) {
		if constexpr (vectorize)
			tiles.v = p.v;
		else
			memcpy(tiles.b, p.tiles.b, sizeof(tiles));
		return *this;
	}

	template <int idx = 0>
	Position2048V insert_idx_u64(uint64_t a) {
		static_assert(idx >= 0 && idx < count);

		if constexpr (vectorize) {
			return EXTRACT_VEC_ENTRY<idx>(tiles.v);
		} else {
			b[idx] = a;
		}
	}

	template <int idx = 0>
	uint64_t extract_idx_u64() {

	}

	Position2048V copy() {
		return Position2048V(tiles.v);
	}

};
