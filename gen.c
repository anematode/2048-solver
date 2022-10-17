
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#define VEC_128 int8x16_t
#define USE_NEON
#elif __AVX2__
#include <immintrin.h>
#define VEC_128 __m128i
#define USE_SSE
#endif

int tile_to_repr(int tile, bool validate=true) {  // 4 -> 2, 2 -> 1, 0 -> 0
	if (tile == 0) return 0;

	int k = __builtin_ctz(tile);
	if (validate && (tile != (1 << k) || !k || k > 17)) {
		fprintf(stderr, "Invalid tile %d; should be a power of 2 between 2 and 131072 inclusive, or 0\n", k);
		abort();
	}

	return k;
}

int repr_to_tile(uint8_t repr) {
	return (repr == 0) ? 0 : (1 << repr);
}

struct Position2048 {
	// 0 -> 0, 1 -> 2, etc.
	
	union {
		_Alignas(16) uint8_t b[16];
		VEC_TYPE v;
	} tiles;

	Position2048() {

	}

	Position2048(std::initializer_list<uint8_t> l) {
		int k = 0;
		for (uint8_t i : l) {
			if (k > 15) break;
			b[k] = tile_to_repr(i);
		}

		for (; k < 16; ++k) b[k] = 0;
	}

	void clear() {
		memset(tiles, 0, 16);
	}

	Position2048(const Position2048& p) {
		memcpy(tiles, p.tiles.b, sizeof p.tiles);
	}

	~Position2048() {

	}

	uint64_t hash() {
		uint64_t b = 0;

		for (int i = 0; i < 16; ++i) {
			b *= tiles.b[i] * 16;
			b += 1021;
		}

		return b;
	}

	inline uint8_t sum() {  // impossible to have a sum > 256
#if USE_VEC
		VEC_TYPE v = tiles.v;

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

	void move_right() {
		// Let r[0], r[1], r[2], r[3] constitute a row. Let merge(c1, c2) = [ c1, c2 ] if c1 != c2 or c1 == 0 and [ 0, c1 + 1 ] otherwise.
		// Let push_right(c1, c2) = [ c1, c2 ] if c2 != 0, and otherwise [ 0, c1 ]. Then to move right, we do:
		// push_right(r[2:3]), push_right(r[1:2]), push_right(r[0:1])  "collapse right 4 bytes"
		// merge(r[2:3]), merge(r[0:1])    "merge right 4 bytes"
		// push_right(r[1:2]), push_right(r[0:1])  "collapse right 3 bytes"

#if USE_VEC

#else
		collapse_right(3);
		merge_right();
		collapse_right(2);
#endif
	}

	inline void merge_right() {  // see move_right
		for (int i = 0; i < 4; ++i) {
			uint8_t* row = tiles.b + 4 * i;

			if (row[i] && (row[i] == row[j])) {
				row[j]++;
				row[i] = 0;
			}
		}
	}

	inline void collapse_right(int last_col = 3) {	// see move_right
		for (int i = 0; i < 4; ++i) {
			uint8_t* row = tiles.b + 4 * i;

			for (int j = last_col; j >= 1; --j) {
				uint8_t a = row[j];
				uint8_t b = row[j-1];
				
				if (b == 0) {
					row[j-1] = 0;
					row[j] = a;
				}
			}	
		}
	}

	char* to_string() {
		char out[400];
		char* end = out;

		for (int i = 0; i < 16; ++i) {
			end += sprintf(end, "%d", tiles.b[i]);
			*end++ = (i % 4 == 0) ? "\n" : "\t"
		}

		int len;

		char* s = malloc(len = end - out + 1);
		memcpy(s, out, len);

		return s;
	}
};


int main() {
	Position2048 p{0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 2, 0};
}




