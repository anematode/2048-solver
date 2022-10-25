#include "move_lut.h"
#include "shuffle.h"

namespace Analysis {
	namespace detail {
		uint16_t* move_right_lut16 = nullptr;
		uint32_t* move_right_lut32 = nullptr;

		int generate_move_right_luts() {
			static bool called = false;
			if (called) return 0;
			called = true;

			const int SZ = 1 << 16;

			move_right_lut16 = (uint16_t*)malloc(SZ * sizeof(uint16_t));
			move_right_lut32 = (uint32_t*)malloc(SZ * sizeof(uint32_t));

			for (uint32_t a = 0; a < (1 << 16); ++a) {
				uint16_t v16 = 0;

				uint8_t tt[4] = { (uint8_t)(a & 0xf), (uint8_t)((a & 0xf0) >> 4), (uint8_t)((a & 0xf00) >> 8), (uint8_t)((a & 0xf000) >> 12)  };

				auto collapse_right = [&] () -> void {
					for (int i = 2; i >= 0; --i) {
						if (tt[i+1] == 0) {
							tt[i+1] = tt[i];
							tt[i] = 0;
						}
					}
				};

				collapse_right();
				collapse_right();
				collapse_right();
				for (int i = 2; i >= 0; --i) {
					if (tt[i] == tt[i+1] && tt[i]) {
						tt[i+1] = 1 + tt[i];
						tt[i] = 0;
					}
				}
				collapse_right();
				collapse_right();

				//v32 = tt[0] + (tt[1] << 8) + (tt[2] << 16) + (tt[3] << 24);
				v16 = tt[0] + (tt[1] << 4) + (tt[2] << 8) + (tt[3] << 12);

				move_right_lut32[a] = v16;
				move_right_lut16[a] = v16;
			}

			return 0;
		}
	}

	uint64_t move_right(uint64_t tiles) {	
		using namespace detail;
		uint16_t msk = -1;

		tiles = (uint64_t)move_right_lut16[tiles & msk] |
			((uint64_t)move_right_lut16[(tiles >> 16) & msk] << 16) |
			((uint64_t)move_right_lut16[(tiles >> 32) & msk] << 32) |
			((uint64_t)move_right_lut16[(tiles >> 48) & msk] << 48);

		return tiles;
	}

	uint64_t move_perm(uint64_t tiles, uint64_t perm, uint64_t inv_perm) {
		tiles = shuffle_nibbles(tiles, perm);
		tiles = move_right(tiles);
		tiles = shuffle_nibbles(tiles, inv_perm);

		return tiles;
	}

#define DEFINE_MOVE(name, perm, inv_perm) uint64_t name(uint64_t tiles) { \
	return move_perm(tiles, perm, inv_perm); }

	DEFINE_MOVE(move_left, constants::rotate_180, constants::rotate_180)
	DEFINE_MOVE(move_up, constants::rotate_270, constants::rotate_90)
	DEFINE_MOVE(move_down, constants::rotate_90, constants::rotate_270)

#ifdef USE_X86_VECTORIZE
	__m128i move_right(__m128i tiles) {
		__m128i lo_16_msk = _mm_set1_epi32(0xffff);
		__m128i lo_16 = _mm_and_si128(lo_16_msk, tiles.v);
		__m128i hi_16 = _mm_srli_epi32(tiles.v, 16);

		// tp ~4
		__m128i lo_16_l = _mm_i32gather_epi32((const int*) mr_lut_32, lo_16, 4);
		__m128i hi_16_l = _mm_i32gather_epi32((const int*) mr_lut_32, hi_16, 4);

		return  _mm_slli_epi32(hi_16_l, 16) | lo_16_l;
	}

	__m256i move_right(__m256i tiles) {
		__m256i lo_16_msk = _mm256_set1_epi32(0xffff);
		__m256i lo_16 = _mm256_and_si256(lo_16_msk, tiles.v);
		__m256i hi_16 = _mm256_srli_epi32(tiles.v, 16);

		// 2x vpgatherdd ymm, ymm, ymm -> tp 8 or so on ICL. Maybe consider fancy shuffling techniques, though.
		__m256i lo_16_l = _mm256_i32gather_epi32((const int*) mr_lut_32, lo_16, 4);
		__m256i hi_16_l = _mm256_i32gather_epi32((const int*) mr_lut_32, hi_16, 4);

		return  _mm256_slli_epi32(hi_16_l, 16) | lo_16_l;
	}

#ifdef USE_AVX512_VECTORIZE
	__m512i move_right(__m512i tiles) {
		__m512i lo_16_msk = _mm512_set1_epi32(0xffff);
		__m512i lo_16 = _mm512_and_si512(lo_16_msk, tiles);
		__m512i hi_16 = _mm512_srli_epi32(tiles.v, 16);

		__m512i lo_16_l = _mm512_i32gather_epi32(lo_16, (const int*) mr_lut_32, 4);
		__m512i hi_16_l = _mm512_i32gather_epi32(hi_16, (const int*) mr_lut_32, 4);

		return _mm512_slli_epi32(hi_16_l, 16) | lo_16_l;
	}
#endif

#endif

	uint64_t set_tile(uint64_t tiles, uint8_t tile, int idx) {
		assert(idx >= 0 && idx < 16);

		uint64_t msk = 0xfULL << (4 * idx);
		return (tiles & ~msk) | ((uint64_t)(tiles & 0xf) << (4 * idx));
	}

	uint8_t get_tile(uint64_t tiles, int idx) {
		assert(idx >= 0 && idx < 16);

		idx *= 4;
		return (tiles & (0xf << idx)) >> idx;
	}

	static const int8_t com_x_weights[16] = {
		-63, -1, 1, 63, -63, -1, 1, 63, -63, -1, 1, 63, -63, -1, 1, 63
	};

	static const int8_t com_y_weights[16] = {
		-63, -63, -63, -63, -1, -1, -1, -1, 1, 1, 1, 1, 63, 63, 63, 63
	};

	void compute_center_of_mass(uint64_t tiles, int* com_x, int* com_y) {
		// Dot product

		*com_x = *com_y = 0;

		for (int i = 0; i < 16; ++i) {
			*com_x += (int32_t)get_tile(tiles, i) * com_x_weights[i];
			*com_y += (int32_t)get_tile(tiles, i) * com_y_weights[i];
		}
	}

	uint64_t canonical_position(uint64_t tiles) {

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
		
		using namespace constants;
	
		int com_x, com_y;
		compute_center_of_mass(tiles, &com_x, &com_y);

		if (com_x < 0) {
			tiles = shuffle_nibbles(tiles, reflect_h);
			com_x *= -1;
		}
		if (com_y < 0) {
			tiles = shuffle_nibbles(tiles, reflect_v);
			com_y *= -1;
		}
		if (com_x > com_y) {
			tiles = shuffle_nibbles(tiles, reflect_tl);

			int tmp = com_x;
			com_x = com_y;
			com_y = tmp;
		}

		uint64_t cc = tiles;
		if (__builtin_expect(com_x == 0, 0)) {
			if (__builtin_expect(com_y == 0, 0)) {
				// Try all 8 combinations...
				// TODO: optimize with SSE
				tiles = shuffle_nibbles(tiles, reflect_v);
				tiles = max(tiles, cc);
				tiles = shuffle_nibbles(tiles, reflect_h);
				tiles = max(tiles, cc);
				tiles = shuffle_nibbles(tiles, reflect_v);
				tiles = max(tiles, cc);
				tiles = shuffle_nibbles(tiles, reflect_tr);
				tiles = max(tiles, cc);
				tiles = shuffle_nibbles(tiles, reflect_h);
				tiles = max(tiles, cc);
				tiles = shuffle_nibbles(tiles, reflect_v);
				tiles = max(tiles, cc);
				tiles = shuffle_nibbles(tiles, reflect_h);
				tiles = max(tiles, cc);
			} else {
				tiles = shuffle_nibbles(tiles, reflect_h);	
				tiles = max(tiles, cc);
			}
		} else if (__builtin_expect(com_y == 0, 0)) {
			tiles = shuffle_nibbles(tiles, reflect_v);

			tiles = max(tiles, cc);
		}

		return tiles;
	}
}

