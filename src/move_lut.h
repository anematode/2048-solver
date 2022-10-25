/**
 * LUTs for moving right. We use two LUTs: One intended for 32-bit vpgatherdd, and one for 16-bit scalar loads.
 */
#pragma once

#include "defs.h"
#include "shuffle.h"

namespace Analysis {
	// By convention, the lowest significant nibble is index 0 and corresponds to the top left corner.
	uint8_t get_tile(uint64_t tiles, int idx);
	uint64_t set_tile(uint64_t tiles, uint8_t tile, int idx);

	namespace detail {
		extern uint16_t* move_right_lut16;
		extern uint32_t* move_right_lut32;

		int generate_move_right_luts();
		namespace {
			static int _ = generate_move_right_luts();
		}
	}

	uint64_t move_right(uint64_t tiles);  // All movements reduce to this
	uint64_t move_up(uint64_t tiles);
	uint64_t move_down(uint64_t tiles);
	uint64_t move_left(uint64_t tiles);


#ifdef USE_X86_VECTORIZE
	__m128i move_right(__m128i tiles);
	__m256i move_right(__m256i tiles);
#ifdef USE_AVX512_VECTORIZE
	__m512i move_right(__m512i tiles);
#endif


	__m128i canonical_position(__m128i tiles);	
	__m256i canonical_position(__m256i tiles);	
#ifdef USE_AVX512_VECTORIZE
	__m512i canonical_position(__m512i tiles);	
#endif

#endif // USE_X86_VECTORIZE


	// See impl for details
	uint64_t canonical_position(uint64_t tiles);
	void compute_center_of_mass(uint64_t tiles, int* com_x, int* com_y);
}
