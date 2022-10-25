/**
 * LUTs for moving right. We use two LUTs: One intended for 32-bit vpgatherdd, and one for 16-bit scalar loads.
 */
#pragma once

#include "defs.h"

namespace Analysis {
	namespace detail {
		extern uint16_t* move_right_lut16;
		extern uint32_t* move_right_lut32;

		int generate_move_right_luts();
		namespace {
			static int _ = generate_move_right_luts();
		}
	}

	namespace fallback {
		uint64_t move_right(uint64_t tiles);  // All movements reduce to this
		uint64_t move_up(uint64_t tiles);
		uint64_t move_down(uint64_t tiles);
		uint64_t move_left(uint64_t tiles);
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
}
