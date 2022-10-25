/**
 * Implementation of a scalar 2048 position, and a position vector (i.e., computing 2, 4, or 8 positions
 * in parallel). We only really consider positions with a tile value of 32768 or less, so that each one
 * can fit in one nibble, and each position in one 64-bit chunk. Scalar fallbacks for PositionV are found
 * in this file. The scalar implementation for Position is found in position.cc. The vector implementation
 * for PositionV is found in position_v.cc.
 *
 * For now, we don't try to handle >32768 tiles. Conceivably we could support some "extended format", or even
 * store things as eight bytes and compress to, say, 80 bits, but that's nontrivial. It also means that
 * positions would have to be processed in 128-bit chunks, halving throughput in some cases. Perhaps later.
 */

#pragma once

#include "defs.h"
#include <array>

namespace Analysis {
	uint32_t repr_to_tile(uint8_t repr);
	uint8_t tile_to_repr(uint32_t tile, bool validate=false);

	/**
	 * Position class containing a single 64-bit entry. Not terribly optimized compared to the vector
	 * implementation; the latter should be preferred if usable in context. This is more of a reference
	 * implementation.
	 */
	struct Position {
		uint64_t tiles;

		Position& set_tile(int idx, uint8_t value);	
		uint8_t get_tile(int idx) const;

		// Takes in the powers of two, NOT the underlying representation.	
		Position(std::array<int, 16>);

		Position();
		Position(const Position&);
		// For efficient input, a hexadecimal constant is preferred.
		Position(uint64_t);

		Position& operator=(const Position&);

		uint32_t tile_sum() const;   // note: we compute the tile sum in tiles, not their representation
		Position perm(uint64_t nibble_shuffle);

		Position identity() const;
		Position rotate_90() const;
		Position rotate_180() const;
		Position rotate_270() const;
		// Reflect horizontal <-> y-axis, vertical <-> x-axis
		Position reflect_h() const;
		Position reflect_v() const;
		// Transpose
		Position reflect_tl() const;
		Position reflect_tr() const;

		Position move_right() const;

#ifdef USE_X86_VECTORIZE
		__m128i to_sse_bytes();	
		static from_sse_bytes(__m128i bytes);
#endif

		char* to_string() const;
		Position canonical() const;

		bool operator==(const Position& b) const noexcept;
		bool operator!=(const Position& b) const noexcept;

		uint64_t lo_nibbles() const;
		uint64_t hi_nibbles() const;
	};

	/**
	 * Vector of positions with a given count. 2, 4, or 8 positions may be included if vectorization
	 * is desired (and the target processor has the requisite instructions).
	 *
	 * We implement scalar fallback implementations in this file and specialized vector implementations
	 * in the .cc file.
	 */
	template <int count>
	class PositionV {

	};
}
