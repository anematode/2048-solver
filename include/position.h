/**
 * Implementation of a scalar 2048 position, and a position vector (i.e., computing 2, 4, or 8 positions
 * in parallel). We only really consider positions with a tile value of 32768 or less, so that each one
 * can fit in one nibble, and each position in one 64-bit chunk. Scalar fallbacks for PositionV are found
 * in this file. The scalar implementation for Position is found in position.cc. The vector implementation
 * for PositionV is found in position_v.cc.
 */
#pragma once



namespace Analysis {
	/**
	 * Position class containing a single 64-bit entry.
	 */
	class Position {

	};

	/**
	 * Vector of positions with a given count. 2, 4, or 8 positions may be included if vectorization
	 * is desired (and the target processor has the requisite instructions).
	 */
	template <int count>
	class PositionV {

	};
}
