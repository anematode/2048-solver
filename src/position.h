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
#include "shuffle.h"
#include "move_lut.h"
#include "rng.h"
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

		Position move_right(bool* successful) const;
		Position move_up(bool* successful) const;
		Position move_left(bool* successful) const;
		Position move_down(bool* successful) const;

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

		// Insert a random 2 or 4
		Position get_next_random(bool* successful, Rng* rng=&thread_rng) const;
		void gen_next(Position* pp2, Position* pp4,
				int* pp2p, int* pp4p, int* pp2c, int* pp4c,
				int* pp2allowed, int*pp4allowed, int* pp2disallowed, int* pp4disallowed);
		void gen_new_tiles(Position* pp2, Position* pp4, int* pp2c, int* pp4c);
		
		static Position start(int seed=-1);
		static std::array<Position, 32> get_all_starting();	
	};

	template <int count>
	constexpr bool can_vectorize = 
#ifdef USE_X86_VECTORIZE
		count == 2 || count == 4
#ifdef USE_AVX512_VECTORIZE
		|| count == 8
#endif

#else
		false
#endif		
		;

	/**
	 * Vector of positions with a given count. 2, 4, or 8 positions may be included if vectorization
	 * is desired (and the target processor has the requisite instructions).
	 * 
	 * We implement both vector and scalar methods in this file.
	 */
	template <int _count, bool _vectorize=can_vectorize<_count> >
	class PositionV {
		constexpr static int count = _count;
		constexpr static bool vectorize = _vectorize;

		static_assert(_count > 0);
		static_assert(!(vectorize && !can_vectorize<count>));
		
		private:

		using _VEC_TYPE = std::array<uint64_t, count>;

		public:

#ifdef USE_X86_VECTORIZE
		using VEC_TYPE = vectorize ? ((count == 2) ? __m128i : ((count == 4) ? __m256i : __m512i)) : _VEC_TYPE;
#else
		using VEC_TYPE = _VEC_TYPE;
#endif

		VEC_TYPE tiles;

		PositionV();
		PositionV(const PositionV& p) {
			tiles = p.tiles;
		}

		PositionV& operator=(const PositionV& p) {
			tiles = p.tiles;
			return *this;
		}

		// Defined below
#if 0
		char* to_string() const;
#endif

		// These functions have corresponding vectorized versions
		PositionV perm(uint64_t nibble_shuffle);

		VEC_TYPE tile_sum() const;
		PositionV move_right() const;
		PositionV canonical() const;
		Position get_idx(int idx) const;
		void set_idx(int idx, Position p);

		/**
		 * Implementations used for both scalar and vector
		 */
		PositionV identity() const {
			return PositionV { tiles };
		}

		/**
		 * Scalar-only implementations
		 */

		PositionV(VEC_TYPE tiles) {
			this->tiles = tiles;
		}

		const _VEC_TYPE as_array() const requires (!vectorize) {
			return tiles;	
		}

		PositionV() requires (!vectorize) {
			memset(&tiles[0], 0, _count * sizeof(uint64_t));
		}


		PositionV perm(uint64_t nibble_shuffle) const requires (!vectorize) {
			PositionV v;

			fallback::shuffle_nibbles_arr_same(&v.tiles[0], &tiles[0], count, nibble_shuffle);
			return v;
		}

		VEC_TYPE tile_sum() const requires (!vectorize) {
			VEC_TYPE r;

			for (int i = 0; i < count; ++i)
				r[i] = tiles[i].tile_sum();	
		}

#define SCALAR_IMPL_PERM(name, constant)  \
		PositionV name() const requires (!vectorize) { \
			return perm(constant); \
		}

		SCALAR_IMPL_PERM(rotate_90, constants::rotate_90)
		SCALAR_IMPL_PERM(rotate_180, constants::rotate_180)
		SCALAR_IMPL_PERM(rotate_270, constants::rotate_270)
		SCALAR_IMPL_PERM(reflect_h, constants::reflect_h)
		SCALAR_IMPL_PERM(reflect_v, constants::reflect_v)
		SCALAR_IMPL_PERM(reflect_tr, constants::reflect_tr)
		SCALAR_IMPL_PERM(reflect_tl, constants::reflect_tl)

#undef SCALAR_IMPL_PERM

		PositionV move_right() const requires (!vectorize) {
			VEC_TYPE a;

			for (int i = 0; i < count; ++i)
				a[i] = Analysis::move_right(tiles[i]);

			return a;
		}

		Position get_idx(int idx) const requires (!vectorize) {
			assert(0 <= idx && idx < count);
			return tiles[idx];
		}

		void set_idx(int idx, Position p) requires (!vectorize) {
			assert(0 <= idx && idx < count);

			tiles[idx] = p.tiles;
		}

		char* to_string() const {
			char* ss = (char*)malloc(count * 200);
			char* w = ss;

			for (const Position& p : as_array()) {
				char* ps = p.to_string();

				w = stpcpy(stpcpy(w, ps), "\n");

				free(ps);
			}

			return ss;
		}

		PositionV get_next_random(Rng* rng=nullptr) requires (!vectorize) {
			return *this;
		}

		static PositionV start_all() requires (!vectorize) {
			PositionV p;
			for (int i = 0; i < count; ++i) {
				p.tiles[i] = Position::start().tiles;
			}
			return p;
		}

		// Compare positions into mask
		static uint64_t cmp_mask(const PositionV& p1, const PositionV& p2) requires (count <= 64) {
			return 0;
		}

#ifdef USE_X86_VECTORIZE
		static uint64_t cmp_mask(const PositionV& p1, const PositionV& p2) requires (vectorize) {
			return cmp64_to_mask(p1.tiles, p2.tiles);
		}
#endif

#if 0
#ifdef USE_X86_VECTORIZE
		static void _get_next_positions_all_same(__m256i, Position* result2, Position* result4, int* count2, int* count4);
#endif

		// p required to be an array of all 0s. Order of output is guaranteed ... explain
		static void _get_next_positions_all_same(std::array<Position, 4> pv, Position* result2, Position* result4, int* count2, int* count4) requires (!vectorize) {
			Position p = pv[0];

			Position pr = p.move_right();
			for (int tile : { 2, 4 }) {
				auto pv_p = (tile == 2) ? &result2 : &result4;

				auto pv_c = (tile == 2) ? count2 : count4;

				int insert_idx = 0;
				for (int i = 0; i < 4; ++i) {
					// Insert a 2 or 4 in each row at the specified column, if possible, and move
					Position q = p;
					bool allowed[4];

					for (int j = 0; j < 4; ++j) {
						if (q.get_tile(i * 4 + j)) {
							q.set_tile(i * 4 + j, tile);
							allowed[i] = true;
						} else {
							allowed[i] = false;
						}
					}

					q.move_right();

					for (int k = 0; k < 4; ++k) {
						if (allowed[k]) {
							uint64_t row_msk = 0xffff << (16 * k);

							Position next = Position{ (q.tiles & row_msk) | (pr.tiles & ~row_msk) };
							(*pv_p)[insert_idx++] = next;
						}
					}
				}

				*pv_c = insert_idx;
			}
		}

		// result should be at least 30 in length and ideally aligned to a 64-byte boundary if using 512-bit vectors
		static void get_next_positions(Position p, Position* result2, Position* result4, int* count2, int* count4) requires (!vectorize) {
			if constexpr (vectorize) {
				// saves a couple annoying shuffling operations if using vectors
				// _get_next_positions_all_same(_mm256_set1_epi64(p.tiles), result2, result4, count2, count4);
			} else {
				std::array<Position, 4> pv;
				for (int i = 0; i < 4; ++i) pv[i] = p;
				_get_next_positions_all_same(pv, result2, result4, count2, count4);
			}
		}
#endif
	};

}

		namespace std {
		    template <>
		    struct hash<Analysis::Position> {
			size_t operator ()(Analysis::Position p) const {
			    return p.tiles;
			}
		    };
		}
