/**
 * Implementation of scalar position manipulation
 */

#include "position.h"
#include "shuffle.h"
#include "move_lut.h"

namespace Analysis {
	Position::Position() {
		tiles = 0;
	}

	Position::Position(const Position& p) {
		tiles = p.tiles;
	}

	Position::Position(uint64_t a) {
		tiles = a;
	}

	Position::Position(std::array<int, 16> l) {
		tiles = 0;

		for (uint32_t i : l) {
			tiles |= (tile_to_repr(i));
			tiles = ((tiles & 0xf) << 60) | (tiles >> 4);

		}
	}

	Position& Position::set_tile(int idx, uint8_t value) {
		uint64_t msk = 0xfULL << (4 * idx);
		tiles = (tiles & ~msk) | ((uint64_t)(value & 0xf) << (4 * idx));

		return *this;
	}

	uint8_t Position::get_tile(int idx) const {
		uint64_t msk = 0xfULL << (4 * idx);

		return (tiles & msk) >> (4 * idx);
	}

	Position& Position::operator=(const Position& p) {
		tiles = p.tiles;
		return *this;
	}

	uint64_t Position::lo_nibbles() const {
		return tiles & LO_NIBBLES;
	}

	uint64_t Position::hi_nibbles() const {
		return tiles & HI_NIBBLES;
	}

	uint32_t Position::tile_sum() const {  // impossible to have a sum > 256
		uint32_t sum = 0;
		for (int i = 0; i < 16; ++i) {
			sum += repr_to_tile(get_tile(i));
		}

		return sum;
	}

#if 0 //USE_VEC && defined(__BMI2__)
	__m128i _to_sse_bytes() const {
		return _mm_set_epi64x(_pdep_u64(tiles >> 32, LO_NIBBLES), _pdep_u64(tiles, LO_NIBBLES));
	}

	Position& _from_sse_bytes(__m128i a) {
		tiles = _pext_u64(_mm_cvtsi128_si64(a), LO_NIBBLES) |
			((_pext_u64(_mm_extract_epi64(a, 1), LO_NIBBLES)) << 32);

		return *this;
	}

	Position& perm_self_lut128(const uint8_t* p) {
#if USE_VEC && defined(__BMI2__)
		__m128i b = _to_sse_bytes();
		__m128i shuf = _mm_shuffle_epi8(b, _mm_load_si128((const __m128i*) p));

		return _from_sse_bytes(shuf);
#endif
		uint64_t v = 0;
		for (int i = 0; i < 16; ++i) {
			v |= get_tile(p[i]);
			v = ((v & 0xf) << 60) | (v >> 4);
		}

		tiles = v;

		return *this;
	}
#endif

	Position Position::perm(uint64_t nibble_shuffle) {
		return Position{ shuffle_nibbles(tiles, nibble_shuffle) };
	}

	// Rotations are counterclockwise by convention
	Position Position::rotate_90() const {
		return Position{ shuffle_nibbles(tiles, constants::rotate_90) };
	}
	Position Position::rotate_180() const {
		return Position{ shuffle_nibbles(tiles, constants::rotate_180) };
	}
	Position Position::rotate_270() const {
		return Position{ shuffle_nibbles(tiles, constants::rotate_270) };
	}
	Position Position::identity() const {
		return Position{ tiles };
	}
	Position Position::reflect_tl() const {
		return Position{ shuffle_nibbles(tiles, constants::reflect_tl) };
	}
	Position Position::reflect_tr() const {
		return Position{ shuffle_nibbles(tiles, constants::reflect_tr) };
	}
	Position Position::reflect_v() const {
		return Position{ shuffle_nibbles(tiles, constants::reflect_v) };
	}
	Position Position::reflect_h() const {
		return Position{ shuffle_nibbles(tiles, constants::reflect_h) };
	}

	Position Position::move_right() const {
		return Position{ ::Analysis::move_right(tiles) };
	}


	char* Position::to_string() const {
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

	bool Position::operator==(const Position& b) const noexcept {
		return tiles == b.tiles;
	}

	bool Position::operator !=(const Position& b) const noexcept {
		return !(*this == b);
	}

	Position Position::canonical() const {
		return canonical_position(tiles);
	}

	uint8_t tile_to_repr(uint32_t tile, bool validate) {
		 // 4 -> 2, 2 -> 1, 0 -> 0
		if (tile == 0) return 0;

		int k = __builtin_ctzl(tile);

		if (validate && (tile != ((uint32_t)1 << k) || !k || k > 17)) {
			fprintf(stderr, "Invalid tile %d; should be a power of 2 between 2 and 131072 inclusive, or 0\n", tile);
			abort();
		}

		return k;
	}

	uint32_t repr_to_tile(uint8_t repr) {
		return (repr == 0) ? 0 : (1U << repr);
	}

	// Randomly insert a 2 or 4 into the current position, and do not move it
	Position Position::get_next_random(bool* successful, Rng* r) const {
		uint8_t tile = (r->next() < (1U << 31) / 5) ? 2 : 1;

		Position q = identity();

#ifndef __BMI2__
		uint8_t idxs[15];
		int count;
		grab_empty_idxs(tiles, idxs, &count);

		if (unlikely(count == 0)) { 
			*successful = false;
			return *this;
		}

		int idx = idxs[r->next() % count];

		q.set_tile(idx, tile);
		*successful = true;
#else
		// BMI2 enjoyer
		uint64_t m = ~zero_mask_zero_nibbles(tiles);
		uint64_t empty_idxs = _pext_u64(0xfedcba9876543210, m);
		int count = _popcnt_u64(m) >> 2;

		if (unlikely(!count)) {
			*successful = false;
		} else {
			int random_zero_idx = empty_idxs >> (4 * (r->next() % count));

			q.set_tile(random_zero_idx, tile);
			*successful = true;
		}
#endif

		return q;
	}

	// Generate a random starting position, optionally with some "seed"; a seed of -1 indicates a random one should be chosen.
	// By iterating all seeds from 0 to 31 inclusive, all potential starting positions are created. Note that a "starting position"
	// actually constitutes a base position, rather than a position including two random tiles. Therefore, a starting 2048 position
	// contains exactly one tile. The position is not guaranteed to be canonical.
	Position Position::starting_position(int seed) {
		int idx, tile;

		if (seed == -1) {
			idx = thread_rng.next() % 16;
			tile = (thread_rng.next() % 10 == 0) ? 4 : 2;
		} else {
			seed &= 31;
			idx = seed & 0xf;
			tile = seed >> 4;
		}

		return Position{}.set_tile(idx, tile);
	}

	void Position::gen_next(Position* pp2, Position* pp4,
				int* pp2p, int* pp4p, int* pp2c, int* pp4c,
				int* pp2allowed, int*pp4allowed, int* pp2disallowed, int* pp4disallowed) {


	}

	void Position::gen_new_tiles(Position* pp2, Position* pp4, int* pp2c, int* pp4c) {
		// This is rather tricky. We first generate all indices where a 2 or 4 can be inserted, then compute all such positions,
		// store them, and remove duplicates.
		uint8_t idxs[16];
		int count;
		grab_empty_idxs(tiles, idxs, &count);

		*pp2c = *pp4c = 0;


		for (int i = 0; i < count; ++i) {
			uint8_t idx = idxs[i];

			// Insert a 2 and a 4
			Position q = identity();
			q.set_tile(idx, 1);

			*pp2++ = q;
			*pp2c += 1;

			q.set_tile(idx, 2);

			*pp4++ = q;
			*pp4c += 1;
		}
	}
#if 0
	static void canonicalize(__restrict__ uint64_t* input, 

	// Canonicalize and deduplicate, recording frequencies of each entry. That is, take every entry, convert it to canonical form, remove
	// duplicates, and record the number of each duplicate. The output guarantees that the first entry in each
	static void canonicalize_dedup(__restrict__ uint64_t* input, int input_len, __restrict__ uint64_t* output, int* output_freq, int* count) {

#endif
}
