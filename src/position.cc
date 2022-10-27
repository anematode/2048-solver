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

	Position Position::get_next_random(Rng* r, bool* successful) const {
		uint8_t tile = (r->next() < (1U << 31) / 5) ? 2 : 1;

		Position q = identity();
		int tries = -20;
		int idx;

		for (idx = r->next() & 0xf; get_tile(idx) && (++tries); idx = r->next() & 0xf);

		printf("PP %i\n", idx);

		uint8_t idxs[15];
		int write_i = 0;

		if (!tries) {
			for (uint8_t idx = 0; idx < 16; ++idx) {
				if (!get_tile(idx)) {
					idxs[write_i++] = idx;
				}
			}

			if (write_i == 0) { 
				*successful = false;
				return *this;
			}

			assert(write_i != 0);
			idx = r->next() % write_i;
		}

		q.set_tile(idx, tile);
		*successful = true;

		return q;
	}

}
