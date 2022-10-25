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

	namespace fallback {
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

		using namespace constants;

		DEFINE_MOVE(move_left, rotate_180, rotate_180)
		DEFINE_MOVE(move_up, rotate_270, rotate_90)
		DEFINE_MOVE(move_down, rotate_90, rotate_270)
	}
}

