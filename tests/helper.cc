#include "helper.h"

namespace Analysis {
	namespace Test {
		Position random_positions[RANDOM_POSITIONS_CNT];

		Position from_u32(uint32_t a) {
			uint64_t v = 0;

			for (int i = 0; i < 16; ++i) {
				v |= (a & (0x3 << (2 * i))) << (2 * i);
			}

			return Position(v);

		}

		void fill_random_test_positions() {
			uint64_t kk = 0;
			for (int i = 0; i < 10000; ++i) {
				random_positions[i] = from_u32(kk >> 13);

				kk *= 4029302011;
				kk += 35021;
			}
		}
	}
}
