
#include "2048.h"

uint16_t mr_lut_16[65536];
uint32_t mr_lut_32[65536];

int detail::fill_mr_luts() {
	uint32_t* u32 = mr_lut_32;
	uint16_t* u16 = mr_lut_16;

	for (uint32_t a = 0; a < (1 << 16); ++a) {
		uint32_t v32 = 0;
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

		u32[a] = v16;
		u16[a] = v16;
	}

	return 0;
}
