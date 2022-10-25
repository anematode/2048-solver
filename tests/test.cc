#include <catch2/catch_test_macros.hpp>
#include "../src/shuffle.h"
#include "../src/move_lut.h"

using namespace Analysis;

TEST_CASE("Nibble shuffle is correct", "[nibble shuffle]") {

	SECTION("Fixed tests") {
		uint64_t r1 = shuffle_nibbles(0xfedcba9876543210, 0xaa025411fe034102);
		REQUIRE(r1 ==  0xaa025411fe034102);
		uint64_t r2 = shuffle_nibbles(0x0123456789abcdef, 0xaa025411fe034102);
		REQUIRE(r2 == 0x55fdabee01fcbefd);
		uint64_t r3 = shuffle_nibbles(0xaa025411fe034102, 0xeeee11110000ffff);
		REQUIRE(r3 == 0xaaaa00002222aaaa);
	}

	SECTION("Compare vector to scalar") {
		uint64_t a = 0, idx = 0;
		for (int i = 0; i < 10000; ++i) {
			a = 3082 * a + 1010;
			idx = 2308208 * idx + 102;

			uint64_t expected = fallback::shuffle_nibbles(a, idx);
			uint64_t given = shuffle_nibbles(a, idx);

			REQUIRE(expected == given);
		}
	}

}

TEST_CASE("Moves are correct", "[moves]") {
	SECTION("Test move LUT") {
		REQUIRE(fallback::move_right(0) == 0);
		REQUIRE(fallback::move_right(0x0100) == 0x1000);
		REQUIRE(fallback::move_right(0x0022'0100) == 0x3000'1000);
		REQUIRE(fallback::move_right(0x2222'0100) == 0x3300'1000);
		REQUIRE(fallback::move_right(0x4004'0102) == 0x5000'1200);
	}
}

