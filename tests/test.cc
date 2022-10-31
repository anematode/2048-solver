
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include "../src/shuffle.h"
#include "../src/move_lut.h"
#include "../src/position.h"
#include "helper.h"

#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING
#define ANALYSIS_BENCH(mm) [&] () -> auto 
#else
#define ANALYSIS_BENCH(mm) BENCHMARK(mm)
#endif

using namespace Analysis;
using namespace Analysis::Test;

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

TEST_CASE("Moves", "[moves]") {
	SECTION("Test move LUT") {
		REQUIRE(move_right(0) == 0);
		REQUIRE(move_right(0x0100) == 0x1000);
		REQUIRE(move_right(0x0022'0100) == 0x3000'1000);
		REQUIRE(move_right(0x2222'0100) == 0x3300'1000);
		REQUIRE(move_right(0x4004'0102) == 0x5000'1200);
	}

	ANALYSIS_BENCH("Random position move right (10000 cases)") {
		uint64_t sum = 0;

		for (const Position& p : random_positions) {
			bool s;
			sum = p.move_right(&s).tiles;	
		}

		return sum;
	};

}

TEST_CASE("Gen next", "[gen next]") {
	// Given a base position, generate all possible next base positions when moving right, optionally with a probability attached to them.
	// That is, compute all possible next base positions after a move right has been committed. There are also four extra values: The
	// EV of cases where a 2-move is allowed, where a 2-move is disallowed, and analogously for 4-moves.
	std::array<Position, 32> pp2;    // unique, potential next positions containing a 2
	std::array<Position, 32> pp4;
	std::array<uint8_t, 32> pp2p;	 // integer probability (# cases * 9 for 2s, # cases * 1 for 4s)
	std::array<uint8_t, 32> pp4p;

	int pp2c, pp4c;  // count of each unique positions written to the arrays pp2 and pp4
	int pp2allowed, pp4allowed;
	int pp2disallowed, pp4disallowed;

	SECTION("Gen next") {
		Position p { 0x0 };

		//p.gen_next(pp2, pp4, pp2p, pp4p, &pp2c, &pp4c, &pp2allowed, &pp4allowed, &pp2disallowed, &pp4disallowed);

	}
}

TEST_CASE("Deduplicate entries", "[dedup entries]") {
	SECTION("Scalar") {
		uint64_t test[] = { 0x0, 0x0, 0x1, 0x5, 0x2, 0x71172, 0x71172, 0x71172, 0x3, 0x51 };
		uint64_t correct_result[] = { 0x0, 0x1, 0x5, 0x2, 0x71172, 0x3, 0x51 };
		int correct_freqs[] = { 2, 1, 1, 1, 3, 1, 1 };

		uint64_t result[sizeof(correct_result) / sizeof(uint64_t)];
		int freqs[sizeof(correct_freqs) / sizeof(int)];

		int result_count;

		dedup_positions_consecutive(test, 10, result, freqs, &result_count);

		REQUIRE(result_count == 7);
		REQUIRE(memcmp(correct_result, result, sizeof(correct_result)) == 0);
		REQUIRE(memcmp(correct_freqs, freqs, sizeof(correct_result)) == 0);
	}
}

TEST_CASE("Gen new tiles", "[gen new]") {
	SECTION("Scalar") {
		Position p{0x5020000012520};

		Position new2[16];
		Position new4[16];

		int ii[10] = {
			0, 5, 6, 7, 8, 9, 11, 13, 14, 15
		};		

		int new2c, new4c;
		p.gen_new_tiles(new2, new4, &new2c, &new4c);

		REQUIRE(new2c == 10);
		REQUIRE(new4c == 10);

		// Tiles should be eagerly placed into the lowest index
		for (int i = 0; i < new2c; ++i) {
			REQUIRE(p.identity().set_tile(ii[i], 1) == new2[i]);
			REQUIRE(p.identity().set_tile(ii[i], 2) == new4[i]);
		}
	}

	/*SECTION("Scalar move dedup") {
		// Get all next 
		Position p{
	}*/
}

TEST_CASE("Random", "[random]") {
	SECTION("Get next random") {
		Position p{0x040241111};
		 /*
		Position new2[16], new4[16];
		int new2c, new4c;
		p.gen_new_tiles(new2, new4, &new2c, &new4c);*/

		bool successful;
		Position next = p.get_next_random(&successful);

		REQUIRE(is_valid_gen_tile(next.tiles, p.tiles));
		REQUIRE(successful);
	}

	SECTION("Get next random with full board") {
		Position p{0x1234123412341234};

		bool successful;
		Position next = p.get_next_random(&successful);

		REQUIRE(!is_valid_gen_tile(next.tiles, p.tiles));
		REQUIRE(!successful);
	}
}

TEST_CASE("Canonical hashing", "[canonical]") {
	SECTION("Is consistent") {
		REQUIRE((Position { 0 }).canonical() == 0);
		REQUIRE((Position { 0x1 }).canonical().tiles == 1ULL << 60);

		REQUIRE((Position { 0x12 }).canonical().tiles == 0x21ULL << 56);

		for (const Position& p : random_positions) {
			Position q = p.canonical();

			REQUIRE(p.rotate_90().canonical() == q);
			REQUIRE(p.rotate_180().canonical() == q);
			REQUIRE(p.rotate_270().canonical() == q);
			REQUIRE(p.reflect_h().canonical() == q);
			REQUIRE(p.reflect_v().canonical() == q);
			REQUIRE(p.reflect_tr().canonical() == q);
			REQUIRE(p.reflect_tl().canonical() == q);
		}
	}
}

TEST_CASE("Tile sum", "[tile sum]") {
	SECTION("Fixed cases") {
		REQUIRE(Position{ { 0, 2, 8, 4,
				4, 4, 4, 4,
				0, 0, 0, 2,
				0, 0, 0, 16 } }.tile_sum() == 48);

	}

	ANALYSIS_BENCH("Random position tile sum (10000 cases)") {
		uint64_t sum = 0;

		for (const Position& p : random_positions) {
			sum += p.tile_sum();
		}

		return sum;
	};
}

TEST_CASE("Nibble ops", "[nibble ops]") {
	uint64_t tc[2][2] = {
		{0x0000f00020003004, 0xffff0fff0fff0ff0},
		{0x0000f00ff030241f, 0xffff0ff00f0f0000}
	};

	uint64_t tc2[4][2] = {
		{ 0x0123456789abcdef, 15 },
		{ 0x0102030405060707, 8 },
		{ 0x0, 0},
		{ 0x1500000000402, 4 }
	};

	uint64_t tc3[4][2] = {	
		{ 0x0123456789abcdef, 0x0003000400040004 },
		{ 0x0102030405060707, 0x0002000200020002 },
		{ 0x0, 0 },
		{ 0x1500000000402, 0x0001000100000002 }
	};

	uint64_t tc4[4][2] = {
		{ 0x0, 0 },
		{ 0x10000000000, 1 },
		{ 0xf000000f000, 0xf },
		{ 0x012340987baa0, 0xa }
	};

	uint64_t tc5[4][2] = {
		{ 0x0, 0 },
		{ 0x123456789, 0x3fe },
		{ 0x22241100042ff, 0x10034 },
		{ 0xffffffffffffffff, (1 << 15) * 16 }
	};

	SECTION("scalar cmp") {
		for (uint64_t *a : tc) {
			REQUIRE(mask_zero_nibbles(a[0]) == a[1]);
		}
	}

	SECTION("scalar count") {
		for (uint64_t *a : tc2) {
			REQUIRE(count_tiles(a[0]) == a[1]);
		}
	}

	SECTION("scalar row count") {
		for (uint64_t *a : tc3) {
			REQUIRE(count_rows(a[0]) == a[1]);
		}
	}

	/*SECTION("nibble max") {
		for (uint64_t *a : tc4) {
			REQUIRE(nibble_max(a[0]) == a[1]);
		}
	}*/

	SECTION("tile sum") {
		for (uint64_t *a : tc5) {
			REQUIRE(tile_sum(a[0]) == a[1]);
		}
	}

#ifdef USE_X86_VECTORIZE
	SECTION("x86 cmp") {
		for (uint64_t *a : tc) {
			REQUIRE(_mm_cvtsi128_si64x(mask_zero_nibbles(_mm_cvtsi64_si128(a[0]))) == a[1]);
			REQUIRE(_mm256_cvtsi256_si64x(mask_zero_nibbles(_mm256_cvtsi64_si256(a[0]))) == a[1]);
		}
	}
#endif
}


#if 0
uint64_t test_canonical_2() {
	uint64_t cases = 0;
	for_each_position_0_thru_8([&] (const Position2048& p, uint64_t) {
		Position2048 q = p.copy().make_canonical();
		cases++;

		expect_eq(p.copy().rotate_90().make_canonical(), q, "canonical", __LINE__);
		expect_eq(p.copy().rotate_180().make_canonical(), q, "canonical", __LINE__);
		expect_eq(p.copy().rotate_270().make_canonical(), q, "canonical", __LINE__);
		expect_eq(p.copy().reflect_h().make_canonical(), q, "canonical", __LINE__);
		expect_eq(p.copy().reflect_v().make_canonical(), q, "canonical", __LINE__);
		expect_eq(p.copy().reflect_tr().make_canonical(), q, "canonical", __LINE__);
		expect_eq(p.copy().reflect_tl().make_canonical(), q, "canonical", __LINE__);
	}, 103);

	return cases;
}
	}
}

#endif
