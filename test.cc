// Correctness testing, performance testing
#include "2048.h"
#include <vector>
#include <functional>

enum class TestType {
	CORRECTNESS,
	NEON_CMP,
	X86_CMP,
	SCALAR_PERF,
	NEON_PERF,
	X86_PERF
};

struct TestResult {
	const char* summary;    // memory leak enjoyer
	int correct;
	double ns_per_subtest;

	char* to_string() {
		char* mm = (char*)malloc(strlen(summary) + 1000);
		char* w = mm;

		if (!correct) {
			w += sprintf(mm, "TEST FAILED!\n");
		}
		w += sprintf(mm, "\n\tSummary: %s\n", summary);
		w += sprintf(mm, "\n\tns per subtest: %.3E\n", ns_per_subtest);

		return mm;
	}
};

struct Test {
	TestType type;
	const char* test_name;
	const char* description;

	std::function<TestResult()> callback;
	TestResult* result = nullptr;

	void summary() const {
		printf("Test: %s\n", test_name);
		char* s = nullptr;
		printf("Result: %s\n", result ? (s = result->to_string()) : "not run");
		free(s);

		printf("Description: %s\n\n", description);
	}

	void run() {
		TestResult rr = callback();
		result = (TestResult*)malloc(sizeof (TestResult));
		memcpy(result, &rr, sizeof (TestResult));

		printf("Finished test %s\n", test_name);
	}
};

std::vector<Test> tests;

void add_test(std::function<TestResult()> callback, TestType type, const char* test_name, const char* description) {
	Test tt;

	tt.type = type;
	tt.test_name = test_name;
	tt.description = description;
	tt.callback = callback;
	
	tests.push_back(tt);
}

uint64_t timespec_to_ns(struct timespec* ts) {
	return ts->tv_nsec + ts->tv_sec * 1'000'000'000;
}

void add_test(std::function<uint64_t()> callback, TestType type, const char* test_name, const char* description) {
	auto wrapped = [=] () -> TestResult {
		struct timespec start;
		clock_gettime(CLOCK_REALTIME, &start);

		auto tests = callback();

		struct timespec end;
		clock_gettime(CLOCK_REALTIME, &end);

		uint64_t ns = (timespec_to_ns(&end) - timespec_to_ns(&start));
		printf("%llu %llu\n", ns, tests);

		return TestResult{  .summary="", .correct = true, .ns_per_subtest = (double)ns / (double)tests };
	};

	add_test(wrapped, type, test_name, description);
}


void summarize_all_tests() {
	for (const Test& test : tests) {
		test.summary();
	}
}

void run_all_tests() {
	printf("Running all tests.\n");
	for (Test& test : tests) {
		test.run();
		fflush(stdout);
	}
}

// For correctness testing
template <typename T>
void expect_eq(const T& a, const T& b, const char* msg="unnamed", int line = -1) {
	if (a != b) {
		printf("Expected equality (test %s, line %i)\n", msg, line);
		printf("Found\n%s\n", a.to_string());
		printf("Expected\n%s", b.to_string());

		abort();
	}
}

// Test whether the rotations/reflections work as expected
uint64_t test_perm8x16() {
	using namespace Perm8x16;
	Position2048 p{
		0, 0, 4, 0,
		8, 0, 4, 0,
		0, 16, 0, 0,
		0, 0, 32, 0
	};

	expect_eq(p, p, "identity equals", __LINE__);
	expect_eq(p.copy().rotate_90(), Position2048{
		0, 0, 0, 0,
		4, 4, 0, 32,
		0, 0, 16, 0,
		0, 8, 0, 0	
			}, "rotate 90", __LINE__);
	expect_eq(p.copy().rotate_180(), Position2048{
		0, 32, 0, 0,
		0, 0, 16, 0,
		0, 4, 0, 8,
		0, 4, 0, 0
			}, "rotate 180", __LINE__);
	expect_eq(p.copy().rotate_270(), Position2048{
		0, 0, 8, 0,
		0, 16, 0, 0,
		32, 0, 4, 4,
		0, 0, 0, 0
			}, "rotate 270", __LINE__);
	expect_eq(p.copy().reflect_h(), Position2048{
		0, 4, 0, 0,
		0, 4, 0, 8,
		0, 0, 16, 0,
		0, 32, 0, 0
			}, "reflect horiz", __LINE__);
	expect_eq(p.copy().reflect_v(), Position2048{
		0, 0, 32, 0,
		0, 16, 0, 0,
		8, 0, 4, 0,
		0, 0, 4, 0
			}, "reflect vertical", __LINE__);
	expect_eq(p.copy().reflect_tl(), Position2048{
		0, 8, 0, 0,
		0, 0, 16, 0,
		4, 4, 0, 32,
		0, 0, 0, 0
			}, "reflect top-left", __LINE__);
	expect_eq(p.copy().reflect_tr(), Position2048{
		0, 0, 0, 0,
		32, 0, 4, 4,
		0, 16, 0, 0,
		0, 0, 8, 0
			}, "reflect top_right", __LINE__);

	// Sanity check the permutations some more....
	int v = 0;
	for (const uint8_t* mm : { 
			rotate_90, rotate_180, rotate_270, reflect_h, reflect_v, reflect_tr, reflect_tl
			}) {
		uint8_t cc[16] = { 0 };
		for (int i = 0; i < 16; ++i) {
			cc[mm[i]] = 1;
		}

		for (int i = 0; i < 16; ++i) {
			if (!cc[i]) {
				fprintf(stderr, "Permutation at index %i is missing element %i\n", v, i);
				abort();
			}
		}

		++v;
	}

	return 1;
}

// Test different cases of the move right functionality
uint64_t test_move() {
	expect_eq(Position2048{
			0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 0
			}.move_right(),
			Position2048{
			0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 0
			}, "move", __LINE__);
	expect_eq(Position2048{
			0, 2, 0, 0,
			4, 0, 0, 0,
			0, 0, 8, 0,
			0, 0, 0, 16
			}.move_right(),
			Position2048{
			0, 0, 0, 2,
			0, 0, 0, 4,
			0, 0, 0, 8,
			0, 0, 0, 16
			}, "move", __LINE__);
	expect_eq(Position2048{
			0, 2, 2, 0,
			4, 0, 4, 0,
			0, 0, 8, 8,
			16, 0, 0, 16
			}.move_right(),
			Position2048{
			0, 0, 0, 4,
			0, 0, 0, 8,
			0, 0, 0, 16,
			0, 0, 0, 32
			}, "move", __LINE__);
	expect_eq(Position2048{
			0, 2, 2, 2,
			4, 0, 4, 4,
			8, 8, 8, 0,
			16, 16, 16, 16
			}.move_right(),
			Position2048{
			0, 0, 2, 4,
			0, 0, 4, 8,
			0, 0, 8, 16,
			0, 0, 32, 32
			}, "move", __LINE__);
	expect_eq(Position2048{
			2, 2, 4, 0,
			4, 0, 8, 8,
			8, 0, 4, 0,
			0, 32, 0, 16
			}.move_right(),
			Position2048{
			0, 0, 4, 4,
			0, 0, 4, 16,
			0, 0, 8, 4,
			0, 0, 32, 16
			}, "move", __LINE__);
	expect_eq(Position2048{
			4, 2, 4, 2,
			2, 4, 2, 2,
			8, 4, 0, 0,
			0, 4, 0, 4
			}.move_right(),
			Position2048{
			4, 2, 4, 2,
			0, 2, 4, 4,
			0, 0, 8, 4,
			0, 0, 0, 8
			}, "move", __LINE__);

	return 1;
}

Position2048 from_u32(uint32_t a) {
	Position2048 p;
	for (int i = 0; i < 16; ++i) {
		p.tiles.b[i] = (a & (0x3 << (2 * i))) >> (2 * i);
	}

	return p;
}

// Call a function on every position with numbers between 0 and 8. Intended for rigorous move testing.
template <typename L>
void for_each_position_0_thru_8(L callback, uint64_t step) {
	for (uint64_t a = 0; a < 1ULL << 32; a += step) {
		// spread bit pairs -> bytes
		
		Position2048 p = from_u32(a);
		if constexpr (!std::is_void_v<decltype(callback(p, 0))>)
			callback(p, a);
		else
			callback(p);
	}
}

const int RANDOM_POSITION_CNT = 10000;
Position2048 random_test_positions[RANDOM_POSITION_CNT];

void fill_random_test_positions() {
	uint64_t kk = 0;
	for (int i = 0; i < 10000; ++i) {
		random_test_positions[i] = from_u32(kk >> 13);
		kk *= 4029302011;
		kk += 35021;
	}
}

uint64_t test_scalar_move_perf() {
	uint64_t cases = 0;
	Position2048 q;

	for (int i = 0; i < 1000; ++i) {
		for (const Position2048& p : random_test_positions) {
			q = p.copy().move_right();
			cases++;
		}
		free(q.to_string());
	}


	return cases;
}

uint64_t test_neon_move_perf() {

}

uint64_t test_avx2_move_perf() {

}

uint64_t test_avx512_move_perf() {

}

uint64_t test_avx512_vbmi2_move_perf() {

}

void perf_move_right() {

}

void add_neon_tests() {

}

void add_x86_tests() {

}

int main() {
	fill_random_test_positions();

	add_test(
			test_perm8x16,
			TestType::CORRECTNESS,
			"Test whether the in-place permutations on 16 elements are correct",
			"test_perm8x16"
			);
	add_test(
		test_move,
		TestType::CORRECTNESS,
		"Test whether the moves are performed correctly relative to a few test cases",
		"test_move"
			);

	add_test(
		test_scalar_move_perf,
		TestType::SCALAR_PERF,
		"Test the performance of the scalar move function, for random inputs -- how many ns per move?",
		"test_scalar_move_perf"
		);

	add_neon_tests();
	add_x86_tests();

	run_all_tests();
	summarize_all_tests();
}

#if 0
int main() {
	Position2048 p{
		0, 0, 0, 2,
		2, 8, 4, 4,
		2, 0, 0, 2,
		8, 8, 8, 8
	};

	p.move_right();

	//test_perm8x16();
	//test_move();
	
	for_each_position_0_thru_8([&] (Position2048& p) {
			Position2048 correct = p.copy();
			correct._move_right_scalar();
			Position2048 vec = p.copy();
			vec._move_right_neon();

			if (correct != vec) {
				puts("Original:");	
				puts(p.to_string());
				puts("Correct:");	
				puts(correct.to_string());
				puts("Vector:");	
				puts(vec.to_string());
				abort();
			}
	});
}
#endif
