#pragma once

#include <cstdint>
#include <cstdio>
#include <cassert>

/**
 * Minimum supported vectorization instruction set is AVX2. AVX512VBMI is ideal for fast shuffling.
 * ARM NEON is not of interest at the moment.
 */
#if defined(__AVX2__) && defined(__BMI2__) /* no CPU I'm aware of has AVX2 and not BMI2, but... */
#include <immintrin.h>

#define USE_X86_VECTORIZE

#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512BW__)
#define USE_AVX512_VECTORIZE

#if defined(__AVX512VL__) && defined(__AVX512VBMI__)
#define USE_VBMI_VECTORIZE
#endif

#else
#define USE_AVX2_VECTORIZE
#endif

#if defined(__AVXVNNI__) || defined(__AVX512VNNI__)
#define USE_VNNI_VECTORIZE
#endif

#endif

namespace Analysis {
	inline void print_features() {
		const char* features =
			"2048 analysis compiled features:\n"
#ifdef USE_AVX2_VECTORIZE
			"USE_AVX2_VECTORIZE\n"
#endif
#ifdef USE_AVX512_VECTORIZE
			"USE_AVX512_VECTORIZE\n"
#endif
#ifdef USE_VBMI_VECTORIZE
			"USE_VBMI_VECTORIZE\n"
#endif
#ifdef USE_VNNI_VECTORIZE
			"USE_VNNI_VECTORIZE\n"
#endif
			;

		puts(features);
	}

	constexpr uint64_t LO_NIBBLES = 0x0f0f'0f0f'0f0f'0f0f;
	constexpr uint64_t HI_NIBBLES = 0xf0f0'f0f0'f0f0'f0f0;

	template <typename T>
	T max(T a, T b) {
		return a < b ? b : a;
	}

	__attribute__((always_inline)) inline int likely(int cond) {
		return __builtin_expect(cond, 1);
	}

	__attribute__((always_inline)) inline int unlikely(int cond) {
		return __builtin_expect(cond, 0);
	}
}
