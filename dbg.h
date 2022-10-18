#pragma once

#include <signal.h>

#ifdef __AVX2__
#include <immintrin.h>
#elif __ARM_NEON__
#include <arm_neon.h>
#endif

// Convenience functions, etc

inline void bkp() {
#ifdef __X86_64__
	__asm__ (
		"int3;"
	    );
#elif __aarch64__
#ifndef __APPLE__
	__asm__ ("trap;");
#else
	__asm__ ("brk #0x1");
#endif
#else
	raise(SIGINT);
#endif
}

namespace {
	static char* volatile a;
	alignas(64) static volatile char* dd;
}

template <typename T, std::enable_if_t<!std::is_integral_v<T>, bool> = true >
void prevent_opt(T v) {   // attempt to prevent v from being optimized
	*a = *((char*)&v);
}

template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true >
void prevent_opt(T v) {
	*a = v;
}

#ifdef __SSE__
template <>
void prevent_opt(__m128i a) {
	__asm__ __volatile__ {
		"vmovdqa %[a], %[a];"
		::: "memory"
	};
}

#endif

#ifdef __AVX__
template <>
void prevent_opt(__m256i a) {
	__asm__ __volatile__ {
		"vmovdqa %[a], %[a];"
		::: "memory"
	};
}
#endif

#ifdef __AVX512F__
template <>
void prevent_opt(__m512i a) {
	__asm__ __volatile__ {
		"vmovdqa32 %[a], %[a];"
		::: "memory"
	};
}
#endif

#ifdef __NEON__
template <>
void prevent_opt(uint8x16_t a) {
	__asm__ __volatile__ {
		"vmov %[a], %[a];"
		::: "memory"
	};
}
#endif


inline void vec_to_string() {
	
}

inline void print_vec() {

}
