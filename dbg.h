#pragma once

#include <type_traits>
#include <inttypes.h>
#include <stdint.h>
#include <signal.h>

#ifdef __AVX2__
#include <immintrin.h>
#elif __ARM_NEON__
#include <arm_neon.h>
#endif

// Convenience functions, etc

inline void bkp() {
#ifdef __X86_64__
	__asm__  (
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
template <typename VT>
constexpr bool is_int_vec = std::is_same_v<VT, __m128i> || std::is_same_v<VT, __m256i> || std::is_same_v<VT, __m512i>;
template <typename VT>
constexpr bool is_double_vec = std::is_same_v<VT, __m128d> || std::is_same_v<VT, __m256d> || std::is_same_v<VT, __m512d>;

template <typename T>
constexpr const char* fmt_string_d =
    std::is_same_v<T, uint64_t> ? "%" PRId64 : (
    std::is_same_v<T, int64_t> ? "%" PRIu64 : (
    std::is_same_v<T, uint32_t> ? "%" PRId32 : (
    std::is_same_v<T, int32_t> ? "%" PRIu32 : (
    std::is_same_v<T, uint16_t> ? "%" PRId16 : (
    std::is_same_v<T, int16_t> ? "%" PRIu16 : (
    std::is_same_v<T, uint8_t> ? "%" PRId8 : (
    std::is_same_v<T, int8_t> ? "%" PRIu8 : (
    std::is_same_v<T, double> ? "%.17g" : (
    std::is_same_v<T, float> ? "%.9g" : "$"
    )))))))));

template <typename T>
constexpr const char* fmt_string_x =
    std::is_same_v<T, uint64_t> ? "%" PRIx64 : (
    std::is_same_v<T, int64_t> ? "%" PRIx64 : (
    std::is_same_v<T, uint32_t> ? "%" PRIx32 : (
    std::is_same_v<T, int32_t> ? "%" PRIx32 : (
    std::is_same_v<T, uint16_t> ? "%" PRIx16 : (
    std::is_same_v<T, int16_t> ? "%" PRIx16 : (
    std::is_same_v<T, uint8_t> ? "%" PRIx8 : (
    std::is_same_v<T, int8_t> ? "%" PRIx8 : (
    std::is_same_v<T, double> ? "%.17g" : (
    std::is_same_v<T, float> ? "%.9g" : "$"
    )))))))));

namespace {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
	    template <typename ElemType, typename VT, bool hex=false>
		    void print_vec_base(const VT v) {
			    constexpr int cnt = sizeof(VT) / sizeof(ElemType);
			    constexpr const char* fmt = hex ? fmt_string_x<ElemType> : fmt_string_d<ElemType>;

			    ElemType st[cnt];

			    if constexpr (std::is_same_v<VT, __m128i>)
				    _mm_storeu_si128((__m128i*) st, v);
			    else if constexpr (std::is_same_v<VT, __m256i>)
				    _mm256_storeu_si256((__m256i*) st, v);
			    else
				    _mm512_storeu_si512((__m512i*) st, v);

			    for (int i = 0; i < cnt; ++i) {
				    printf(fmt, st[i]);
				    putchar(' ');
			    }

			    puts("");
		    }
#pragma GCC diagnostic pop
    }

template <typename ElemType, typename VT>
std::enable_if_t<is_int_vec<VT> && std::is_fundamental_v<ElemType>, void>
print_vec(const VT v) {
	print_vec_base<ElemType, VT, false>(v);
}

template <typename ElemType, typename VT>
std::enable_if_t<is_int_vec<VT> && std::is_fundamental_v<ElemType>, void>
print_vec_hex(const VT v) {
	print_vec_base<ElemType, VT, true>(v);
}

template <typename ElemType, typename VT>
std::enable_if_t<is_double_vec<VT> && std::is_fundamental_v<ElemType>, void>
print_vec(const VT v) {
	if constexpr (std::is_same_v<VT, __m128d>) {
		print_vec<ElemType>(_mm_castpd_si128(v));
	} else if constexpr (std::is_same_v<VT, __m256d>) {
		print_vec<ElemType>(_mm256_castpd_si256(v));
	} else {
		print_vec<ElemType>(_mm512_castpd_si512(v));
	}
}


template <typename T, std::enable_if_t<is_int_vec<T>, bool> = true>
inline void _prevent_opt(T a) {
	if constexpr (std::is_same_v<T, __m256i>) {
		volatile uint64_t m = _mm_cvtsi128_si64(_mm256_castsi256_si128(a));
	} else if constexpr (std::is_same_v<T, __m128i>) {
		volatile uint64_t m = _mm_cvtsi128_si64(a);
	} else {
		volatile uint64_t m = _mm_cvtsi128_si64(_mm512_castsi512_si128(a));
	}

	asm volatile("" :: "X"(a) : "memory");

}

template <>
inline void prevent_opt(__m128i a) {
	_prevent_opt(a);
}

template <>
inline void prevent_opt(__m256i a) {
	_prevent_opt(a);
}

template <>
inline void prevent_opt(__m512i a) {
	_prevent_opt(a);
}
#endif
