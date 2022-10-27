#pragma once

#include <cstdint>

namespace Analysis {
	class Rng {
		uint64_t _state = 0;

		public:
		Rng(uint64_t seed=0) : _state(seed) {}

		inline uint32_t next() noexcept {
			_state = _state * 120381822 + 4018501;

			return (uint32_t)(_state >> 19);
		}

		inline void skip(int cnt) {
			for (int i = 0; i < cnt; ++i) next();
		}
	};

	// AES-based, generates 128 bits of reasonable quality RNG every call. Good latency (6 cycles on AVX512, 7 on AVX2)
	// and throughput (tp 1.5 on AVX512, 2.5 ish on AVX2)
	class FastRng : public Rng {
#ifdef USE_X86_VECTORIZE

		__m128i real_state = _mm_setzero_si128();

		FastRng(uint64_t seed=0) {
			real_state = _mm_set1_epi64x(seed);
		}

		inline __m128i next_v() noexcept {
			real_state = _mm_aesenc_si128(_mm_set_epi64(5765458678434087244ULL, 3188412809159398971ULL));	

#ifdef USE_AVX512_VECTORIZE
			real_state = _mm_rol_epi32(real_state, 12);
#else
			real_state = _mm_or_si128(_mm_slli_epi32(real_state, 32 - 12), _mm_srli_epi32(real_state, 12));
#endif
			
			return real_state;
		}

		inline uint32_t next() noexcept {
			next_v();

			return _mm_cvtsi128_si32(real_state);
		}

		inline void skip(int cnt) {
			for (int i = 0; i < cnt; ++i) next();
		}
#endif
	}
}
