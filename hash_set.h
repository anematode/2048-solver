
#pragma once

// Fast multithreaded hash set
#include <thread>
#include "dbg.h"
#include <stdlib.h>
#include <stdint.h>
#include <memory.h>
#include <string.h>

// Insert nonzero elements into the queue to have the hashing thread deal them out

/*
template <int size>
class HashSetU64Queue {
	private:
	uint64_t* _data;

	// Whether the queue is ready to accept more elements	
	volatile sig_atomic_t ready;

	public:

	volatile uint64_t* wt;  // thread should insert elements here (and the inserter should stop here)
	uint64_t* rd;  // already read and inserted up to here

	uint64_t* end; // thread should stop here

	HashSetU64Queue() {
		contents = (uint64_t)malloc(size * sizeof(uint64_t));
		ready = 0;
	}

	~HashSetU64Queue() {
		free(contents);
	}

	// NOT thread safe. Only one thread should call this function.
	inline uint64_t enqueue(uint64_t a) {
		// Wait until hashing thread is ready
		while (!ready);

		*wt++ = a;
	}

	void wait_till_ready() {

	}
	
	void deposit_into_hs(HashSetU64* hs) {
		// Deposit all entries until rd
		if (wt == rd) return;

		for (; rd < wt; ++rd) {
			hs->insert(*wt);
		}
	}
};*/

template <int use_hashing_thread=false>
class HashSetU64 {
	static constexpr int MAX_SIZE_BITS = 42;
	static constexpr int MIN_SIZE_BITS = 8;

	static constexpr uint64_t DEFAULT_SIZE_BITS = 8;

	private:
	void _resize_bits_in_range(int size_bits) {
		if (size_bits < 4 || size_bits > 48) {
			printf("Size bits %d out of range\n", size_bits);
			abort();
		}
	}

	HashSetU64(int size, bool no_touch_pages) {
		_data = nullptr;
		resize_bits(size);

		_used_count = 0;
		_insertion_thread_in_use = false;

		// Touches pages to avoid strangeness
		if (!no_touch_pages)
			clear();
	}


	public:

	HashSetU64(int size_bits=DEFAULT_SIZE_BITS) : HashSetU64(size_bits, false) {
		
	}

	~HashSetU64() {
		free(_data);
	}

	HashSetU64(HashSetU64&& hs) {
		free(_data);
		_data = hs._data;
		hs._data = nullptr;	
		_sz_bits = hs._sz_bits;
		_used_count = hs._used_count;
	}

	HashSetU64(const HashSetU64& hs) : HashSetU64(hs._sz_bits, true /* no touch pages */) {
		memcpy(_data, hs._data, size());
	}

	/*HashSetU64& operator=(const HashSetU64& hs) {
		free(_data);
		_data = hs._data;
		
	}
	HashSetU64& operator=(HashSetU64&& hs) : HashSetU64(hs) {}
	*/


	uint64_t size() const {
		return 1ULL << _sz_bits;
	}

	uint64_t size_bits() const {
		return _sz_bits;
	}

	void resize_bits(int size_bits) {
		_resize_bits_in_range(size_bits);

		if (size_bits != _sz_bits) {
			_sz_bits = size_bits;
			
			uint64_t desired_size = size() * sizeof(uint64_t);
			if (_data) {
				_data = (uint64_t*)realloc(_data, desired_size);
			} else {
				_data = (uint64_t*)malloc(desired_size);
			}
		}

	}

	// Resize to smallest possible power of two
	void resize(int size) {
		resize_bits( 65 - __builtin_clzll (size - 1));
	}

	void clear() {
		memset(_data, 0, size() * sizeof(uint64_t));
	}

	uint64_t hash_elem(uint64_t e) {
		uint64_t hsh = e;

		for (int i = 0; i < 6; ++i) {
			hsh *= -1849381202ULL; hsh += 4;
		}

		return hsh;
	}

	uint64_t insert(uint64_t a) {
		// 0 is a reserved element
		uint64_t msk_idx = (1 << _sz_bits) - 1;
		uint64_t idx = hash_elem(a) & msk_idx;
		uint64_t i = idx, d = 0;

		for (; i != idx - 1; ++d, i = (i + 1) & msk_idx) {
			uint64_t c = _data[i];

			if (c == 0 || c == a) {
				_data[i] = a;
				break;
			}
		}

		_used_count++;	
		maybe_resize();

		return a;
	}

	inline void maybe_resize() {
		if (__builtin_expect(_used_count > (size() << 1), 0)) {
			// Need to resize to double the size 

			resize(_sz_bits + 1);
		}
	}


	// Should be used sparingly
	uint64_t remove(uint64_t a) {
		uint64_t msk_idx = (1 << _sz_bits) - 1;
		uint64_t idx = hash_elem(a) & msk_idx;
		uint64_t i = idx;

		for (; i != idx - 1; i = (i + 1) & msk_idx) {
			uint64_t c = _data[i];

			if (c == 0 || c == a) {

				// Zero contents, then rehash all subsequent entries
				// max disp will only decrease (but we don't do the work of figuring out whether it actually
				// decreased, as removal is a rare operation for us)
				_data[i] = 0;
				++i;

				for (; i != idx - 1; i = (i + 1) & msk_idx) {
					uint64_t entry = _data[i];

					if (entry == 0)
						goto out;

					_data[i] = 0;
					insert(entry);
				}
			}
		}

out:

		_used_count++;	
	}

	bool contains(uint64_t a) {
		uint64_t msk_idx = (1 << _sz_bits) - 1;
		uint64_t idx = hash_elem(a) & msk_idx;
		uint64_t i = idx;

		for (; i != idx - 1; i = (i + 1) & msk_idx) {
			uint64_t c = _data[i];

			if (c == 0) return false;
			else if (c == a) return true;
		}

		printf("Issue! %i\n", __LINE__);
		abort();
	}

	private:
	bool _insertion_thread_in_use;

	uint64_t* _data;

	int _sz_bits;
	uint64_t _used_count;

};
