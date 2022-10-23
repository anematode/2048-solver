
#pragma once

// Fast multithreaded hash set
#include <thread>
#include "dbg.h"

// Insert elements into the queue to have the hashing thread deal them out
class HashSetU64Queue {

};

template <int use_hashing_thread>
class HashSetU64 {
	constexpr int MAX_SIZE_BITS = 42;
	constexpr int MIN_SIZE_BITS = 8;

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

	HashSetU64(int size) : HashSetU64(size, false) {
	}

	~HashSetU64() {
		free(_data);
	}

	HashSetU64(HashU64&& hs) {
		free(_data);
		_data = hs._data;
		hs._data = nullptr;	
		_sz_bits = hs._sz_bits;
		_used_count = hs._used_count;
	}

	HashSetU64(const HashSetU64& hs) : HashSetU64(hs._sz_bits, true /* no touch pages */) {
		memcpy(_data, hs._data, size());
	}

	HashSetU64& operator=(const HashSetU64& hs) : HashSetU64(hs) {}
	HashSetU64& operator=(HashSetU64&& hs) : HashSetU64(hs) {}

	bool hash_thread_in_use() const {
		return use_hashing_thread && hash_thread_in_use;
	}

	uint64_t size() const {
		return 1ULL << _sz_bits;
	}

	uint64_t size_bits() const {
		return _sz_bits;
	}

	void resize_bits(int size_bits) {
		_resize_bits_in_range(size_bits);
	}

	// Resize to smallest possible power of two
	void resize(int size) {
		resize_bits( 65 - __builtin_clzll (size - 1));
	}

	void clear() {
		memset(_data, 0, size() * sizeof(uint64_t));
	}

	uint64_t hash_elem(uint64_t e) {
		// We want something fast, so we use the AES instruction set. 
	}

	uint64_t add_scalar(uint64_t a) {

	}

	private:
	bool _insertion_thread_in_use;

	uint64_t _data;

	int _sz_bits;
	uint64_t _used_count;

};
