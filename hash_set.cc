#include "hash_set.h"
#include <assert.h>

#include <stdio.h>

// Test hash set
int main() {
	HashSetU64 hs;

	assert(hs.contains(4) == false);
	assert(hs.contains(1) == false);
	assert(hs.contains(2) == false);
	hs.insert(1);
	assert(hs.contains(1) == true);
	hs.insert(4);
	assert(hs.contains(4) == true);
}
