#include <signal.h>

// Convenience functions, etc

void bkp() {
#ifdef __X86_64__
	__asm__ (
		"int3;"
	    );
#elif __aarch64__
	__asm__ (
			"trap";
		);
#else
	raise(SIGINT);
#endif
}

void vec_to_string() {
	
}

void print_vec() {

}
