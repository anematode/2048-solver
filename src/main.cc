#include "defs.h"
#include "position.h"

int main() {
	using namespace Analysis;

	print_features();
		Rng rng{0};
		
		rng.skip(19);

		Position p{0x002404201211458};

		char* s = p.to_string();
		puts(s);
		free(s);

		bool successful;
		//s = p.move_right().to_string();
		s = p.get_next_random(&successful).to_string();
		puts(s);
		free(s);

	/*PositionV<2> p {
		{ 0, 0 }
	};

	p.set_idx(1, 0x101011);

	printf("%s", p.to_string());*/
}
