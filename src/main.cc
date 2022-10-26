#include "defs.h"
#include "position.h"

int main() {
	using namespace Analysis;

	print_features();

	PositionV<2> p {
		{ 0, 0 }
	};

	printf("%s", p.to_string());
}
