#include "defs.h"
#include "position.h"

#include <unordered_set>
	using namespace Analysis;

int play_dumb_game() {
	bool s;

	int cnt = 0;
	Position p = Position::start();

	while (1) {
		++cnt;
		int i = 0;
		for (; i < 4; ++i) {
			p = p.move_right(&s);
			if (s) {
				break;
			}

			p = p.rotate_90();
		}

		if (i == 4) // oof!
			break;

		p = p.get_next_random(&s);
		// puts(p.to_string());
	}

	//puts(p.to_string());
	return cnt;
}

int play_dumb_game_v() {
	using PV = PositionV<4>;
	PV pv = PV::start_all();

	int cnt = 0;

	bool s = false;
	while (0) {
		++cnt;
		PV mu = pv.move_right();
		if (!s) {

		}

		pv = pv.get_next_random();
	}

	return 0;
}

int main() {

	/*std::unordered_set<Position> set;

	auto starts = Position::get_all_starting();
	for (auto& s : starts) {
		set.insert(s);
	}

	decltype(set) next;

	for (auto& p : set) {
		char* s = p.to_string();
		puts(s);
		free(s);
	}*/

	uint64_t total = 0;

	for (int i = 0; i < 10000000; ++i) total += play_dumb_game();

	printf("Total moves: %" PRIu64 "\n", total);
}
