
#include "position.h"

// Implementation of the vectorized versions of position for 2, 4, and 8 elements

namespace Analysis {

}

// Explicitly instantiate allowed vectorized templates
template class PositionV<2>;
template class PositionV<4>;
template class PositionV<8>;
