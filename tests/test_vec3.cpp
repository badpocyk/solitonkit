#include <iostream>
#include <cassert>
#include <cmath>
#include "solitonkit/core/Vec3.hpp"

using namespace solitonkit;

int main() {
    Vec3 a{ 1.0, 0.0, 0.0 };
    Vec3 b{ 0.0, 1.0, 0.0 };

    Vec3 c = cross(a, b);

    assert(c.x == 0.0);
    assert(c.y == 0.0);
    assert(c.z == 1.0);

    assert(dot(a, b) == 0.0);
    assert(dot(a, a) == 1.0);

    Vec3 d{ 3.0, 4.0, 0.0 };
    Vec3 e = d.normalized();

    assert(std::abs(e.norm() - 1.0) < 1e-12);
	std::cout << "All tests passed!" << std::endl;
    return 0;
}