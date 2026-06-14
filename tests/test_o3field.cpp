#include <cassert>
#include <cmath>
#include <iostream>

#include "solitonkit/core/O3Field.hpp"

using namespace solitonkit;

int main() {
    Lattice2D lat{ 5, 4 };

    O3Field field{ lat };

    assert(field.size() == 20);
    assert(field.lattice().nx() == 5);
    assert(field.lattice().ny() == 4);

    // Default field should be vacuum: (0, 0, 1)
    for (std::size_t k = 0; k < field.size(); ++k) {
        const Vec3& v = field.at_index(k);

        assert(std::abs(v.x - 0.0) < 1e-12);
        assert(std::abs(v.y - 0.0) < 1e-12);
        assert(std::abs(v.z - 1.0) < 1e-12);
        assert(std::abs(v.norm() - 1.0) < 1e-12);
    }

    // set_uniform should normalize the input vector
    field.set_uniform(Vec3{ 10.0, 0.0, 0.0 });

    for (std::size_t k = 0; k < field.size(); ++k) {
        const Vec3& v = field.at_index(k);

        assert(std::abs(v.x - 1.0) < 1e-12);
        assert(std::abs(v.y - 0.0) < 1e-12);
        assert(std::abs(v.z - 0.0) < 1e-12);
        assert(std::abs(v.norm() - 1.0) < 1e-12);
    }

    // Periodic boundary access
    field(0, 0) = Vec3{ 0.0, 1.0, 0.0 };

    assert(std::abs(field(5, 0).y - 1.0) < 1e-12);
    assert(std::abs(field(0, 4).y - 1.0) < 1e-12);
    assert(std::abs(field(5, 4).y - 1.0) < 1e-12);

    // normalize_all should restore unit length
    field(2, 2) = Vec3{ 3.0, 4.0, 0.0 };
    field.normalize_all();

    assert(std::abs(field(2, 2).norm() - 1.0) < 1e-12);

    // Random field should also live on S^2
    O3Field random_field = O3Field::random(lat, 42);

    for (std::size_t k = 0; k < random_field.size(); ++k) {
        assert(std::abs(random_field.at_index(k).norm() - 1.0) < 1e-12);
    }

    std::cout << "O3Field tests passed successfully\n";

    return 0;
}