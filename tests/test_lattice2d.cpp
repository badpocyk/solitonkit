#include <cassert>
#include <cmath>
#include <iostream>

#include "solitonkit/core/Lattice2D.hpp"

using namespace solitonkit;

int main() {
    Lattice2D lat{ 4, 3, 0.5, 0.25 };

    assert(lat.nx() == 4);
    assert(lat.ny() == 3);
    assert(lat.size() == 12);

    assert(std::abs(lat.dx() - 0.5) < 1e-12);
    assert(std::abs(lat.dy() - 0.25) < 1e-12);

    assert(lat.index(0, 0) == 0);
    assert(lat.index(1, 0) == 1);
    assert(lat.index(0, 1) == 4);
    assert(lat.index(3, 2) == 11);

    assert(lat.index(4, 0) == 0);
    assert(lat.index(0, 3) == 0);

    assert(lat.index_signed(-1, 0) == 3);
    assert(lat.index_signed(0, -1) == 8);
    assert(lat.index_signed(-1, -1) == 11);

    assert(lat.left(0) == 3);
    assert(lat.right(3) == 0);

    assert(lat.down(0) == 2);
    assert(lat.up(2) == 0);

    assert(lat.boundary_condition() == BoundaryCondition::Periodic);
    assert(lat.is_boundary(0, 0));
    assert(lat.is_boundary(3, 2));
    assert(!lat.is_boundary(1, 1));
    assert(!lat.is_fixed_boundary(0, 0));

    for (const auto condition : {
        BoundaryCondition::Fixed,
        BoundaryCondition::Neumann,
        BoundaryCondition::Dirichlet
    }) {
        Lattice2D non_periodic{ 4, 3, 0.5, 0.25, condition };

        assert(non_periodic.left(0) == 0);
        assert(non_periodic.right(3) == 3);
        assert(non_periodic.down(0) == 0);
        assert(non_periodic.up(2) == 2);

        assert(non_periodic.index(4, 3) == 11);
        assert(non_periodic.index_signed(-1, -1) == 0);
    }

    Lattice2D fixed{ 4, 3, 0.5, 0.25, BoundaryCondition::Fixed };
    assert(fixed.is_fixed_boundary(0, 1));
    assert(!fixed.is_fixed_boundary(1, 1));

    Lattice2D dirichlet{
        4,
        3,
        0.5,
        0.25,
        BoundaryCondition::Dirichlet
    };
    assert(dirichlet.is_fixed_boundary(0, 1));
    assert(dirichlet.is_dirichlet_boundary(0, 1));
    assert(!dirichlet.is_dirichlet_boundary(1, 1));

    std::cout << "Lattice2D tests passed successfully\n";

    return 0;
}
