#pragma once

#include <cstddef>

namespace solitonkit {

    enum class BoundaryCondition {
        Periodic,
        Fixed,
        Neumann,
        Dirichlet
    };

    inline bool pins_boundary(BoundaryCondition condition) {
        return condition == BoundaryCondition::Fixed
            || condition == BoundaryCondition::Dirichlet;
    }

    struct BoundaryConditions2D {
        BoundaryCondition x{ BoundaryCondition::Periodic };
        BoundaryCondition y{ BoundaryCondition::Periodic };

        BoundaryConditions2D() = default;

        explicit BoundaryConditions2D(BoundaryCondition uniform)
            : x(uniform), y(uniform) {}

        BoundaryConditions2D(BoundaryCondition x_, BoundaryCondition y_)
            : x(x_), y(y_) {}

        bool uniform() const {
            return x == y;
        }
    };

    struct BoundaryConditions3D {
        BoundaryCondition x{ BoundaryCondition::Periodic };
        BoundaryCondition y{ BoundaryCondition::Periodic };
        BoundaryCondition z{ BoundaryCondition::Periodic };

        BoundaryConditions3D() = default;

        explicit BoundaryConditions3D(BoundaryCondition uniform)
            : x(uniform), y(uniform), z(uniform) {}

        BoundaryConditions3D(
            BoundaryCondition x_,
            BoundaryCondition y_,
            BoundaryCondition z_
        ) : x(x_), y(y_), z(z_) {}

        bool uniform() const {
            return x == y && y == z;
        }
    };

    inline std::size_t map_boundary_index(
        std::ptrdiff_t value,
        std::size_t extent,
        BoundaryCondition condition
    ) {
        const auto n = static_cast<std::ptrdiff_t>(extent);

        if (condition == BoundaryCondition::Periodic) {
            auto result = value % n;
            if (result < 0) {
                result += n;
            }
            return static_cast<std::size_t>(result);
        }

        if (value < 0) {
            return 0;
        }

        const auto result = static_cast<std::size_t>(value);
        return result < extent ? result : extent - 1;
    }

} // namespace solitonkit
