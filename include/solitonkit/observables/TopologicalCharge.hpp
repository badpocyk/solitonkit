#pragma once

#include <cstddef>

#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/core/Vec3.hpp"

namespace solitonkit {

    class TopologicalCharge {
    public:
        static Vec3 derivative_x(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) {
            const auto& lat = field.lattice();

            const std::size_t il = lat.left(i);
            const std::size_t ir = lat.right(i);

            return (field(ir, j) - field(il, j)) / (2.0 * lat.dx());
        }

        static Vec3 derivative_y(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) {
            const auto& lat = field.lattice();

            const std::size_t jd = lat.down(j);
            const std::size_t ju = lat.up(j);

            return (field(i, ju) - field(i, jd)) / (2.0 * lat.dy());
        }

        static double density_at(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) {
            constexpr double pi = 3.141592653589793238462643383279502884;

            const Vec3 phi = field(i, j);
            const Vec3 dx_phi = derivative_x(field, i, j);
            const Vec3 dy_phi = derivative_y(field, i, j);

            return dot(phi, cross(dx_phi, dy_phi)) / (4.0 * pi);
        }

        static double total(const O3Field& field) {
            const auto& lat = field.lattice();

            const double cell_area = lat.dx() * lat.dy();

            double q = 0.0;

            const std::ptrdiff_t nx = static_cast<std::ptrdiff_t>(lat.nx());
            const std::ptrdiff_t ny = static_cast<std::ptrdiff_t>(lat.ny());

#ifdef SOLITONKIT_USE_OPENMP
#pragma omp parallel for reduction(+:q) collapse(2)
#endif
            for (std::ptrdiff_t j = 0; j < ny; ++j) {
                for (std::ptrdiff_t i = 0; i < nx; ++i) {
                    q += density_at(
                        field,
                        static_cast<std::size_t>(i),
                        static_cast<std::size_t>(j)
                    ) * cell_area;
                }
            }

            return q;
        }
    };

} // namespace solitonkit