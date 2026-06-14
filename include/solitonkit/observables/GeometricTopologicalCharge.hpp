#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/core/Vec3.hpp"

namespace solitonkit {

    class GeometricTopologicalCharge {
    public:
        static double solid_angle(
            const Vec3& a_raw,
            const Vec3& b_raw,
            const Vec3& c_raw
        ) {
            const Vec3 a = a_raw.normalized();
            const Vec3 b = b_raw.normalized();
            const Vec3 c = c_raw.normalized();

            const double numerator = dot(a, cross(b, c));

            const double denominator =
                1.0
                + dot(a, b)
                + dot(b, c)
                + dot(c, a);

            return 2.0 * std::atan2(numerator, denominator);
        }

        static double plaquette_charge(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) {
            const auto& lat = field.lattice();

            const std::size_t ir = lat.right(i);
            const std::size_t ju = lat.up(j);

            const Vec3& n00 = field(i, j);
            const Vec3& n10 = field(ir, j);
            const Vec3& n01 = field(i, ju);
            const Vec3& n11 = field(ir, ju);

            const double omega_1 = solid_angle(n00, n10, n11);
            const double omega_2 = solid_angle(n00, n11, n01);

            constexpr double pi = 3.141592653589793238462643383279502884;

            return (omega_1 + omega_2) / (4.0 * pi);
        }

        static double compute(const O3Field& field) {
            const auto& lat = field.lattice();

            if (lat.nx() < 2 || lat.ny() < 2) {
                throw std::runtime_error(
                    "Geometric topological charge requires at least a 2x2 field"
                );
            }

            double total = 0.0;

            const std::ptrdiff_t nx = static_cast<std::ptrdiff_t>(lat.nx());
            const std::ptrdiff_t ny = static_cast<std::ptrdiff_t>(lat.ny());

#ifdef SOLITONKIT_USE_OPENMP
#pragma omp parallel for reduction(+:total) collapse(2)
#endif
            for (std::ptrdiff_t j = 0; j < ny; ++j) {
                for (std::ptrdiff_t i = 0; i < nx; ++i) {
                    total += plaquette_charge(
                        field,
                        static_cast<std::size_t>(i),
                        static_cast<std::size_t>(j)
                    );
                }
            }

            return total;
        }
    };

} // namespace solitonkit