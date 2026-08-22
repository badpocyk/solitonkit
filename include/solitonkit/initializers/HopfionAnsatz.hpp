#pragma once

#include <cmath>
#include <complex>
#include <stdexcept>

#include "solitonkit/core/Lattice3D.hpp"
#include "solitonkit/core/O3Field3D.hpp"
#include "solitonkit/core/Vec3.hpp"

namespace solitonkit {

    struct HopfionSpec {
        double scale{ 1.0 };
        int winding_p{ 1 };
        int winding_q{ 1 };

        int hopf_charge() const {
            return winding_p * winding_q;
        }
    };

    class HopfionAnsatz {
    public:
        static O3Field3D create(
            const Lattice3D& lattice,
            const HopfionSpec& spec = {}
        ) {
            if (spec.scale <= 0.0) {
                throw std::runtime_error("Hopfion scale must be positive");
            }
            if (spec.winding_p <= 0 || spec.winding_q <= 0) {
                throw std::runtime_error("Hopfion windings must be positive");
            }

            O3Field3D field(lattice);
            const double cx = 0.5 * static_cast<double>(lattice.nx() - 1);
            const double cy = 0.5 * static_cast<double>(lattice.ny() - 1);
            const double cz = 0.5 * static_cast<double>(lattice.nz() - 1);
            const double scale_squared = spec.scale * spec.scale;

            for (std::size_t k = 0; k < lattice.nz(); ++k) {
                const double z = (static_cast<double>(k) - cz) * lattice.dz();
                for (std::size_t j = 0; j < lattice.ny(); ++j) {
                    const double y = (static_cast<double>(j) - cy) * lattice.dy();
                    for (std::size_t i = 0; i < lattice.nx(); ++i) {
                        const double x = (static_cast<double>(i) - cx) * lattice.dx();
                        const double radius_squared = x * x + y * y + z * z;
                        const double denominator = radius_squared + scale_squared;

                        const std::complex<double> z1 =
                            2.0 * spec.scale * std::complex<double>(x, y)
                            / denominator;
                        const std::complex<double> z2 = std::complex<double>(
                            2.0 * spec.scale * z,
                            radius_squared - scale_squared
                        ) / denominator;

                        const std::complex<double> u = std::pow(z1, spec.winding_p);
                        const std::complex<double> v = std::pow(z2, spec.winding_q);
                        const double normalization = std::norm(u) + std::norm(v);

                        if (normalization < 1e-30) {
                            field(i, j, k) = { 0.0, 0.0, 1.0 };
                            continue;
                        }

                        const std::complex<double> product = u * std::conj(v);
                        field(i, j, k) = Vec3{
                            2.0 * product.real() / normalization,
                            2.0 * product.imag() / normalization,
                            (std::norm(v) - std::norm(u)) / normalization
                        }.normalized();
                    }
                }
            }
            field.enforce_boundary_condition();
            return field;
        }
    };

} // namespace solitonkit
