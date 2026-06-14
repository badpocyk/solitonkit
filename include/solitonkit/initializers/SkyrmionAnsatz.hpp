#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "solitonkit/core/Lattice2D.hpp"
#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/core/Vec3.hpp"

namespace solitonkit {

    struct SkyrmionSpec {
        double x0{ 0.0 };
        double y0{ 0.0 };
        int charge{ 1 };
        double scale{ 1.0 };
        double phase{ 0.0 };
    };

    class SkyrmionAnsatz {
    public:
        static O3Field charge_one(const Lattice2D& lattice, double scale) {
            return make_skyrmion(lattice, scale, -1.0);
        }

        static O3Field anti_charge_one(const Lattice2D& lattice, double scale) {
            return make_skyrmion(lattice, scale, 1.0);
        }

        static O3Field charge_one_at(
            const Lattice2D& lattice,
            double scale,
            double x0,
            double y0,
            double phase = 0.0
        ) {
            return multi_skyrmion(
                lattice,
                {
                    SkyrmionSpec{
                        x0,
                        y0,
                        1,
                        scale,
                        phase
                    }
                }
            );
        }

        static O3Field anti_charge_one_at(
            const Lattice2D& lattice,
            double scale,
            double x0,
            double y0,
            double phase = 0.0
        ) {
            return multi_skyrmion(
                lattice,
                {
                    SkyrmionSpec{
                        x0,
                        y0,
                        -1,
                        scale,
                        phase
                    }
                }
            );
        }

        static O3Field multi_skyrmion(
            const Lattice2D& lattice,
            const std::vector<SkyrmionSpec>& specs
        ) {
            if (specs.empty()) {
                throw std::runtime_error(
                    "At least one skyrmion specification is required"
                );
            }

            O3Field field{ lattice };

            const double cx = 0.5 * static_cast<double>(lattice.nx() - 1);
            const double cy = 0.5 * static_cast<double>(lattice.ny() - 1);

            for (std::size_t j = 0; j < lattice.ny(); ++j) {
                for (std::size_t i = 0; i < lattice.nx(); ++i) {
                    const double x =
                        (static_cast<double>(i) - cx) * lattice.dx();

                    const double y =
                        (static_cast<double>(j) - cy) * lattice.dy();

                    std::complex<double> W{ 1.0, 0.0 };

                    for (const auto& spec : specs) {
                        if (spec.scale <= 0.0) {
                            throw std::runtime_error(
                                "Skyrmion scale must be positive"
                            );
                        }

                        if (spec.charge == 0) {
                            throw std::runtime_error(
                                "Skyrmion charge must be non-zero"
                            );
                        }

                        const double dx = x - spec.x0;
                        const double dy = y - spec.y0;

                        const double orientation =
                            spec.charge > 0 ? -1.0 : 1.0;

                        std::complex<double> z{
                            dx / spec.scale,
                            orientation * dy / spec.scale
                        };

                        const std::complex<double> phase_factor{
                            std::cos(spec.phase),
                            std::sin(spec.phase)
                        };

                        z *= phase_factor;

                        const int abs_charge =
                            spec.charge < 0 ? -spec.charge : spec.charge;

                        W *= std::pow(z, abs_charge);
                    }

                    const double absW2 = std::norm(W);
                    const double denom = 1.0 + absW2;

                    const double phi_x = 2.0 * W.real() / denom;
                    const double phi_y = 2.0 * W.imag() / denom;
                    const double phi_z = (absW2 - 1.0) / denom;

                    field(i, j) = Vec3{ phi_x, phi_y, phi_z };
                }
            }

            field.normalize_all();

            return field;
        }

    private:
        static O3Field make_skyrmion(
            const Lattice2D& lattice,
            double scale,
            double orientation
        ) {
            if (scale <= 0.0) {
                throw std::runtime_error("Skyrmion scale must be positive");
            }

            O3Field field{ lattice };

            const double cx = 0.5 * static_cast<double>(lattice.nx() - 1);
            const double cy = 0.5 * static_cast<double>(lattice.ny() - 1);

            for (std::size_t j = 0; j < lattice.ny(); ++j) {
                for (std::size_t i = 0; i < lattice.nx(); ++i) {
                    const double x =
                        (static_cast<double>(i) - cx) * lattice.dx();

                    const double y =
                        (static_cast<double>(j) - cy) * lattice.dy();

                    const double r2 = x * x + y * y;
                    const double s2 = scale * scale;
                    const double denom = r2 + s2;

                    const double phi_x = 2.0 * scale * x / denom;
                    const double phi_y = orientation * 2.0 * scale * y / denom;
                    const double phi_z = (r2 - s2) / denom;

                    field(i, j) = Vec3{ phi_x, phi_y, phi_z };
                }
            }

            field.normalize_all();

            return field;
        }
    };

} // namespace solitonkit