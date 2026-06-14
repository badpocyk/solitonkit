#pragma once

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>

#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/models/O3SigmaModel.hpp"
#include "solitonkit/observables/TopologicalCharge.hpp"

namespace solitonkit {

    class CSVWriter {
    public:
        static void write_field(
            const O3Field& field,
            const std::string& filename
        ) {
            std::ofstream out(filename);

            if (!out) {
                throw std::runtime_error("Could not open file for writing: " + filename);
            }

            const auto& lat = field.lattice();

            out << "i,j,x,y,phi_x,phi_y,phi_z\n";

            for (std::size_t j = 0; j < lat.ny(); ++j) {
                for (std::size_t i = 0; i < lat.nx(); ++i) {
                    const Vec3& phi = field(i, j);

                    const double x = static_cast<double>(i) * lat.dx();
                    const double y = static_cast<double>(j) * lat.dy();

                    out
                        << i << ","
                        << j << ","
                        << x << ","
                        << y << ","
                        << phi.x << ","
                        << phi.y << ","
                        << phi.z << "\n";
                }
            }
        }

        static void write_energy_density(
            const O3Field& field,
            const O3SigmaModel& model,
            const std::string& filename
        ) {
            std::ofstream out(filename);

            if (!out) {
                throw std::runtime_error("Could not open file for writing: " + filename);
            }

            const auto& lat = field.lattice();

            out << "i,j,x,y,energy_density\n";

            for (std::size_t j = 0; j < lat.ny(); ++j) {
                for (std::size_t i = 0; i < lat.nx(); ++i) {
                    const double x = static_cast<double>(i) * lat.dx();
                    const double y = static_cast<double>(j) * lat.dy();

                    out
                        << i << ","
                        << j << ","
                        << x << ","
                        << y << ","
                        << model.energy_density_at(field, i, j) << "\n";
                }
            }
        }

        static void write_topological_density(
            const O3Field& field,
            const std::string& filename
        ) {
            std::ofstream out(filename);

            if (!out) {
                throw std::runtime_error("Could not open file for writing: " + filename);
            }

            const auto& lat = field.lattice();

            out << "i,j,x,y,topological_charge_density\n";

            for (std::size_t j = 0; j < lat.ny(); ++j) {
                for (std::size_t i = 0; i < lat.nx(); ++i) {
                    const double x = static_cast<double>(i) * lat.dx();
                    const double y = static_cast<double>(j) * lat.dy();

                    out
                        << i << ","
                        << j << ","
                        << x << ","
                        << y << ","
                        << TopologicalCharge::density_at(field, i, j) << "\n";
                }
            }
        }
    };

} // namespace solitonkit