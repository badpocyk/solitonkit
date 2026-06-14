#include <filesystem>
#include <iostream>
#include <string>

#include "solitonkit/core/Lattice2D.hpp"
#include "solitonkit/initializers/SkyrmionAnsatz.hpp"
#include "solitonkit/io/CSVWriter.hpp"
#include "solitonkit/models/O3SigmaModel.hpp"
#include "solitonkit/observables/TopologicalCharge.hpp"

using namespace solitonkit;

int main(int argc, char** argv) {
    std::filesystem::path output_dir = "output";

    if (argc >= 2) {
        output_dir = argv[1];
    }

    std::filesystem::create_directories(output_dir);

    const std::size_t nx = 128;
    const std::size_t ny = 128;

    const double dx = 1.0;
    const double dy = 1.0;

    const double skyrmion_scale = 16.0;

    Lattice2D lattice{ nx, ny, dx, dy };

    O3Field field = SkyrmionAnsatz::charge_one(lattice, skyrmion_scale);

    O3SigmaModel model{ 1.0 };

    const double energy = model.total_energy(field);
    const double charge = TopologicalCharge::total(field);

    const auto field_file = output_dir / "skyrmion_field.csv";
    const auto energy_file = output_dir / "skyrmion_energy_density.csv";
    const auto topo_file = output_dir / "skyrmion_topological_density.csv";

    CSVWriter::write_field(field, field_file.string());
    CSVWriter::write_energy_density(field, model, energy_file.string());
    CSVWriter::write_topological_density(field, topo_file.string());

    std::cout << "Skyrmion export demo\n";
    std::cout << "Grid size: " << nx << " x " << ny << "\n";
    std::cout << "Energy: " << energy << "\n";
    std::cout << "Topological charge: " << charge << "\n";
    std::cout << "Output directory: " << std::filesystem::absolute(output_dir) << "\n";

    std::cout << "Exported files:\n";
    std::cout << "  " << field_file << "\n";
    std::cout << "  " << energy_file << "\n";
    std::cout << "  " << topo_file << "\n";

    return 0;
}