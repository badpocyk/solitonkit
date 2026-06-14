#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "solitonkit/core/Lattice2D.hpp"
#include "solitonkit/flows/GradientFlow.hpp"
#include "solitonkit/initializers/SkyrmionAnsatz.hpp"
#include "solitonkit/io/CSVWriter.hpp"
#include "solitonkit/models/O3SigmaModel.hpp"
#include "solitonkit/observables/TopologicalCharge.hpp"

using namespace solitonkit;

void write_history_csv(
    const std::vector<FlowRecord>& history,
    const std::filesystem::path& filename
) {
    std::ofstream out(filename);

    if (!out) {
        throw std::runtime_error("Could not open history file for writing");
    }

    out << "step,energy,topological_charge\n";

    for (const auto& record : history) {
        out
            << record.step << ","
            << record.energy << ","
            << record.topological_charge << "\n";
    }
}

int main(int argc, char** argv) {
    std::filesystem::path output_dir = "output_relaxation";

    if (argc >= 2) {
        output_dir = argv[1];
    }

    std::filesystem::create_directories(output_dir);

    const std::size_t nx = 128;
    const std::size_t ny = 128;

    const double dx = 1.0;
    const double dy = 1.0;

    const double skyrmion_scale = 16.0;

    const std::size_t steps = 500;
    const std::size_t record_every = 10;
    const double step_size = 0.01;

    Lattice2D lattice{ nx, ny, dx, dy };

    O3Field field = SkyrmionAnsatz::charge_one(lattice, skyrmion_scale);

    O3SigmaModel model{ 1.0 };
    GradientFlow flow{ step_size };

    const double initial_energy = model.total_energy(field);
    const double initial_charge = TopologicalCharge::total(field);

    std::cout << "Skyrmion relaxation demo\n";
    std::cout << "Grid size: " << nx << " x " << ny << "\n";
    std::cout << "Steps: " << steps << "\n";
    std::cout << "Step size: " << step_size << "\n";
    std::cout << "Initial energy: " << initial_energy << "\n";
    std::cout << "Initial topological charge: " << initial_charge << "\n";

    const auto history = flow.run(field, model, steps, record_every);

    const double final_energy = model.total_energy(field);
    const double final_charge = TopologicalCharge::total(field);

    const auto field_file = output_dir / "relaxed_skyrmion_field.csv";
    const auto energy_file = output_dir / "relaxed_skyrmion_energy_density.csv";
    const auto topo_file = output_dir / "relaxed_skyrmion_topological_density.csv";
    const auto history_file = output_dir / "relaxation_history.csv";

    CSVWriter::write_field(field, field_file.string());
    CSVWriter::write_energy_density(field, model, energy_file.string());
    CSVWriter::write_topological_density(field, topo_file.string());
    write_history_csv(history, history_file);

    std::cout << "Final energy: " << final_energy << "\n";
    std::cout << "Final topological charge: " << final_charge << "\n";
    std::cout << "Output directory: " << std::filesystem::absolute(output_dir) << "\n";

    std::cout << "Exported files:\n";
    std::cout << "  " << field_file << "\n";
    std::cout << "  " << energy_file << "\n";
    std::cout << "  " << topo_file << "\n";
    std::cout << "  " << history_file << "\n";

    return 0;
}