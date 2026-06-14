#include <cassert>
#include <filesystem>
#include <iostream>

#include "solitonkit/core/Lattice2D.hpp"
#include "solitonkit/initializers/SkyrmionAnsatz.hpp"
#include "solitonkit/io/CSVWriter.hpp"
#include "solitonkit/models/O3SigmaModel.hpp"

using namespace solitonkit;

int main() {
    Lattice2D lat{ 16, 16, 1.0, 1.0 };

    O3Field field = SkyrmionAnsatz::charge_one(lat, 4.0);
    O3SigmaModel model{ 1.0 };

    const std::string field_file = "test_field.csv";
    const std::string energy_file = "test_energy_density.csv";
    const std::string topo_file = "test_topological_density.csv";

    CSVWriter::write_field(field, field_file);
    CSVWriter::write_energy_density(field, model, energy_file);
    CSVWriter::write_topological_density(field, topo_file);

    assert(std::filesystem::exists(field_file));
    assert(std::filesystem::exists(energy_file));
    assert(std::filesystem::exists(topo_file));

    assert(std::filesystem::file_size(field_file) > 0);
    assert(std::filesystem::file_size(energy_file) > 0);
    assert(std::filesystem::file_size(topo_file) > 0);

    std::filesystem::remove(field_file);
    std::filesystem::remove(energy_file);
    std::filesystem::remove(topo_file);

    std::cout << "CSVWriter tests passed successfully\n";

    return 0;
}