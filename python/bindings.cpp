// python/bindings.cpp
#ifdef SOLITONKIT_USE_OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "solitonkit/analysis/LinearStability.hpp"
#include "solitonkit/analysis/Continuation.hpp"
#include "solitonkit/core/Lattice2D.hpp"
#include "solitonkit/core/Lattice3D.hpp"
#include "solitonkit/core/O3Field3D.hpp"
#include "solitonkit/core/ScalarField2D.hpp"
#include "solitonkit/core/Vec2.hpp"
#include "solitonkit/core/Vec3.hpp"
#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/dynamics/LLGDynamics.hpp"
#include "solitonkit/dynamics/LandauLifshitzDynamics.hpp"
#include "solitonkit/flows/BabySkyrmeGradientFlow.hpp"
#include "solitonkit/flows/BabySkyrmeOptimizers.hpp"
#include "solitonkit/flows/GradientFlow.hpp"
#include "solitonkit/initializers/SkyrmionAnsatz.hpp"
#include "solitonkit/initializers/HopfionAnsatz.hpp"
#include "solitonkit/models/BabySkyrmeModel.hpp"
#include "solitonkit/models/HopfionModel.hpp"
#include "solitonkit/models/MicromagneticModel.hpp"
#include "solitonkit/models/O3SigmaModel.hpp"
#include "solitonkit/models/ScalarModels.hpp"
#include "solitonkit/observables/GeometricTopologicalCharge.hpp"
#include "solitonkit/operators/DifferentialOperators.hpp"
#include "solitonkit/solvers/Solvers.hpp"
#include "solitonkit/solvers/StationarySolvers.hpp"
#include "solitonkit/topology/Topology.hpp"

namespace py = pybind11;

namespace solitonkit_binding {

    constexpr double PI = 3.141592653589793238462643383279502884;

    solitonkit::BoundaryCondition parse_boundary_condition(
        const std::string& boundary
    ) {
        if (boundary == "periodic") {
            return solitonkit::BoundaryCondition::Periodic;
        }

        if (boundary == "fixed") {
            return solitonkit::BoundaryCondition::Fixed;
        }

        if (boundary == "neumann") {
            return solitonkit::BoundaryCondition::Neumann;
        }

        if (boundary == "dirichlet") {
            return solitonkit::BoundaryCondition::Dirichlet;
        }

        throw std::invalid_argument(
            "boundary must be 'periodic', 'fixed', 'neumann', or 'dirichlet'"
        );
    }

    std::string boundary_condition_name(
        solitonkit::BoundaryCondition boundary
    ) {
        switch (boundary) {
        case solitonkit::BoundaryCondition::Periodic:
            return "periodic";
        case solitonkit::BoundaryCondition::Fixed:
            return "fixed";
        case solitonkit::BoundaryCondition::Neumann:
            return "neumann";
        case solitonkit::BoundaryCondition::Dirichlet:
            return "dirichlet";
        }

        throw std::runtime_error("unknown boundary condition");
    }

    double sk_dot(const solitonkit::Vec3& a, const solitonkit::Vec3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    solitonkit::Vec3 sk_cross(const solitonkit::Vec3& a, const solitonkit::Vec3& b) {
        return solitonkit::Vec3{
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        };
    }

    double sk_norm(const solitonkit::Vec3& v) {
        return std::sqrt(sk_dot(v, v));
    }

    solitonkit::Vec3 sk_normalized(const solitonkit::Vec3& v) {
        const double n = sk_norm(v);

        if (n == 0.0) {
            return solitonkit::Vec3{ 0.0, 0.0, 1.0 };
        }

        return solitonkit::Vec3{
            v.x / n,
            v.y / n,
            v.z / n
        };
    }

    solitonkit::O3Field make_empty_field(
        std::size_t nx,
        std::size_t ny,
        double dx,
        double dy,
        solitonkit::BoundaryCondition boundary =
            solitonkit::BoundaryCondition::Periodic
    ) {
        solitonkit::Lattice2D lattice(nx, ny, dx, dy, boundary);
        solitonkit::O3Field field(lattice);

        return field;
    }

    solitonkit::O3Field make_uniform_field(
        std::size_t nx,
        std::size_t ny,
        double dx,
        double dy,
        double x,
        double y,
        double z,
        solitonkit::BoundaryCondition boundary =
            solitonkit::BoundaryCondition::Periodic
    ) {
        solitonkit::O3Field field = make_empty_field(nx, ny, dx, dy, boundary);

        const solitonkit::Vec3 value = sk_normalized(
            solitonkit::Vec3{ x, y, z }
        );

        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                field(i, j) = value;
            }
        }

        field.enforce_boundary_condition();

        return field;
    }

    solitonkit::O3Field make_skyrmion_field(
        std::size_t nx,
        std::size_t ny,
        double dx,
        double dy,
        double radius,
        int charge,
        solitonkit::BoundaryCondition boundary =
            solitonkit::BoundaryCondition::Periodic
    ) {
        if (radius <= 0.0) {
            throw std::invalid_argument("radius must be positive");
        }

        if (charge == 0) {
            throw std::invalid_argument("charge must be non-zero");
        }

        solitonkit::O3Field field = make_empty_field(nx, ny, dx, dy, boundary);

        const double cx = 0.5 * static_cast<double>(nx - 1);
        const double cy = 0.5 * static_cast<double>(ny - 1);

        constexpr double eps = 1e-12;

        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                const double x = (static_cast<double>(i) - cx) * dx;
                const double y = (static_cast<double>(j) - cy) * dy;

                const double r = std::sqrt(x * x + y * y);
                const double theta = std::atan2(y, x);

                const double profile = 2.0 * std::atan2(radius, r + eps);
                const double angle = -static_cast<double>(charge) * theta;

                const solitonkit::Vec3 n{
                    std::sin(profile) * std::cos(angle),
                    std::sin(profile) * std::sin(angle),
                    std::cos(profile)
                };

                field(i, j) = sk_normalized(n);
            }
        }

        field.enforce_boundary_condition();

        return field;
    }

    solitonkit::O3Field make_skyrmion_field_at(
        std::size_t nx,
        std::size_t ny,
        double dx,
        double dy,
        double radius,
        double center_x,
        double center_y,
        int charge,
        solitonkit::BoundaryCondition boundary =
            solitonkit::BoundaryCondition::Periodic
    ) {
        if (radius <= 0.0) {
            throw std::invalid_argument("radius must be positive");
        }

        if (charge == 0) {
            throw std::invalid_argument("charge must be non-zero");
        }

        solitonkit::O3Field field = make_empty_field(nx, ny, dx, dy, boundary);

        constexpr double eps = 1e-12;

        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                const double x = (static_cast<double>(i) - center_x) * dx;
                const double y = (static_cast<double>(j) - center_y) * dy;

                const double r = std::sqrt(x * x + y * y);
                const double theta = std::atan2(y, x);

                const double profile = 2.0 * std::atan2(radius, r + eps);
                const double angle = -static_cast<double>(charge) * theta;

                const solitonkit::Vec3 n{
                    std::sin(profile) * std::cos(angle),
                    std::sin(profile) * std::sin(angle),
                    std::cos(profile)
                };

                field(i, j) = sk_normalized(n);
            }
        }

        field.enforce_boundary_condition();

        return field;
    }

    solitonkit::O3Field make_multi_skyrmion_field(
        std::size_t nx,
        std::size_t ny,
        double dx,
        double dy,
        const std::vector<solitonkit::SkyrmionSpec>& specs,
        solitonkit::BoundaryCondition boundary =
            solitonkit::BoundaryCondition::Periodic
    ) {
        solitonkit::Lattice2D lattice(nx, ny, dx, dy, boundary);
        solitonkit::O3Field field = solitonkit::SkyrmionAnsatz::multi_skyrmion(
            lattice,
            specs
        );

        field.enforce_boundary_condition();

        return field;
    }

    py::array_t<double> field_to_numpy(const solitonkit::O3Field& field) {
        const auto& lattice = field.lattice();

        const std::size_t nx = lattice.nx();
        const std::size_t ny = lattice.ny();

        py::array_t<double> result({
            static_cast<py::ssize_t>(ny),
            static_cast<py::ssize_t>(nx),
            static_cast<py::ssize_t>(3)
            });

        auto out = result.mutable_unchecked<3>();

        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                const solitonkit::Vec3& v = field(i, j);

                out(
                    static_cast<py::ssize_t>(j),
                    static_cast<py::ssize_t>(i),
                    0
                ) = v.x;

                out(
                    static_cast<py::ssize_t>(j),
                    static_cast<py::ssize_t>(i),
                    1
                ) = v.y;

                out(
                    static_cast<py::ssize_t>(j),
                    static_cast<py::ssize_t>(i),
                    2
                ) = v.z;
            }
        }

        return result;
    }

    solitonkit::O3Field field_from_numpy(
        py::array_t<double, py::array::c_style | py::array::forcecast> array,
        double dx,
        double dy,
        const std::string& boundary
    ) {
        const py::buffer_info info = array.request();

        if (info.ndim != 3) {
            throw std::invalid_argument("expected array with shape (height, width, 3)");
        }

        if (info.shape[2] != 3) {
            throw std::invalid_argument("expected last dimension to be 3");
        }

        const std::size_t ny = static_cast<std::size_t>(info.shape[0]);
        const std::size_t nx = static_cast<std::size_t>(info.shape[1]);

        solitonkit::O3Field field = make_empty_field(
            nx,
            ny,
            dx,
            dy,
            parse_boundary_condition(boundary)
        );

        auto data = array.unchecked<3>();

        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                const solitonkit::Vec3 v{
                    data(
                        static_cast<py::ssize_t>(j),
                        static_cast<py::ssize_t>(i),
                        0
                    ),
                    data(
                        static_cast<py::ssize_t>(j),
                        static_cast<py::ssize_t>(i),
                        1
                    ),
                    data(
                        static_cast<py::ssize_t>(j),
                        static_cast<py::ssize_t>(i),
                        2
                    )
                };

                field(i, j) = sk_normalized(v);
            }
        }

        field.enforce_boundary_condition();

        return field;
    }

    solitonkit::O3Field field_from_numpy_with_boundaries(
        py::array_t<double, py::array::c_style | py::array::forcecast> array,
        double dx,
        double dy,
        const std::string& boundary_x,
        const std::string& boundary_y
    ) {
        const py::buffer_info info = array.request();
        if (info.ndim != 3 || info.shape[2] != 3) {
            throw std::invalid_argument(
                "expected array with shape (height, width, 3)"
            );
        }

        const std::size_t ny = static_cast<std::size_t>(info.shape[0]);
        const std::size_t nx = static_cast<std::size_t>(info.shape[1]);
        solitonkit::O3Field field(solitonkit::Lattice2D(
            nx,
            ny,
            dx,
            dy,
            solitonkit::BoundaryConditions2D{
                parse_boundary_condition(boundary_x),
                parse_boundary_condition(boundary_y)
            }
        ));
        auto data = array.unchecked<3>();
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                field(i, j) = sk_normalized(solitonkit::Vec3{
                    data(j, i, 0),
                    data(j, i, 1),
                    data(j, i, 2)
                });
            }
        }
        field.enforce_boundary_condition();
        return field;
    }

    solitonkit::O3Field3D field3d_from_numpy(
        py::array_t<double, py::array::c_style | py::array::forcecast> array,
        double dx,
        double dy,
        double dz,
        const std::string& boundary_x,
        const std::string& boundary_y,
        const std::string& boundary_z
    ) {
        const py::buffer_info info = array.request();
        if (info.ndim != 4 || info.shape[3] != 3) {
            throw std::invalid_argument(
                "expected array with shape (depth, height, width, 3)"
            );
        }

        const std::size_t nz = static_cast<std::size_t>(info.shape[0]);
        const std::size_t ny = static_cast<std::size_t>(info.shape[1]);
        const std::size_t nx = static_cast<std::size_t>(info.shape[2]);
        solitonkit::O3Field3D field(solitonkit::Lattice3D(
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            solitonkit::BoundaryConditions3D{
                parse_boundary_condition(boundary_x),
                parse_boundary_condition(boundary_y),
                parse_boundary_condition(boundary_z)
            }
        ));
        auto data = array.unchecked<4>();
        for (std::size_t k = 0; k < nz; ++k) {
            for (std::size_t j = 0; j < ny; ++j) {
                for (std::size_t i = 0; i < nx; ++i) {
                    field(i, j, k) = sk_normalized(solitonkit::Vec3{
                        data(k, j, i, 0),
                        data(k, j, i, 1),
                        data(k, j, i, 2)
                    });
                }
            }
        }
        field.enforce_boundary_condition();
        return field;
    }

    solitonkit::Vec3 derivative_x(
        const solitonkit::O3Field& field,
        std::size_t i,
        std::size_t j
    ) {
        return solitonkit::differential::derivative_x(field, i, j);
    }

    solitonkit::Vec3 derivative_y(
        const solitonkit::O3Field& field,
        std::size_t i,
        std::size_t j
    ) {
        return solitonkit::differential::derivative_y(field, i, j);
    }

    py::array_t<double> energy_density(const solitonkit::O3Field& field) {
        const auto& lattice = field.lattice();

        const std::size_t nx = lattice.nx();
        const std::size_t ny = lattice.ny();

        py::array_t<double> result({
            static_cast<py::ssize_t>(ny),
            static_cast<py::ssize_t>(nx)
            });

        auto out = result.mutable_unchecked<2>();

        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                const solitonkit::Vec3 dx_vec = derivative_x(field, i, j);
                const solitonkit::Vec3 dy_vec = derivative_y(field, i, j);

                out(
                    static_cast<py::ssize_t>(j),
                    static_cast<py::ssize_t>(i)
                ) = 0.5 * (sk_dot(dx_vec, dx_vec) + sk_dot(dy_vec, dy_vec));
            }
        }

        return result;
    }

    double total_energy(const solitonkit::O3Field& field) {
        const auto& lattice = field.lattice();

        const std::size_t nx = lattice.nx();
        const std::size_t ny = lattice.ny();
        const double area = lattice.dx() * lattice.dy();

        double result = 0.0;

        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                const solitonkit::Vec3 dx_vec = derivative_x(field, i, j);
                const solitonkit::Vec3 dy_vec = derivative_y(field, i, j);

                result += 0.5 * (
                    sk_dot(dx_vec, dx_vec) +
                    sk_dot(dy_vec, dy_vec)
                    ) * area;
            }
        }

        return result;
    }

    py::array_t<double> topological_density(const solitonkit::O3Field& field) {
        const auto& lattice = field.lattice();

        const std::size_t nx = lattice.nx();
        const std::size_t ny = lattice.ny();

        py::array_t<double> result({
            static_cast<py::ssize_t>(ny),
            static_cast<py::ssize_t>(nx)
            });

        auto out = result.mutable_unchecked<2>();

        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                const solitonkit::Vec3& n = field(i, j);
                const solitonkit::Vec3 dx_vec = derivative_x(field, i, j);
                const solitonkit::Vec3 dy_vec = derivative_y(field, i, j);

                out(
                    static_cast<py::ssize_t>(j),
                    static_cast<py::ssize_t>(i)
                ) = sk_dot(n, sk_cross(dx_vec, dy_vec)) / (4.0 * PI);
            }
        }

        return result;
    }

    double topological_charge(const solitonkit::O3Field& field) {
        const auto& lattice = field.lattice();

        const std::size_t nx = lattice.nx();
        const std::size_t ny = lattice.ny();
        const double area = lattice.dx() * lattice.dy();

        double result = 0.0;

        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                const solitonkit::Vec3& n = field(i, j);
                const solitonkit::Vec3 dx_vec = derivative_x(field, i, j);
                const solitonkit::Vec3 dy_vec = derivative_y(field, i, j);

                result += sk_dot(n, sk_cross(dx_vec, dy_vec)) / (4.0 * PI) * area;
            }
        }

        return result;
    }

    solitonkit::FlowRecord make_record(
        const solitonkit::O3Field& field,
        std::size_t step
    ) {
        solitonkit::FlowRecord record;

        record.step = step;
        record.energy = total_energy(field);
        record.topological_charge = topological_charge(field);

        return record;
    }

    std::vector<solitonkit::FlowRecord> run_gradient_flow_inplace(
        solitonkit::O3Field& field,
        double step_size,
        std::size_t steps,
        std::size_t record_every
    ) {
        if (step_size <= 0.0) {
            throw std::invalid_argument("step_size must be positive");
        }

        if (record_every == 0) {
            throw std::invalid_argument("record_every must be positive");
        }

        solitonkit::GradientFlow flow(step_size);

        std::vector<solitonkit::FlowRecord> records;
        records.push_back(make_record(field, 0));

        for (std::size_t step = 1; step <= steps; ++step) {
            flow.step(field);

            if (step % record_every == 0 || step == steps) {
                records.push_back(make_record(field, step));
            }
        }

        return records;
    }

    std::tuple<solitonkit::O3Field, std::vector<solitonkit::FlowRecord>>
        run_gradient_flow(
            const solitonkit::O3Field& input,
            double step_size,
            std::size_t steps,
            std::size_t record_every
        ) {
        solitonkit::O3Field field = input;

        std::vector<solitonkit::FlowRecord> records =
            run_gradient_flow_inplace(
                field,
                step_size,
                steps,
                record_every
            );

        return std::make_tuple(field, records);
    }

    std::vector<solitonkit::FlowRecord> run_baby_skyrme_gradient_flow_inplace(
        solitonkit::O3Field& field,
        double kappa,
        double mass,
        double step_size,
        std::size_t steps,
        std::size_t record_every,
        double dmi
    ) {
        const solitonkit::BabySkyrmeModel model(kappa, mass, dmi);
        const solitonkit::BabySkyrmeGradientFlow flow(step_size);

        return flow.run(field, model, steps, record_every);
    }

    std::tuple<solitonkit::O3Field, std::vector<solitonkit::FlowRecord>>
        run_baby_skyrme_gradient_flow(
            const solitonkit::O3Field& input,
            double kappa,
            double mass,
            double step_size,
            std::size_t steps,
            std::size_t record_every,
            double dmi
        ) {
        solitonkit::O3Field field = input;

        std::vector<solitonkit::FlowRecord> records =
            run_baby_skyrme_gradient_flow_inplace(
                field,
                kappa,
                mass,
                step_size,
                steps,
                record_every,
                dmi
            );

        return std::make_tuple(field, records);
    }

    std::vector<solitonkit::FlowRecord>
        run_baby_skyrme_riemannian_gradient_descent_inplace(
            solitonkit::O3Field& field,
            double kappa,
            double mass,
            double step_size,
            std::size_t steps,
            std::size_t record_every,
            double dmi
        ) {
        const solitonkit::BabySkyrmeModel model(kappa, mass, dmi);
        const solitonkit::BabySkyrmeRiemannianGradientDescent optimizer(
            step_size
        );

        return optimizer.run(field, model, steps, record_every);
    }

    std::tuple<solitonkit::O3Field, std::vector<solitonkit::FlowRecord>>
        run_baby_skyrme_riemannian_gradient_descent(
            const solitonkit::O3Field& input,
            double kappa,
            double mass,
            double step_size,
            std::size_t steps,
            std::size_t record_every,
            double dmi
        ) {
        solitonkit::O3Field field = input;

        std::vector<solitonkit::FlowRecord> records =
            run_baby_skyrme_riemannian_gradient_descent_inplace(
                field,
                kappa,
                mass,
                step_size,
                steps,
                record_every,
                dmi
            );

        return std::make_tuple(field, records);
    }

    std::vector<solitonkit::FlowRecord>
        run_baby_skyrme_barzilai_borwein_inplace(
            solitonkit::O3Field& field,
            double kappa,
            double mass,
            double initial_step_size,
            double min_step_size,
            double max_step_size,
            std::size_t max_line_search_steps,
            std::size_t steps,
            std::size_t record_every,
            double dmi
        ) {
        const solitonkit::BabySkyrmeModel model(kappa, mass, dmi);
        const solitonkit::BabySkyrmeBarzilaiBorweinGradient optimizer(
            initial_step_size,
            min_step_size,
            max_step_size,
            max_line_search_steps
        );

        return optimizer.run(field, model, steps, record_every);
    }

    std::tuple<solitonkit::O3Field, std::vector<solitonkit::FlowRecord>>
        run_baby_skyrme_barzilai_borwein(
            const solitonkit::O3Field& input,
            double kappa,
            double mass,
            double initial_step_size,
            double min_step_size,
            double max_step_size,
            std::size_t max_line_search_steps,
            std::size_t steps,
            std::size_t record_every,
            double dmi
        ) {
        solitonkit::O3Field field = input;

        std::vector<solitonkit::FlowRecord> records =
            run_baby_skyrme_barzilai_borwein_inplace(
                field,
                kappa,
                mass,
                initial_step_size,
                min_step_size,
                max_step_size,
                max_line_search_steps,
                steps,
                record_every,
                dmi
            );

        return std::make_tuple(field, records);
    }

    std::vector<solitonkit::FlowRecord> run_baby_skyrme_lbfgs_inplace(
        solitonkit::O3Field& field,
        double kappa,
        double mass,
        double initial_step_size,
        std::size_t memory,
        std::size_t max_line_search_steps,
        std::size_t steps,
        std::size_t record_every,
        double dmi
    ) {
        const solitonkit::BabySkyrmeModel model(kappa, mass, dmi);
        const solitonkit::BabySkyrmeLBFGSOptimizer optimizer(
            initial_step_size,
            memory,
            max_line_search_steps
        );

        return optimizer.run(field, model, steps, record_every);
    }

    std::tuple<solitonkit::O3Field, std::vector<solitonkit::FlowRecord>>
        run_baby_skyrme_lbfgs(
            const solitonkit::O3Field& input,
            double kappa,
            double mass,
            double initial_step_size,
            std::size_t memory,
            std::size_t max_line_search_steps,
            std::size_t steps,
            std::size_t record_every,
            double dmi
        ) {
        solitonkit::O3Field field = input;

        std::vector<solitonkit::FlowRecord> records =
            run_baby_skyrme_lbfgs_inplace(
                field,
                kappa,
                mass,
                initial_step_size,
                memory,
                max_line_search_steps,
                steps,
                record_every,
                dmi
            );

        return std::make_tuple(field, records);
    }

    std::vector<solitonkit::FlowRecord>
        run_baby_skyrme_semi_implicit_flow_inplace(
            solitonkit::O3Field& field,
            double kappa,
            double mass,
            double step_size,
            std::size_t implicit_iterations,
            std::size_t steps,
            std::size_t record_every,
            double dmi
        ) {
        const solitonkit::BabySkyrmeModel model(kappa, mass, dmi);
        const solitonkit::BabySkyrmeSemiImplicitFlow optimizer(
            step_size,
            implicit_iterations
        );

        return optimizer.run(field, model, steps, record_every);
    }

    std::tuple<solitonkit::O3Field, std::vector<solitonkit::FlowRecord>>
        run_baby_skyrme_semi_implicit_flow(
            const solitonkit::O3Field& input,
            double kappa,
            double mass,
            double step_size,
            std::size_t implicit_iterations,
            std::size_t steps,
            std::size_t record_every,
            double dmi
        ) {
        solitonkit::O3Field field = input;

        std::vector<solitonkit::FlowRecord> records =
            run_baby_skyrme_semi_implicit_flow_inplace(
                field,
                kappa,
                mass,
                step_size,
                implicit_iterations,
                steps,
                record_every,
                dmi
            );

        return std::make_tuple(field, records);
    }

    std::vector<solitonkit::DynamicsRecord> run_landau_lifshitz_inplace(
        solitonkit::O3Field& field,
        double kappa,
        double mass,
        double time_step,
        double damping,
        std::size_t steps,
        std::size_t record_every,
        double dmi
    ) {
        const solitonkit::BabySkyrmeModel model(kappa, mass, dmi);
        const solitonkit::LandauLifshitzDynamics dynamics(time_step, damping);

        return dynamics.run(field, model, steps, record_every);
    }

    std::tuple<solitonkit::O3Field, std::vector<solitonkit::DynamicsRecord>>
        run_landau_lifshitz(
            const solitonkit::O3Field& input,
            double kappa,
            double mass,
            double time_step,
            double damping,
            std::size_t steps,
            std::size_t record_every,
            double dmi
        ) {
        solitonkit::O3Field field = input;

        std::vector<solitonkit::DynamicsRecord> records =
            run_landau_lifshitz_inplace(
                field,
                kappa,
                mass,
                time_step,
                damping,
                steps,
                record_every,
                dmi
            );

        return std::make_tuple(field, records);
    }

    py::array_t<double> make_skyrmion_numpy(
        int width,
        int height,
        double radius
    ) {
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("width and height must be positive");
        }

        solitonkit::O3Field field = make_skyrmion_field(
            static_cast<std::size_t>(width),
            static_cast<std::size_t>(height),
            1.0,
            1.0,
            radius,
            1,
            solitonkit::BoundaryCondition::Periodic
        );

        return field_to_numpy(field);
    }

    py::array_t<double> make_skyrmion_at_numpy(
        int width,
        int height,
        double radius,
        double center_x,
        double center_y
    ) {
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("width and height must be positive");
        }

        solitonkit::O3Field field = make_skyrmion_field_at(
            static_cast<std::size_t>(width),
            static_cast<std::size_t>(height),
            1.0,
            1.0,
            radius,
            center_x,
            center_y,
            1,
            solitonkit::BoundaryCondition::Periodic
        );

        return field_to_numpy(field);
    }

    template <typename Field, typename ConcreteModel>
    std::tuple<Field, std::vector<solitonkit::SolverRecord>> minimize_copy(
        const Field& input,
        const ConcreteModel& model,
        const solitonkit::MinimizeOptions& options
    ) {
        Field field = input;
        auto history = solitonkit::minimize(field, model, options);
        return std::make_tuple(field, history);
    }

    template <typename Field, typename ConcreteModel>
    std::vector<solitonkit::SolverRecord> minimize_inplace(
        Field& field,
        const ConcreteModel& model,
        const solitonkit::MinimizeOptions& options
    ) {
        return solitonkit::minimize(field, model, options);
    }

    template <typename Field, typename ConcreteModel>
    std::tuple<Field, std::vector<solitonkit::SolverRecord>> solve_copy(
        const Field& input,
        const ConcreteModel& model,
        const solitonkit::SolveOptions& options
    ) {
        Field field = input;
        auto history = solitonkit::solve(field, model, options);
        return std::make_tuple(field, history);
    }

    template <typename Field, typename ConcreteModel>
    std::vector<solitonkit::SolverRecord> solve_inplace(
        Field& field,
        const ConcreteModel& model,
        const solitonkit::SolveOptions& options
    ) {
        return solitonkit::solve(field, model, options);
    }

    template <typename Field, typename ConcreteModel>
    void bind_solver_overloads(py::module_& module) {
        module.def(
            "minimize",
            &minimize_copy<Field, ConcreteModel>,
            py::arg("field"),
            py::arg("model"),
            py::arg("options") = solitonkit::MinimizeOptions{}
        );
        module.def(
            "minimize_inplace",
            &minimize_inplace<Field, ConcreteModel>,
            py::arg("field"),
            py::arg("model"),
            py::arg("options") = solitonkit::MinimizeOptions{}
        );
        module.def(
            "solve",
            &solve_copy<Field, ConcreteModel>,
            py::arg("field"),
            py::arg("model"),
            py::arg("options") = solitonkit::SolveOptions{}
        );
        module.def(
            "solve_inplace",
            &solve_inplace<Field, ConcreteModel>,
            py::arg("field"),
            py::arg("model"),
            py::arg("options") = solitonkit::SolveOptions{}
        );
    }

    template <typename Field, typename ConcreteModel>
    std::tuple<Field, std::vector<solitonkit::StationaryRecord>>
        solve_stationary_copy(
            const Field& input,
            const ConcreteModel& model,
            const solitonkit::StationaryOptions& options
        ) {
        Field field = input;
        auto history = solitonkit::solve_stationary(field, model, options);
        return std::make_tuple(field, history);
    }

    template <typename Field, typename ConcreteModel>
    std::vector<solitonkit::StationaryRecord> solve_stationary_inplace(
        Field& field,
        const ConcreteModel& model,
        const solitonkit::StationaryOptions& options
    ) {
        return solitonkit::solve_stationary(field, model, options);
    }

    template <typename Field, typename ConcreteModel>
    void bind_stationary_overloads(py::module_& module) {
        module.def(
            "solve_stationary",
            &solve_stationary_copy<Field, ConcreteModel>,
            py::arg("field"),
            py::arg("model"),
            py::arg("options") = solitonkit::StationaryOptions{},
            py::call_guard<py::gil_scoped_release>()
        );
        module.def(
            "solve_stationary_inplace",
            &solve_stationary_inplace<Field, ConcreteModel>,
            py::arg("field"),
            py::arg("model"),
            py::arg("options") = solitonkit::StationaryOptions{},
            py::call_guard<py::gil_scoped_release>()
        );
    }

    py::array_t<double> scalar_mode_to_numpy(
        const std::vector<double>& mode,
        const solitonkit::ScalarField2D& field
    ) {
        const auto& lattice = field.lattice();
        if (mode.size() != field.size()) {
            throw std::runtime_error("stability mode size does not match field");
        }
        py::array_t<double> result({
            static_cast<py::ssize_t>(lattice.ny()),
            static_cast<py::ssize_t>(lattice.nx())
        });
        auto output = result.mutable_unchecked<2>();
        for (std::size_t j = 0; j < lattice.ny(); ++j) {
            for (std::size_t i = 0; i < lattice.nx(); ++i) {
                output(
                    static_cast<py::ssize_t>(j),
                    static_cast<py::ssize_t>(i)
                ) = mode[lattice.index(i, j)];
            }
        }
        return result;
    }

    py::array_t<double> vector_mode_to_numpy(
        const std::vector<solitonkit::Vec3>& mode,
        const solitonkit::O3Field& field
    ) {
        const auto& lattice = field.lattice();
        if (mode.size() != field.size()) {
            throw std::runtime_error("stability mode size does not match field");
        }
        py::array_t<double> result({
            static_cast<py::ssize_t>(lattice.ny()),
            static_cast<py::ssize_t>(lattice.nx()),
            static_cast<py::ssize_t>(3)
        });
        auto output = result.mutable_unchecked<3>();
        for (std::size_t j = 0; j < lattice.ny(); ++j) {
            for (std::size_t i = 0; i < lattice.nx(); ++i) {
                const auto& value = mode[lattice.index(i, j)];
                output(j, i, 0) = value.x;
                output(j, i, 1) = value.y;
                output(j, i, 2) = value.z;
            }
        }
        return result;
    }

    py::array_t<double> vector_mode_to_numpy(
        const std::vector<solitonkit::Vec3>& mode,
        const solitonkit::O3Field3D& field
    ) {
        const auto& lattice = field.lattice();
        if (mode.size() != field.size()) {
            throw std::runtime_error("stability mode size does not match field");
        }
        py::array_t<double> result({
            static_cast<py::ssize_t>(lattice.nz()),
            static_cast<py::ssize_t>(lattice.ny()),
            static_cast<py::ssize_t>(lattice.nx()),
            static_cast<py::ssize_t>(3)
        });
        auto output = result.mutable_unchecked<4>();
        for (std::size_t k = 0; k < lattice.nz(); ++k) {
            for (std::size_t j = 0; j < lattice.ny(); ++j) {
                for (std::size_t i = 0; i < lattice.nx(); ++i) {
                    const auto& value = mode[lattice.index(i, j, k)];
                    output(k, j, i, 0) = value.x;
                    output(k, j, i, 1) = value.y;
                    output(k, j, i, 2) = value.z;
                }
            }
        }
        return result;
    }

    template <typename Field>
    py::dict stability_result_to_python(
        const solitonkit::StabilityResult<double>& result,
        const Field& field
    ) {
        py::list modes;
        for (const auto& mode : result.modes) {
            modes.append(scalar_mode_to_numpy(mode, field));
        }
        py::dict output;
        output["eigenvalues"] = result.eigenvalues;
        output["residual_norms"] = result.residual_norms;
        output["modes"] = std::move(modes);
        output["gradient_norm"] = result.gradient_norm;
        output["iterations"] = result.iterations;
        output["degrees_of_freedom"] = result.degrees_of_freedom;
        output["converged"] = result.converged;
        output["stationary"] = result.stationary;
        output["stable"] = result.stable;
        return output;
    }

    template <typename Field>
    py::dict stability_result_to_python(
        const solitonkit::StabilityResult<solitonkit::Vec3>& result,
        const Field& field
    ) {
        py::list modes;
        for (const auto& mode : result.modes) {
            modes.append(vector_mode_to_numpy(mode, field));
        }
        py::dict output;
        output["eigenvalues"] = result.eigenvalues;
        output["residual_norms"] = result.residual_norms;
        output["modes"] = std::move(modes);
        output["gradient_norm"] = result.gradient_norm;
        output["iterations"] = result.iterations;
        output["degrees_of_freedom"] = result.degrees_of_freedom;
        output["converged"] = result.converged;
        output["stationary"] = result.stationary;
        output["stable"] = result.stable;
        return output;
    }

    template <typename Field, typename ConcreteModel>
    py::dict stability_analysis_python(
        const Field& field,
        const ConcreteModel& model,
        const solitonkit::StabilityOptions& options
    ) {
        return stability_result_to_python(
            solitonkit::stability_analysis(field, model, options),
            field
        );
    }

    template <typename Field, typename ConcreteModel>
    void bind_stability_overload(py::module_& module) {
        module.def(
            "stability_analysis",
            &stability_analysis_python<Field, ConcreteModel>,
            py::arg("field"),
            py::arg("model"),
            py::arg("options") = solitonkit::StabilityOptions{}
        );
    }

    template <typename Field, typename ConcreteModel>
    py::dict continue_solution_python(
        const Field& initial,
        const py::function& model_factory,
        const solitonkit::ContinuationOptions& options
    ) {
        const auto branch = solitonkit::continue_solution(
            initial,
            [&model_factory](double parameter) {
                return model_factory(parameter).cast<ConcreteModel>();
            },
            options
        );
        py::list points;
        for (const auto& point : branch.points) {
            py::dict item;
            item["parameter"] = point.parameter;
            item["energy"] = point.energy;
            item["residual_norm"] = point.residual_norm;
            item["lowest_eigenvalue"] = point.lowest_eigenvalue;
            item["corrector_steps"] = point.corrector_steps;
            item["converged"] = point.converged;
            item["stable"] = point.stable;
            item["bifurcation_candidate"] = point.bifurcation_candidate;
            item["field"] = py::cast(point.field);
            points.append(std::move(item));
        }
        py::dict output;
        output["points"] = std::move(points);
        output["reached_stop"] = branch.reached_stop;
        output["converged"] = branch.converged;
        return output;
    }

    template <typename Field, typename ConcreteModel>
    void bind_continuation_overload(
        py::module_& module,
        const char* name
    ) {
        module.def(
            name,
            &continue_solution_python<Field, ConcreteModel>,
            py::arg("field"),
            py::arg("model_factory"),
            py::arg("options")
        );
    }

    py::dict gmres_python(
        const py::function& apply,
        const std::vector<double>& right_hand_side,
        const solitonkit::GMRESOptions& options,
        const std::vector<double>& inverse_diagonal
    ) {
        const auto result = solitonkit::gmres(
            [&apply](const std::vector<double>& value) {
                return apply(value).cast<std::vector<double>>();
            },
            right_hand_side,
            options,
            inverse_diagonal
        );
        py::dict output;
        output["solution"] = result.solution;
        output["residual_norm"] = result.residual_norm;
        output["iterations"] = result.iterations;
        output["converged"] = result.converged;
        return output;
    }

} // namespace solitonkit_binding


PYBIND11_MODULE(_core, m) {
    using namespace solitonkit_binding;

    m.doc() = "C++ backend for solitonkit numerical field analysis";

    py::enum_<solitonkit::BoundaryCondition>(m, "BoundaryCondition")
        .value("Periodic", solitonkit::BoundaryCondition::Periodic)
        .value("Fixed", solitonkit::BoundaryCondition::Fixed)
        .value("Neumann", solitonkit::BoundaryCondition::Neumann)
        .value("Dirichlet", solitonkit::BoundaryCondition::Dirichlet);

    py::class_<solitonkit::BoundaryConditions2D>(m, "BoundaryConditions2D")
        .def(py::init<solitonkit::BoundaryCondition>())
        .def(
            py::init<
                solitonkit::BoundaryCondition,
                solitonkit::BoundaryCondition
            >(),
            py::arg("x"),
            py::arg("y")
        )
        .def_readwrite("x", &solitonkit::BoundaryConditions2D::x)
        .def_readwrite("y", &solitonkit::BoundaryConditions2D::y)
        .def_property_readonly(
            "uniform",
            &solitonkit::BoundaryConditions2D::uniform
        );

    py::class_<solitonkit::BoundaryConditions3D>(m, "BoundaryConditions3D")
        .def(py::init<solitonkit::BoundaryCondition>())
        .def(
            py::init<
                solitonkit::BoundaryCondition,
                solitonkit::BoundaryCondition,
                solitonkit::BoundaryCondition
            >(),
            py::arg("x"),
            py::arg("y"),
            py::arg("z")
        )
        .def_readwrite("x", &solitonkit::BoundaryConditions3D::x)
        .def_readwrite("y", &solitonkit::BoundaryConditions3D::y)
        .def_readwrite("z", &solitonkit::BoundaryConditions3D::z)
        .def_property_readonly(
            "uniform",
            &solitonkit::BoundaryConditions3D::uniform
        );

    py::enum_<solitonkit::FieldKind>(m, "FieldKind")
        .value("Scalar2D", solitonkit::FieldKind::Scalar2D)
        .value("XY2D", solitonkit::FieldKind::XY2D)
        .value("O3_2D", solitonkit::FieldKind::O3_2D)
        .value("O3_3D", solitonkit::FieldKind::O3_3D);

    py::class_<solitonkit::Model>(m, "Model")
        .def_property_readonly("name", &solitonkit::Model::name)
        .def_property_readonly("dimensions", &solitonkit::Model::dimensions)
        .def_property_readonly("field_kind", &solitonkit::Model::field_kind);

    py::class_<solitonkit::Vec2>(m, "Vec2")
        .def(py::init<>())
        .def(py::init<double, double>())
        .def_readwrite("x", &solitonkit::Vec2::x)
        .def_readwrite("y", &solitonkit::Vec2::y)
        .def("norm", &solitonkit::Vec2::norm)
        .def("squared_norm", &solitonkit::Vec2::norm_squared)
        .def("__repr__", [](const solitonkit::Vec2& value) {
        return "Vec2(" + std::to_string(value.x) + ", "
            + std::to_string(value.y) + ")";
    });

    py::class_<solitonkit::Vec3>(m, "Vec3")
        .def(py::init<>())
        .def(py::init<double, double, double>())
        .def_readwrite("x", &solitonkit::Vec3::x)
        .def_readwrite("y", &solitonkit::Vec3::y)
        .def_readwrite("z", &solitonkit::Vec3::z)
        .def("norm", [](const solitonkit::Vec3& v) {
        return sk_norm(v);
    })
        .def("squared_norm", [](const solitonkit::Vec3& v) {
        return sk_dot(v, v);
    })
        .def("normalized", [](const solitonkit::Vec3& v) {
        return sk_normalized(v);
    })
        .def("__repr__", [](const solitonkit::Vec3& v) {
        return "Vec3(" +
            std::to_string(v.x) + ", " +
            std::to_string(v.y) + ", " +
            std::to_string(v.z) + ")";
    });

    py::class_<solitonkit::O3Field>(m, "O3Field")
        .def(py::init([](
            std::size_t nx,
            std::size_t ny,
            double dx,
            double dy,
            const std::string& boundary
            ) {
        return make_empty_field(
            nx,
            ny,
            dx,
            dy,
            parse_boundary_condition(boundary)
        );
    }),
            py::arg("nx"),
            py::arg("ny"),
            py::arg("dx") = 1.0,
            py::arg("dy") = 1.0,
            py::arg("boundary") = "periodic"
        )
        .def_property_readonly("nx", [](const solitonkit::O3Field& field) {
        return field.lattice().nx();
    })
        .def_property_readonly("ny", [](const solitonkit::O3Field& field) {
        return field.lattice().ny();
    })
        .def_property_readonly("dx", [](const solitonkit::O3Field& field) {
        return field.lattice().dx();
    })
        .def_property_readonly("dy", [](const solitonkit::O3Field& field) {
        return field.lattice().dy();
    })
        .def_property_readonly("spacing", [](const solitonkit::O3Field& field) {
        return field.lattice().dx();
    })
        .def_property_readonly("boundary", [](const solitonkit::O3Field& field) {
        return boundary_condition_name(field.lattice().boundary_condition());
    })
        .def_property_readonly("boundary_x", [](const solitonkit::O3Field& field) {
        return boundary_condition_name(field.lattice().boundary_x());
    })
        .def_property_readonly("boundary_y", [](const solitonkit::O3Field& field) {
        return boundary_condition_name(field.lattice().boundary_y());
    })
        .def_property(
            "dirichlet_value",
            &solitonkit::O3Field::dirichlet_value,
            &solitonkit::O3Field::set_dirichlet_value
        )
        .def_static("with_boundaries", [](
            std::size_t nx,
            std::size_t ny,
            double dx,
            double dy,
            const std::string& boundary_x,
            const std::string& boundary_y
        ) {
        return solitonkit::O3Field(solitonkit::Lattice2D(
            nx,
            ny,
            dx,
            dy,
            solitonkit::BoundaryConditions2D{
                parse_boundary_condition(boundary_x),
                parse_boundary_condition(boundary_y)
            }
        ));
    },
            py::arg("nx"),
            py::arg("ny"),
            py::arg("dx") = 1.0,
            py::arg("dy") = 1.0,
            py::arg("boundary_x") = "periodic",
            py::arg("boundary_y") = "periodic"
        )
        .def("get", [](const solitonkit::O3Field& field, std::size_t i, std::size_t j) {
        return field(i, j);
    })
        .def("set", [](solitonkit::O3Field& field, std::size_t i, std::size_t j, const solitonkit::Vec3& value) {
        field(i, j) = sk_normalized(value);
        field.enforce_boundary_condition();
    })
        .def("to_numpy", &field_to_numpy)
        .def("__repr__", [](const solitonkit::O3Field& field) {
        return "O3Field(nx=" +
            std::to_string(field.lattice().nx()) +
            ", ny=" +
            std::to_string(field.lattice().ny()) +
            ", dx=" +
            std::to_string(field.lattice().dx()) +
            ", dy=" +
            std::to_string(field.lattice().dy()) +
            ", boundary='" +
            boundary_condition_name(field.lattice().boundary_condition()) +
            "'" +
            ")";
    });

    py::class_<solitonkit::ScalarField2D>(m, "ScalarField2D")
        .def(py::init([](
            std::size_t nx,
            std::size_t ny,
            double dx,
            double dy,
            double value,
            double dirichlet_value,
            const std::string& boundary_x,
            const std::string& boundary_y
        ) {
        return solitonkit::ScalarField2D(
            solitonkit::Lattice2D(
                nx,
                ny,
                dx,
                dy,
                solitonkit::BoundaryConditions2D{
                    parse_boundary_condition(boundary_x),
                    parse_boundary_condition(boundary_y)
                }
            ),
            value,
            dirichlet_value
        );
    }),
            py::arg("nx"),
            py::arg("ny"),
            py::arg("dx") = 1.0,
            py::arg("dy") = 1.0,
            py::arg("value") = 0.0,
            py::arg("dirichlet_value") = 0.0,
            py::arg("boundary_x") = "periodic",
            py::arg("boundary_y") = "periodic"
        )
        .def_property_readonly("nx", [](const solitonkit::ScalarField2D& field) {
        return field.lattice().nx();
    })
        .def_property_readonly("ny", [](const solitonkit::ScalarField2D& field) {
        return field.lattice().ny();
    })
        .def_property_readonly("dx", [](const solitonkit::ScalarField2D& field) {
        return field.lattice().dx();
    })
        .def_property_readonly("dy", [](const solitonkit::ScalarField2D& field) {
        return field.lattice().dy();
    })
        .def_property_readonly("boundary_x", [](const solitonkit::ScalarField2D& field) {
        return boundary_condition_name(field.lattice().boundary_x());
    })
        .def_property_readonly("boundary_y", [](const solitonkit::ScalarField2D& field) {
        return boundary_condition_name(field.lattice().boundary_y());
    })
        .def_property(
            "dirichlet_value",
            &solitonkit::ScalarField2D::dirichlet_value,
            &solitonkit::ScalarField2D::set_dirichlet_value
        )
        .def("get", [](const solitonkit::ScalarField2D& field, std::size_t i, std::size_t j) {
        return field(i, j);
    })
        .def("set", [](solitonkit::ScalarField2D& field, std::size_t i, std::size_t j, double value) {
        field(i, j) = value;
        field.enforce_boundary_condition();
    })
        .def("to_numpy", [](const solitonkit::ScalarField2D& field) {
        const auto& lattice = field.lattice();
        py::array_t<double> result({ lattice.ny(), lattice.nx() });
        auto output = result.mutable_unchecked<2>();
        for (std::size_t j = 0; j < lattice.ny(); ++j) {
            for (std::size_t i = 0; i < lattice.nx(); ++i) {
                output(j, i) = field(i, j);
            }
        }
        return result;
    });

    py::class_<solitonkit::XYField, solitonkit::ScalarField2D>(m, "XYField")
        .def(py::init([](
            std::size_t nx,
            std::size_t ny,
            double dx,
            double dy,
            double angle,
            double dirichlet_value,
            const std::string& boundary_x,
            const std::string& boundary_y
        ) {
        return solitonkit::XYField(
            solitonkit::Lattice2D(
                nx,
                ny,
                dx,
                dy,
                solitonkit::BoundaryConditions2D{
                    parse_boundary_condition(boundary_x),
                    parse_boundary_condition(boundary_y)
                }
            ),
            angle,
            dirichlet_value
        );
    }),
            py::arg("nx"),
            py::arg("ny"),
            py::arg("dx") = 1.0,
            py::arg("dy") = 1.0,
            py::arg("angle") = 0.0,
            py::arg("dirichlet_value") = 0.0,
            py::arg("boundary_x") = "periodic",
            py::arg("boundary_y") = "periodic"
        );

    py::class_<solitonkit::O3Field3D>(m, "O3Field3D")
        .def(py::init([](
            std::size_t nx,
            std::size_t ny,
            std::size_t nz,
            double dx,
            double dy,
            double dz,
            const std::string& boundary_x,
            const std::string& boundary_y,
            const std::string& boundary_z
        ) {
        return solitonkit::O3Field3D(solitonkit::Lattice3D(
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            solitonkit::BoundaryConditions3D{
                parse_boundary_condition(boundary_x),
                parse_boundary_condition(boundary_y),
                parse_boundary_condition(boundary_z)
            }
        ));
    }),
            py::arg("nx"),
            py::arg("ny"),
            py::arg("nz"),
            py::arg("dx") = 1.0,
            py::arg("dy") = 1.0,
            py::arg("dz") = 1.0,
            py::arg("boundary_x") = "periodic",
            py::arg("boundary_y") = "periodic",
            py::arg("boundary_z") = "periodic"
        )
        .def_property_readonly("nx", [](const solitonkit::O3Field3D& field) {
        return field.lattice().nx();
    })
        .def_property_readonly("ny", [](const solitonkit::O3Field3D& field) {
        return field.lattice().ny();
    })
        .def_property_readonly("nz", [](const solitonkit::O3Field3D& field) {
        return field.lattice().nz();
    })
        .def_property_readonly("dx", [](const solitonkit::O3Field3D& field) {
        return field.lattice().dx();
    })
        .def_property_readonly("dy", [](const solitonkit::O3Field3D& field) {
        return field.lattice().dy();
    })
        .def_property_readonly("dz", [](const solitonkit::O3Field3D& field) {
        return field.lattice().dz();
    })
        .def_property_readonly("boundary", [](const solitonkit::O3Field3D& field) {
        return boundary_condition_name(field.lattice().boundary_condition());
    })
        .def_property_readonly("boundary_x", [](const solitonkit::O3Field3D& field) {
        return boundary_condition_name(field.lattice().boundary_x());
    })
        .def_property_readonly("boundary_y", [](const solitonkit::O3Field3D& field) {
        return boundary_condition_name(field.lattice().boundary_y());
    })
        .def_property_readonly("boundary_z", [](const solitonkit::O3Field3D& field) {
        return boundary_condition_name(field.lattice().boundary_z());
    })
        .def_property(
            "dirichlet_value",
            &solitonkit::O3Field3D::dirichlet_value,
            &solitonkit::O3Field3D::set_dirichlet_value
        )
        .def("get", [](const solitonkit::O3Field3D& field, std::size_t i, std::size_t j, std::size_t k) {
        return field(i, j, k);
    })
        .def("set", [](solitonkit::O3Field3D& field, std::size_t i, std::size_t j, std::size_t k, const solitonkit::Vec3& value) {
        field(i, j, k) = value.normalized();
        field.enforce_boundary_condition();
    })
        .def("to_numpy", [](const solitonkit::O3Field3D& field) {
        const auto& lattice = field.lattice();
        py::array_t<double> result({
            lattice.nz(), lattice.ny(), lattice.nx(), std::size_t(3)
        });
        auto output = result.mutable_unchecked<4>();
        for (std::size_t k = 0; k < lattice.nz(); ++k) {
            for (std::size_t j = 0; j < lattice.ny(); ++j) {
                for (std::size_t i = 0; i < lattice.nx(); ++i) {
                    const auto value = field(i, j, k);
                    output(k, j, i, 0) = value.x;
                    output(k, j, i, 1) = value.y;
                    output(k, j, i, 2) = value.z;
                }
            }
        }
        return result;
    });

    py::class_<solitonkit::FlowRecord>(m, "FlowRecord")
        .def_readwrite("step", &solitonkit::FlowRecord::step)
        .def_readwrite("energy", &solitonkit::FlowRecord::energy)
        .def_readwrite(
            "topological_charge",
            &solitonkit::FlowRecord::topological_charge
        )
        .def("__repr__", [](const solitonkit::FlowRecord& record) {
        return "FlowRecord(step=" +
            std::to_string(record.step) +
            ", energy=" +
            std::to_string(record.energy) +
            ", topological_charge=" +
            std::to_string(record.topological_charge) +
            ")";
    });

    py::class_<solitonkit::DynamicsRecord>(m, "DynamicsRecord")
        .def_readwrite("step", &solitonkit::DynamicsRecord::step)
        .def_readwrite("time", &solitonkit::DynamicsRecord::time)
        .def_readwrite("energy", &solitonkit::DynamicsRecord::energy)
        .def_readwrite(
            "topological_charge",
            &solitonkit::DynamicsRecord::topological_charge
        )
        .def("__repr__", [](const solitonkit::DynamicsRecord& record) {
        return "DynamicsRecord(step=" +
            std::to_string(record.step) +
            ", time=" +
            std::to_string(record.time) +
            ", energy=" +
            std::to_string(record.energy) +
            ", topological_charge=" +
            std::to_string(record.topological_charge) +
            ")";
    });

    py::class_<solitonkit::SolverRecord>(m, "SolverRecord")
        .def_readwrite("step", &solitonkit::SolverRecord::step)
        .def_readwrite("time", &solitonkit::SolverRecord::time)
        .def_readwrite("energy", &solitonkit::SolverRecord::energy)
        .def_readwrite("gradient_norm", &solitonkit::SolverRecord::gradient_norm)
        .def_readwrite("converged", &solitonkit::SolverRecord::converged)
        .def("__repr__", [](const solitonkit::SolverRecord& record) {
        return "SolverRecord(step=" + std::to_string(record.step)
            + ", time=" + std::to_string(record.time)
            + ", energy=" + std::to_string(record.energy)
            + ", gradient_norm=" + std::to_string(record.gradient_norm)
            + ", converged=" + (record.converged ? "True" : "False") + ")";
    });

    py::class_<solitonkit::MinimizeOptions>(m, "MinimizeOptions")
        .def(py::init<>())
        .def_readwrite("max_steps", &solitonkit::MinimizeOptions::max_steps)
        .def_readwrite("step_size", &solitonkit::MinimizeOptions::step_size)
        .def_readwrite("tolerance", &solitonkit::MinimizeOptions::tolerance)
        .def_readwrite("record_every", &solitonkit::MinimizeOptions::record_every)
        .def_readwrite("line_search", &solitonkit::MinimizeOptions::line_search)
        .def_readwrite("min_step_size", &solitonkit::MinimizeOptions::min_step_size);

    py::class_<solitonkit::SolveOptions>(m, "SolveOptions")
        .def(py::init<>())
        .def_readwrite("steps", &solitonkit::SolveOptions::steps)
        .def_readwrite("time_step", &solitonkit::SolveOptions::time_step)
        .def_readwrite("record_every", &solitonkit::SolveOptions::record_every)
        .def_readwrite("tolerance", &solitonkit::SolveOptions::tolerance);

    py::class_<solitonkit::StabilityOptions>(m, "StabilityOptions")
        .def(py::init<>())
        .def_readwrite("modes", &solitonkit::StabilityOptions::modes)
        .def_readwrite(
            "max_iterations",
            &solitonkit::StabilityOptions::max_iterations
        )
        .def_readwrite(
            "subspace_dimension",
            &solitonkit::StabilityOptions::subspace_dimension
        )
        .def_readwrite("tolerance", &solitonkit::StabilityOptions::tolerance)
        .def_readwrite(
            "finite_difference_step",
            &solitonkit::StabilityOptions::finite_difference_step
        )
        .def_readwrite(
            "stationarity_tolerance",
            &solitonkit::StabilityOptions::stationarity_tolerance
        )
        .def_readwrite(
            "eigenvalue_tolerance",
            &solitonkit::StabilityOptions::eigenvalue_tolerance
        )
        .def_readwrite("seed", &solitonkit::StabilityOptions::seed);

    py::class_<solitonkit::GMRESOptions>(m, "GMRESOptions")
        .def(py::init<>())
        .def_readwrite("restart", &solitonkit::GMRESOptions::restart)
        .def_readwrite(
            "max_iterations",
            &solitonkit::GMRESOptions::max_iterations
        )
        .def_readwrite("tolerance", &solitonkit::GMRESOptions::tolerance);

    py::class_<solitonkit::StationaryOptions>(m, "StationaryOptions")
        .def(py::init<>())
        .def_readwrite("max_steps", &solitonkit::StationaryOptions::max_steps)
        .def_readwrite("tolerance", &solitonkit::StationaryOptions::tolerance)
        .def_readwrite(
            "finite_difference_step",
            &solitonkit::StationaryOptions::finite_difference_step
        )
        .def_readwrite(
            "initial_damping",
            &solitonkit::StationaryOptions::initial_damping
        )
        .def_readwrite(
            "minimum_damping",
            &solitonkit::StationaryOptions::minimum_damping
        )
        .def_readwrite(
            "trust_radius",
            &solitonkit::StationaryOptions::trust_radius
        )
        .def_readwrite("line_search", &solitonkit::StationaryOptions::line_search)
        .def_readwrite(
            "preconditioner_probes",
            &solitonkit::StationaryOptions::preconditioner_probes
        )
        .def_readwrite(
            "preconditioner_floor",
            &solitonkit::StationaryOptions::preconditioner_floor
        )
        .def_readwrite("seed", &solitonkit::StationaryOptions::seed)
        .def_readwrite("gmres", &solitonkit::StationaryOptions::gmres);

    py::class_<solitonkit::StationaryRecord>(m, "StationaryRecord")
        .def_readonly("step", &solitonkit::StationaryRecord::step)
        .def_readonly("energy", &solitonkit::StationaryRecord::energy)
        .def_readonly(
            "residual_norm",
            &solitonkit::StationaryRecord::residual_norm
        )
        .def_readonly("damping", &solitonkit::StationaryRecord::damping)
        .def_readonly(
            "linear_iterations",
            &solitonkit::StationaryRecord::linear_iterations
        )
        .def_readonly(
            "linear_residual",
            &solitonkit::StationaryRecord::linear_residual
        )
        .def_readonly(
            "linear_converged",
            &solitonkit::StationaryRecord::linear_converged
        )
        .def_readonly("converged", &solitonkit::StationaryRecord::converged);

    py::class_<solitonkit::ContinuationOptions>(m, "ContinuationOptions")
        .def(py::init<>())
        .def_readwrite("start", &solitonkit::ContinuationOptions::start)
        .def_readwrite("stop", &solitonkit::ContinuationOptions::stop)
        .def_readwrite("step", &solitonkit::ContinuationOptions::step)
        .def_readwrite(
            "minimum_step", &solitonkit::ContinuationOptions::minimum_step
        )
        .def_readwrite(
            "maximum_step", &solitonkit::ContinuationOptions::maximum_step
        )
        .def_readwrite(
            "step_growth", &solitonkit::ContinuationOptions::step_growth
        )
        .def_readwrite(
            "step_shrink", &solitonkit::ContinuationOptions::step_shrink
        )
        .def_readwrite(
            "parameter_scale", &solitonkit::ContinuationOptions::parameter_scale
        )
        .def_readwrite("max_points", &solitonkit::ContinuationOptions::max_points)
        .def_readwrite(
            "max_corrector_steps",
            &solitonkit::ContinuationOptions::max_corrector_steps
        )
        .def_readwrite(
            "target_corrector_steps",
            &solitonkit::ContinuationOptions::target_corrector_steps
        )
        .def_readwrite(
            "corrector_tolerance",
            &solitonkit::ContinuationOptions::corrector_tolerance
        )
        .def_readwrite(
            "finite_difference_step",
            &solitonkit::ContinuationOptions::finite_difference_step
        )
        .def_readwrite(
            "minimum_damping",
            &solitonkit::ContinuationOptions::minimum_damping
        )
        .def_readwrite(
            "trust_radius", &solitonkit::ContinuationOptions::trust_radius
        )
        .def_readwrite(
            "preconditioner_probes",
            &solitonkit::ContinuationOptions::preconditioner_probes
        )
        .def_readwrite(
            "preconditioner_floor",
            &solitonkit::ContinuationOptions::preconditioner_floor
        )
        .def_readwrite(
            "analyze_stability",
            &solitonkit::ContinuationOptions::analyze_stability
        )
        .def_readwrite(
            "bifurcation_tolerance",
            &solitonkit::ContinuationOptions::bifurcation_tolerance
        )
        .def_readwrite("seed", &solitonkit::ContinuationOptions::seed)
        .def_readwrite("gmres", &solitonkit::ContinuationOptions::gmres)
        .def_readwrite(
            "stationary", &solitonkit::ContinuationOptions::stationary
        )
        .def_readwrite("stability", &solitonkit::ContinuationOptions::stability);

    py::class_<solitonkit::topology::VortexDefect>(m, "VortexDefect")
        .def_readonly("x", &solitonkit::topology::VortexDefect::x)
        .def_readonly("y", &solitonkit::topology::VortexDefect::y)
        .def_readonly("charge", &solitonkit::topology::VortexDefect::charge)
        .def_readonly("i", &solitonkit::topology::VortexDefect::i)
        .def_readonly("j", &solitonkit::topology::VortexDefect::j);

    py::class_<solitonkit::topology::HopfChargeOptions>(m, "HopfChargeOptions")
        .def(py::init<>())
        .def_readwrite(
            "max_iterations",
            &solitonkit::topology::HopfChargeOptions::max_iterations
        )
        .def_readwrite(
            "tolerance", &solitonkit::topology::HopfChargeOptions::tolerance
        );

    py::class_<solitonkit::topology::HopfChargeResult>(m, "HopfChargeResult")
        .def_readonly("charge", &solitonkit::topology::HopfChargeResult::charge)
        .def_readonly(
            "poisson_residual",
            &solitonkit::topology::HopfChargeResult::poisson_residual
        )
        .def_readonly(
            "divergence_norm",
            &solitonkit::topology::HopfChargeResult::divergence_norm
        )
        .def_readonly(
            "iterations", &solitonkit::topology::HopfChargeResult::iterations
        )
        .def_readonly(
            "converged", &solitonkit::topology::HopfChargeResult::converged
        );

    py::class_<solitonkit::Phi4Model, solitonkit::Model>(m, "Phi4Model")
        .def(
            py::init<double, double>(),
            py::arg("lambda_") = 1.0,
            py::arg("vacuum") = 1.0
        )
        .def_property_readonly("lambda_", &solitonkit::Phi4Model::lambda)
        .def_property_readonly("vacuum", &solitonkit::Phi4Model::vacuum)
        .def("energy", &solitonkit::Phi4Model::energy)
        .def("energy_density_at", &solitonkit::Phi4Model::energy_density_at);

    py::class_<solitonkit::SineGordonModel, solitonkit::Model>(m, "SineGordonModel")
        .def(
            py::init<double, double>(),
            py::arg("mass") = 1.0,
            py::arg("beta") = 1.0
        )
        .def_property_readonly("mass", &solitonkit::SineGordonModel::mass)
        .def_property_readonly("beta", &solitonkit::SineGordonModel::beta)
        .def("energy", &solitonkit::SineGordonModel::energy)
        .def("energy_density_at", &solitonkit::SineGordonModel::energy_density_at);

    py::class_<solitonkit::XYModel, solitonkit::Model>(m, "XYModel")
        .def(
            py::init<double, double>(),
            py::arg("coupling") = 1.0,
            py::arg("field_strength") = 0.0
        )
        .def_property_readonly("coupling", &solitonkit::XYModel::coupling)
        .def_property_readonly("field_strength", &solitonkit::XYModel::field_strength)
        .def("energy", &solitonkit::XYModel::energy);

    py::class_<solitonkit::O3SigmaModel, solitonkit::Model>(m, "O3SigmaModel")
        .def(py::init<double>(), py::arg("coupling") = 1.0)
        .def_property_readonly("coupling", &solitonkit::O3SigmaModel::coupling)
        .def("energy", &solitonkit::O3SigmaModel::energy)
        .def("energy_density_at", &solitonkit::O3SigmaModel::energy_density_at);

    py::class_<solitonkit::BabySkyrmeModel, solitonkit::Model>(m, "BabySkyrmeModel")
        .def(
            py::init<double, double, double>(),
            py::arg("kappa") = 1.0,
            py::arg("mass") = 1.0,
            py::arg("dmi") = 0.0
        )
        .def_property_readonly("kappa", &solitonkit::BabySkyrmeModel::kappa)
        .def_property_readonly("mass", &solitonkit::BabySkyrmeModel::mass)
        .def_property_readonly("dmi", &solitonkit::BabySkyrmeModel::dmi)
        .def("energy", &solitonkit::BabySkyrmeModel::energy)
        .def("energy_terms", [](const solitonkit::BabySkyrmeModel& model, const solitonkit::O3Field& field) {
        const auto terms = model.energy_terms(field);
        py::dict result;
        result["sigma"] = terms.sigma;
        result["skyrme"] = terms.skyrme;
        result["potential"] = terms.potential;
        result["dmi"] = terms.dmi;
        result["total"] = terms.total();
        return result;
    });

    py::enum_<solitonkit::DMIType>(m, "DMIType")
        .value("None_", solitonkit::DMIType::None)
        .value("Bulk", solitonkit::DMIType::Bulk)
        .value("Interfacial", solitonkit::DMIType::Interfacial);

    py::class_<solitonkit::MicromagneticModel, solitonkit::Model>(
        m,
        "MicromagneticModel"
    )
        .def(
            py::init<
                double,
                double,
                double,
                const solitonkit::Vec3&,
                const solitonkit::Vec3&,
                solitonkit::DMIType
            >(),
            py::arg("exchange") = 1.0,
            py::arg("dmi") = 0.0,
            py::arg("anisotropy") = 0.0,
            py::arg("applied_field") = solitonkit::Vec3{},
            py::arg("easy_axis") = solitonkit::Vec3{ 0.0, 0.0, 1.0 },
            py::arg("dmi_type") = solitonkit::DMIType::Bulk
        )
        .def_property_readonly("exchange", &solitonkit::MicromagneticModel::exchange)
        .def_property_readonly("dmi", &solitonkit::MicromagneticModel::dmi)
        .def_property_readonly("anisotropy", &solitonkit::MicromagneticModel::anisotropy)
        .def_property_readonly("applied_field", &solitonkit::MicromagneticModel::applied_field)
        .def_property_readonly("easy_axis", &solitonkit::MicromagneticModel::easy_axis)
        .def_property_readonly("dmi_type", &solitonkit::MicromagneticModel::dmi_type)
        .def("energy", &solitonkit::MicromagneticModel::energy)
        .def("effective_field_at", &solitonkit::MicromagneticModel::effective_field_at)
        .def("energy_terms", [](const solitonkit::MicromagneticModel& model, const solitonkit::O3Field& field) {
        const auto terms = model.energy_terms(field);
        py::dict result;
        result["exchange"] = terms.exchange;
        result["dmi"] = terms.dmi;
        result["anisotropy"] = terms.anisotropy;
        result["zeeman"] = terms.zeeman;
        result["total"] = terms.total();
        return result;
    });

    py::class_<solitonkit::HopfionModel, solitonkit::Model>(m, "HopfionModel")
        .def(
            py::init<double, double, double>(),
            py::arg("coupling") = 1.0,
            py::arg("kappa") = 1.0,
            py::arg("mass") = 0.0
        )
        .def_property_readonly("coupling", &solitonkit::HopfionModel::coupling)
        .def_property_readonly("kappa", &solitonkit::HopfionModel::kappa)
        .def_property_readonly("mass", &solitonkit::HopfionModel::mass)
        .def("energy", &solitonkit::HopfionModel::energy)
        .def("energy_terms", [](const solitonkit::HopfionModel& model, const solitonkit::O3Field3D& field) {
        const auto terms = model.energy_terms(field);
        py::dict result;
        result["sigma"] = terms.sigma;
        result["skyrme"] = terms.skyrme;
        result["potential"] = terms.potential;
        result["total"] = terms.total();
        return result;
    });

    py::class_<solitonkit::HopfionSpec>(m, "HopfionSpec")
        .def(
            py::init<double, int, int>(),
            py::arg("scale") = 1.0,
            py::arg("winding_p") = 1,
            py::arg("winding_q") = 1
        )
        .def_readwrite("scale", &solitonkit::HopfionSpec::scale)
        .def_readwrite("winding_p", &solitonkit::HopfionSpec::winding_p)
        .def_readwrite("winding_q", &solitonkit::HopfionSpec::winding_q)
        .def_property_readonly("hopf_charge", &solitonkit::HopfionSpec::hopf_charge);

    py::class_<solitonkit::LLGDynamics>(m, "LLGDynamics")
        .def(
            py::init<double, double, double>(),
            py::arg("time_step"),
            py::arg("damping") = 0.0,
            py::arg("gyromagnetic_ratio") = 1.0
        )
        .def_property_readonly("time_step", &solitonkit::LLGDynamics::time_step)
        .def_property_readonly("damping", &solitonkit::LLGDynamics::damping)
        .def_property_readonly(
            "gyromagnetic_ratio",
            &solitonkit::LLGDynamics::gyromagnetic_ratio
        )
        .def("step", &solitonkit::LLGDynamics::step)
        .def("run", &solitonkit::LLGDynamics::run,
            py::arg("field"),
            py::arg("model"),
            py::arg("steps"),
            py::arg("record_every") = 1
        );

    py::class_<solitonkit::SkyrmionSpec>(m, "SkyrmionSpec")
        .def(
            py::init<double, double, int, double, double>(),
            py::arg("x0") = 0.0,
            py::arg("y0") = 0.0,
            py::arg("charge") = 1,
            py::arg("scale") = 1.0,
            py::arg("phase") = 0.0
        )
        .def_readwrite("x0", &solitonkit::SkyrmionSpec::x0)
        .def_readwrite("y0", &solitonkit::SkyrmionSpec::y0)
        .def_readwrite("charge", &solitonkit::SkyrmionSpec::charge)
        .def_readwrite("scale", &solitonkit::SkyrmionSpec::scale)
        .def_readwrite("phase", &solitonkit::SkyrmionSpec::phase)
        .def("__repr__", [](const solitonkit::SkyrmionSpec& spec) {
        return "SkyrmionSpec(x0=" +
            std::to_string(spec.x0) +
            ", y0=" +
            std::to_string(spec.y0) +
            ", charge=" +
            std::to_string(spec.charge) +
            ", scale=" +
            std::to_string(spec.scale) +
            ", phase=" +
            std::to_string(spec.phase) +
            ")";
    });

    py::class_<solitonkit::GradientFlow>(m, "GradientFlow")
        .def(py::init<double>())
        .def_property_readonly(
            "step_size",
            &solitonkit::GradientFlow::step_size
        )
        .def_static(
            "laplacian_at",
            &solitonkit::GradientFlow::laplacian_at
        );

    bind_solver_overloads<
        solitonkit::ScalarField2D,
        solitonkit::Phi4Model
    >(m);
    bind_solver_overloads<
        solitonkit::ScalarField2D,
        solitonkit::SineGordonModel
    >(m);
    bind_solver_overloads<solitonkit::XYField, solitonkit::XYModel>(m);
    bind_solver_overloads<solitonkit::O3Field, solitonkit::O3SigmaModel>(m);
    bind_solver_overloads<solitonkit::O3Field, solitonkit::BabySkyrmeModel>(m);
    bind_solver_overloads<
        solitonkit::O3Field,
        solitonkit::MicromagneticModel
    >(m);
    bind_solver_overloads<
        solitonkit::O3Field3D,
        solitonkit::HopfionModel
    >(m);

    bind_stationary_overloads<
        solitonkit::ScalarField2D,
        solitonkit::Phi4Model
    >(m);
    bind_stationary_overloads<
        solitonkit::ScalarField2D,
        solitonkit::SineGordonModel
    >(m);
    bind_stationary_overloads<solitonkit::XYField, solitonkit::XYModel>(m);
    bind_stationary_overloads<
        solitonkit::O3Field,
        solitonkit::O3SigmaModel
    >(m);
    bind_stationary_overloads<
        solitonkit::O3Field,
        solitonkit::BabySkyrmeModel
    >(m);
    bind_stationary_overloads<
        solitonkit::O3Field,
        solitonkit::MicromagneticModel
    >(m);
    bind_stationary_overloads<
        solitonkit::O3Field3D,
        solitonkit::HopfionModel
    >(m);

    bind_stability_overload<
        solitonkit::ScalarField2D,
        solitonkit::Phi4Model
    >(m);
    bind_stability_overload<
        solitonkit::ScalarField2D,
        solitonkit::SineGordonModel
    >(m);
    bind_stability_overload<solitonkit::XYField, solitonkit::XYModel>(m);
    bind_stability_overload<
        solitonkit::O3Field,
        solitonkit::O3SigmaModel
    >(m);
    bind_stability_overload<
        solitonkit::O3Field,
        solitonkit::BabySkyrmeModel
    >(m);
    bind_stability_overload<
        solitonkit::O3Field,
        solitonkit::MicromagneticModel
    >(m);
    bind_stability_overload<
        solitonkit::O3Field3D,
        solitonkit::HopfionModel
    >(m);

    bind_continuation_overload<
        solitonkit::ScalarField2D,
        solitonkit::Phi4Model
    >(m, "_continue_phi4");
    bind_continuation_overload<
        solitonkit::ScalarField2D,
        solitonkit::SineGordonModel
    >(m, "_continue_sine_gordon");
    bind_continuation_overload<
        solitonkit::XYField,
        solitonkit::XYModel
    >(m, "_continue_xy");
    bind_continuation_overload<
        solitonkit::O3Field,
        solitonkit::O3SigmaModel
    >(m, "_continue_o3_sigma");
    bind_continuation_overload<
        solitonkit::O3Field,
        solitonkit::BabySkyrmeModel
    >(m, "_continue_baby_skyrme");
    bind_continuation_overload<
        solitonkit::O3Field,
        solitonkit::MicromagneticModel
    >(m, "_continue_micromagnetic");
    bind_continuation_overload<
        solitonkit::O3Field3D,
        solitonkit::HopfionModel
    >(m, "_continue_hopfion");

    m.def(
        "gmres",
        &gmres_python,
        py::arg("apply"),
        py::arg("right_hand_side"),
        py::arg("options") = solitonkit::GMRESOptions{},
        py::arg("inverse_diagonal") = std::vector<double>{}
    );

    m.def("degree", &solitonkit::topology::degree, py::arg("field"));
    m.def(
        "detect_defects",
        &solitonkit::topology::detect_defects,
        py::arg("field"),
        py::arg("threshold") = 0.5
    );
    m.def(
        "vortex_number",
        &solitonkit::topology::vortex_number,
        py::arg("field")
    );
    m.def(
        "winding_number",
        &solitonkit::topology::winding_number,
        py::arg("field")
    );
    m.def(
        "hopf_charge",
        &solitonkit::topology::hopf_charge,
        py::arg("field"),
        py::arg("options") = solitonkit::topology::HopfChargeOptions{},
        py::call_guard<py::gil_scoped_release>()
    );

    m.def("derivative_x", [](
        const solitonkit::ScalarField2D& field,
        std::size_t i,
        std::size_t j
    ) {
        return solitonkit::differential::derivative_x(field, i, j);
    });
    m.def("derivative_y", [](
        const solitonkit::ScalarField2D& field,
        std::size_t i,
        std::size_t j
    ) {
        return solitonkit::differential::derivative_y(field, i, j);
    });
    m.def("laplacian", [](
        const solitonkit::ScalarField2D& field,
        std::size_t i,
        std::size_t j
    ) {
        return solitonkit::differential::laplacian(field, i, j);
    });
    m.def("gradient", [](
        const solitonkit::ScalarField2D& field,
        std::size_t i,
        std::size_t j
    ) {
        return solitonkit::differential::gradient(field, i, j);
    });
    m.def("derivative_x", [](
        const solitonkit::O3Field& field,
        std::size_t i,
        std::size_t j
    ) {
        return solitonkit::differential::derivative_x(field, i, j);
    });
    m.def("derivative_y", [](
        const solitonkit::O3Field& field,
        std::size_t i,
        std::size_t j
    ) {
        return solitonkit::differential::derivative_y(field, i, j);
    });
    m.def("laplacian", [](
        const solitonkit::O3Field& field,
        std::size_t i,
        std::size_t j
    ) {
        return solitonkit::differential::laplacian(field, i, j);
    });
    m.def("curl", [](
        const solitonkit::O3Field& field,
        std::size_t i,
        std::size_t j
    ) {
        return solitonkit::differential::curl(field, i, j);
    });
    m.def("derivative_x", [](
        const solitonkit::O3Field3D& field,
        std::size_t i,
        std::size_t j,
        std::size_t k
    ) {
        return solitonkit::differential::derivative_x(field, i, j, k);
    });
    m.def("derivative_y", [](
        const solitonkit::O3Field3D& field,
        std::size_t i,
        std::size_t j,
        std::size_t k
    ) {
        return solitonkit::differential::derivative_y(field, i, j, k);
    });
    m.def("derivative_z", [](
        const solitonkit::O3Field3D& field,
        std::size_t i,
        std::size_t j,
        std::size_t k
    ) {
        return solitonkit::differential::derivative_z(field, i, j, k);
    });
    m.def("laplacian", [](
        const solitonkit::O3Field3D& field,
        std::size_t i,
        std::size_t j,
        std::size_t k
    ) {
        return solitonkit::differential::laplacian(field, i, j, k);
    });
    m.def("curl", [](
        const solitonkit::O3Field3D& field,
        std::size_t i,
        std::size_t j,
        std::size_t k
    ) {
        return solitonkit::differential::curl(field, i, j, k);
    });

    m.def("make_hopfion_field", [](
        std::size_t nx,
        std::size_t ny,
        std::size_t nz,
        double spacing,
        double scale,
        int winding_p,
        int winding_q,
        const std::string& boundary
    ) {
        const solitonkit::Lattice3D lattice{
            nx,
            ny,
            nz,
            spacing,
            spacing,
            spacing,
            parse_boundary_condition(boundary)
        };
        return solitonkit::HopfionAnsatz::create(
            lattice,
            solitonkit::HopfionSpec{ scale, winding_p, winding_q }
        );
    },
        py::arg("nx"),
        py::arg("ny"),
        py::arg("nz"),
        py::arg("spacing") = 1.0,
        py::arg("scale") = 3.0,
        py::arg("winding_p") = 1,
        py::arg("winding_q") = 1,
        py::arg("boundary") = "dirichlet"
    );

    m.def("run_llg", [](
        const solitonkit::O3Field& input,
        const solitonkit::MicromagneticModel& model,
        double time_step,
        double damping,
        double gyromagnetic_ratio,
        std::size_t steps,
        std::size_t record_every
    ) {
        solitonkit::O3Field field = input;
        solitonkit::LLGDynamics dynamics{
            time_step,
            damping,
            gyromagnetic_ratio
        };
        const auto history = dynamics.run(field, model, steps, record_every);
        return std::make_tuple(field, history);
    },
        py::arg("field"),
        py::arg("model"),
        py::arg("time_step") = 1e-3,
        py::arg("damping") = 0.1,
        py::arg("gyromagnetic_ratio") = 1.0,
        py::arg("steps") = 1000,
        py::arg("record_every") = 10
    );

    m.def(
        "make_uniform_field",
        [](
            std::size_t nx,
            std::size_t ny,
            double dx,
            double dy,
            double x,
            double y,
            double z,
            const std::string& boundary
            ) {
        return make_uniform_field(
            nx,
            ny,
            dx,
            dy,
            x,
            y,
            z,
            parse_boundary_condition(boundary)
        );
    },
        py::arg("nx"),
        py::arg("ny"),
        py::arg("dx") = 1.0,
        py::arg("dy") = 1.0,
        py::arg("x") = 0.0,
        py::arg("y") = 0.0,
        py::arg("z") = 1.0,
        py::arg("boundary") = "periodic"
    );

    m.def(
        "make_skyrmion_field",
        [](
            std::size_t nx,
            std::size_t ny,
            double dx,
            double dy,
            double radius,
            int charge,
            const std::string& boundary
            ) {
        return make_skyrmion_field(
            nx,
            ny,
            dx,
            dy,
            radius,
            charge,
            parse_boundary_condition(boundary)
        );
    },
        py::arg("nx"),
        py::arg("ny"),
        py::arg("dx") = 1.0,
        py::arg("dy") = 1.0,
        py::arg("radius") = 20.0,
        py::arg("charge") = 1,
        py::arg("boundary") = "periodic"
    );

    m.def(
        "make_multi_skyrmion_field",
        [](
            std::size_t nx,
            std::size_t ny,
            const std::vector<solitonkit::SkyrmionSpec>& specs,
            double dx,
            double dy,
            const std::string& boundary
            ) {
        return make_multi_skyrmion_field(
            nx,
            ny,
                dx,
                dy,
                specs,
                parse_boundary_condition(boundary)
        );
    },
        py::arg("nx"),
        py::arg("ny"),
        py::arg("specs"),
        py::arg("dx") = 1.0,
        py::arg("dy") = 1.0,
        py::arg("boundary") = "periodic"
    );

    m.def(
        "make_skyrmion",
        &make_skyrmion_numpy,
        py::arg("width"),
        py::arg("height"),
        py::arg("radius") = 20.0
    );

    m.def(
        "make_skyrmion_at",
        &make_skyrmion_at_numpy,
        py::arg("width"),
        py::arg("height"),
        py::arg("radius"),
        py::arg("center_x"),
        py::arg("center_y")
    );

    m.def(
        "field_to_numpy",
        &field_to_numpy,
        py::arg("field")
    );

    m.def(
        "field_from_numpy",
        &field_from_numpy,
        py::arg("array"),
        py::arg("dx") = 1.0,
        py::arg("dy") = 1.0,
        py::arg("boundary") = "periodic"
    );

    m.def(
        "field_from_numpy_with_boundaries",
        &field_from_numpy_with_boundaries,
        py::arg("array"),
        py::arg("dx") = 1.0,
        py::arg("dy") = 1.0,
        py::arg("boundary_x") = "periodic",
        py::arg("boundary_y") = "periodic"
    );

    m.def(
        "field3d_from_numpy",
        &field3d_from_numpy,
        py::arg("array"),
        py::arg("dx") = 1.0,
        py::arg("dy") = 1.0,
        py::arg("dz") = 1.0,
        py::arg("boundary_x") = "periodic",
        py::arg("boundary_y") = "periodic",
        py::arg("boundary_z") = "periodic"
    );

    m.def(
        "energy_density",
        &energy_density,
        py::arg("field")
    );

    m.def(
        "total_energy",
        &total_energy,
        py::arg("field")
    );

    m.def(
        "topological_density",
        &topological_density,
        py::arg("field")
    );

    m.def(
        "topological_charge",
        &topological_charge,
        py::arg("field")
    );

    m.def(
        "run_gradient_flow_inplace",
        &run_gradient_flow_inplace,
        py::arg("field"),
        py::arg("step_size"),
        py::arg("steps"),
        py::arg("record_every") = 10
    );

    m.def(
        "run_gradient_flow",
        &run_gradient_flow,
        py::arg("field"),
        py::arg("step_size"),
        py::arg("steps"),
        py::arg("record_every") = 10
    );

    m.def(
        "baby_skyrme_energy",
        [](const solitonkit::O3Field& field, double kappa, double mass, double dmi) {
        const solitonkit::BabySkyrmeModel model(kappa, mass, dmi);
        return model.energy(field);
    },
        py::arg("field"),
        py::arg("kappa") = 1.0,
        py::arg("mass") = 1.0,
        py::arg("dmi") = 0.0
    );

    m.def(
        "baby_skyrme_energy_terms",
        [](const solitonkit::O3Field& field, double kappa, double mass, double dmi) {
        const solitonkit::BabySkyrmeModel model(kappa, mass, dmi);
        const solitonkit::BabySkyrmeEnergyTerms terms = model.energy_terms(field);

        py::dict result;
        result["sigma"] = terms.sigma;
        result["skyrme"] = terms.skyrme;
        result["potential"] = terms.potential;
        result["dmi"] = terms.dmi;
        result["total"] = terms.total();

        return result;
    },
        py::arg("field"),
        py::arg("kappa") = 1.0,
        py::arg("mass") = 1.0,
        py::arg("dmi") = 0.0
    );

    m.def(
        "run_baby_skyrme_gradient_flow_inplace",
        &run_baby_skyrme_gradient_flow_inplace,
        py::arg("field"),
        py::arg("kappa") = 1.0,
        py::arg("mass") = 1.0,
        py::arg("step_size") = 1e-4,
        py::arg("steps") = 1000,
        py::arg("record_every") = 10,
        py::arg("dmi") = 0.0
    );

    m.def(
        "run_baby_skyrme_gradient_flow",
        &run_baby_skyrme_gradient_flow,
        py::arg("field"),
        py::arg("kappa") = 1.0,
        py::arg("mass") = 1.0,
        py::arg("step_size") = 1e-4,
        py::arg("steps") = 1000,
        py::arg("record_every") = 10,
        py::arg("dmi") = 0.0
    );

    m.def(
        "run_baby_skyrme_riemannian_gradient_descent_inplace",
        &run_baby_skyrme_riemannian_gradient_descent_inplace,
        py::arg("field"),
        py::arg("kappa") = 1.0,
        py::arg("mass") = 1.0,
        py::arg("step_size") = 1e-4,
        py::arg("steps") = 1000,
        py::arg("record_every") = 10,
        py::arg("dmi") = 0.0
    );

    m.def(
        "run_baby_skyrme_riemannian_gradient_descent",
        &run_baby_skyrme_riemannian_gradient_descent,
        py::arg("field"),
        py::arg("kappa") = 1.0,
        py::arg("mass") = 1.0,
        py::arg("step_size") = 1e-4,
        py::arg("steps") = 1000,
        py::arg("record_every") = 10,
        py::arg("dmi") = 0.0
    );

    m.def(
        "run_baby_skyrme_barzilai_borwein_inplace",
        &run_baby_skyrme_barzilai_borwein_inplace,
        py::arg("field"),
        py::arg("kappa") = 1.0,
        py::arg("mass") = 1.0,
        py::arg("initial_step_size") = 1e-4,
        py::arg("min_step_size") = 1e-8,
        py::arg("max_step_size") = 1e-2,
        py::arg("max_line_search_steps") = 12,
        py::arg("steps") = 1000,
        py::arg("record_every") = 10,
        py::arg("dmi") = 0.0
    );

    m.def(
        "run_baby_skyrme_barzilai_borwein",
        &run_baby_skyrme_barzilai_borwein,
        py::arg("field"),
        py::arg("kappa") = 1.0,
        py::arg("mass") = 1.0,
        py::arg("initial_step_size") = 1e-4,
        py::arg("min_step_size") = 1e-8,
        py::arg("max_step_size") = 1e-2,
        py::arg("max_line_search_steps") = 12,
        py::arg("steps") = 1000,
        py::arg("record_every") = 10,
        py::arg("dmi") = 0.0
    );

    m.def(
        "run_baby_skyrme_lbfgs_inplace",
        &run_baby_skyrme_lbfgs_inplace,
        py::arg("field"),
        py::arg("kappa") = 1.0,
        py::arg("mass") = 1.0,
        py::arg("initial_step_size") = 1.0,
        py::arg("memory") = 5,
        py::arg("max_line_search_steps") = 12,
        py::arg("steps") = 1000,
        py::arg("record_every") = 10,
        py::arg("dmi") = 0.0
    );

    m.def(
        "run_baby_skyrme_lbfgs",
        &run_baby_skyrme_lbfgs,
        py::arg("field"),
        py::arg("kappa") = 1.0,
        py::arg("mass") = 1.0,
        py::arg("initial_step_size") = 1.0,
        py::arg("memory") = 5,
        py::arg("max_line_search_steps") = 12,
        py::arg("steps") = 1000,
        py::arg("record_every") = 10,
        py::arg("dmi") = 0.0
    );

    m.def(
        "run_baby_skyrme_semi_implicit_flow_inplace",
        &run_baby_skyrme_semi_implicit_flow_inplace,
        py::arg("field"),
        py::arg("kappa") = 1.0,
        py::arg("mass") = 1.0,
        py::arg("step_size") = 1e-3,
        py::arg("implicit_iterations") = 20,
        py::arg("steps") = 1000,
        py::arg("record_every") = 10,
        py::arg("dmi") = 0.0
    );

    m.def(
        "run_baby_skyrme_semi_implicit_flow",
        &run_baby_skyrme_semi_implicit_flow,
        py::arg("field"),
        py::arg("kappa") = 1.0,
        py::arg("mass") = 1.0,
        py::arg("step_size") = 1e-3,
        py::arg("implicit_iterations") = 20,
        py::arg("steps") = 1000,
        py::arg("record_every") = 10,
        py::arg("dmi") = 0.0
    );

    m.def(
        "run_landau_lifshitz_inplace",
        &run_landau_lifshitz_inplace,
        py::arg("field"),
        py::arg("kappa") = 1.0,
        py::arg("mass") = 1.0,
        py::arg("time_step") = 1e-5,
        py::arg("damping") = 0.0,
        py::arg("steps") = 1000,
        py::arg("record_every") = 10,
        py::arg("dmi") = 0.0
    );

    m.def(
        "run_landau_lifshitz",
        &run_landau_lifshitz,
        py::arg("field"),
        py::arg("kappa") = 1.0,
        py::arg("mass") = 1.0,
        py::arg("time_step") = 1e-5,
        py::arg("damping") = 0.0,
        py::arg("steps") = 1000,
        py::arg("record_every") = 10,
        py::arg("dmi") = 0.0
    );

    m.def(
        "topological_charge_geometric",
        [](const solitonkit::O3Field& field) {
        return solitonkit::GeometricTopologicalCharge::compute(field);
    },
        py::arg("field")
    );

    m.def(
        "openmp_enabled",
        []() {
#ifdef SOLITONKIT_USE_OPENMP
        return true;
#else
        return false;
#endif
    }
    );

    m.def(
        "openmp_max_threads",
        []() {
#ifdef SOLITONKIT_USE_OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }
    );
}
