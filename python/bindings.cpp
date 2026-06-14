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

#include "solitonkit/core/Lattice2D.hpp"
#include "solitonkit/core/Vec3.hpp"
#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/dynamics/LandauLifshitzDynamics.hpp"
#include "solitonkit/flows/BabySkyrmeGradientFlow.hpp"
#include "solitonkit/flows/GradientFlow.hpp"
#include "solitonkit/initializers/SkyrmionAnsatz.hpp"
#include "solitonkit/models/BabySkyrmeModel.hpp"
#include "solitonkit/observables/GeometricTopologicalCharge.hpp"

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

        throw std::invalid_argument(
            "boundary must be 'periodic', 'fixed', or 'neumann'"
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

    solitonkit::Vec3 sk_sub(const solitonkit::Vec3& a, const solitonkit::Vec3& b) {
        return solitonkit::Vec3{
            a.x - b.x,
            a.y - b.y,
            a.z - b.z
        };
    }

    solitonkit::Vec3 sk_mul(const solitonkit::Vec3& v, double s) {
        return solitonkit::Vec3{
            v.x * s,
            v.y * s,
            v.z * s
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

        return solitonkit::SkyrmionAnsatz::multi_skyrmion(
            lattice,
            specs
        );
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

        return field;
    }

    solitonkit::Vec3 derivative_x(
        const solitonkit::O3Field& field,
        std::size_t i,
        std::size_t j
    ) {
        const auto& lattice = field.lattice();

        const std::size_t ip = lattice.right(i);
        const std::size_t im = lattice.left(i);
        const double dx = lattice.dx();

        return sk_mul(
            sk_sub(field(ip, j), field(im, j)),
            1.0 / (2.0 * dx)
        );
    }

    solitonkit::Vec3 derivative_y(
        const solitonkit::O3Field& field,
        std::size_t i,
        std::size_t j
    ) {
        const auto& lattice = field.lattice();

        const std::size_t jp = lattice.up(j);
        const std::size_t jm = lattice.down(j);
        const double dy = lattice.dy();

        return sk_mul(
            sk_sub(field(i, jp), field(i, jm)),
            1.0 / (2.0 * dy)
        );
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
        std::size_t record_every
    ) {
        const solitonkit::BabySkyrmeModel model(kappa, mass);
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
            std::size_t record_every
        ) {
        solitonkit::O3Field field = input;

        std::vector<solitonkit::FlowRecord> records =
            run_baby_skyrme_gradient_flow_inplace(
                field,
                kappa,
                mass,
                step_size,
                steps,
                record_every
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
        std::size_t record_every
    ) {
        const solitonkit::BabySkyrmeModel model(kappa, mass);
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
            std::size_t record_every
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
                record_every
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

} // namespace solitonkit_binding


PYBIND11_MODULE(_core, m) {
    using namespace solitonkit_binding;

    m.doc() = "C++ backend for solitonkit";

    py::enum_<solitonkit::BoundaryCondition>(m, "BoundaryCondition")
        .value("Periodic", solitonkit::BoundaryCondition::Periodic)
        .value("Fixed", solitonkit::BoundaryCondition::Fixed)
        .value("Neumann", solitonkit::BoundaryCondition::Neumann);

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
        .def("get", [](const solitonkit::O3Field& field, std::size_t i, std::size_t j) {
        return field(i, j);
    })
        .def("set", [](solitonkit::O3Field& field, std::size_t i, std::size_t j, const solitonkit::Vec3& value) {
        field(i, j) = sk_normalized(value);
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
        [](const solitonkit::O3Field& field, double kappa, double mass) {
        const solitonkit::BabySkyrmeModel model(kappa, mass);
        return model.energy(field);
    },
        py::arg("field"),
        py::arg("kappa") = 1.0,
        py::arg("mass") = 1.0
    );

    m.def(
        "baby_skyrme_energy_terms",
        [](const solitonkit::O3Field& field, double kappa, double mass) {
        const solitonkit::BabySkyrmeModel model(kappa, mass);
        const solitonkit::BabySkyrmeEnergyTerms terms = model.energy_terms(field);

        py::dict result;
        result["sigma"] = terms.sigma;
        result["skyrme"] = terms.skyrme;
        result["potential"] = terms.potential;
        result["total"] = terms.total();

        return result;
    },
        py::arg("field"),
        py::arg("kappa") = 1.0,
        py::arg("mass") = 1.0
    );

    m.def(
        "run_baby_skyrme_gradient_flow_inplace",
        &run_baby_skyrme_gradient_flow_inplace,
        py::arg("field"),
        py::arg("kappa") = 1.0,
        py::arg("mass") = 1.0,
        py::arg("step_size") = 1e-4,
        py::arg("steps") = 1000,
        py::arg("record_every") = 10
    );

    m.def(
        "run_baby_skyrme_gradient_flow",
        &run_baby_skyrme_gradient_flow,
        py::arg("field"),
        py::arg("kappa") = 1.0,
        py::arg("mass") = 1.0,
        py::arg("step_size") = 1e-4,
        py::arg("steps") = 1000,
        py::arg("record_every") = 10
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
        py::arg("record_every") = 10
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
        py::arg("record_every") = 10
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
