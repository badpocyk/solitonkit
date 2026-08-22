#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace solitonkit {

    enum class FieldKind {
        Scalar2D,
        XY2D,
        O3_2D,
        O3_3D
    };

    class Model {
    public:
        virtual ~Model() = default;
        virtual std::string name() const = 0;
        virtual std::size_t dimensions() const = 0;
        virtual FieldKind field_kind() const = 0;
    };

    template <typename Field, typename Value>
    class DifferentiableModel : public Model {
    public:
        virtual double energy(const Field& field) const = 0;
        virtual std::vector<Value> negative_gradient(const Field& field) const = 0;
    };

} // namespace solitonkit
