from pathlib import Path

import solitonkit as sk
from solitonkit.core import (
    make_skyrmion_field,
    field_to_numpy,
    total_energy,
    topological_charge,
)

Path("outputs").mkdir(exist_ok=True)

# Real C++ O3Field
field_cpp = make_skyrmion_field(
    nx=128,
    ny=128,
    spacing=0.2,
    radius=4.0,
    charge=1,
)

# Convert to NumPy for visualization
field = field_to_numpy(field_cpp)

print("HAS_CPP_CORE:", sk.HAS_CPP_CORE)
print("C++ field:", field_cpp)
print("field shape:", field.shape)
print("energy:", total_energy(field_cpp))
print("topological charge:", topological_charge(field_cpp))

sk.save_skyrmion_plot(
    field,
    "outputs/skyrmion_from_cpp.png",
    quiver_step=6,
)

sk.plot_surface(field, component="z")
sk.show()