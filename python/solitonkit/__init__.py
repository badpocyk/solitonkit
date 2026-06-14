# python/solitonkit/__init__.py

from __future__ import annotations

"""
solitonkit

Python interface for visualizing and running C++ soliton field simulations.
"""

__version__ = "0.1.0"


# ---------------------------------------------------------------------
# Core Python API over the C++ extension
# ---------------------------------------------------------------------

from .core import (
    HAS_CPP_CORE,
    HAS_CPP_BACKEND,
    CPP_BACKEND_AVAILABLE,
    CPP_CORE_AVAILABLE,
    BACKEND,
    SkyrmionConfig,
    Vec3,
    O3Field,
    Field2D,
    FlowRecord,
    DynamicsRecord,
    GradientFlow,
    BoundaryCondition,
    require_cpp_core,
    core_info,
    backend_name,
    make_skyrmion,
    make_skyrmion_at,
    make_skyrmion_from_config,
    make_skyrmion_field,
    make_skyrmion_field_xy,
    make_uniform_field,
    make_field2d,
    field_from_numpy,
    field_to_numpy,
    field2d_to_numpy,
    to_numpy,
    energy_density,
    total_energy,
    topological_density,
    topological_charge,
    topological_charge_geometric,
    run_gradient_flow,
    run_gradient_flow_inplace,
    gradient_flow,
    gradient_flow_description,
    make_multi_skyrmion_field,
    SkyrmionSpec,
    baby_skyrme_energy,
    baby_skyrme_energy_terms,
    run_baby_skyrme_gradient_flow,
    run_baby_skyrme_gradient_flow_inplace,
    run_landau_lifshitz,
    run_landau_lifshitz_inplace,
    openmp_enabled,
    openmp_max_threads,
)

from .adaptive import (
    AdaptiveFlowRecord,
    run_adaptive_gradient_flow,
)

# ---------------------------------------------------------------------
# Runner API
# ---------------------------------------------------------------------

from .runner import (
    SimulationConfig,
    SimulationResult,
    run_skyrmion_relaxation,
    run_and_save_skyrmion_relaxation,
    save_simulation_result,
    save_records_csv,
    save_field_csv,
    save_summary_txt,
    flow_record_to_dict,
    flow_records_to_dicts,
    flow_records_to_numpy,
    dynamics_record_to_dict,
    save_dynamics_records_csv,
    print_records,
)


# ---------------------------------------------------------------------
# Visualization API
# ---------------------------------------------------------------------

from .viz import (
    ensure_field,
    normalize_field,
    field_to_rgb,
    plot_component,
    plot_rgb,
    plot_quiver,
    plot_density,
    plot_skyrmion,
    plot_surface,
    save_figure,
    save_skyrmion_plot,
    show,
    show_skyrmion_plot,
    show_field_components,
    show_energy_density,
    show_topological_density,
    show_skyrmion_diagnostics,
    save_skyrmion_diagnostics,
)

# ---------------------------------------------------------------------
# Datasets API
# ---------------------------------------------------------------------

from .datasets import (
    generate_skyrmion_dataset,
    load_skyrmion_dataset,
)

# ---------------------------------------------------------------------
# Field I/O and animations
# ---------------------------------------------------------------------

from .io import (
    FIELD_FORMAT_VERSION,
    SUPPORTED_FIELD_FORMAT_VERSIONS,
    BOUNDARY_CONDITIONS,
    save_field_npz,
    load_field_npz,
)

from .animation import (
    AnimationMode,
    FlowSnapshot,
    run_baby_skyrme_gradient_flow_snapshots,
    save_field_animation,
    save_flow_animation,
)

__all__ = [
    "__version__",

    # Backend flags
    "HAS_CPP_CORE",
    "HAS_CPP_BACKEND",
    "CPP_BACKEND_AVAILABLE",
    "CPP_CORE_AVAILABLE",
    "BACKEND",

    # Core classes
    "SkyrmionConfig",
    "Vec3",
    "O3Field",
    "Field2D",
    "FlowRecord",
    "DynamicsRecord",
    "GradientFlow",
    "BoundaryCondition",

    # Core backend helpers
    "require_cpp_core",
    "core_info",
    "backend_name",
    "gradient_flow_description",

    # Field creation
    "make_skyrmion",
    "make_skyrmion_at",
    "make_skyrmion_from_config",
    "make_skyrmion_field",
    "make_skyrmion_field_xy",
    "make_uniform_field",
    "make_field2d",

    # Conversion
    "field_from_numpy",
    "field_to_numpy",
    "field2d_to_numpy",
    "to_numpy",

    # Observables
    "energy_density",
    "total_energy",
    "topological_density",
    "topological_charge",

    # Gradient flow
    "run_gradient_flow",
    "run_gradient_flow_inplace",
    "gradient_flow",
    "AdaptiveFlowRecord",
    "run_adaptive_gradient_flow",
    "run_baby_skyrme_gradient_flow",
    "run_baby_skyrme_gradient_flow_inplace",
    "run_landau_lifshitz",
    "run_landau_lifshitz_inplace",

    # Runner
    "SimulationConfig",
    "SimulationResult",
    "run_skyrmion_relaxation",
    "run_and_save_skyrmion_relaxation",
    "save_simulation_result",
    "save_records_csv",
    "save_field_csv",
    "save_summary_txt",
    "flow_record_to_dict",
    "flow_records_to_dicts",
    "flow_records_to_numpy",
    "dynamics_record_to_dict",
    "save_dynamics_records_csv",
    "print_records",

    # Visualization
    "ensure_field",
    "normalize_field",
    "field_to_rgb",
    "plot_component",
    "plot_rgb",
    "plot_quiver",
    "plot_density",
    "plot_skyrmion",
    "plot_surface",
    "save_figure",
    "save_skyrmion_plot",
    "show",

    # Datasets
    "generate_skyrmion_dataset",
    "load_skyrmion_dataset",

    # Field I/O
    "FIELD_FORMAT_VERSION",
    "SUPPORTED_FIELD_FORMAT_VERSIONS",
    "BOUNDARY_CONDITIONS",
    "save_field_npz",
    "load_field_npz",

    # Animations
    "AnimationMode",
    "FlowSnapshot",
    "run_baby_skyrme_gradient_flow_snapshots",
    "save_field_animation",
    "save_flow_animation",

    # Baby Skyrme model
    "baby_skyrme_energy",
    "baby_skyrme_energy_terms",

    # OpenMP
    "openmp_enabled",
    "openmp_max_threads",
]
