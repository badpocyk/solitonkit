from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import solitonkit as sk


class ExtendedPythonAPITests(unittest.TestCase):
    def test_gmres_stationary_and_continuation(self) -> None:
        linear = sk.gmres(
            lambda value: np.asarray([2.0 * value[0], 4.0 * value[1]]),
            [2.0, 8.0],
            tolerance=1e-12,
        )
        self.assertTrue(linear.converged)
        np.testing.assert_allclose(linear.solution, [1.0, 2.0], atol=1e-10)

        field = sk.ScalarField2D(4, 4, value=0.8)
        model = sk.Phi4Model(lambda_=1.0, vacuum=1.0)
        stationary, records = sk.solve_stationary(
            field,
            model,
            max_steps=10,
            tolerance=1e-9,
            preconditioner_probes=2,
            gmres_tolerance=1e-9,
        )
        self.assertTrue(records[-1].converged)
        np.testing.assert_allclose(stationary.to_numpy(), 1.0, atol=1e-8)

        branch = sk.continue_solution(
            sk.ScalarField2D(3, 3, value=0.8),
            lambda vacuum: sk.Phi4Model(lambda_=1.0, vacuum=vacuum),
            start=0.8,
            stop=1.05,
            step=0.1,
            parameter_name="vacuum",
            analyze_stability=False,
            stationary_tolerance=1e-9,
        )
        self.assertTrue(branch.reached_stop)
        self.assertGreaterEqual(len(branch.points), 3)
        self.assertEqual(branch.parameter_name, "vacuum")
        for point in branch.points:
            self.assertAlmostEqual(
                point.field.get(0, 0), point.parameter, places=6
            )

    def test_topology_sweeps_storage_and_benchmarks(self) -> None:
        vortex = sk.XYField(
            6, 6, boundary_x="neumann", boundary_y="neumann"
        )
        for j in range(6):
            for i in range(6):
                vortex.set(i, j, np.arctan2(j - 2.5, i - 2.5))
        defects = sk.detect_defects(vortex)
        self.assertEqual(len(defects), 1)
        self.assertEqual(defects[0].charge, 1)
        self.assertEqual(sk.vortex_number(vortex), 1)

        uniform3d = sk.O3Field3D(5, 5, 5)
        hopf = sk.hopf_charge(uniform3d, return_diagnostics=True)
        self.assertTrue(hopf.converged)
        self.assertAlmostEqual(hopf.charge, 0.0)

        initial = sk.ScalarField2D(3, 3, value=0.8)
        diagram = sk.phase_diagram(
            initial,
            lambda lambda_, vacuum: sk.Phi4Model(
                lambda_=lambda_, vacuum=vacuum
            ),
            {"lambda_": [0.8, 1.0], "vacuum": [0.9, 1.0]},
            workers=2,
            solver_kwargs={
                "max_steps": 10,
                "tolerance": 1e-8,
                "preconditioner_probes": 1,
            },
            keep_fields=True,
        )
        self.assertEqual(diagram.shape, (2, 2))
        self.assertEqual(diagram.energy_grid.shape, (2, 2))
        self.assertTrue(all(point.converged for point in diagram.points))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.h5"
            sk.save_run(diagram, path, metadata={"purpose": "test"})
            stored = sk.resume(path)
            self.assertEqual(stored.kind, "sweep")
            self.assertEqual(stored.metadata["purpose"], "test")
            self.assertEqual(stored.latest_field.to_numpy().shape, (3, 3))

            pinned = sk.ScalarField2D(
                4,
                4,
                value=1.0,
                dirichlet_value=1.0,
                boundary_x="dirichlet",
                boundary_y="dirichlet",
            )
            checkpoint_path = Path(directory) / "checkpoint.h5"
            sk.save_checkpoint(pinned, checkpoint_path, step=7)
            checkpoint = sk.resume(checkpoint_path)
            self.assertEqual(checkpoint.metadata["checkpoint_step"], 7)
            self.assertAlmostEqual(
                checkpoint.latest_field.dirichlet_value, 1.0
            )
            np.testing.assert_allclose(
                checkpoint.latest_field.to_numpy(), 1.0
            )

        benchmark = sk.run_benchmarks(sizes=(4,), repeats=1)
        self.assertEqual(len(benchmark.entries), 3)
        self.assertIn("sites/s", benchmark.format_table())

    def test_scalar_models_and_solver(self) -> None:
        field = sk.ScalarField2D(
            12,
            10,
            dx=0.5,
            dy=0.5,
            value=1.0,
            dirichlet_value=1.0,
            boundary_x="dirichlet",
            boundary_y="dirichlet",
        )
        field.set(6, 5, 0.0)
        model = sk.Phi4Model(lambda_=1.0, vacuum=1.0)
        before = model.energy(field)

        relaxed, history = sk.minimize(
            field,
            model,
            max_steps=6,
            step_size=0.05,
            record_every=1,
        )

        self.assertLess(model.energy(relaxed), before)
        self.assertEqual(model.name, "phi4")
        self.assertGreaterEqual(len(history), 2)
        self.assertEqual(relaxed.to_numpy().shape, (10, 12))

        sine = sk.SineGordonModel()
        vacuum = sk.ScalarField2D(6, 6)
        self.assertAlmostEqual(sine.energy(vacuum), 0.0)

    def test_xy_model(self) -> None:
        field = sk.XYField(8, 8, angle=0.0)
        model = sk.XYModel(coupling=1.0)
        field.set(4, 4, 1.0)
        before = model.energy(field)
        evolved, records = sk.solve(
            field,
            model,
            steps=5,
            time_step=0.02,
            record_every=1,
        )
        self.assertLess(model.energy(evolved), before)
        self.assertEqual(len(records), 6)

    def test_micromagnetics_and_llg(self) -> None:
        field = sk.O3Field(10, 10)
        field.set(5, 5, sk.Vec3(1.0, 0.0, 0.0))
        model = sk.MicromagneticModel(
            exchange=1.0,
            dmi=0.1,
            anisotropy=0.2,
            applied_field=sk.Vec3(0.0, 0.0, 0.5),
            dmi_type=sk.DMIType.Interfacial,
        )
        before = model.energy(field)
        evolved, records = sk.run_llg(
            field,
            model,
            time_step=1e-3,
            damping=0.5,
            steps=5,
            record_every=1,
        )
        self.assertLess(model.energy(evolved), before)
        self.assertEqual(len(records), 6)
        norms = np.linalg.norm(evolved.to_numpy(), axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_hopfion_field(self) -> None:
        field = sk.make_hopfion_field(
            9,
            9,
            9,
            spacing=0.5,
            scale=1.5,
            winding_p=1,
            winding_q=1,
        )
        values = field.to_numpy()
        self.assertEqual(values.shape, (9, 9, 9, 3))
        np.testing.assert_allclose(
            np.linalg.norm(values, axis=-1),
            1.0,
            atol=1e-12,
        )
        model = sk.HopfionModel(coupling=1.0, kappa=0.25, mass=0.1)
        terms = model.energy_terms(field)
        self.assertGreater(terms["sigma"], 0.0)
        self.assertGreater(terms["skyrme"], 0.0)
        self.assertGreater(terms["potential"], 0.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hopfion.npz"
            sk.save_field_npz(field, path, metadata={"charge": 1})
            loaded, metadata = sk.load_field_npz(path, return_metadata=True)
            self.assertIsInstance(loaded, sk.O3Field3D)
            self.assertEqual(loaded.boundary_z, "dirichlet")
            self.assertAlmostEqual(loaded.dz, 0.5)
            self.assertEqual(metadata["charge"], 1)
            np.testing.assert_allclose(loaded.to_numpy(), values, atol=1e-12)

    def test_mixed_boundaries_survive_io(self) -> None:
        field = sk.O3Field.with_boundaries(
            8,
            6,
            dx=0.5,
            dy=0.25,
            boundary_x="periodic",
            boundary_y="dirichlet",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.npz"
            sk.save_field_npz(field, path)
            loaded = sk.load_field_npz(path)
        self.assertEqual(loaded.boundary_x, "periodic")
        self.assertEqual(loaded.boundary_y, "dirichlet")

    def test_linear_stability_and_eigenmode_plot(self) -> None:
        field = sk.ScalarField2D(4, 4, value=0.0)
        model = sk.Phi4Model(lambda_=1.0, vacuum=1.0)
        stability = sk.stability_analysis(
            field,
            model,
            modes=1,
            subspace_dimension=16,
            tolerance=1e-6,
        )

        self.assertTrue(stability.converged)
        self.assertTrue(stability.stationary)
        self.assertFalse(stability.stable)
        self.assertEqual(stability.negative_mode_count, 1)
        self.assertAlmostEqual(stability.eigenvalues[0], -1.0, places=5)
        self.assertEqual(stability.modes[0].shape, (4, 4))
        self.assertLess(stability.residual_norms[0], 1e-6)

        axis = sk.plot_eigenmode(stability)
        self.assertIn("lambda", axis.get_title())
        plt.close(axis.figure)

        magnetization = sk.make_uniform_field(4, 4)
        magnetic_model = sk.MicromagneticModel(
            exchange=1.0,
            anisotropy=0.5,
            applied_field=sk.Vec3(0.0, 0.0, 0.25),
        )
        magnetic_stability = sk.stability_analysis(
            magnetization,
            magnetic_model,
            modes=2,
            subspace_dimension=32,
            tolerance=1e-6,
        )

        self.assertTrue(magnetic_stability.stable)
        np.testing.assert_allclose(
            magnetic_stability.eigenvalues,
            np.asarray([1.25, 1.25]),
            atol=1e-5,
        )
        self.assertEqual(magnetic_stability.modes[0].shape, (4, 4, 3))
        self.assertAlmostEqual(
            np.linalg.norm(magnetic_stability.modes[0]),
            1.0,
            places=10,
        )
        self.assertAlmostEqual(
            float(np.sum(
                magnetic_stability.modes[0]
                * magnetic_stability.modes[1]
            )),
            0.0,
            places=10,
        )
        np.testing.assert_allclose(
            magnetic_stability.modes[0][..., 2],
            0.0,
            atol=1e-12,
        )

        field3d = sk.O3Field3D(3, 3, 3)
        hopfion_model = sk.HopfionModel(
            coupling=1.0,
            kappa=0.1,
            mass=0.5,
        )
        stability3d = sk.stability_analysis(
            field3d,
            hopfion_model,
            modes=1,
            subspace_dimension=16,
            tolerance=1e-6,
        )
        self.assertTrue(stability3d.stable)
        self.assertAlmostEqual(stability3d.eigenvalues[0], 0.25, places=5)
        self.assertEqual(stability3d.modes[0].shape, (3, 3, 3, 3))

        axis = sk.plot_eigenmode(stability3d, slice_index=1)
        plt.close(axis.figure)


if __name__ == "__main__":
    unittest.main()
