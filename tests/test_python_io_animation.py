from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

import solitonkit as sk
from solitonkit.cli import main as cli_main


class FieldIoAnimationTests(unittest.TestCase):
    def test_boundary_conditions(self) -> None:
        fixed = sk.make_uniform_field(8, 8, boundary="fixed")
        fixed.set(0, 4, sk.Vec3(1.0, 0.0, 0.0))
        fixed_before = fixed.to_numpy().copy()
        sk.run_gradient_flow_inplace(fixed, step_size=0.05, steps=1)

        self.assertEqual(fixed.boundary, "fixed")
        np.testing.assert_allclose(
            fixed.to_numpy()[0, :, :],
            fixed_before[0, :, :],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            fixed.to_numpy()[:, 0, :],
            fixed_before[:, 0, :],
            atol=1e-12,
        )

        neumann = sk.make_uniform_field(8, 8, boundary="neumann")
        neumann.set(0, 4, sk.Vec3(1.0, 0.0, 0.0))
        sk.run_gradient_flow_inplace(neumann, step_size=0.05, steps=1)

        self.assertEqual(neumann.boundary, "neumann")
        self.assertLess(neumann.get(0, 4).x, 1.0)
        self.assertGreater(neumann.get(0, 4).z, 0.0)

        with self.assertRaises(ValueError):
            sk.make_uniform_field(8, 8, boundary="unknown")

    def test_skyrmion_charge_parameter_matches_observable(self) -> None:
        positive = sk.make_skyrmion_field(
            64,
            64,
            spacing=0.5,
            radius=4.0,
            charge=1,
        )
        negative = sk.make_skyrmion_field(
            64,
            64,
            spacing=0.5,
            radius=4.0,
            charge=-1,
        )

        self.assertGreater(sk.topological_charge(positive), 0.8)
        self.assertLess(sk.topological_charge(negative), -0.8)

    def test_npz_round_trip(self) -> None:
        field = sk.make_skyrmion_field(
            24,
            20,
            dx=0.25,
            dy=0.4,
            radius=2.0,
            boundary="fixed",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.npz"

            sk.save_field_npz(
                field,
                path,
                metadata={"model": "baby-skyrme", "charge": 1},
            )

            loaded, metadata = sk.load_field_npz(
                path,
                return_metadata=True,
            )

            self.assertTrue(path.is_file())
            self.assertAlmostEqual(loaded.dx, 0.25)
            self.assertAlmostEqual(loaded.dy, 0.4)
            self.assertEqual(loaded.boundary, "fixed")
            self.assertEqual(metadata["model"], "baby-skyrme")
            self.assertEqual(metadata["charge"], 1)
            np.testing.assert_allclose(
                sk.field_to_numpy(loaded),
                sk.field_to_numpy(field),
                atol=1e-12,
            )

    def test_flow_snapshots_and_gif(self) -> None:
        field = sk.make_skyrmion_field(
            24,
            24,
            spacing=0.5,
            radius=2.5,
        )

        relaxed, snapshots = sk.run_baby_skyrme_gradient_flow_snapshots(
            field,
            step_size=1e-4,
            steps=5,
            frame_every=2,
        )

        self.assertEqual([snapshot.step for snapshot in snapshots], [0, 2, 4, 5])
        self.assertLess(snapshots[-1].energy, snapshots[0].energy)
        self.assertEqual(relaxed.nx, field.nx)
        self.assertEqual(relaxed.ny, field.ny)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flow.gif"
            sk.save_flow_animation(snapshots, path, fps=5, dpi=50)

            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_landau_lifshitz_dynamics(self) -> None:
        field = sk.make_skyrmion_field(
            24,
            24,
            spacing=0.5,
            radius=2.5,
            boundary="fixed",
        )
        before = field.to_numpy().copy()

        evolved, records = sk.run_landau_lifshitz(
            field,
            time_step=1e-5,
            damping=0.5,
            steps=5,
            record_every=1,
        )
        after = evolved.to_numpy()

        self.assertEqual(len(records), 6)
        self.assertAlmostEqual(records[-1].time, 5e-5)
        self.assertLess(records[-1].energy, records[0].energy)
        np.testing.assert_allclose(field.to_numpy(), before, atol=1e-12)
        np.testing.assert_allclose(after[0], before[0], atol=1e-12)
        np.testing.assert_allclose(after[-1], before[-1], atol=1e-12)
        np.testing.assert_allclose(after[:, 0], before[:, 0], atol=1e-12)
        np.testing.assert_allclose(after[:, -1], before[:, -1], atol=1e-12)
        np.testing.assert_allclose(
            np.linalg.norm(after, axis=2),
            1.0,
            atol=1e-12,
        )

        with self.assertRaises(ValueError):
            sk.run_landau_lifshitz(field, time_step=0.0, steps=1)

        with self.assertRaises(ValueError):
            sk.run_landau_lifshitz(field, damping=-0.1, steps=1)

    def test_cli_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.npz"
            relaxed = root / "relaxed.npz"
            records = root / "records.csv"
            evolved = root / "evolved.npz"
            dynamics_records = root / "dynamics.csv"
            plot = root / "relaxed.png"

            self.assertEqual(
                cli_main(
                    [
                        "generate",
                        "--nx", "20",
                        "--ny", "20",
                        "--spacing", "0.5",
                        "--radius", "2.0",
                        "--boundary", "fixed",
                        "--output", str(initial),
                    ]
                ),
                0,
            )

            self.assertEqual(
                cli_main(
                    [
                        "relax",
                        "--input", str(initial),
                        "--output", str(relaxed),
                        "--records", str(records),
                        "--steps", "2",
                        "--record-every", "1",
                    ]
                ),
                0,
            )

            self.assertEqual(
                cli_main(
                    [
                        "evolve",
                        "--input", str(relaxed),
                        "--output", str(evolved),
                        "--records", str(dynamics_records),
                        "--time-step", "0.00001",
                        "--damping", "0.5",
                        "--steps", "2",
                        "--record-every", "1",
                    ]
                ),
                0,
            )

            self.assertEqual(
                cli_main(
                    [
                        "plot",
                        "--input", str(relaxed),
                        "--output", str(plot),
                        "--no-quiver",
                        "--dpi", "50",
                    ]
                ),
                0,
            )

            for path in (
                initial,
                relaxed,
                records,
                evolved,
                dynamics_records,
                plot,
            ):
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

            loaded = sk.load_field_npz(evolved)
            self.assertEqual(loaded.boundary, "fixed")

            header = dynamics_records.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(header, "step,time,energy,topological_charge")

    def test_legacy_npz_defaults_to_periodic(self) -> None:
        field = sk.make_uniform_field(4, 4)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            np.savez(
                path,
                field=field.to_numpy(),
                dx=np.asarray(0.5),
                dy=np.asarray(0.75),
                format_version=np.asarray(1),
                metadata_json=np.asarray("{}"),
            )

            loaded = sk.load_field_npz(path)

            self.assertEqual(loaded.boundary, "periodic")
            self.assertAlmostEqual(loaded.dx, 0.5)
            self.assertAlmostEqual(loaded.dy, 0.75)


if __name__ == "__main__":
    unittest.main()
