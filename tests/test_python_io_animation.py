from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

import solitonkit as sk
from solitonkit.cli import main as cli_main


class FieldIoAnimationTests(unittest.TestCase):
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

    def test_cli_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.npz"
            relaxed = root / "relaxed.npz"
            records = root / "records.csv"
            plot = root / "relaxed.png"

            self.assertEqual(
                cli_main(
                    [
                        "generate",
                        "--nx", "20",
                        "--ny", "20",
                        "--spacing", "0.5",
                        "--radius", "2.0",
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
                        "plot",
                        "--input", str(relaxed),
                        "--output", str(plot),
                        "--no-quiver",
                        "--dpi", "50",
                    ]
                ),
                0,
            )

            for path in (initial, relaxed, records, plot):
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
