import argparse

import solitonkit as sk


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        default="output",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
    )

    args = parser.parse_args()

    outputs = sk.viz.plot_skyrmion(
        data_dir=args.data_dir,
        save=True,
        show=not args.no_show,
    )

    print("Saved plots:")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()