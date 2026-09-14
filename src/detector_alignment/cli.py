from __future__ import annotations

import argparse

from .pipeline import AlignmentPipeline


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="detector-align",
        description="Align detector point clouds from a C-frame plane and in-plane references.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("process", help="Process one measurement configuration")
    run.add_argument("config", help="YAML configuration file")
    args = parser.parse_args(argv)

    if args.command == "process":
        result = AlignmentPipeline.from_yaml(args.config).run()
        sigma = (result.covariance_transform_params.diagonal().clip(min=0.0)) ** 0.5
        print("Alignment complete")
        print("R =")
        print(result.rotation)
        print("T [mm] =", result.translation)
        print("1-sigma [rx ry rz tx ty tz] =", sigma)


if __name__ == "__main__":
    main()
