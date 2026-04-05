from pathlib import Path

from structure_generation import StructureConfig, process_directory


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    layout_dir = repo_root / "images" / "layouts"
    edge_dir = repo_root / "images" / "edges"

    processed = process_directory(
        input_dir=layout_dir,
        output_dir=edge_dir,
        config=StructureConfig(),
        save_rgb=False,
        seed=0,
    )
    print(f"Prepared {processed} structural map(s) in {edge_dir}")


if __name__ == "__main__":
    main()
