import argparse
import yaml

from train import parse_args, train_from_args


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="Start training Creative-GNN using a config file.")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to YAML config.")
    args_cli = parser.parse_args()

    cfg = load_config(args_cli.config)
    base_args = parse_args([])  # defaults
    for k, v in cfg.items():
        if hasattr(base_args, k):
            setattr(base_args, k, v)
        else:
            print(f"Warning: unknown config key '{k}' ignored.")
    train_from_args(base_args)


if __name__ == "__main__":
    main()
