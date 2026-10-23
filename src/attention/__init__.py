"""Transformer français standard ↔ Gen Z entraîné depuis zéro."""


def main():
    import argparse
    import sys
    from importlib import import_module

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "inference", "evaluate", "smoke"])
    args, remainder = parser.parse_known_args()
    sys.argv = [f"attention {args.command}", *remainder]
    import_module(f"attention.{args.command}").main()
