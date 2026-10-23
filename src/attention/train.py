"""python -m attention.train --train data/train.jsonl --validation data/validation.jsonl"""

import argparse
import hashlib
import json
import platform
import random
import time
from dataclasses import asdict
from functools import partial
from importlib.metadata import version
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from attention.data import collate_batch, examples, load_pairs
from attention.tokenizer import TextTokenizer
from attention.transformers import ModelConfig, Transformer


def select_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def epoch(model, loader, device, optimizer=None, clip=1.0):
    model.train(optimizer is not None)
    total_loss, tokens = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for source, target, labels in loader:
            source, target, labels = (
                source.to(device),
                target.to(device),
                labels.to(device),
            )
            logits = model(source, target)
            count = labels.ne(model.config.pad_id).sum().item()
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=model.config.pad_id,
                reduction="sum",
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Perte non finie")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                (loss / count).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip, error_if_nonfinite=True
                )
                optimizer.step()
            total_loss += loss.item()
            tokens += count
    return total_loss / tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", default="checkpoints/local-v1")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    train_pairs, validation_pairs = load_pairs(args.train), load_pairs(args.validation)
    if set(p["group_id"] for p in train_pairs) & set(
        p["group_id"] for p in validation_pairs
    ):
        raise ValueError("Fuite de groupes train/validation")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = (
        TextTokenizer.load(output)
        if args.resume
        else TextTokenizer.train(train_pairs, args.vocab_size)
    )
    tokenizer.save(output)
    config = ModelConfig(
        **json.loads(Path(args.config).read_text()),
        vocab_size=len(tokenizer),
        pad_id=tokenizer.ids["<pad>"],
    )
    train_data = examples(train_pairs, tokenizer, config.max_length)
    val_data = examples(validation_pairs, tokenizer, config.max_length)
    collate = partial(collate_batch, pad_id=config.pad_id)
    loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    validation_loader = DataLoader(
        val_data, batch_size=args.batch_size, collate_fn=collate
    )
    model = Transformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start, best, stale, history = 0, float("inf"), 0, []
    split = {
        key: {
            "path": str(Path(path).resolve()),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "ids": [p["id"] for p in pairs],
        }
        for key, path, pairs in [
            ("train", args.train, train_pairs),
            ("validation", args.validation, validation_pairs),
        ]
    }
    if args.resume:
        saved = torch.load(output / "last.pt", map_location=device, weights_only=True)
        same_split = all(
            saved["split"][key]["sha256"] == split[key]["sha256"]
            and saved["split"][key]["ids"] == split[key]["ids"]
            for key in split
        )
        tokenizer_hash = hashlib.sha256(
            (output / "tokenizer.json").read_bytes()
        ).hexdigest()
        if (
            saved["config"] != asdict(config)
            or not same_split
            or saved.get("tokenizer_sha256") != tokenizer_hash
        ):
            raise ValueError(
                "Configuration, tokenizer ou données différentes du run à reprendre"
            )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best, stale, history = (
            saved["epoch"],
            saved["best"],
            saved["stale"],
            saved["history"],
        )
        torch.set_rng_state(saved["rng_state"].cpu())
        random.setstate(saved["python_rng_state"])
        if device.type == "cuda" and "cuda_rng_state" in saved:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        if device.type == "mps" and "mps_rng_state" in saved:
            torch.mps.set_rng_state(saved["mps_rng_state"].cpu())
    metadata = {
        "arguments": vars(args),
        "config": asdict(config),
        "device": str(device),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "versions": {
            name: version(name) for name in ["tokenizers", "fastapi", "numpy"]
        },
        "tokenizer_sha256": hashlib.sha256(
            (output / "tokenizer.json").read_bytes()
        ).hexdigest(),
        "split": split,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2))
    begun = time.monotonic()
    for index in range(start, args.epochs):
        train_loss = epoch(model, loader, device, optimizer)
        validation_loss = epoch(model, validation_loader, device)
        history.append(
            {
                "epoch": index + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "elapsed_seconds": time.monotonic() - begun,
            }
        )
        improved = validation_loss < best
        best, stale = (validation_loss, 0) if improved else (best, stale + 1)
        saved = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": index + 1,
            "best": best,
            "stale": stale,
            "history": history,
            "config": asdict(config),
            "model_id": output.name,
            "split": split,
            "tokenizer_sha256": metadata["tokenizer_sha256"],
            "rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
        }
        if device.type == "cuda":
            saved["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        if device.type == "mps":
            saved["mps_rng_state"] = torch.mps.get_rng_state()
        torch.save(saved, output / "last.pt")
        if improved:
            torch.save(saved, output / "best.pt")
        (output / "metrics.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(history[-1]), flush=True)
        if args.patience and stale >= args.patience:
            break


if __name__ == "__main__":
    main()
