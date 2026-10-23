import json
from dataclasses import asdict

import torch

from attention.data import collate_batch, examples
from attention.inference import Translator
from attention.tokenizer import TextTokenizer
from attention.train import epoch
from attention.transformers import ModelConfig, MultiHeadAttention, Transformer


def tiny_config(vocab=32):
    return ModelConfig(
        vocab_size=vocab,
        d_model=16,
        heads=2,
        d_ff=32,
        encoder_layers=1,
        decoder_layers=1,
        dropout=0,
        max_length=16,
    )


def test_attention_padding_and_cross_lengths():
    torch.manual_seed(42)
    attention = MultiHeadAttention(2, 16).eval()
    keys, queries = torch.randn(2, 7, 16), torch.randn(2, 3, 16)
    mask = torch.tensor([[False] * 4 + [True] * 3] * 2)
    output = attention(keys, queries, mask)
    changed = keys.clone()
    changed[:, 4:] = 1000
    torch.testing.assert_close(output, attention(changed, queries, mask))
    assert output.shape == (2, 3, 16)
    full_mask = torch.ones_like(mask)
    assert torch.isfinite(attention(keys, queries, full_mask)).all()


def test_causal_mask_and_parameter_gradients():
    torch.manual_seed(42)
    model = Transformer(tiny_config()).eval()
    source = torch.tensor([[4, 8, 9, 2, 0], [5, 10, 2, 0, 0]])
    target = torch.tensor([[1, 7, 8, 2], [1, 8, 9, 2]])
    original = model(source, target)
    changed = target.clone()
    changed[:, 2:] = 12
    torch.testing.assert_close(original[:, :2], model(source, changed)[:, :2])
    loss = torch.nn.functional.cross_entropy(
        original.reshape(-1, 32), target.reshape(-1)
    )
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert any("heads.0.WQ" in name for name, _ in model.named_parameters())
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.step()
    assert torch.isfinite(model(source, target)).all()


def test_checkpoint_logits_roundtrip(tmp_path):
    model = Transformer(tiny_config()).eval()
    source, target = torch.tensor([[4, 9, 2]]), torch.tensor([[1, 8]])
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    restored = Transformer(tiny_config()).eval()
    restored.load_state_dict(torch.load(tmp_path / "weights.pt", weights_only=True))
    torch.testing.assert_close(model(source, target), restored(source, target))


def test_bpe_unicode_and_teacher_forcing(tmp_path):
    pairs = [
        {
            "id": "1",
            "group_id": "1",
            "standard": "Élodie n’a pas 42 € 😊.",
            "genz": "Élodie a pas 42 € 😊.",
        }
    ]
    tokenizer = TextTokenizer.train(pairs, 300)
    tokenizer.save(tmp_path)
    loaded = TextTokenizer.load(tmp_path)
    for text in ["Élodie n’a pas 42 € 😊.", "café 🪩", "e\u0301"]:
        import unicodedata

        assert loaded.decode(loaded.encode(text)) == unicodedata.normalize("NFC", text)
    batch = examples(pairs, loaded, 64)
    source, target, labels = collate_batch(batch)
    assert source.shape[0] == 2
    assert target[:, 0].eq(loaded.ids["<bos>"]).all()
    assert (labels == loaded.ids["<eos>"]).any()


def test_generation_immediate_eos_and_limit(tmp_path):
    pairs = [{"standard": "Salut", "genz": "Yo"}]
    tokenizer = TextTokenizer.train(pairs, 270)
    tokenizer.save(tmp_path)
    c = ModelConfig(
        vocab_size=len(tokenizer),
        d_model=8,
        heads=2,
        encoder_layers=1,
        decoder_layers=1,
        d_ff=16,
        max_length=16,
        dropout=0,
    )
    model = Transformer(c)
    with torch.no_grad():
        model.projection.weight.zero_()
        model.projection.bias.zero_()
        model.projection.bias[tokenizer.ids["<eos>"]] = 10
    (tmp_path / "run.json").write_text("{}")

    def save():
        torch.save(
            {"config": asdict(c), "model": model.state_dict(), "model_id": "test"},
            tmp_path / "best.pt",
        )

    save()
    translated = Translator(tmp_path, "cpu").translate("Salut", "to_genz")
    assert translated.translation == "" and translated.truncated is False
    with torch.no_grad():
        model.projection.bias[tokenizer.ids["<eos>"]] = 0
        model.projection.bias[tokenizer.encode("a")[0]] = 10
    save()
    translated = Translator(tmp_path, "cpu").translate("Salut", "to_genz")
    assert translated.truncated is True


def test_training_resume_preserves_history(tmp_path, monkeypatch):
    from attention.train import main

    train = [{"id": "a", "group_id": "a", "standard": "Bonjour.", "genz": "Salut."}]
    val = [{"id": "b", "group_id": "b", "standard": "Au revoir.", "genz": "À plus."}]
    for name, rows in [("train", train), ("val", val)]:
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    configuration = {
        "d_model": 8,
        "heads": 2,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "d_ff": 16,
        "dropout": 0,
        "max_length": 32,
    }
    (tmp_path / "config.json").write_text(json.dumps(configuration))
    output = tmp_path / "run"
    args = [
        "train",
        "--train",
        str(tmp_path / "train.jsonl"),
        "--validation",
        str(tmp_path / "val.jsonl"),
        "--output",
        str(output),
        "--config",
        str(tmp_path / "config.json"),
        "--epochs",
        "1",
        "--threads",
        "2",
    ]
    monkeypatch.setattr("sys.argv", args)
    main()
    args[args.index("--epochs") + 1] = "2"
    monkeypatch.setattr("sys.argv", args + ["--resume"])
    main()
    history = json.loads((output / "metrics.json").read_text())
    assert [entry["epoch"] for entry in history] == [1, 2]
    translator = Translator(output, "cpu")
    assert translator.model_id == "run"
    import pytest

    from attention.inference import TranslationError

    with pytest.raises(TranslationError) as error:
        translator.translate("x" * 200, "to_genz")
    assert error.value.code == "INPUT_TOO_LONG"
    tokenizer = json.loads((output / "tokenizer.json").read_text())
    tokenizer["normalizer"] = {"type": "NFD"}
    (output / "tokenizer.json").write_text(json.dumps(tokenizer))
    with pytest.raises(ValueError, match="Tokenizer différent"):
        Translator(output, "cpu")
