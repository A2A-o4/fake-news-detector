import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import build_pipeline  # noqa: E402


def test_pipeline_learns_simple_split():
    texts = [
        "government announces budget",
        "central bank keeps rates",
        "SHOCKING miracle trick cures everything",
        "you won't believe this secret miracle",
    ]
    labels = ["real", "real", "fake", "fake"]
    model = build_pipeline().fit(texts, labels)
    assert model.predict(["shocking secret miracle cure"])[0] == "fake"


def test_liar_label_mapping_is_binary():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from download_data import LIAR_BINARY

    assert set(LIAR_BINARY.values()) == {"fake", "real"}
    assert len(LIAR_BINARY) == 6
