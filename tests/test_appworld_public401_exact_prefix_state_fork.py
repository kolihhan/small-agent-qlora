"""No protected AppWorld data. Synthetic checkpoint bytes test."""
from pathlib import Path
from types import SimpleNamespace
import pytest

from scripts.appworld_train_public401_exact_prefix_state_fork import checkpoint_state_hash


class SyntheticWorld:
    def __init__(self,root,data:bytes):
        self.output_checkpoints_directory=str(root)
        self.data=data

    def save_state(self,tag):
        p=Path(self.output_checkpoints_directory)/tag
        p.mkdir(parents=True,exist_ok=True)
        (p/"changes.sqlite").write_bytes(self.data)


def test_private_snapshot_hash_compares_same_content_not_absolute_path(tmp_path):
    a=SyntheticWorld(tmp_path/"original",b"identical private database bytes")
    b=SyntheticWorld(tmp_path/"treatment",b"identical private database bytes")
    c=SyntheticWorld(tmp_path/"different",b"modified task database bytes")
    original=checkpoint_state_hash(a,"orig_first_public401")
    treatment=checkpoint_state_hash(b,"treat_first_public401")
    assert original==treatment
    assert checkpoint_state_hash(c,"treat_first_public401")!=original


def test_snapshot_fails_closed_if_world_did_not_persist(tmp_path):
    w=SimpleNamespace(output_checkpoints_directory=str(tmp_path),
                      save_state=lambda tag:None)
    with pytest.raises(RuntimeError,match="directory missing"):
        checkpoint_state_hash(w,"missing")
