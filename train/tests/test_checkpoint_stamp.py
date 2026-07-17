"""Base ``on_save_checkpoint`` stamps ``model_class`` onto every checkpoint."""
from train.test_utils import TinyLightningModel, TinyModule


def test_on_save_checkpoint_stamps_model_class():
    model = TinyLightningModel(TinyModule())
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    # The stamp records the eager module's class name (unaffected by torch.compile).
    assert checkpoint["model_class"] == "TinyModule"


def test_subclass_super_keeps_both_keys():
    class Sub(TinyLightningModel):
        def on_save_checkpoint(self, checkpoint):
            super().on_save_checkpoint(checkpoint)
            checkpoint["extra_key"] = 123

    model = Sub(TinyModule())
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    assert checkpoint["model_class"] == "TinyModule"
    assert checkpoint["extra_key"] == 123
