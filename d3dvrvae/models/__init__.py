from .basic_model import BasicModelOption, create_basic_model
from .vr_model import VRModelOption, create_vr_model
from .option import ModelOption


def create_model(
    opt: ModelOption,
    n_epoch: int,
    steps_per_epoch: int,
):
    if isinstance(opt, BasicModelOption) and type(opt) is BasicModelOption:
        return create_basic_model(opt, n_epoch, steps_per_epoch)
    if isinstance(opt, VRModelOption) and type(opt) is VRModelOption:
        return create_vr_model(opt, n_epoch, steps_per_epoch)
    raise NotImplementedError(f"{opt.__class__.__name__} not implemented")
