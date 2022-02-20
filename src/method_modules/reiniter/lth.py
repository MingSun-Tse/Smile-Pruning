import torch
import torch.nn as nn

class Reiniter():
    def __init__(self, model, loader, args, logger, passer):
        self.model = model
        self.ckpt_init = logger.passer['ckpt_init']

    def reinit(self):
        assert hasattr(self.model, 'mask'), "'model' should has attr 'mask'."
        state_dict = torch.load(self.ckpt_init)['state_dict']
        self.model.load_state_dict(state_dict)
        for name, m in self.model.named_modules():
            if name in self.model.mask:
                m.weight.data.mul_(self.model.mask[name])
        print('==> Reinit model: use LTH-like reinitialization - apply masks to initial weights')