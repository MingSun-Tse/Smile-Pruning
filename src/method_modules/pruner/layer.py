import torch.nn as nn
import torch

class Layer():
    '''A neat class to maintain network layer structure'''
    def __init__(self, name, size, layer_index, layer_type=None, res=False, last=None):
        self.name = name
        self.size = []
        for x in size:
            self.size.append(x)
        self.layer_index = layer_index # layer id
        self.layer_type = layer_type
        self.last = last # last layer
        self.is_shortcut = True if "downsample" in name else False
        if res:
            self.stage, self.seq_index, self.block_index = self._get_various_index_by_name(name)
    
    def _get_various_index_by_name(self, name):
        '''Get the indeces including stage, seq_ix, blk_ix.
            Same stage means the same feature map size.
        '''
        global lastest_stage # an awkward impel, just for now
        if name.startswith('module.'):
            name = name[7:] # remove the prefix caused by pytorch data parallel

        if "conv1" == name: # TODO: this might not be so safe
            lastest_stage = 0
            return 0, None, None
        if "linear" in name or 'fc' in name: # Note: this can be risky. Check it fully. TODO: @mingsun-tse
            return lastest_stage + 1, None, None # fc layer should always be the last layer
        else:
            try:
                stage  = int(name.split(".")[0][-1]) # ONLY work for standard resnets. name example: layer2.2.conv1, layer4.0.downsample.0
                seq_ix = int(name.split(".")[1])
                if 'conv' in name.split(".")[-1]:
                    blk_ix = int(name[-1]) - 1
                else:
                    blk_ix = -1 # shortcut layer  
                lastest_stage = stage
                return stage, seq_ix, blk_ix
            except:
                print('!Parsing the layer name failed: %s. Please check.' % name)

def register_modulename(model):
    for name, module in model.named_modules():
        module.name = name

def register_hook(model, **kwargs):
    """ 'learnable_layers' is a constant """
    last_module = [None]
    def hook(module, input, output):
        kwargs['max_len_name'] = max(kwargs['max_len_name'], len(module.name))
        kwargs['layers'][module.name] = Layer(name=module.name, 
                                            size=module.weight.size(), 
                                            layer_index=len(kwargs['layers']),
                                            layer_type=module.__class__.__name__,
                                            res=kwargs['res'],
                                            last=last_module[0])
        last_module[0] = module.name

    def register(module, handles):
        children = list(module.children())
        if len(children) == 0:
            if isinstance(module, kwargs['learnable_layers']):
                handles += [module.register_forward_hook(hook)]
        else:
            for c in children:
                register(c, handles)

    handles = [] # used for remove hooks
    register(model, handles)
    return handles

def rm_hook(handles):
    [x.remove() for x in handles]