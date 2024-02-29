from collections import OrderedDict


def count_parameters(model, verbose=True):
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    from utils.utils import count_parameters
    count_parameters(model)
    import sys
    sys.exit(1)
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))
    return n_all, n_trainable


def state_dict_without_module(model, module_name):
    state_dict = model.state_dict()
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if module_name in key:
            continue
        new_state_dict[key] = state_dict[key]
    return new_state_dict


def merge_state_dict_with_module(state_dict, module_state_dict, module_name):
    new_state_dict = OrderedDict()
    for key in module_state_dict.keys():
        new_key = module_name + '.' + key
        new_state_dict[new_key] = module_state_dict[key]
    state_dict.update(new_state_dict)
    return state_dict
