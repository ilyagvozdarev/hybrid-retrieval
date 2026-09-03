import inspect, json, torch

_SCALAR = (int, float, bool, str, type(None))

def describe_loss(obj):
    cfg = {"class": type(obj).__name__}

    # 1) штатный API sentence-transformers
    if hasattr(obj, "get_config_dict"):
        try:
            cfg.update(obj.get_config_dict())
        except Exception:
            pass

    # 2) добираем то, чего в нём нет, + разворачиваем вложенные лоссы
    for cls in type(obj).__mro__:
        init = cls.__dict__.get("__init__")
        if init is None:
            continue
        for name in inspect.signature(init).parameters:
            if name in ("self", "model", "args", "kwargs"):
                continue
            if not hasattr(obj, name):
                continue
            val = getattr(obj, name)
            if isinstance(val, torch.nn.Module):
                cfg[name] = describe_loss(val)          # перекрывает строку из get_config_dict
            elif name in cfg:
                continue
            elif isinstance(val, _SCALAR):
                cfg[name] = val
            elif callable(val):
                cfg[name] = getattr(val, "__name__", repr(val))
    return cfg