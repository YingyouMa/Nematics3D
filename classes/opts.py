from dataclasses import fields, is_dataclass, replace

from Nematics3D.logging_decorator import logging_and_warning_decorator

@logging_and_warning_decorator()
def merge_opts(opts, kwargs, prefix="", logger=None):
    """
    Update a dataclass instance `opts` with values from `kwargs` whose
    keys start with a given prefix. The prefix is removed before matching
    the remaining part of the key to a field name in the dataclass.

    Parameters
    ----------
    opts : dataclass instance
        The target dataclass object to be updated.
    kwargs : dict
        A dictionary of keyword arguments that may contain keys with the
        specified prefix. Matching keys will be consumed (removed) from
        this dictionary.
    prefix : str, optional
        The prefix used to identify relevant keys in `kwargs`. Defaults to "".

    Returns
    -------
    dataclass instance
        A new dataclass object with updated field values.

    Raises
    ------
    TypeError
        If `opts` is not a dataclass instance.
    """
    if not is_dataclass(opts):
        raise TypeError("opts must be a dataclass instance")
        
    field_names = {f.name for f in fields(opts)}

    updates = {}
    for key, val in list(kwargs.items()):
        if key.startswith(prefix):
            name = key[len(prefix) :]  # strip prefix
            if name in field_names:
                updates[name] = val
                kwargs.pop(key)  # consume the key
            else:
                raise AttributeError(f"Invalid option '{key}' for {type(opts).__name__}")


    return replace(opts, **updates), kwargs

@logging_and_warning_decorator()
def merge_opts_all(prefix_to_opts: dict, kwargs: dict, name: str, logger=None):
    """
    Merge kwargs into multiple dataclass option objects according to prefix.

    Parameters
    ----------
    prefix_to_opts : dict
        Mapping from prefix string to dataclass instance.
        Example: {"opts_": opts, "vis_": vis}
    kwargs : dict
        Keyword arguments to consume.

    Returns
    -------
    dict
        New dataclass objects, same keys as prefix_to_opts.
    """
    kwargs = dict(kwargs)
    results = {}
    for prefix, opts in prefix_to_opts.items():
        new_opts, kwargs = merge_opts(opts, kwargs, prefix)
        results[prefix] = new_opts

    if kwargs:  
        msg = f">>> Unexpected keyword arguments for class {name!r}: {list(kwargs.keys())}. \n"
        msg += ">>> Ignore them in the following."
        logger.warning(msg)

    return results


# def make_opts_property(attr):
#     """Generate a property that proxies access to self._opts[field]."""
#     def getter(self):
#         return getattr(self._opts_all, attr)
#     def setter(self, value):
#         setattr(self._opts_all, attr, value)
        
#     return property(getter, setter)


def auto_opts_tubes(bindings: dict):

    def decorator(cls):
        for name, path in bindings.items():
            if name.startswith("_"):
                raise AttributeError(
                    f"Invalid binding for '{name}': internal fields cannot be exposed."
                )
        for name, path in bindings.items():
            attrs = path.split(".")  
            key = name[len("opts_") :]  

            def getter(self, _key=key):
                return getattr(self._opts_all, _key)

            def setter(self, value, _attrs=attrs, _key=key):
                setattr(self._opts_all, _key, value)

                processed = getattr(self._opts_all, _key)

                for item in self._entities:
                    target = item
                    for attr in _attrs[:-1]:
                        target = getattr(target, attr)
                    setattr(target, _attrs[-1], processed)

            setattr(cls, name, property(getter, setter))

        return cls

    return decorator


def auto_opts_tubes(bindings: dict):
    def decorator(cls):
        for name, path in bindings.items():
            if name.startswith("_"):
                raise AttributeError(
                    f"Invalid binding for '{name}': internal fields cannot be exposed."
                )

            key = name[len("opts_") :]

            attrs = path.split(".")

            def getter(self, _key=key):
                return getattr(self._opts_all, _key)
            
            def setter(self, value, _attrs=attrs, _key=key, _name=name):

                setattr(self._opts_all, _key, value)
                processed = getattr(self._opts_all, _key)

                for item in self._entities:
                    target = item
                    for attr in _attrs[:-1]:
                        target = getattr(target, attr)
                    setattr(target, _attrs[-1], processed)

            setattr(cls, name, property(getter, setter))


        return cls
    return decorator
