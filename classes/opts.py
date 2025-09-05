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



def register_opts_aliases(cls):
    """
    Class decorator to register alias properties without the ``opts_`` prefix.

    This function scans all properties defined in the class whose names start 
    with ``opts_`` and automatically creates new properties with the same 
    getter, setter, and deleter, but without the prefix.

    Example:
        >>> @register_opts_aliases
        ... class SceneWrapper:
        ...     @property
        ...     def opts_fgcolor(self):
        ...         return (1, 1, 1)
        ...
        ...     @opts_fgcolor.setter
        ...     def opts_fgcolor(self, value):
        ...         pass
        ...
        >>> wrapper = SceneWrapper()
        >>> wrapper.fgcolor        # Equivalent to wrapper.opts_fgcolor
        >>> wrapper.fgcolor = (0,0,0)  # Equivalent to wrapper.opts_fgcolor = (0,0,0)

    Args:
        cls (type): The class to decorate.

    Returns:
        type: The same class with additional alias properties.
    """
    for name, attr in list(cls.__dict__.items()):
        if isinstance(attr, property) and name.startswith("opts_"):
            alias = name[len("opts_"):]
            if not hasattr(cls, alias):
                setattr(cls, alias, property(attr.fget, attr.fset, attr.fdel, attr.__doc__))
    return cls


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
                    target = item._entities[0]
                    for attr in _attrs[:-1]:
                        target = getattr(target, attr)
                    setattr(target, _attrs[-1], processed)

            setattr(cls, name, property(getter, setter))


        return cls
    return decorator

def auto_opts_tubes_each(bindings: dict):
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
                
                target = self._entities[0]
                for attr in _attrs[:-1]:
                    target = getattr(target, attr)
                setattr(target, _attrs[-1], processed)

            setattr(cls, name, property(getter, setter))


        return cls
    return decorator

