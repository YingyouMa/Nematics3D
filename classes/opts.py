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

@logging_and_warning_decorator(start_finish_level=5)
def merge_opts_all(prefix_to_opts: dict, kwargs: dict, name: str, logger=None):
    """
    Distribute keyword arguments into multiple dataclass-based option objects
    according to their name prefixes, with automatic validation and leftover
    detection.

    This function supports a flexible configuration style in which a single call
    (typically an ``__init__`` method) receives a large pool of keyword arguments,
    and different subsets of those keywords are intended for different option
    objects.  Each option object is associated with a prefix that identifies which
    kwargs belong to it.  All matched arguments are merged into the corresponding
    dataclass instance via ``merge_opts()``, which applies field-level validation
    according to each dataclass's ``__setattr__`` logic.

    Any keyword arguments that do not match *any* known prefix are collected and
    reported as unexpected via ``logger.warning``.  They are ignored in the final
    output.

    This mechanism makes it possible to cleanly support large, modular
    configuration schemas such as:

    - ``opts_*``   → simulation options
    - ``vis_*``    → visualization options
    - ``io_*``     → import/export settings
    - ``smooth_*`` → smoothing and filtering parameters
    - ``""`` (empty prefix) → default or “main” configuration block

    The function is also compatible with the simple case where only a single
    dataclass is used.  In that case the caller may pass ``{"": obj}`` so that
    *all* kwargs are treated as fields of that object.

    Parameters
    ----------
    prefix_to_opts : dict[str, object]
        Mapping from prefix string to dataclass instance.  For each entry:

        ``prefix : option_object``

        all kwargs whose keys begin with ``prefix`` will be stripped of that
        prefix and applied to ``option_object``.

        Example:
            ``{"opts_": sim_opts, "vis_": vis_opts}``

    kwargs : dict
        The raw keyword argument dictionary to be consumed and distributed.

    name : str
        Name of the parent class or component invoking this function.  Used only
        for producing clearer warning messages when unexpected arguments appear.

    logger : Logger, optional
        Logger used to emit warnings about unmatched keyword arguments.

    Returns
    -------
    dict[str, object]
        A dictionary mapping prefixes to *newly merged* dataclass instances.
        The keys are the same as ``prefix_to_opts``.  All matched keyword
        arguments are consumed; only unknown leftovers generate a warning.

    Notes
    -----
    - Dataclass validation rules (defined via ``__setattr__`` or custom
      validators) are automatically triggered during merging.
    - If multiple prefixes match no arguments, their option objects are returned
      unchanged.
    - Leftover kwargs are never silently ignored; a warning is always issued.

    Examples
    --------
    >>> # Case 1: Multiple option groups
    >>> merged = merge_opts_all(
    ...     {"opts_": sim_opts, "vis_": vis_opts},
    ...     {"opts_steps": 100, "vis_color": "red", "unknown": 1},
    ...     name="MyClass",
    ...     logger=logger,
    ... )
    >>> merged["opts_"].steps
    100
    >>> merged["vis_"].color
    'red'
    >>> # A warning is emitted for {"unknown": 1}

    >>> # Case 2: Single configuration block via empty prefix
    >>> merged = merge_opts_all(
    ...     {"": inputQ},
    ...     {"Q": Qfield, "grid_offset": (1, 2, 3)},
    ...     name="QFieldObject",
    ...     logger=logger,
    ... )
    >>> new_inputQ = merged[""]
    """
    kwargs = dict(kwargs)
    results = {}
    for prefix, opts in prefix_to_opts.items():
        new_opts, kwargs = merge_opts(opts, kwargs, prefix)
        results[prefix] = new_opts

    if kwargs:  
        msg = f"Unexpected keyword arguments for class {name!r}: {list(kwargs.keys())}. \n"
        msg += "Ignore them in the following."
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

