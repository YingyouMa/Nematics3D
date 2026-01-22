from dataclasses import fields, is_dataclass, replace

from Nematics3D.logging_decorator import logging_and_warning_decorator

@logging_and_warning_decorator(start_finish_level=5)
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
                try:
                    raise AttributeError(f"Invalid option '{key}' for {type(opts).__name__}")
                except:
                    logger.exception("Please check input.")
                    logger.recovery("Ignore this key in the following.")
                    kwargs.pop(key)

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


@logging_and_warning_decorator(start_finish_level=5)
def build_dict_override(
    dict_origin: dict,
    dict_override: dict | None = None,
    *,
    name: str = "input",
    logger = None
):

    if dict_override is None:
        dict_override = {}

    defaults = dict(dict_origin)

    for k, v in dict_override.items():
        if k not in defaults:
            try:
                raise KeyError(
                    f"Invalid key {k!r} in dict_origin; "
                    f"not a valid {name} option."
                )
            except KeyError:
                logger.exception("Check input.")
                logger.recovery("This key will be ignored.")
                continue
        defaults[k] = v

    return defaults


# @logging_and_warning_decorator(start_finish_level=5)
# def merge_opts_with_kwargs(
#     opts,
#     kwargs: dict,
#     *,
#     opts_type,
#     logger=None,
# ):
#     """
#     Merge an optional options object with keyword arguments.

#     Parameters
#     ----------
#     opts : object or None
#         Options object to be merged. If provided, must be an instance of opts_type
#         and implement `act_asdict()`.
#     kwargs : dict
#         Keyword-based configuration. Takes precedence over opts on overlap.
#     opts_type : type
#         Expected type of the opts object.

#     Returns
#     -------
#     merged_opts : dict
#         Merged configuration dictionary, where kwargs override opts.
#     """

#     if opts is not None:
#         if not isinstance(opts, opts_type):
#             try:
#                 raise TypeError(
#                     f"`opts` must be {opts_type.__name__} object. "
#                     f"Got type={type(opts)} instead."
#                 )
#             except:
#                 logger.exception("Check input.")
#                 logger.recovery("Ignoring `opts` and using `kwargs` only.")
#                 opts_dict = {}
#         else:
#             opts_dict = opts.act_asdict()
#             overlap = opts_dict.keys() & kwargs.keys()
#             if overlap:
#                 logger.warning(
#                     f"Overlapping configuration detected: {list(overlap)}. "
#                     f"The values in **kwargs will take precedence over `opts`.",
#                 )
#     else:
#         opts_dict = {}

#     return opts_dict | kwargs




