from dataclasses import fields, is_dataclass, replace

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import UNSET
from Nematics3D.general import is_equal

@logging_and_warning_decorator(start_finish_level=5)
def merge_opts(opts, kwargs, prefix="", logger=None):
    """
    Merge prefixed keyword arguments into a dataclass-based option object.

    This function extracts entries from ``kwargs`` whose keys start with the
    specified ``prefix``, strips the prefix, and applies the resulting key–value
    pairs to a dataclass instance ``opts``.  All successfully matched keys are
    consumed (removed) from ``kwargs``.

    The merge is performed by constructing a *new* dataclass instance via
    ``dataclasses.replace``.  As a result, all field-level validation defined
    inside ``opts`` (e.g. via ``__init__``, ``__post_init__``, ``__setattr__``,
    or custom validators in the option class hierarchy) is automatically
    triggered during the merge process.

    Parameters
    ----------
    opts : dataclass instance
        The target option object.  Must be an instance of a dataclass.
        A new instance will be returned; the original object is not modified.

    kwargs : dict
        A dictionary of keyword arguments to be consumed.  Any key that matches
        the given prefix and a valid field name will be removed from this
        dictionary.

    prefix : str, optional
        Prefix used to identify which entries in ``kwargs`` belong to this
        option object.  The prefix is stripped before matching against
        dataclass field names.  Defaults to ``""``.

    Returns
    -------
    new_opts : dataclass instance
        A new dataclass instance with updated field values.  All merged values
        have already passed the internal validation logic of ``opts``.

    remaining_kwargs : dict
        The same ``kwargs`` dictionary after in-place consumption of all
        recognized keys.

    Raises
    ------
    TypeError
        If ``opts`` is not a dataclass instance.
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
    """
    Merge an override dictionary into a base dictionary with key validation.
    
    This function creates a shallow copy of ``dict_origin`` and applies values
    from ``dict_override`` on top of it.  Only keys that already exist in
    ``dict_origin`` are allowed to be overridden; any unknown keys appearing
    in ``dict_override`` are considered invalid and will be ignored with a
    warning.
    
    The merge operation is non-destructive to the input dictionaries:
    ``dict_origin`` is never modified, and a new dictionary is returned.
    
    Parameters
    ----------
    dict_origin : dict
        The base dictionary defining the allowed set of keys and their default
        values.
    
    dict_override : dict or None, optional
        A dictionary containing override values.  Only keys present in
        ``dict_origin`` are applied.  If ``None``, an empty override is assumed.
    
    name : str, optional
        A human-readable name used in warning or error messages to identify
        the configuration context (e.g. ``"input"``, ``"options"``,
        ``"visual"``).
    
    Returns
    -------
    dict
        A new dictionary consisting of ``dict_origin`` updated with all valid
        overrides from ``dict_override``.
    """

    if dict_override is None:
        dict_override = {}

    defaults = dict(dict_origin)

    for k, v in dict_override.items():
        if k not in defaults:
            try:
                raise KeyError(
                    f"Invalid key {k!r} in dict_override; not a valid {name} option. "
                    "This key will be ignored."
                )
            except KeyError:
                logger.exception("Check input.")
                logger.recovery("This key will be ignored.")
                continue
        defaults[k] = v

    return defaults


@logging_and_warning_decorator(start_finish_level=5)
def cover_value(
    obj,
    is_allow_cover_target_set: bool = True,
    is_allow_unset_source: bool = False,
    logger=None,
    **kwargs,
):
    """
    Conditionally assign values from kwargs to attributes of obj.

    Parameters
    ----------
    obj : object
        Target object whose attributes will be updated.
    is_allow_cover_target_set : bool, default True
        If False, attributes on obj that are already set (not UNSET)
        will not be overwritten.
    is_allow_unset_source : bool, default False
        If False, kwargs entries whose value is UNSET will be ignored.
    **kwargs
        Attribute-value pairs used as assignment sources.
    """

    for key, value in kwargs.items():

        # Source-side constraint: whether UNSET is allowed as an input value
        if not is_allow_unset_source and value is UNSET:
            continue

        # Target-side constraint: whether existing values may be overwritten
        if not is_allow_cover_target_set and getattr(obj, key, UNSET) is not UNSET:
            continue
        
        try:
            setattr(obj, key, value)
        except Exception:
            logger.exception("Check input.")
            logger.recovery("Automatically ignore this modification")
            
            
@logging_and_warning_decorator(start_finish_level=5)
def diff_dict_values(dict1: dict, dict2: dict, logger=None):

    diff1 = {}
    diff2 = {}

    keys = set(dict1.keys()) | set(dict2.keys())
    for k in keys:
        has1 = k in dict1
        has2 = k in dict2

        if (not has1) or (not has2):
            if has1:
                diff1[k] = dict1[k]
            if has2:
                diff2[k] = dict2[k]
            continue

        if not is_equal(dict1[k], dict2[k]):
            diff1[k] = dict1[k]
            diff2[k] = dict2[k]

    return diff1, diff2




