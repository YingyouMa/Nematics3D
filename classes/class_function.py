from ..datatypes import UNSET
from ..logging_decorator import logging_and_warning_decorator

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
        if not is_allow_cover_target_set and getattr(obj, key) is not UNSET:
            continue
        
        try:
            setattr(obj, key, value)
        except:
            logger.exception("Check input.")
            logger.recovery("Automatically ignore this modification")
            
