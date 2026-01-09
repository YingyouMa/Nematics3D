from ..datatypes import UNSET

def cover_value(obj, is_cover_set=True, **kwargs):
    
    for key, value in kwargs.items():
        if not is_cover_set and getattr(obj, key) is not UNSET:
            continue
        setattr(obj, key, value)