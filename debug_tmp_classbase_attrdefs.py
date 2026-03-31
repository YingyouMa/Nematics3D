from pathlib import Path

path = Path(r'D:\Document\GitHub\Nematics3D\src\nematics3d\classes\class_base.py')
text = path.read_text(encoding='utf-8').replace('\r\n', '\n')

old_attr_defs = '''    __attr_defs__ = {
        "raw_name": {
            "doc": "The underlying string identifier for this instance.",
            "validator": as_str,
            "is_public_settable": True,
            "is_protected": False,
        },
        "owner": {
            "doc": "The object that owns this instance.",
            "kind": "relation",
            "validator": None,
            "is_public_settable": False,
            "is_protected": False,
            "is_weak_by_default": True,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "registry": {
            "doc": "The Registry object where this instance is registered.",
            "kind": "relation",
            "validator": None,
            "is_public_settable": False,
            "is_protected": False,
            "is_weak_by_default": True,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "impl_attrs": {
            "doc": "Runtime attribute metadata copied from the class template.",
            "validator": None,
            "is_public_settable": False,
            "is_protected": False,
        },
    }
'''
new_attr_defs = '''    __attr_defs__ = {
        "raw_name": {
            "doc": "The underlying string identifier for this instance.",
            "validator": as_str,
            "is_public_settable": True,
            "is_protected": False,
        },
        "owner": {
            "doc": "The object that owns this instance.",
            "kind": "relation",
            "is_public_settable": False,
            "is_weak_by_default": True,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "registry": {
            "doc": "The Registry object where this instance is registered.",
            "kind": "relation",
            "is_public_settable": False,
            "is_weak_by_default": True,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "impl_attrs": {
            "doc": "Runtime attribute metadata copied from the class template.",
            "is_public_settable": False,
        },
    }
'''
assert old_attr_defs in text
text = text.replace(old_attr_defs, new_attr_defs)

old_register = '''        attr_info = {
            "doc": as_str(doc, name=f"Definition doc for {name!r}"),
            "validator": validator,
            "is_public_settable": bool(is_public_settable),
            "is_protected": False,
        }
'''
new_register = '''        attr_info = {
            "doc": as_str(doc, name=f"Definition doc for {name!r}"),
            "validator": validator,
            "is_public_settable": bool(is_public_settable),
        }
        if attr_info["is_public_settable"]:
            attr_info["is_protected"] = False
'''
assert old_register in text
text = text.replace(old_register, new_register)

path.write_text(text, encoding='utf-8')