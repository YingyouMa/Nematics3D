# class ClassBase:
#     __attrs__ = {
#         "raw_name": "The underlying string identifier for this instance.",
#         "_impl_extra_attrs": (
#             "A dictionary storing dynamic user-defined attributes. "
#             "Managed via 'act_add_attr'."
#         ),
#         "_impl_extra_attrs_docs": "Documentation strings for user-defined extra attributes.",
#         "_impl_relations": (
#             "Runtime storage for object relations. "
#             "Key: relation name; value: target object or weakref.ref(target)."
#         ),
#         "_impl_relation_docs": (
#             "A dictionary storing relation documentation strings. "
#             "Key: relation name; value: description."
#         ),
#         "_impl_getattr_names": (
#             "A set storing all public names that __getattr__ is allowed to resolve."
#         ),
#         "_impl_attrs_protected": (
#             "A set storing protected field names. Protected names cannot be modified "
#             "through normal setattr."
#         ),
#     }

#     # Each instance has at most one owner and at most one registry relation at a time.
#     __relations__ = {
#         "owner": "The object that owns this instance.",
#         "registry": "The Registry object where this instance is registered.",
#     }
#     __properties__ = {}

#     __slots__ = tuple(__attrs__.keys()) + ("__weakref__",)

#     # ------------------------------------------------------------------
#     # Initialization
#     # ------------------------------------------------------------------

#     def __init__(self, *, name: str | None, name_replace: str):

#         if not hasattr(self, "_impl_extra_attrs"):
#             object.__setattr__(self, "_impl_extra_attrs", {})
#         if not hasattr(self, "_impl_extra_attrs_docs"):
#             object.__setattr__(self, "_impl_extra_attrs_docs", {})
#         if not hasattr(self, "_impl_relations"):
#             object.__setattr__(self, "_impl_relations", {})
#         if not hasattr(self, "_impl_relation_docs"):
#             object.__setattr__(self, "_impl_relation_docs", {})
#         if not hasattr(self, "_impl_getattr_names"):
#             object.__setattr__(self, "_impl_getattr_names", set())
#         if not hasattr(self, "_impl_attrs_protected"):
#             object.__setattr__(self, "_impl_attrs_protected", set())

#         self._helper_init_getattr_names_basic()
#         self._helper_init_relations_basic()

#         if name is None:
#             name = name_replace
#         else:
#             name = as_str(name, name=self.__attrs__["raw_name"], replace=name_replace)

#         self.act_set_name(name if name else name_replace)

#     def _helper_init_getattr_names_basic(self):
#         names = object.__getattribute__(self, "_impl_getattr_names")
#         for key in type(self).__attrs__.keys():
#             names.add(key)
#             if key.startswith("raw_"):
#                 names.add(key[4:])
#         names.update(type(self).__properties__.keys())
#         names.update(type(self).__relations__.keys())
#         names.update(object.__getattribute__(self, "_impl_extra_attrs_docs").keys())

#     def _helper_init_relations_basic(self):
#         relations = object.__getattribute__(self, "_impl_relations")
#         relation_docs = object.__getattribute__(self, "_impl_relation_docs")
#         for key, doc in type(self).__relations__.items():
#             relations.setdefault(key, None)
#             relation_docs.setdefault(key, doc)

#     # ------------------------------------------------------------------
#     # Readable-name registry
#     # ------------------------------------------------------------------

#     def _helper_register_getattr_name(self, name, *, allow_existing=False):
#         name = as_str(name, name="Readable attribute name")
#         names = self._impl_getattr_names
#         if (name in names) and (not allow_existing):
#             raise AttributeError(
#                 f"Cannot register readable name {name!r}: "
#                 f"it conflicts with an existing readable name of {type(self).__name__}."
#             )
#         names.add(name)
#         return name

#     # ------------------------------------------------------------------
#     # Relations
#     # ------------------------------------------------------------------

#     def _helper_resolve_relation_value(self, key):
#         value = self._impl_relations.get(key, None)
#         if isinstance(value, weakref.ReferenceType):
#             return value()
#         return value

#     def act_bind_relation_base(
#         self,
#         name: str,
#         target,
#         *,
#         doc: str | None = None,
#         is_weak: bool = True,
#         is_replace: bool = True,
#     ):
#         name = as_str(name, name=f"Relation name for instance {self.raw_name!r}")
#         if not name.isidentifier():
#             raise ValueError(
#                 f"Invalid relation name {name!r}: must be a valid Python identifier."
#             )

#         if name not in self._impl_relations:
#             self._helper_register_getattr_name(name)
#             if doc is None:
#                 doc = "Newly added relation."
#         elif doc is not None:
#             doc = as_str(doc, name=f"Relation doc for instance {self.raw_name!r}")

#         if doc is not None:
#             self._impl_relation_docs[name] = doc
#         old_target = self._helper_resolve_relation_value(name)
#         if old_target is not None and old_target is not target and (not is_replace):
#             raise RuntimeError(
#                 f"Relation {name!r} of {type(self).__name__} is already bound."
#             )

#         self._impl_relations[name] = (
#             weakref.ref(target) if (is_weak and target is not None) else target
#         )
#         return target

#     def act_unbind_relation_base(self, name: str):
#         name = as_str(name, name=f"Relation name for instance {self.raw_name!r}")
#         if name in self._impl_relations:
#             self._impl_relations[name] = None

#     # ------------------------------------------------------------------
#     # Core identity
#     # ------------------------------------------------------------------

#     @logging_and_warning_decorator(start_finish_level=5)
#     def act_set_name(self, name, logger=None):
#         try:
#             name = as_str(name, name=self.__attrs__["raw_name"])
#         except (TypeError, ValueError):
#             logger.exception("Invalid name.")
#             logger.recovery("Ignore this modification.")
#             return None

#         check_name = getattr(self.registry, "_helper_check_name", None)
#         if callable(check_name):
#             name = check_name(name)
#         object.__setattr__(self, "raw_name", name)

#         return name

#     # ------------------------------------------------------------------
#     # Protected attributes
#     # ------------------------------------------------------------------

#     def _helper_resolve_protected_target(self, name: str) -> str:
#         name = as_str(name, name=f"Protected attribute name for instance {self.name!r}")
#         if name in type(self).__attrs__:
#             return name
#         if name in self._impl_extra_attrs_docs:
#             return name
#         potential_raw = f"raw_{name}"
#         if potential_raw in type(self).__attrs__:
#             return potential_raw
#         raise AttributeError(
#             f"Cannot protect {name!r}: it is not a field in {type(self).__name__}.__attrs__ "
#             "and is not a registered extra attribute."
#         )

#     def act_register_protected_attr(self, attrs):
#         if isinstance(attrs, str):
#             attrs = [attrs]
#         elif not isinstance(attrs, (list, tuple, set)):
#             raise TypeError(
#                 "attrs must be a string or a sequence of strings, "
#                 f"got {type(attrs).__name__}."
#             )

#         for attr in attrs:
#             self._impl_attrs_protected.add(self._helper_resolve_protected_target(attr))

#     def act_unregister_protected_attr(self, attrs):
#         if isinstance(attrs, str):
#             attrs = [attrs]
#         elif not isinstance(attrs, (list, tuple, set)):
#             raise TypeError(
#                 "attrs must be a string or a sequence of strings, "
#                 f"got {type(attrs).__name__}."
#             )

#         for attr in attrs:
#             self._impl_attrs_protected.discard(
#                 self._helper_resolve_protected_target(attr)
#             )

#     # ------------------------------------------------------------------
#     # Attribute inspection
#     # ------------------------------------------------------------------

#     @classmethod
#     def _helper_is_writable_property(cls, name: str) -> bool:
#         desc = cls.__properties__.get(name, "")
#         return isinstance(desc, str) and desc.startswith("Writable:")

#     def show_attr_desc(self, attr_name: str) -> str:
#         descriptions_attrs = self.__class__.__attrs__
#         if attr_name in descriptions_attrs:
#             return f"{attr_name!r}: {descriptions_attrs[attr_name]}"

#         potential_raw = f"raw_{attr_name}"
#         if potential_raw in descriptions_attrs:
#             return (
#                 f"{attr_name!r}: Alias of {potential_raw!r}. "
#                 f"{descriptions_attrs[potential_raw]}"
#             )

#         descriptions_properties = self.__class__.__properties__
#         if attr_name in descriptions_properties:
#             return f"{attr_name!r}: {descriptions_properties[attr_name]}"

#         descriptions_relations = self._impl_relation_docs
#         if attr_name in descriptions_relations:
#             return f"{attr_name!r}: {descriptions_relations[attr_name]}"

#         descriptions_extra = self._impl_extra_attrs_docs
#         if attr_name in descriptions_extra:
#             return f"{attr_name!r}: {descriptions_extra[attr_name]}"

#         raise KeyError(
#             f"Attribute {attr_name!r} was not found in {type(self).__name__}.__attrs__ / "
#             "__properties__ / __relations__ / extra attrs."
#         )

#     @logging_and_warning_decorator(start_finish_level=5)
#     def show_getattrs(self, is_return=False, logger=None):
#         names = sorted(
#             name for name in self._impl_getattr_names if not name.startswith("_impl_")
#         )

#         lines = [
#             "When reading or assigning, the 'raw_' prefix may be omitted where a public alias exists."
#         ]
#         for name in names:
#             try:
#                 lines.append(self.show_attr_desc(name))
#             except KeyError:
#                 continue

#         if not lines:
#             lines.append("<none>")

#         output = "\n".join(lines)
#         logger.info(output)
#         if is_return:
#             return output

#     @logging_and_warning_decorator(start_finish_level=5)
#     def show_modifiable_attrs(self, is_return=False, logger=None):
#         protected = set(self._impl_attrs_protected)
#         attrs_fields = []
#         attrs_extra = []
#         attrs_properties = []

#         for attr_name in type(self).__attrs__:
#             if attr_name in protected:
#                 continue
#             if (
#                 attr_name.startswith("raw_")
#                 or attr_name.startswith("state_")
#                 or attr_name.startswith("default_")
#             ):
#                 attrs_fields.append(attr_name)

#         for attr_name in self._impl_extra_attrs_docs:
#             if attr_name not in protected:
#                 attrs_extra.append(attr_name)

#         for attr_name in type(self).__properties__.keys():
#             if self.__class__._helper_is_writable_property(attr_name):
#                 attrs_properties.append(attr_name)

#         lines = [
#             "When assigning, the 'raw_' prefix may be omitted.",
#         ]
#         if attrs_fields:
#             lines.extend([f"    * {name}" for name in sorted(attrs_fields)])
#         else:
#             lines.append("    * <none>")

#         lines.append(
#             "  - extra attrs: these are dynamically registered user attributes."
#         )
#         if attrs_extra:
#             lines.extend([f"    * {name}" for name in sorted(attrs_extra)])
#         else:
#             lines.append("    * <none>")

#         lines.append(
#             "  - writable properties: these are public properties whose setters are supported."
#         )
#         if attrs_properties:
#             lines.extend([f"    * {name}" for name in sorted(attrs_properties)])
#         else:
#             lines.append("    * <none>")

#         if protected:
#             lines.append(
#                 "Protected fields are excluded from the lists above and cannot be modified through normal setattr."
#             )

#         output = "\n".join(lines)
#         logger.info(output)
#         if is_return:
#             return output

#     @logging_and_warning_decorator(start_finish_level=5)
#     def show_relations(self, is_return=False, logger=None):
#         lines = []

#         for name in self._impl_relations.keys():
#             target = getattr(self, name, None)
#             if target is None:
#                 continue
#             desc = self._impl_relation_docs.get(name, "Newly added relation.")
#             lines.append(f"{name}: {desc}")
#             lines.append(f"  current: {target}")

#         if not lines:
#             lines.append("<none>")

#         output = "\n".join(lines)
#         logger.info(output)
#         if is_return:
#             return output

#     def _helper_relation_tree_node_label(self):
#         return str(self)

#     def _helper_relation_tree_walk(
#         self,
#         *,
#         depth: int,
#         is_include_none: bool,
#         is_follow_registry: bool,
#         _prefix: str = "",
#         _visited: set[int] | None = None,
#     ) -> list[str]:
#         if _visited is None:
#             _visited = set()

#         node_id = id(self)
#         lines = [f"{_prefix}{self._helper_relation_tree_node_label()}"]
#         if node_id in _visited:
#             lines[-1] += " [visited]"
#             return lines
#         _visited.add(node_id)

#         if depth <= 0:
#             return lines

#         entries = []
#         for name in self._impl_relations.keys():
#             target = self._helper_resolve_relation_value(name)
#             if target is None and not is_include_none:
#                 continue
#             entries.append((name, target))

#         if is_follow_registry and hasattr(self, "_entity"):
#             try:
#                 registry_items = list(self._entity)
#             except TypeError:
#                 registry_items = []
#             for index, item in enumerate(registry_items):
#                 if item is None and not is_include_none:
#                     continue
#                 entries.append((f"[{index}]", item))

#         if not entries:
#             lines.append(f"{_prefix}  <none>")
#             return lines

#         last_index = len(entries) - 1
#         for index, (name, target) in enumerate(entries):
#             branch = "└─ " if index == last_index else "├─ "
#             child_prefix = _prefix + ("   " if index == last_index else "│  ")
#             if target is None:
#                 lines.append(f"{_prefix}{branch}{name}: <none>")
#                 continue

#             lines.append(f"{_prefix}{branch}{name}:")
#             walk = getattr(target, "_helper_relation_tree_walk", None)
#             if callable(walk):
#                 lines.extend(
#                     walk(
#                         depth=depth - 1,
#                         is_include_none=is_include_none,
#                         is_follow_registry=is_follow_registry,
#                         _prefix=child_prefix,
#                         _visited=_visited,
#                     )
#                 )
#             else:
#                 lines.append(f"{child_prefix}{target}")

#         return lines

#     @logging_and_warning_decorator(start_finish_level=5)
#     def show_relation_tree(
#         self,
#         depth: int = 2,
#         is_return=False,
#         is_include_none: bool = False,
#         is_follow_registry: bool = False,
#         logger=None,
#     ):
#         depth = int(depth)
#         if depth < 0:
#             raise ValueError("depth must be >= 0.")

#         output = "\n".join(
#             self._helper_relation_tree_walk(
#                 depth=depth,
#                 is_include_none=is_include_none,
#                 is_follow_registry=is_follow_registry,
#             )
#         )
#         logger.info(output)
#         if is_return:
#             return output

#     # ------------------------------------------------------------------
#     # Dynamic extra attributes
#     # ------------------------------------------------------------------

#     def act_add_attr(
#         self,
#         name: str,
#         doc: str,
#         default=None,
#         overwrite: bool = False,
#     ):

#         name = as_str(name, name=f"Extra attribute name for instance {self.name!r}")
#         doc = as_str(doc, name=f"Extra attribute doc for instance {self.name!r}")

#         if not name.isidentifier():
#             raise ValueError(
#                 f"Invalid extra attribute name {name!r}: must be a valid Python identifier."
#             )

#         docs = self._impl_extra_attrs_docs
#         data = self._impl_extra_attrs

#         if (name in docs) and (not overwrite):
#             raise KeyError(
#                 f"Extra attribute {name!r} is already registered. Use overwrite=True to override."
#             )

#         if name not in docs:
#             self._helper_register_getattr_name(name)

#         docs[name] = doc
#         if overwrite or (name not in data):
#             data[name] = default

#     # ------------------------------------------------------------------
#     # Attribute access
#     # ------------------------------------------------------------------

#     def __getattr__(self, key):
#         if key in object.__getattribute__(self, "_impl_relations"):
#             return self._helper_resolve_relation_value(key)

#         potential_raw = f"raw_{key}"
#         if potential_raw in type(self).__attrs__:
#             return object.__getattribute__(self, potential_raw)

#         extra = object.__getattribute__(self, "_impl_extra_attrs")
#         if key in extra:
#             return extra[key]

#         cls_name = type(self).__name__
#         try:
#             obj_name = object.__getattribute__(self, "raw_name")
#         except AttributeError:
#             obj_name = "Uninitialized"
#         raise AttributeError(f"[{cls_name}: {obj_name!r}] has no attribute {key!r}.")

#     @logging_and_warning_decorator(start_finish_level=5)
#     def _helper_setattr_basic(self, key, value, logger=None):

#         if key in self._impl_extra_attrs_docs:
#             if key in self._impl_attrs_protected:
#                 logger.warning(
#                     f"{key!r} is protected on {type(self).__name__}. "
#                     "Please unprotect it before modifying."
#                 )
#                 return
#             self._impl_extra_attrs[key] = value
#             return

#         if key in self._impl_relations:
#             logger.warning(
#                 f"{key!r} is a relation of {type(self).__name__}. "
#                 "Please modify it via act_bind_relation_base() / act_unbind_relation_base()."
#             )
#             return

#         target_key = key
#         attrs_now = type(self).__attrs__
#         if key not in attrs_now:
#             potential_raw = f"raw_{key}"
#             if potential_raw in attrs_now:
#                 target_key = potential_raw
#             else:
#                 cls_name = self.__class__.__name__
#                 obj_name = getattr(self, "raw_name", "Uninitialized")
#                 raise AttributeError(
#                     f"[{cls_name}: {obj_name!r}] Assignment blocked: "
#                     f"{key!r} is not a valid or registered attribute."
#                 )

#         if target_key in self._impl_attrs_protected:
#             logger.warning(
#                 f"{target_key!r} is protected on {type(self).__name__}. "
#                 "Please unprotect it before modifying."
#             )
#             return

#         if target_key.startswith("_") or (
#             not target_key.startswith("raw_")
#             and not target_key.startswith("state_")
#             and not target_key.startswith("default_")
#         ):
#             cls_name = self.__class__.__name__
#             obj_name = getattr(self, "raw_name", "Uninitialized")
#             raise AttributeError(
#                 f"[{cls_name}: {obj_name!r}] Assignment blocked: "
#                 f"{key!r} resolves to internal attribute {target_key!r}, "
#                 "which cannot be assigned through the public setattr path."
#             )

#         self._helper_setattr_final(target_key, value)

#     def _helper_setattr_final(self, key, value):
#         if key == "raw_name":
#             self.act_set_name(value)
#             return
#         object.__setattr__(self, key, value)

#     def __setattr__(self, key, value):
#         self._helper_setattr_basic(key, value)

#     # ------------------------------------------------------------------
#     # Representation
#     # ------------------------------------------------------------------

#     def __repr__(self) -> str:
#         cls_name = self.__class__.__name__
#         msg = f"{cls_name}({self.name!r})"
#         return msg
