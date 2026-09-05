from dataclasses import dataclass
import tempfile
import unittest
from types import MappingProxyType

from nematics3d.datatypes import UNSET
from nematics3d.core.class_base import AttrDef
from nematics3d.core.host_base import HostBase, OptsBase


@dataclass(slots=True, repr=False)
class DemoOpts(OptsBase):
    width: int = 1

    __attrs__ = {
        **OptsBase.__attrs__,
        "width": "The width of the demo object.",
    }

    impl_validators = {
        **OptsBase.impl_validators,
        "width": lambda value, desc: int(value),
    }

    impl_defaults_frozen = MappingProxyType(
        {
            **dict(OptsBase.impl_defaults_frozen),
            "width": 1,
        }
    )


class DemoHost(HostBase):
    __attr_defs__ = {
        "raw_level": AttrDef(
            doc="The level of the demo host.",
            kind="raw",
            validator=lambda value, desc: int(value),
            is_reapply_opts_after_raw=True,
        ),
        "state_count": AttrDef(
            doc="Writable runtime count.",
            kind="state",
            validator=lambda value, desc: int(value),
        ),
        "theme": AttrDef(
            doc="Writable display theme.",
            kind="property",
            is_public_settable=True,
        ),
    }

    __slots__ = ("raw_level", "state_count", "_theme", "impl_apply_calls")

    def __init__(self, opts=None, opts_defaults_override=None, **kwargs):
        super().__init__(
            opts_type=DemoOpts,
            opts=opts,
            opts_defaults_override=opts_defaults_override,
            name="demo",
            name_replace="demo",
            **kwargs,
        )
        if not hasattr(self, "raw_level"):
            object.__setattr__(self, "raw_level", 1)
        if not hasattr(self, "state_count"):
            object.__setattr__(self, "state_count", 0)
        object.__setattr__(self, "_theme", "light")
        object.__setattr__(self, "impl_apply_calls", [])

    @property
    def theme(self):
        return self._theme

    @theme.setter
    def theme(self, value):
        object.__setattr__(self, "_theme", value)

    def _helper_commit_apply_opts_main(self, is_reapply_opts=False, **kwargs):
        self.impl_apply_calls.append((bool(is_reapply_opts), dict(kwargs)))
        with self.opts.act_internal_update():
            for key, value in kwargs.items():
                setattr(self.opts, key, value)


@dataclass(slots=True, repr=False)
class WrapperOpts(OptsBase):
    alpha: int = 1

    __attrs__ = {
        **OptsBase.__attrs__,
        "alpha": "Wrapper-only option.",
    }

    impl_validators = {
        **OptsBase.impl_validators,
        "alpha": lambda value, desc: int(value),
    }

    impl_defaults_frozen = MappingProxyType(
        {
            **dict(OptsBase.impl_defaults_frozen),
            "alpha": 1,
        }
    )


class WrapperHost(HostBase):
    __slots__ = ("impl_apply_calls",)

    def __init__(self):
        super().__init__(
            opts_type=WrapperOpts,
            name="wrapper",
            name_replace="wrapper",
        )
        object.__setattr__(self, "impl_apply_calls", [])
        self.opts.act_finalize(defaults=self.opts_defaults)

    def _helper_commit_apply_opts_main(self, is_reapply_opts=False, **kwargs):
        self.impl_apply_calls.append((bool(is_reapply_opts), dict(kwargs)))
        with self.opts.act_internal_update():
            for key, value in kwargs.items():
                setattr(self.opts, key, value)


class BadReturnHost(DemoHost):
    def _helper_commit_apply_opts_main(self, is_reapply_opts=False, **kwargs):
        del is_reapply_opts, kwargs
        return "bad-return"


@dataclass(slots=True, repr=False)
class MissingAttrFieldOpts(OptsBase):
    __attrs__ = {
        **OptsBase.__attrs__,
        "ghost": "Missing dataclass field.",
    }


@dataclass(slots=True, repr=False)
class InvalidValidatorKeyOpts(OptsBase):
    width: int = 1

    __attrs__ = {
        **OptsBase.__attrs__,
        "width": "Width field.",
    }

    impl_validators = {
        **OptsBase.impl_validators,
        "ghost": int,
    }


@dataclass(slots=True, repr=False)
class InvalidDefaultKeyOpts(OptsBase):
    width: int = 1

    __attrs__ = {
        **OptsBase.__attrs__,
        "width": "Width field.",
    }

    impl_defaults_frozen = MappingProxyType(
        {
            **dict(OptsBase.impl_defaults_frozen),
            "ghost": 3,
        }
    )


class TestHostBase(unittest.TestCase):
    def test_opts_attrs_keys_must_be_dataclass_fields(self):
        with self.assertRaisesRegex(ValueError, "__attrs__ keys"):
            MissingAttrFieldOpts()

    def test_opts_validators_keys_must_belong_to_attrs(self):
        with self.assertRaisesRegex(ValueError, "impl_validators keys"):
            InvalidValidatorKeyOpts()

    def test_opts_defaults_keys_must_belong_to_attrs(self):
        with self.assertRaisesRegex(ValueError, "impl_defaults_frozen keys"):
            InvalidDefaultKeyOpts()

    def test_show_modifiable_attrs_excludes_forbidden_properties(self):
        host = DemoHost()
        host.act_register_protected_attr("theme")

        output = host.show_modifiable_attrs(is_return=True)

        self.assertNotIn("'theme': Writable display theme.", output)
        self.assertIn("Protected or wrapped fields are excluded", output)

    def test_internal_impl_assignment_rejects_unknown_field(self):
        opts = DemoOpts()

        with self.assertRaisesRegex(AttributeError, "Invalid internal option field"):
            opts.impl_missing = 1

    def test_commit_apply_opts_main_must_return_two_tuple(self):
        host = BadReturnHost()

        with self.assertRaisesRegex(TypeError, "must return None or a 2-tuple"):
            host.act_commit(width=3)

    def test_commit_extra_ignores_protected_extra_attr(self):
        host = DemoHost()
        host.act_add_attr("note", "Extra note.", default="a")
        host.act_register_protected_attr("note")

        host.act_commit(note="b")

        self.assertEqual(host.note, "a")
        self.assertIn("note", host.attrs_protected)
        self.assertIn("note", host.attrs_forbidden)

    def test_fixed_opts_protects_and_unprotects_all_opts_fields(self):
        host = DemoHost()
        host.act_register_protected_opts_all()

        host.act_commit(width=5)
        host.width = 6
        host.opts.width = 7

        self.assertEqual(host.opts.width, 1)
        self.assertIn("tag", host.attrs_protected)
        self.assertIn("width", host.attrs_protected)

        host.act_unregister_protected_opts_all()
        host.act_commit(width=5)

        self.assertEqual(host.opts.width, 5)
        self.assertNotIn("width", host.attrs_protected)

    def test_show_modifiable_attrs_excludes_forbidden_extra_attr(self):
        host = DemoHost()
        host.act_add_attr("note", "Extra note.", default="a")
        host.act_register_protected_attr("note")

        output = host.show_modifiable_attrs(is_return=True)

        self.assertNotIn("'note': Extra note.", output)
        self.assertIn("note", host.attrs_forbidden)

    def test_register_wrapped_attr_accepts_extra_attr(self):
        host = DemoHost()
        host.act_add_attr("note", "Extra note.", default="a")

        host.act_register_wrapped_attr("note")

        self.assertIn("note", host.attrs_wrapped)
        self.assertIn("note", host.attrs_forbidden)

    def test_opts_finalize_fills_frozen_defaults(self):
        opts = DemoOpts(tag=UNSET, width=UNSET)

        opts.act_finalize()

        self.assertTrue(opts.impl_is_functioning)
        self.assertEqual(opts.tag, "options")
        self.assertEqual(opts.width, 1)

    def test_opts_finalize_prefers_explicit_defaults(self):
        opts = DemoOpts(tag=UNSET, width=UNSET)

        opts.act_finalize(defaults={"tag": "custom", "width": 7})

        self.assertEqual(opts.tag, "custom")
        self.assertEqual(opts.width, 7)

    def test_opts_finalize_rejects_missing_default_unless_allowed(self):
        @dataclass(slots=True, repr=False)
        class NoDefaultOpts(OptsBase):
            value: int | object = UNSET

            __attrs__ = {**OptsBase.__attrs__, "value": "No frozen default."}
            impl_defaults_frozen = MappingProxyType(dict(OptsBase.impl_defaults_frozen))

        opts = NoDefaultOpts()
        with self.assertRaisesRegex(KeyError, "Missing default"):
            opts.act_finalize()

        opts = NoDefaultOpts()
        opts.act_finalize(is_allow_unset=True)
        self.assertIs(opts.value, UNSET)

    def test_opts_cannot_finalize_twice(self):
        opts = DemoOpts()
        opts.act_finalize()

        with self.assertRaisesRegex(RuntimeError, "already been finalized"):
            opts.act_finalize()

    def test_opts_asdict_can_include_or_exclude_unset(self):
        opts = DemoOpts(tag=UNSET, width=UNSET)

        self.assertEqual(opts.act_asdict(), {})
        self.assertEqual(
            opts.act_asdict(is_include_unset=True),
            {"tag": UNSET, "width": UNSET},
        )

    def test_internal_update_restores_functioning_state_after_exception(self):
        opts = DemoOpts()
        opts.act_finalize()

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with opts.act_internal_update():
                self.assertFalse(opts.impl_is_functioning)
                raise RuntimeError("boom")

        self.assertTrue(opts.impl_is_functioning)

    def test_functioning_opts_assignment_forwards_to_host(self):
        host = DemoHost()
        host.opts.act_finalize(defaults=host.opts_defaults)

        host.opts.width = 4

        self.assertEqual(host.opts.width, 4)
        self.assertEqual(host.impl_apply_calls[-1], (False, {"width": 4}))

    def test_host_assignment_to_opts_field_uses_commit_pipeline(self):
        host = DemoHost()
        host.opts.act_finalize(defaults=host.opts_defaults)

        host.width = 5

        self.assertEqual(host.opts.width, 5)
        self.assertEqual(host.impl_apply_calls[-1], (False, {"width": 5}))

    def test_functioning_opts_rejects_unset(self):
        host = DemoHost()
        host.opts.act_finalize(defaults=host.opts_defaults)

        host.opts.width = UNSET

        self.assertEqual(host.opts.width, 1)

    def test_invalid_functioning_opts_assignment_is_ignored(self):
        host = DemoHost()
        host.opts.act_finalize(defaults=host.opts_defaults)

        host.opts.width = "not-an-int"

        self.assertEqual(host.opts.width, 1)

    def test_invalid_pre_finalize_opts_assignment_resets_to_unset(self):
        opts = DemoOpts()

        opts.width = "not-an-int"

        self.assertIs(opts.width, UNSET)

    def test_constructor_accepts_opts_instance_and_flat_overrides(self):
        opts = DemoOpts(tag="base", width=2)

        host = DemoHost(opts=opts, width=6)

        self.assertIsNot(host.opts, opts)
        self.assertEqual(host.opts.tag, "base")
        self.assertEqual(host.opts.width, 6)
        self.assertIs(host.opts.host, host)

    def test_constructor_applies_valid_opts_defaults_override(self):
        host = DemoHost(opts_defaults_override={"width": 9})

        self.assertEqual(host.opts_defaults["width"], 9)
        self.assertEqual(host.opts_defaults["tag"], "options")

    def test_constructor_can_initialize_raw_and_state_inputs(self):
        host = DemoHost(level="3", state_count="8")

        self.assertEqual(host.level, 3)
        self.assertEqual(host.state_count, 8)

    def test_commit_raw_alias_validates_and_updates_storage(self):
        host = DemoHost()
        host.opts.act_finalize(defaults=host.opts_defaults)

        host.act_commit(level="4")

        self.assertEqual(host.raw_level, 4)
        self.assertEqual(host.level, 4)

    def test_commit_raw_canonical_name_validates_and_updates_storage(self):
        host = DemoHost()
        host.opts.act_finalize(defaults=host.opts_defaults)

        host.act_commit(raw_level="5")

        self.assertEqual(host.level, 5)

    def test_commit_rejects_ambiguous_raw_alias_pair(self):
        host = DemoHost()

        with self.assertRaisesRegex(TypeError, "Ambiguous input"):
            host.act_commit(level=2, raw_level=3)

    def test_invalid_raw_commit_is_ignored(self):
        host = DemoHost()

        host.act_commit(level="bad")

        self.assertEqual(host.level, 1)

    def test_state_commit_validates_and_updates_storage(self):
        host = DemoHost()

        host.act_commit(state_count="11")

        self.assertEqual(host.state_count, 11)

    def test_raw_change_can_request_opts_reapply(self):
        host = DemoHost()
        host.opts.act_finalize(defaults=host.opts_defaults)

        host.act_commit(level=4)

        self.assertEqual(host.impl_apply_calls[-1], (True, {}))

    def test_explicit_opts_reapply_calls_subclass_hook_without_opts_changes(self):
        host = DemoHost()
        host.opts.act_finalize(defaults=host.opts_defaults)

        host.act_commit(is_reapply_opts=True)

        self.assertEqual(host.impl_apply_calls[-1], (True, {}))

    def test_commit_name_accepts_public_and_raw_forms(self):
        host = DemoHost()

        host.act_commit(name="renamed")
        self.assertEqual(host.name, "renamed")

        host.act_commit(raw_name="renamed-again")
        self.assertEqual(host.name, "renamed-again")

    def test_commit_name_rejects_public_and_raw_forms_together(self):
        host = DemoHost()

        with self.assertRaisesRegex(TypeError, "Ambiguous input"):
            host.act_commit(name="a", raw_name="b")

    def test_commit_extra_attr_validates_and_updates(self):
        host = DemoHost()
        host.act_add_attr(
            "note",
            "Integer note.",
            default=1,
            validator=lambda value, desc: int(value),
        )

        host.act_commit(note="3")

        self.assertEqual(host.note, 3)

    def test_private_commit_keys_are_ignored(self):
        host = DemoHost()

        host.act_commit(_secret=1, impl_secret=2)

        with self.assertRaises(AttributeError):
            _ = host._secret
        with self.assertRaises(AttributeError):
            _ = host.impl_secret

    def test_protected_raw_attr_blocks_commit_and_direct_assignment(self):
        host = DemoHost()
        host.act_register_protected_attr("level")

        host.act_commit(level=5)
        self.assertEqual(host.level, 1)

        with self.assertRaises(AttributeError):
            host.level = 6
        self.assertEqual(host.level, 1)

    def test_protected_opts_attr_blocks_all_public_update_paths(self):
        host = DemoHost()
        host.opts.act_finalize(defaults=host.opts_defaults)
        host.act_register_protected_attr("width")

        host.act_commit(width=4)
        host.width = 5
        host.opts.width = 6

        self.assertEqual(host.opts.width, 1)

    def test_unregister_protected_attr_restores_updates(self):
        host = DemoHost()
        host.opts.act_finalize(defaults=host.opts_defaults)
        host.act_register_protected_attr(["level", "width"])
        host.act_unregister_protected_attr(["level", "width"])

        host.act_commit(level=4, width=5)

        self.assertEqual(host.level, 4)
        self.assertEqual(host.opts.width, 5)

    def test_wrapped_update_temporarily_allows_wrapped_fields_and_restores_flags(self):
        host = DemoHost()
        host.opts.act_finalize(defaults=host.opts_defaults)
        host.act_register_wrapped_attr(["level", "width"])

        with host.act_wrapped_update():
            self.assertNotIn("level", host.attrs_wrapped)
            self.assertNotIn("width", host.attrs_wrapped)
            host.act_commit(level=4, width=5)

        self.assertEqual(host.level, 4)
        self.assertEqual(host.opts.width, 5)
        self.assertIn("level", host.attrs_wrapped)
        self.assertIn("width", host.attrs_wrapped)

    def test_nested_wrapped_update_preserves_outer_unwrapped_state_until_exit(self):
        host = DemoHost()
        host.act_register_wrapped_attr("level")

        with host.act_wrapped_update():
            self.assertNotIn("level", host.attrs_wrapped)
            with host.act_wrapped_update():
                self.assertNotIn("level", host.attrs_wrapped)
            self.assertNotIn("level", host.attrs_wrapped)

        self.assertIn("level", host.attrs_wrapped)

    def test_bind_and_unbind_wrapper_manage_relations_and_wrapped_flags(self):
        wrapper = WrapperHost()
        wrapped = DemoHost()

        wrapped.act_bind_wrapper(wrapper, protected_attrs=["level", "width"])

        self.assertIs(wrapped.wrapper, wrapper)
        self.assertIs(wrapper.wrapped, wrapped)
        self.assertIn("level", wrapped.attrs_wrapped)
        self.assertIn("width", wrapped.attrs_wrapped)

        wrapped.act_unbind_wrapper()

        self.assertIsNone(wrapped.wrapper)
        self.assertIsNone(wrapper.wrapped)
        self.assertFalse(wrapped.attrs_wrapped)

    def test_wrapper_rejects_conflicting_existing_bindings(self):
        wrapper = WrapperHost()
        first = DemoHost()
        second = DemoHost()
        first.act_bind_wrapper(wrapper)

        with self.assertRaisesRegex(RuntimeError, "already wraps"):
            second.act_bind_wrapper(wrapper)

    def test_leftover_kwargs_forward_to_wrapped_host(self):
        wrapper = WrapperHost()
        wrapped = DemoHost()
        wrapped.opts.act_finalize(defaults=wrapped.opts_defaults)
        wrapped.act_bind_wrapper(wrapper, protected_attrs="width")

        wrapper.act_commit(width=7)

        self.assertEqual(wrapped.opts.width, 7)

    def test_opts_wrapped_forwards_explicit_opts_object(self):
        wrapper = WrapperHost()
        wrapped = DemoHost()
        wrapped.opts.act_finalize(defaults=wrapped.opts_defaults)
        wrapped.act_bind_wrapper(wrapper)

        wrapper.act_commit(opts_wrapped=DemoOpts(width=8))

        self.assertEqual(wrapped.opts.width, 8)

    def test_wrapped_kwargs_enrichment_runs_before_forwarding(self):
        wrapper = WrapperHost()
        wrapped = DemoHost()
        wrapped.opts.act_finalize(defaults=wrapped.opts_defaults)
        wrapped.act_bind_wrapper(wrapper)
        wrapper.act_attach_enrich_kwargs_wrapped_task(
            "add-width",
            lambda host, kwargs: {"width": 9},
        )

        wrapper.act_commit(alpha=2)

        self.assertEqual(wrapper.opts.alpha, 2)
        self.assertEqual(wrapped.opts.width, 9)

    def test_sync_task_receives_successfully_applied_raw_and_opts_changes(self):
        host = DemoHost()
        host.opts.act_finalize(defaults=host.opts_defaults)
        calls = []
        host.act_attach_sync_task("record", lambda **kwargs: calls.append(dict(kwargs)))

        host.act_commit(level=3, width=4)

        self.assertEqual(calls, [{"level": 3, "width": 4}])

    def test_sync_enrichment_can_extend_payload(self):
        host = DemoHost()
        host.opts.act_finalize(defaults=host.opts_defaults)
        calls = []
        host.act_attach_enrich_kwargs_sync_task(
            "derived",
            lambda host, kwargs_sync: {"double_level": kwargs_sync["level"] * 2},
        )
        host.act_attach_sync_task("record", lambda **kwargs: calls.append(dict(kwargs)))

        host.act_commit(level=3)

        self.assertEqual(calls, [{"level": 3, "double_level": 6}])

    def test_failing_sync_task_does_not_block_other_tasks(self):
        host = DemoHost()
        calls = []

        def fail(**kwargs):
            raise RuntimeError("expected")

        host.act_attach_sync_task("fail", fail)
        host.act_attach_sync_task("record", lambda **kwargs: calls.append(dict(kwargs)))

        host.act_commit(state_count=2)

        self.assertEqual(calls, [{"state_count": 2}])

    def test_detach_sync_and_enrichment_tasks(self):
        host = DemoHost()
        host.act_attach_sync_task("sync", lambda **kwargs: None)
        host.act_attach_enrich_kwargs_sync_task("sync-enrich", lambda **kwargs: {})
        host.act_attach_enrich_kwargs_wrapped_task(
            "wrapped-enrich", lambda **kwargs: {}
        )

        host.act_detach_sync_task("sync")
        host.act_detach_enrich_kwargs_sync_task("sync-enrich")
        host.act_detach_enrich_kwargs_wrapped_task("wrapped-enrich")

        self.assertEqual(host.impl_sync_func, {})
        self.assertEqual(host.impl_enrich_kwargs_sync_func, {})
        self.assertEqual(host.impl_enrich_kwargs_wrapped_func, {})

    def test_attach_callback_rejects_non_callable(self):
        host = DemoHost()

        with self.assertRaises(TypeError):
            host.act_attach_sync_task("bad", 1)
        with self.assertRaises(TypeError):
            host.act_attach_enrich_kwargs_sync_task("bad", 1)
        with self.assertRaises(TypeError):
            host.act_attach_enrich_kwargs_wrapped_task("bad", 1)

    def test_save_opts_creates_independent_snapshot(self):
        host = DemoHost()
        host.opts.act_finalize(defaults=host.opts_defaults)
        host.act_commit(width=3)

        host.act_save_opts("saved")
        host.act_commit(width=4)

        self.assertEqual(host.opts_backup["saved"]["width"], 3)
        self.assertEqual(host.opts.width, 4)

    def test_show_saved_opts_reports_named_snapshot(self):
        host = DemoHost()
        host.act_save_opts("baseline")

        output = host.show_saved_opts(is_return=True)

        self.assertIn("baseline", output)
        self.assertIn("act_commit", output)

    def test_show_readable_attrs_includes_host_and_opts_surfaces(self):
        host = DemoHost()

        output = host.show_readable_attrs(is_return=True)

        self.assertIn("'level'", output)
        self.assertIn("'state_count'", output)
        self.assertIn("[Opts attributes]", output)
        self.assertIn("'width'", output)

    def test_show_attr_doc_resolves_opts_field(self):
        host = DemoHost()

        self.assertEqual(
            host.show_attr_doc("width", is_return=True), "The width of the demo object."
        )

    def test_show_modifiable_attrs_lists_writable_property(self):
        host = DemoHost()

        output = host.show_modifiable_attrs(is_return=True)

        self.assertIn("[Host writable properties]", output)
        self.assertIn("'theme': Writable display theme.", output)

    def test_writable_property_assignment_uses_property_setter(self):
        host = DemoHost()

        host.theme = "dark"

        self.assertEqual(host.theme, "dark")

    def test_commit_writable_property_uses_setter_and_syncs_applied_value(self):
        host = DemoHost()
        calls = []
        host.act_attach_sync_task("record", lambda **kwargs: calls.append(dict(kwargs)))

        host.act_commit(theme="dark")

        self.assertEqual(host.theme, "dark")
        self.assertEqual(calls, [{"theme": "dark"}])

    def test_opts_json_round_trip_preserves_public_payload(self):
        opts = DemoOpts(tag="saved", width=7)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = opts.act_save_json(f"{tmpdir}/demo_opts.json")
            loaded = DemoOpts(tag="other", width=1)
            result = loaded.act_load_json(path)

        self.assertIs(result, loaded)
        self.assertEqual(loaded.act_asdict(), {"tag": "saved", "width": 7})

    def test_opts_json_load_can_finalize_fresh_opts(self):
        source = DemoOpts(tag="saved", width=7)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = source.act_save_json(f"{tmpdir}/demo_opts.json")
            loaded = DemoOpts(tag=UNSET, width=UNSET)
            loaded.act_load_json(path, is_finalize=True)

        self.assertTrue(loaded.impl_is_functioning)
        self.assertEqual(loaded.tag, "saved")
        self.assertEqual(loaded.width, 7)


if __name__ == "__main__":
    unittest.main()
