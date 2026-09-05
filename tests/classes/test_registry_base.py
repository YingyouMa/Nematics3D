import unittest

from nematics3d.core.class_base import ClassBase
from nematics3d.core.registry_base import RegistryBase


class DemoRegistered(ClassBase):
    __slots__ = ()

    def __init__(self, name):
        super().__init__(name=name, name_replace="demo")


class TestRegistryBase(unittest.TestCase):
    def test_register_returns_term_and_binds_registry_relation(self):
        registry = RegistryBase("demo registry")
        term = DemoRegistered("term")

        result = registry.act_register(term)

        self.assertIs(result, term)
        self.assertIs(term.registry, registry)
        self.assertIs(registry["term"], term)
        self.assertIs(registry[0], term)

    def test_register_duplicate_name_renames_new_object(self):
        registry = RegistryBase("demo registry")
        first = DemoRegistered("same")
        second = DemoRegistered("same")

        registry.act_register(first)
        registry.act_register(second)

        self.assertEqual(first.name, "same")
        self.assertEqual(second.name, "same_1")
        self.assertIs(registry["same"], first)
        self.assertIs(registry["same_1"], second)

    def test_registered_object_can_keep_its_current_name(self):
        registry = RegistryBase("demo registry")
        term = DemoRegistered("same")
        registry.act_register(term)

        result = term.act_set_name("same")

        self.assertEqual(result, "same")
        self.assertEqual(term.name, "same")
        self.assertIs(registry["same"], term)

    def test_registered_object_rename_avoids_other_registered_names(self):
        registry = RegistryBase("demo registry")
        first = DemoRegistered("first")
        second = DemoRegistered("second")
        registry.act_register(first)
        registry.act_register(second)

        result = second.act_set_name("first")

        self.assertEqual(result, "first_1")
        self.assertEqual(second.name, "first_1")
        self.assertIs(registry["first"], first)
        self.assertIs(registry["first_1"], second)

    def test_register_moves_object_from_previous_registry(self):
        first_registry = RegistryBase("first registry")
        second_registry = RegistryBase("second registry")
        term = DemoRegistered("term")
        first_registry.act_register(term)

        result = second_registry.act_register(term)

        self.assertIs(result, term)
        self.assertNotIn(term, first_registry)
        self.assertIn(term, second_registry)
        self.assertIs(term.registry, second_registry)

    def test_register_existing_term_can_return_term_when_allowed(self):
        registry = RegistryBase("demo registry")
        term = DemoRegistered("term")
        registry.act_register(term)

        result = registry.act_register(term, is_contain_ok=True)

        self.assertIs(result, term)
        self.assertEqual(len(registry), 1)

    def test_entity_is_immutable_tuple_view_and_iteration_keeps_order(self):
        registry = RegistryBase("demo registry")
        first = DemoRegistered("first")
        second = DemoRegistered("second")
        registry.act_register(first)
        registry.act_register(second)

        self.assertEqual(registry.entity, (first, second))
        self.assertEqual(tuple(registry), (first, second))
        self.assertEqual(len(registry), 2)
        self.assertIsNone(registry[None])

    def test_unregister_unbinds_registry_relation(self):
        registry = RegistryBase("demo registry")
        term = DemoRegistered("term")
        registry.act_register(term)

        registry.act_unregister(term)

        self.assertNotIn(term, registry)
        self.assertIsNone(term.registry)

    def test_lookup_rejects_unsupported_key_type(self):
        registry = RegistryBase("demo registry")
        registry.act_register(DemoRegistered("term"))

        with self.assertRaises(TypeError):
            registry[1.5]

    def test_act_clear_unregisters_all_and_unbinds_registry(self):
        registry = RegistryBase("demo registry")
        first = DemoRegistered("first")
        second = DemoRegistered("second")
        registry.act_register(first)
        registry.act_register(second)

        result = registry.act_clear(is_show_existing=False)

        self.assertIsNone(result)
        self.assertEqual(len(registry), 0)
        self.assertIsNone(first.registry)
        self.assertIsNone(second.registry)

    def test_act_clear_can_return_removed_objects(self):
        registry = RegistryBase("demo registry")
        first = DemoRegistered("first")
        second = DemoRegistered("second")
        registry.act_register(first)
        registry.act_register(second)

        removed = registry.act_clear(
            is_return_removed=True,
            is_show_existing=False,
        )

        self.assertEqual(removed, (first, second))
        self.assertEqual(len(registry), 0)


if __name__ == "__main__":
    unittest.main()
