def auto_properties(bindings: dict):

    def decorator(cls):
        for name, path in bindings.items():
            attr_chain = path.split('.')  # 例如 ['actor', 'actor', 'property', 'color']

            def make_getter(attrs):
                def getter(self):
                    target = self
                    for attr in attrs[:-1]:
                        target = getattr(target, attr)
                    return getattr(target, attrs[-1])
                return getter

            def make_setter(attrs):
                def setter(self, value):
                    target = self
                    for attr in attrs[:-1]:
                        target = getattr(target, attr)
                    setattr(target, attrs[-1], value)
                return setter

            # 生成并绑定属性
            setattr(cls, name, property(make_getter(attr_chain), make_setter(attr_chain)))

        return cls

    return decorator
