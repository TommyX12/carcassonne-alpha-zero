import traceback
from typing import List


class ValueFactory(object):
    def __init__(self, fn):
        assert callable(fn), 'fn must be callable: {}'.format(fn)
        self._fn = fn

    def __call__(self, ctx):
        return self._fn(ctx)


def factory(fn):
    return ValueFactory(fn)


class DefaultValue(object):
    def __init__(self, value):
        self.value = value


def default(v):
    return DefaultValue(v)


class ValuePatch(object):
    def __init__(self, fn):
        assert callable(fn), 'fn must be callable: {}'.format(fn)
        self._fn = fn

    def __call__(self, ctx, base):
        return self._fn(ctx, base)


def patch(fn):
    return ValuePatch(fn)


def patch_append(v):
    def fn(ctx, base):
        base.append(v)
        return base

    return patch(fn)


def patch_append_factory(factory):
    def fn(ctx, base):
        base.append(factory(ctx))
        return base

    return patch(fn)


class ContextValue(object):
    def __init__(self):
        self._raw_value = None
        self._value = None
        self._value_computed = False
        self._patches: List[ValuePatch] = []

    def set_raw_value(self, value):
        self._raw_value = value
        self._value_computed = False
        self._patches = []

    def add_patch(self, patch):
        if self._value_computed:
            self.set_raw_value(self._value)

        self._patches.append(patch)

    def _compute_value(self, ctx):
        value = self._raw_value
        if isinstance(value, ValueFactory):
            value = value(ctx)

        for patch in self._patches:
            value = patch(ctx, value)

        return value

    def get_value(self, ctx):
        if not self._value_computed:
            self._value = self._compute_value(ctx)
            self._value_computed = True

        return self._value


class Context(object):
    def __init__(self, **kwargs):
        object.__setattr__(self, '_values_', {})
        object.__setattr__(self, '_visiting_', set())
        object.__setattr__(self, '_installed_modules_', set())

        self.bind(**kwargs)

    def __getitem__(self, name):
        if name in self._visiting_:
            raise RuntimeError('Circular dependency detected: {}'.format(name))

        self._visiting_.add(name)
        try:
            if name not in self._values_:
                raise AttributeError('Cannot find context attribute: {}'.format(name))

            return self._values_[name].get_value(self)

        finally:
            self._visiting_.remove(name)

    def __setitem__(self, name, value):
        if name in self.__dict__ or name in Context.__dict__:
            raise ValueError(
                'Cannot override special context attributes: {}'.format(name))

        if isinstance(value, DefaultValue):
            if name in self._values_:
                return

            value = value.value

        if isinstance(value, ValuePatch):
            if name not in self._values_:
                raise KeyError('Patch failed: cannot find context attribute: {}'.format(name))

            self._values_[name].add_patch(value)
            return

        if name not in self._values_:
            self._values_[name] = ContextValue()

        self._values_[name].set_raw_value(value)
        
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def bind(self, fn=None, **kwargs):
        if fn is not None:
            self[fn.__name__] = fn

        for k, v in kwargs.items():
            self[k] = v

        return fn

    def bind_factory(self, fn=None, **kwargs):
        if fn is not None:
            assert not isinstance(fn, ValueFactory), 'Do not use ValueFactory in bind_factory: {}'.format(fn)
            self[fn.__name__] = factory(fn)

        for k, v in kwargs.items():
            assert not isinstance(v, ValueFactory), 'Do not use ValueFactory in bind_factory: {}'.format(v)
            self[k] = factory(v)

        return fn

    def bind_default(self, fn=None, **kwargs):
        if fn is not None:
            assert not isinstance(fn, DefaultValue), 'Do not use DefaultValue in bind_default: {}'.format(fn)
            self[fn.__name__] = default(fn)

        for k, v in kwargs.items():
            assert not isinstance(v, DefaultValue), 'Do not use DefaultValue in bind_default: {}'.format(v)
            self[k] = default(v)

        return fn

    def bind_patch(self, fn=None, **kwargs):
        if fn is not None:
            assert not isinstance(fn, ValuePatch), 'Do not use ValuePatch in bind_patch: {}'.format(fn)
            self[fn.__name__] = patch(fn)

        for k, v in kwargs.items():
            assert not isinstance(v, ValuePatch), 'Do not use ValuePatch in bind_patch: {}'.format(v)
            self[k] = patch(v)

        return fn

    def install(self, module_fn):
        assert callable(module_fn), 'module_fn must be callable: {}'.format(module_fn)
        if module_fn in self._installed_modules_:
            return

        self._installed_modules_.add(module_fn)
        module_fn(self)

    def keys(self):
        return self._values_.keys()


class FrozenContext(Context):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, '_frozen_', True)

    def __setitem__(self, name, value):
        if self._frozen_:
            raise RuntimeError('Cannot modify frozen context')

        super().__setitem__(name, value)

