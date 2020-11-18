import re
from collections import defaultdict


class Tree(defaultdict):
    def __init__(self):
        super(Tree, self).__init__(Tree)

    def flatten(self, prefix=None, result=None):
        if prefix is None:
            prefix = ""
        else:
            prefix = prefix + "."
        if result is None:
            result = {}
        for k, v in self.items():
            if isinstance(v, Tree):
                v.flatten(prefix + k, result)
            else:
                result[prefix + k] = v
        return result


class Registry(object):
    table = Tree()
    def __new__(cls):
        raise ValueError("Registry shouldn't be instantiated.")

    @classmethod
    def register(cls, name):
        def wrapper(obj):
            entry = cls.table
            keys = name.split(".")
            for key in keys[:-1]:
                entry = entry[key]
            if keys[-1] in entry:
                raise KeyError("`%s` has already been registered by %s" % (name, entry[keys[-1]]))

            entry[keys[-1]] = obj
            obj._registry_key = name

            return obj

        return wrapper

    @classmethod
    def get(cls, name):
        entry = cls.table
        keys = name.split(".")
        for i, key in enumerate(keys):
            if key not in entry:
                raise KeyError("Can't find `%s` in `%s`" % (key, ".".join(keys[:i])))
            entry = entry[key]
        return entry

    @classmethod
    def search(cls, name):
        keys = []
        pattern = re.compile(r"\b%s\b" % name)
        for k, v in cls.table.flatten().items():
            if pattern.search(k):
                keys.append(k)
                value = v
        if len(keys) == 0:
            raise KeyError("Can't find any registered key containing `%s`" % name)
        if len(keys) > 1:
            keys = ["`%s`" % key for key in keys]
            raise KeyError("Ambiguous key `%s`. Found %s" % (name, ", ".join(keys)))
        return value


