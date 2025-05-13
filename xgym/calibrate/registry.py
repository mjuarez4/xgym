class Registry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
            cls._instance._endpoints = {}
        return cls._instance

    def endpoint(self):
        def decorator(cls):
            key = cls.__name__
            if key in self._registry:
                raise ValueError(f"Endpoint '{key}' already exists.")
            self._registry[key] = {}
            self._endpoints[cls] = self._registry[key]
            return cls

        return decorator

    def register(self, name):
        def decorator(cls):
            parent_dict = None
            for base in cls.__bases__:
                if base in self._endpoints:
                    parent_dict = self._endpoints[base]
                    break

            if parent_dict is None:
                # Register in top-level
                if name in self._registry:
                    raise ValueError(f"'{name}' already registered at root.")
                self._registry[name] = cls
            else:
                if name in parent_dict:
                    raise ValueError(f"'{name}' already registered under endpoint.")
                parent_dict[name] = cls

            return cls

        return decorator

    def get(self):
        return self._registry


REG = Registry()
