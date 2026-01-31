from abc import ABC, abstractmethod
from typing import List

class Registry:
    def __init__(self, name):
        self.name = name
        self._module_dict = {}

    def register(self, name=None):
        def wrapper(cls):
            key = name if name is not None else cls.__name__
            if key in self._module_dict:
                raise KeyError(f"{key} is already in {self.name}.")
            self._module_dict[key] = cls
            return cls
        return wrapper

    def get(self, name):
        if name not in self._module_dict:
            raise KeyError(f"{name} is not registried in {self.name}.")
        return self._module_dict[name]

AGENT = Registry("agent_backends")
ROUTER = Registry("router_backends")


class VLMInferenceServer(ABC):
    def __init__(self, config:dict):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def run(self, prompt:str)->None:
        pass

class RouterServer(ABC):
    def __init__(self, config:dict):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def run(self, sample)->None:
        pass