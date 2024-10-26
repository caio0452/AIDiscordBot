from typing import Dict, List, Type, TypeVar
from pydantic import BaseModel
from pathlib import Path
import json

T = TypeVar('T', bound=BaseModel)

class ModelFromJSONLoader:
    def __init__(self, *, _json_string: str, _description: str):
        """
        Use the convenience constructor methods instead.
        """        
        self.data = json.loads(_json_string)
        self.description = _description
    
    @classmethod
    def from_file(cls, filename: str) -> 'ModelFromJSONLoader':
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"Cannot create JSON loader for file {filename}: file not found")
        
        with open(filepath, 'r') as f:
            json_string = f.read()
        
        return cls(_json_string=json_string, _description=f"JSON from file {filepath}")
    
    @classmethod
    def from_string(cls, string: str) -> 'ModelFromJSONLoader':
        return cls(_json_string=string, _description=f"JSON containing {string}")
    
    def get_model(self, model_class: Type[T]) -> T:
        if not isinstance(self.data, dict):
            raise ValueError(f"Cannot get object of type {type(T)} from {self.description}")
        return model_class(**self.data)
    
    def get_model_list(self, model_class: Type[T]) -> List[T]:
        if not isinstance(self.data, list):
            raise ValueError(f"Cannot get list of objects of type {type(T)} from {self.description}")
        return [model_class(**item) for item in self.data]
    
    def get_dict(self, model_class: Type[T]) -> Dict[str, T]:
        if not isinstance(self.data, dict):
            raise ValueError(f"Cannot get dict[str,{type(T)}] from {self.description}")
        return {key: model_class(**value) for key, value in self.data.items()}
    
    def get_model_by_name(self, model_class: Type[T], name: str) -> T:
        if not isinstance(self.data, dict):
            raise ValueError(f"Cannot parse dictionary from {self.description}")
        if name not in self.data:
            raise KeyError(f"Dictionary in {self.description} does not contain key {name}")
        return model_class(**self.data[name])