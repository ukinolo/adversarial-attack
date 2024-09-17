from typing import Any, Callable
from types import FunctionType
import copy

class Parameter():
    def __init__(self, name: str, valid_types: tuple, default_value) -> None:
        assert isinstance(name, str)
        assert isinstance(valid_types, tuple)
        for passed_type in valid_types:
            assert isinstance(passed_type, type)
        assert isinstance(default_value, valid_types)
        self._name = name
        self._valid_types = valid_types
        self._default_value = default_value
        self._value = default_value
    
    def set_value(self, value: Any) -> None:
        assert isinstance(value, self._valid_types)
        self._value = value

    def get_value(self) -> Any:
        return self._value
    
    def get_name(self) -> str:
        return self._name
    
    def get_info(self, *, number_of_tabs: int=0) -> str:
        return f'\t{self._name} = {self._value}'.replace('\t', '\t'*number_of_tabs, 1)

    def get_description(self, *, number_of_tabs: int=0) -> str:
        return f'\t{self._name} = {self._default_value} is default and valid types are ({self._valid_types})'.replace('\t', '\t'*number_of_tabs, 1)




class Method():
    def __init__(self, name: str, function: Callable, *, parameters: None|list[Parameter]=None) -> None:
        assert isinstance(name, str)
        assert isinstance(function, FunctionType)
        self._name = name
        self._function = function
        if(parameters != None):
            assert isinstance(parameters, list)
            assert all(isinstance(param, Parameter) for param in parameters)
            self._parameters = parameters
        else:
            self._parameters = list[Parameter]()

    def get_info(self, *, number_of_tabs: int=0) -> str:
        return f'\tMethod {self._name} parameters:\n'.replace('\t', '\t'*number_of_tabs, 1) + \
                '\n'.join([param.get_info(number_of_tabs=number_of_tabs+1) for param in self._parameters])

    def get_description(self, *, number_of_tabs: int=0) -> str:
        if len(self._parameters) > 0:
            return f'\tMethod {self._name} default parameters are:\n'.replace('\t', '\t'*number_of_tabs, 1) + \
                    '\n'.join([param.get_description(number_of_tabs=number_of_tabs+1) for param in self._parameters])
        else:
            return f'\tMethod {self._name} have no parameters\n'.replace('\t', '\t'*number_of_tabs, 1)

    def get_name(self) -> str:
        return self._name
    
    def get_parameter_names(self) -> list[str]:
        return [parameter.get_name() for parameter in self._parameters]
    
    def set_parameter_values(self, **kwargs) -> None:
        for parameter in self._parameters:
            try:
                if kwargs[parameter.get_name()] != None:
                    parameter.set_value(kwargs[parameter.get_name()])
            except KeyError:
                raise KeyError(f'In the passed dictionary there is no key {parameter.get_name()}\t' +\
                                 'Try using get_description() method to see all the parameters for method')

    def __getitem__(self, key: str) -> Any:
        return next(p.get_value() for p in self._parameters if p.get_name() == key)

    def __call__(self, **kwds: Any) -> Any:
        return self._function(self, **kwds)




class Strategy():
    def __init__(self, name: str) -> None:
        assert isinstance(name, str)
        self._name = name
        self._methods = list[Method]()

    def add_method(self, method: Method) -> None:
        assert isinstance(method, Method)
        self._methods.append(method)
        return self

    def get_info(self, *, number_of_tabs :int=0) -> str:
        return f'\tStrategy {self._name} have next methods:\n'.replace('\t', '\t'*number_of_tabs, 1) + \
                '\n'.join([method.get_info(number_of_tabs=number_of_tabs+1) for method in self._methods])

    def get_description(self, *, number_of_tabs :int=0) -> str:
        return f'\tStrategy {self._name} have next methods:\n'.replace('\t', '\t'*number_of_tabs, 1) + \
                '\n'.join([method.get_description(number_of_tabs=number_of_tabs+1) for method in self._methods])

    def get_methods(self) -> list[Method]:
        return self._methods
    
    def get_method_copy(self, method_name: str) -> Method:
        for method in self._methods:
            if method.get_name() == method_name:
                return copy.copy(method)
        raise ValueError(f'Strategy {self._name} does not contain {method_name} method\n' +\
                         'Try using get_description() to see available methods and parameters')
