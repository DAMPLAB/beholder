'''
Author: W.R. Jackson, Damp Lab 2020
'''
from typing import (
    List,
    Union,
)

import yaml

from .logging import BLogger

LOG = BLogger()


class SingletonBaseClass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class BLoader(metaclass=SingletonBaseClass):

    def __init__(self):
        self.configs = {}
        self.config_path = '../example_config'
        # We want to have our configs placed outside the repository.

    def load_config(self, config_name: str, essential: bool = False):
        '''
        Args:
            config_name:
            config_name:
            essential:
        Returns:
        '''
        # Path manipulation goes here
        config_path = f'{self.config_path}/{config_name}_conf.yaml'
        try:
            with open(config_path) as input_file:
                self.configs[config_name] = yaml.load(input_file, Loader=yaml.FullLoader)
                LOG.debug(f'Successfully loaded config file {config_name}')
        except FileNotFoundError as f:
            if essential:
                raise Exception(
                    f'Unable to load essential config file for {config_name} for path {config_name}: Traceback: {f}'
                )
            else:
                LOG.warning(
                    f'Failed to load config file for {config_name} for path {config_name}'
                )
        except yaml.YAMLError as f:
            if essential:
                raise Exception(
                    f'Failed to parse essential config file for {config_name} for path {config_name}: Traceback: {f}'
                )
            else:
                LOG.warning(
                    f'Failed to parse config file for {config_name} for path {config_name}'
                )

    def get(self, key_value: Union[str, List[str]], input_dict: dict = None):
        '''
        Args:
            key_value:
            input_dict:
        Returns:
        '''
        if input_dict is None:
            input_dict = self.configs
        if len(key_value) == 1:
            if type(key_value) == list:
                key_value = key_value[0]
            return input_dict[key_value]
        else:
            intermediate_key = key_value.pop(0)
            input_dict = input_dict[intermediate_key]
            return self.get(key_value, input_dict)
