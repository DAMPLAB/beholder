'''
Author: W.R. Jackson, Damp Lab 2020
'''
import datetime
import logging
import sys

import colorama


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


class BLogger(metaclass=SingletonBaseClass):
    '''
    Simple Custom Logger to enable colorized text output as well as
    discrete control over setting logging levels for debugging.
    '''

    def __init__(self):

        logging.basicConfig(
            filename=f'beholder_{datetime.datetime.now().isoformat()}.log',
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.DEBUG,
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)
        # This only performs any kind of action on a windows machine,
        # otherwise it's a no-op.
        colorama.init(autoreset=True)

    # def write_out(self, output_fp: str):
    #     self.log.()

    def change_logging_level(self, logging_level: str):
        '''
        Args:
            logging_level:

        Returns:
        '''
        logging_level = logging_level.upper()
        if logging_level == "DEBUG":
            self.log.setLevel(logging.DEBUG)
        if logging_level == "INFO":
            self.log.setLevel(logging.INFO)
        if logging_level == "WARNING":
            self.log.setLevel(logging.WARNING)
        if logging_level == "ERROR":
            self.log.setLevel(logging.ERROR)
        if logging_level == "CRITICAL":
            self.log.setLevel(logging.CRITICAL)
        else:
            print(
                f'Unable to recognize passed in logging level {logging_level}'
            )

    def debug(self, message: str):
        '''
        Args:
            message:

        Returns:
        '''
        self.log.debug(
            f'{message}'
        )

    def info(self, message: str):
        '''
        Args:
            message:

        Returns:
        '''
        self.log.info(
            f'{message}'
        )

    def warning(self, message: str):
        '''
        Args:
            message:

        Returns:
        '''
        self.log.warning(
            f'{message}'
        )

    def error(self, message: str):
        '''
        Args:
            message:

        Returns:
        '''
        self.log.error(
            f'{message}'
        )

    def critical(self, message: str):
        '''
        Args:
            message:

        Returns:
        '''
        # Magenta is the most ominous color.
        self.log.critical(
            f'{message}'
        )

    def damp_themed(self, message: str):
        '''
        Args:
            message:

        Returns:
        '''
        self.log.info(
            f'{message}'
        )
