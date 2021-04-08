'''
Author: W.R. Jackson, Damp Lab 2020
'''
import logging

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
        # TODO: Persist to disk via rotating log file handler in
        #  directory outside of repository.
        logging.basicConfig(
            format='%(asctime)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)
        # This only performs any kind of action on a windows machine,
        # otherwise it's a no-op.
        colorama.init(autoreset=True)

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
            f'{colorama.Fore.WHITE}{message}{colorama.Style.RESET_ALL}'
        )

    def info(self, message: str):
        '''
        Args:
            message:

        Returns:
        '''
        self.log.info(
            f'{colorama.Fore.BLUE}{message}{colorama.Style.RESET_ALL}'
        )

    def warning(self, message: str):
        '''
        Args:
            message:

        Returns:
        '''
        self.log.warning(
            f'{colorama.Fore.YELLOW}{message}{colorama.Style.RESET_ALL}'
        )

    def error(self, message: str):
        '''
        Args:
            message:

        Returns:
        '''
        self.log.error(
            f'{colorama.Fore.RED}{message}{colorama.Style.RESET_ALL}'
        )

    def critical(self, message: str):
        '''
        Args:
            message:

        Returns:
        '''
        # Magenta is the most ominous color.
        self.log.critical(
            f'{colorama.Back.RED}{message}{colorama.Style.RESET_ALL}'
        )

    def damp_themed(self, message: str):
        '''
        Args:
            message:

        Returns:
        '''
        self.log.info(
            f'{colorama.Fore.MAGENTA}{message}{colorama.Style.RESET_ALL}'
        )
