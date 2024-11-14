# logger_config.py
import logging
import os
import inspect
from colorama import Fore, Back, Style, init
from typing import Optional


init(autoreset=True)
class ColoredFormatter(logging.Formatter):
    
    COLORS = {
        'DEBUG': Fore.BLUE,
        # 'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE
    }
    
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
            colored_msg = f"{self.COLORS[levelname]}{record.msg}{Style.RESET_ALL}"
            
            original_levelname = record.levelname
            original_msg = record.msg
            
            record.levelname = colored_levelname
            record.msg = colored_msg
            
            result = super().format(record)
            
            record.levelname = original_levelname
            record.msg = original_msg
            
            return result
        return super().format(record)

class CustomLogger:
    _loggers = {}
    _log_file_path: Optional[str] = None
    _log_level = logging.DEBUG
    _file_handler = None
    
    @classmethod
    def set_log_file(cls, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        cls._log_file_path = file_path

        if cls._file_handler:
            for logger in cls._loggers.values():
                logger.removeHandler(cls._file_handler)

        cls._file_handler = logging.FileHandler(file_path, mode='w', encoding='utf-8')
        cls._file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

        for logger in cls._loggers.values():
            logger.addHandler(cls._file_handler)
    
    @classmethod
    def set_level(cls, level):
        cls._log_level = level
        for logger in cls._loggers.values():
            logger.setLevel(level)
    
    @classmethod
    def get_logger(cls):
        caller_frame = inspect.currentframe().f_back
        module_name = caller_frame.f_globals['__name__']
        
        if module_name not in cls._loggers:
            cls._loggers[module_name] = cls._setup_logger(module_name)
        return cls._loggers[module_name]
    
    @classmethod
    def _setup_logger(cls, name):
        logger = logging.getLogger(name)
        logger.setLevel(cls._log_level)
        
        if logger.hasHandlers():
            logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(console_handler)

        if cls._file_handler:
            logger.addHandler(cls._file_handler)
        
        return logger


if __name__ == '__main__':
    CustomLogger.set_log_file("logs/process.log")
    CustomLogger.set_level(logging.DEBUG)
    
    logger = CustomLogger.get_logger()
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")