import logging
import os
from pathlib import Path
from typing import Optional


class Logger:
    _instance = None
    _logger = None

    def __new__(cls, name: str = "BlurProcessor", 
                log_file: Optional[str] = None,
                level: str = "INFO"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._logger = cls._setup_logger(name, log_file, level)
        return cls._instance

    @staticmethod
    def _setup_logger(name: str, log_file: Optional[str], level: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # 避免重复添加handler
        if logger.handlers:
            return logger
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

    @classmethod
    def get_logger(cls) -> logging.Logger:
        if cls._logger is None:
            cls._instance = cls()
        return cls._logger

    @classmethod
    def info(cls, message: str):
        cls._logger.info(message)

    @classmethod
    def debug(cls, message: str):
        cls._logger.debug(message)

    @classmethod
    def warning(cls, message: str):
        cls._logger.warning(message)

    @classmethod
    def error(cls, message: str):
        cls._logger.error(message)

    @classmethod
    def critical(cls, message: str):
        cls._logger.critical(message)
