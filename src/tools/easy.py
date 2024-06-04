# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn


import json
import time
import logging
from functools import wraps


# Timer decorator
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Function {func.__name__} starts!")
        start_time = time.time()
        func_return = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Function {func.__name__} finishes!")
        logging.info(f"Function `{func.__name__}` runtime is {end_time - start_time} seconds.")
        return func_return
    return wrapper


# Initialize a logger
def initialize_logger(file_path, mode = 'w'):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(filename)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(file_path, mode=mode, encoding="utf8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


# Terminate the given logger
def terminate_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


# Load arguments in the given Config
def load_args(Config):
    config = Config()
    parser = config.parser
    try:
        return parser.parse_args()
    except:
        return parser.parse_known_args()[0]


# Save arguments to JSON files
def save_args(args, save_path = None):
    class _MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, type) or isinstance(obj, types.FunctionType):
                return str(obj)
            return json.JSONEncoder.default(self, obj)
    with open(save_path, 'w', encoding="utf8") as f:
        f.write(json.dumps(vars(args), cls=_MyEncoder))


# Update arguments
def update_args(args, **kwargs):
    for key, value in kwargs.items():
        if key not in args:
            logging.warning(f"Key {key} not in arguments but you want to change its value to {value}!")
        args.__setattr__(key, value)
