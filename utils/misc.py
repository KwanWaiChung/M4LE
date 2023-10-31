from logging.handlers import RotatingFileHandler
import logging
import os


level_map = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def get_logger(
    name: str,
    logger_level: str = None,
    console_level: str = None,
    file_level: str = None,
    log_path: str = None,
    maxBytes: int = 1e8,
    backupCount: int = 1,
):
    """Configure the logger and return it.

    Args:
        name (str, optional): Name of the logger, usually __name__.
            Defaults to None. None refers to root logger, usually useful
            when setting the default config at the top level script.
        logger_level (str, optional): level of logger. Defaults to None.
            None is treated as `debug`.
        console_level (str, optional): level of console. Defaults to None.
        file_level (str, optional): level of file. Defaults to None.
            None is treated `debug`.
        log_path (str, optional): The path of the log.
        maxBytes (int): The maximum size of the log file.
            Only used if log_path is not None.
        backupCount (int): Number of rolling backup log files.
            If log_path is `app.log` and backupCount is 3, we will have
            `app.log`, `app.log.1`, `app.log.2` and `app.log.3`.
            Only used if log_path is not None.

    Note that console_level should only be used when configuring the
    root logger.
    """

    logger = logging.getLogger(name)
    if name:
        logger.setLevel(level_map[logger_level or "debug"])
    else:  # root logger default lv should be high to avoid external lib log
        logger.setLevel(level_map[logger_level or "warning"])

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # set up the logfile handler
    if log_path:
        root_logger = logger.root
        if os.path.dirname(log_path):
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = RotatingFileHandler(
            filename=log_path, maxBytes=maxBytes, backupCount=backupCount
        )
        fh.setLevel(level_map[file_level or "debug"])
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

    if console_level:
        root_logger = logger.root
        sh = logging.StreamHandler()
        sh.setLevel(level_map[console_level or "debug"])
        sh.setFormatter(formatter)
        root_logger.addHandler(sh)
    return logger
