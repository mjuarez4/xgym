import logging

colors = {
    "none": "{}",
    "white": "\033[30m{}\033[0m",
    "red": "\033[31m{}\033[0m",
    "green": "\033[32m{}\033[0m",
    "orange": "\033[33m{}\033[0m",
    "blue": "\033[34m{}\033[0m",
    "purple": "\033[35m{}\033[0m",
    "cyan": "\033[36m{}\033[0m",
    "light_gray": "\033[37m{}\033[0m",
    "dark_gray": "\033[90m{}\033[0m",
    "light_red": "\033[91m{}\033[0m",
    "light_green": "\033[92m{}\033[0m",
    "yellow": "\033[93m{}\033[0m",
    "light_blue": "\033[94m{}\033[0m",
    "pink": "\033[95m{}\033[0m",
    "light_cyan": "\033[96m{}\033[0m",
}


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    # format is (filename:lineno) - message
    format = "(%(filename)s:%(lineno)d) %(levelname)s - %(message)s"

    FORMATS = {
        # logging.VERBOSE: grey + format + reset,
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# create logger with 'spam_application'
logger = logging.getLogger("xgym")
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)

logger.debug("debug message")
logger.info("info message")
logger.warning("warning message")
logger.error("error message")
logger.critical("critical message")
print()
