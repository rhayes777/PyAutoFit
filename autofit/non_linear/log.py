import logging
from autoconf import conf

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(conf.instance.general.get("output", "log_level", str)
                .replace(" ", "")
                .upper()
                )