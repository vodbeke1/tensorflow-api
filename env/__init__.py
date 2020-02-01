import os

if os.getenv("ENV") in {"dev", None}:
    from .settings import DevelopmentConfiguration as Config_c
elif os.getenv("ENV") == "test":
    from .settings import TestConfiguration as Config_c
elif os.getenv("ENV") == "prod":
    from .settings import ProductionConfiguration as Config_c

Config = Config_c()