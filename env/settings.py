import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(".") / ".env"

class Configuration:
    DEBUG = os.getenv("DEBUG").lower() == "true"
    ENV = os.getenv("ENV")

class DevelopmentConfiguration(Configuration):
    load_dotenv(dotenv_path=env_path)

class TestConfiguration(Configuration):
    pass

class ProductionConfiguration(Configuration):
    pass

