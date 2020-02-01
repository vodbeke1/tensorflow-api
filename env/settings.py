import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(".") / ".env"

class DevelopmentConfiguration:
    DEBUG = os.getenv("DEBUG").lower() == "true"
    ENV = os.getenv("ENV")
    load_dotenv(dotenv_path=env_path)

class TestConfiguration:
    DEBUG = os.getenv("DEBUG").lower() == "true"
    ENV = os.getenv("ENV")

class ProductionConfiguration:
    DEBUG = os.getenv("DEBUG").lower() == "true"
    ENV = os.getenv("ENV")

