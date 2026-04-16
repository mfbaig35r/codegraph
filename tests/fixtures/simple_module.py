"""A simple module with functions and a class."""

import os  # noqa: F401
from pathlib import Path  # noqa: F401

CONSTANT = 42


def helper(x: int) -> str:
    """Convert int to string."""
    return str(x)


def caller() -> None:
    result = helper(CONSTANT)
    print(result)


class Animal:
    """Base animal class."""

    def __init__(self, name: str) -> None:
        self.name = name

    def speak(self) -> str:
        return f"{self.name} speaks"


class Dog(Animal):
    """A dog that inherits from Animal."""

    def speak(self) -> str:
        return f"{self.name} barks"

    def fetch(self) -> None:
        self.speak()
