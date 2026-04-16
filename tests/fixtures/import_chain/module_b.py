"""Module B — imports from module A."""

from .module_a import BaseProcessor, func_a


def func_b() -> str:
    return func_a() + " via B"


class ChildProcessor(BaseProcessor):
    def process(self) -> None:
        result = func_a()
        print(result)
