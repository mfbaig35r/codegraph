"""Module with decorators and async functions."""

from functools import lru_cache


def my_decorator(func):
    return func


@lru_cache
def cached_compute(n: int) -> int:
    return n * 2


@my_decorator
def decorated_func() -> str:
    return "hello"


class Service:
    @staticmethod
    def static_method() -> None:
        pass

    @classmethod
    def class_method(cls) -> None:
        pass

    @property
    def name(self) -> str:
        return "service"


async def async_handler(request: dict) -> dict:
    result = cached_compute(10)
    return {"value": result}
