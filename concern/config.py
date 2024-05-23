from typing import Any


class ConfigNode(dict):
    def __getattr__(self, __name: str) -> Any:
        if __name in self:
            return self[__name]
        else:
            print(f"Unable to find the key {__name}")
            raise NameError(f"Unable to find the key {__name}")

    def __setattr__(self, __name: str, __value: Any) -> None:
        self[__name] = __value
    