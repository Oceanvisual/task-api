from os import PRIO_USER


def greet(name: str) -> str:
    return f"HELLO, {name}!"

if __name__ == "__main__":
    print(greet("word"))