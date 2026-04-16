import typing

def word(a: str, b: str) -> str:
    return a + b

a: str = input("a : ")
b: str = input("b : ")

print(word(a, b))