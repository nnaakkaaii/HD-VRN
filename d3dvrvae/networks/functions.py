def stride_generator(n: int, reverse: bool = False) -> list[int]:
    strides = [1, 2]*10
    if reverse:
        return list(reversed(strides[:n]))
    return strides[:n]
