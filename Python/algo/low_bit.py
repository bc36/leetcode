"""low bit"""


def low_bit(x: int) -> None:
    """
    如何求 x 最低位的 1
    1.
    x      = 1011000
    ~x     = 0100111
    ~x + 1 = 0101000
    ~x + 1 = -x 补码性质
    得到 low_bit = x ^ -x
    去掉 low_bit -> x -= x & (-x)

    2.
    x     = 1011000
    x - 1 = 1010111
    去掉 low_bit -> x &= x - 1
    """
    return
