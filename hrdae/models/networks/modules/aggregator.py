from abc import ABCMeta, abstractmethod

from torch import nn, Tensor, zeros, bmm, cat
from torch.nn.functional import softmax

from .conv_block import PixelWiseConv2d, PixelWiseConv3d


def create_aggregator2d(aggregator: str, cc: int, cm: int) -> "Aggregator":
    if aggregator == "addition":
        return AdditionAggregator2d(cc, cm)
    if aggregator == "multiplication":
        return MultiplicationAggregator2d(cc, cm)
    if aggregator == "concatenation":
        return ConcatenationAggregator2d()
    if aggregator == "attention":
        return AttentionAggregator2d(cc, cm)
    raise NotImplementedError(f"{aggregator} not implemented")


def create_aggregator3d(aggregator: str, cc: int, cm: int) -> "Aggregator":
    if aggregator == "addition":
        return AdditionAggregator3d(cc, cm)
    if aggregator == "multiplication":
        return MultiplicationAggregator3d(cc, cm)
    if aggregator == "concatenation":
        return ConcatenationAggregator3d()
    if aggregator == "attention":
        return AttentionAggregator3d(cc, cm)
    raise NotImplementedError(f"{aggregator} not implemented")


class Aggregator(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, z: tuple[Tensor, Tensor]) -> Tensor:
        pass


class AdditionAggregator2d(Aggregator):
    def __init__(self, cc: int, cm: int) -> None:
        super().__init__()
        self.conv = None
        if cc != cm:
            self.conv = PixelWiseConv2d(cm, cc)

    def forward(self, z: tuple[Tensor, Tensor]) -> Tensor:
        zc, zm = z
        if self.conv is not None:
            zm = self.conv(zm)
        return zc + zm


class AdditionAggregator3d(Aggregator):
    def __init__(self, cc: int, cm: int) -> None:
        super().__init__()
        self.conv = None
        if cc != cm:
            self.conv = PixelWiseConv3d(cm, cc)

    def forward(self, z: tuple[Tensor, Tensor]) -> Tensor:
        zc, zm = z
        if self.conv is not None:
            zm = self.conv(zm)
        return zc + zm


class MultiplicationAggregator2d(Aggregator):
    def __init__(self, cc: int, cm: int) -> None:
        super().__init__()
        self.conv = None
        if cc != cm:
            self.conv = PixelWiseConv2d(cm, cc)

    def forward(self, z: tuple[Tensor, Tensor]) -> Tensor:
        zc, zm = z
        if self.conv is not None:
            zm = self.conv(zm)
        return zc * zm


class MultiplicationAggregator3d(Aggregator):
    def __init__(self, cc: int, cm: int) -> None:
        super().__init__()
        self.conv = None
        if cc != cm:
            self.conv = PixelWiseConv3d(cm, cc)

    def forward(self, z: tuple[Tensor, Tensor]) -> Tensor:
        zc, zm = z
        if self.conv is not None:
            zm = self.conv(zm)
        return zc * zm


class ConcatenationAggregator2d(Aggregator):
    def forward(self, z: tuple[Tensor, Tensor]) -> Tensor:
        return cat(z, dim=1)


class ConcatenationAggregator3d(Aggregator):
    def forward(self, z: tuple[Tensor, Tensor]) -> Tensor:
        return cat(z, dim=1)


class AttentionAggregator2d(Aggregator):
    def __init__(self, cc: int, cm: int) -> None:
        super().__init__()
        p = cm // 8
        if p == 0:
            p = 1
        self.query_conv = PixelWiseConv2d(cm, p, act_norm=False)
        self.key_conv = PixelWiseConv2d(cm, p, act_norm=False)
        self.value_conv = PixelWiseConv2d(cm, cc, act_norm=False)
        self.gamma = nn.Parameter(zeros(1))

    def forward(self, z: tuple[Tensor, Tensor]) -> Tensor:
        zc, zm = z
        query = self.query_conv(zm)  # (b, p, w, h)
        key = self.key_conv(zm)  # (b, p, w, h)
        value = self.value_conv(zm)  # (b, cc, w, h)

        b, p, w, h = query.size()
        assert query.size() == key.size()
        cc = value.size(1)
        assert value.size() == (b, cc, w, h)

        query = query.reshape(b, p, w * h)
        key = key.reshape(b, p, w * h)
        value = value.reshape(b, cc, w * h)

        attention = bmm(query.permute(0, 2, 1), key)  # (b, w * h, w * h)
        attention = softmax(attention, dim=-1)

        # (b, cc, w * h) * (b, w * h, w * h) -> (b, cc, w * h)
        out = bmm(value, attention)
        out = out.reshape(b, cc, w, h)  # (b, cc, w, h)

        zc += self.gamma * out

        return zc


class AttentionAggregator3d(Aggregator):
    def __init__(self, cc: int, cm: int) -> None:
        super().__init__()
        p = cm // 8
        if p == 0:
            p = 1
        self.query_conv = PixelWiseConv3d(cm, p, act_norm=False)
        self.key_conv = PixelWiseConv3d(cm, p, act_norm=False)
        self.value_conv = PixelWiseConv3d(cm, cc, act_norm=False)
        self.gamma = nn.Parameter(zeros(1))

    def forward(self, z: tuple[Tensor, Tensor]) -> Tensor:
        zc, zm = z
        query = self.query_conv(zm)  # (b, p, d, w, h)
        key = self.key_conv(zm)  # (b, p, d, w, h)
        value = self.value_conv(zm)  # (b, cc, d, w, h)

        b, p, d, w, h = query.size()
        assert query.size() == key.size()
        cc = value.size(1)
        assert value.size() == (b, cc, d, w, h)

        query = query.reshape(b, p, d * w * h)
        key = key.reshape(b, p, d * w * h)
        value = value.reshape(b, cc, d * w * h)

        attention = bmm(query.permute(0, 2, 1), key)  # (b, d * w * h, d * w * h)
        attention = softmax(attention, dim=-1)

        # (b, cc, d * w * h) * (b, d * w * h, d * w * h) -> (b, cc, d * w * h)
        out = bmm(value, attention)
        out = out.reshape(b, cc, d, w, h)  # (b, cc, d, w, h)

        zc += self.gamma * out

        return zc
