from torch import Tensor, cat, nn, sigmoid, split, stack, tanh, zeros


class ConvLSTMCell1d(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        bias: bool,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv1d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(
        self,
        input_tensor: Tensor,
        cur_state: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        h_cur, c_cur = cur_state

        combined = cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = split(combined_conv, self.hidden_dim, dim=1)
        i = sigmoid(cc_i)
        f = sigmoid(cc_f)
        o = sigmoid(cc_o)
        g = tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size: int, image_size: int) -> tuple[Tensor, Tensor]:
        return (
            zeros(
                batch_size,
                self.hidden_dim,
                image_size,
                device=self.conv.weight.device,
            ),
            zeros(
                batch_size,
                self.hidden_dim,
                image_size,
                device=self.conv.weight.device,
            ),
        )


class ConvLSTM1d(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | list[int],
        kernel_size: int | list[int],
        num_layers: int,
        batch_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self._check_kernel_size_consistency(kernel_size)

        _kernel_size: list[int] = []
        if isinstance(kernel_size, list):
            _kernel_size = kernel_size
        else:
            _kernel_size = [kernel_size] * num_layers
        _hidden_dim: list[int] = []
        if isinstance(hidden_dim, list):
            _hidden_dim = hidden_dim
        else:
            _hidden_dim = [hidden_dim] * num_layers
        if not len(_kernel_size) == len(_hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length")

        self.input_dim = input_dim
        self.hidden_dim = _hidden_dim
        self.kernel_size = _kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell1d(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self,
        input_tensor: Tensor,
        hidden_state: list[tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        if not self.batch_first:
            # (t, b, c, h) -> (b, t, c, h)
            input_tensor = input_tensor.permute(1, 0, 2, 3)

        b, _, _, h_ = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=h_)

        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t], cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = stack(output_inner, dim=1)
            cur_layer_input = layer_output

            last_state_list.append((h, c))

        return cur_layer_input, last_state_list

    def _init_hidden(
        self, batch_size: int, image_size: int
    ) -> list[tuple[Tensor, Tensor]]:
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size: int | list[int]) -> None:
        if isinstance(kernel_size, int):
            return
        if isinstance(kernel_size, list) and all(
            [isinstance(elem, int) for elem in kernel_size]
        ):
            return
        raise ValueError("`kernel_size` must be int or list of ints")


class ConvLSTMCell2d(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: tuple[int, int],
        bias: bool,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(
        self,
        input_tensor: Tensor,
        cur_state: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        h_cur, c_cur = cur_state

        combined = cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = split(combined_conv, self.hidden_dim, dim=1)
        i = sigmoid(cc_i)
        f = sigmoid(cc_f)
        o = sigmoid(cc_o)
        g = tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * tanh(c_next)

        return h_next, c_next

    def init_hidden(
        self, batch_size: int, image_size: tuple[int, int]
    ) -> tuple[Tensor, Tensor]:
        height, width = image_size
        return (
            zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
            zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


class ConvLSTM2d(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | list[int],
        kernel_size: tuple[int, int] | list[tuple[int, int]],
        num_layers: int,
        batch_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self._check_kernel_size_consistency(kernel_size)

        _kernel_size: list[tuple[int, int]] = []
        if isinstance(kernel_size, list):
            _kernel_size = kernel_size
        else:
            _kernel_size = [kernel_size] * num_layers
        _hidden_dim: list[int] = []
        if isinstance(hidden_dim, list):
            _hidden_dim = hidden_dim
        else:
            _hidden_dim = [hidden_dim] * num_layers
        if not len(_kernel_size) == len(_hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length")

        self.input_dim = input_dim
        self.hidden_dim = _hidden_dim
        self.kernel_size = _kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell2d(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self,
        input_tensor: Tensor,
        hidden_state: list[tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h_, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h_, w))

        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = stack(output_inner, dim=1)
            cur_layer_input = layer_output

            last_state_list.append((h, c))

        return cur_layer_input, last_state_list

    def _init_hidden(
        self, batch_size: int, image_size: tuple[int, int]
    ) -> list[tuple[Tensor, Tensor]]:
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(
        kernel_size: tuple[int, int] | list[tuple[int, int]]
    ) -> None:
        if isinstance(kernel_size, tuple):
            return
        if isinstance(kernel_size, list) and all(
            [isinstance(elem, tuple) for elem in kernel_size]
        ):
            return
        raise ValueError("`kernel_size` must be tuple or list of tuples")
