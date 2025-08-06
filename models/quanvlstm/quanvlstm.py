import torch
import torch.nn as nn
import torch.nn.functional as F

import torchquantum as tq
import torchquantum.functional as tqf

from torchquantum.layer import U3CU3Layer0


class QuantumConv(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size=3, stride=1, padding=None, n_qubits=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding if padding is not None else patch_size // 2
        self.n_qubits = n_qubits

        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_qubits)
            ]
        )
        self.arch = {"n_wires": self.n_qubits, "n_blocks": 1, "n_layers_per_block": 2} # rx ry rz CONT xx yy zz
        self.q_layer = U3CU3Layer0(self.arch)
        self.measure = tq.MeasureAll(tq.PauliZ)
        
        self.fc1 = nn.Linear(in_channels * patch_size * patch_size, n_qubits)
        self.fc = nn.Linear(n_qubits, out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=B, device=x.device)

        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
        H_pad, W_pad = x.shape[2], x.shape[3]

        out = []

        for i in range(0, H_pad - self.patch_size + 1, self.stride):
            for j in range(0, W_pad - self.patch_size + 1, self.stride):
                patch = x[:, :, i:i + self.patch_size, j:j + self.patch_size]

                if patch.shape[2] < self.patch_size or patch.shape[3] < self.patch_size:
                    patch = F.pad(patch, (0, self.patch_size - patch.shape[3], 0, self.patch_size - patch.shape[2]),
                                    mode='constant', value=0)
                patch = patch.reshape(B, -1)
                patch = self.fc1(patch)  # (B, n_qubits)

                if patch.shape[1] < self.n_qubits:
                    pad_q = torch.zeros(B, self.n_qubits - patch.shape[1], device=patch.device)
                    patch = torch.cat([patch, pad_q], dim=1)
                elif patch.shape[1] > self.n_qubits:
                    patch = patch[:, :self.n_qubits]

                self.encoder(qdev, patch)
                self.q_layer(qdev)
                measured = self.measure(qdev)

                out_patch = self.fc(measured)
                out.append(out_patch)
        
        out = torch.stack(out, dim=1)
        side_h = (H + 2 * self.padding - self.patch_size) // self.stride + 1
        side_w = (W + 2 * self.padding - self.patch_size) // self.stride + 1
        out = out.view(B, self.out_channels, side_h, side_w)
        return out

class QuanvLSTMCell(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: tuple, n_qubits: int = 4):
        """
        Initialize QuanvLSTM cell.

        Args:
            input_dim(int): Number of input channels.
            hidden_dim(int): Number of hidden channels.
            kernel_size(tuple): Size of the convolutional kernel.
            bias(bool): Whether to use bias in the convolutional layers.
        """
        super(QuanvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.n_qubits = n_qubits

        self.quanv = QuantumConv(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            patch_size=self.kernel_size[0],
            stride=1,
            n_qubits=self.n_qubits
        )
    
    def forward(self, input_tensor: torch.Tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1) # concatenate along channel axis

        combined_quanv = self.quanv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_quanv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        device = next(self.parameters()).device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))

class QuanvLSTM(nn.Module):
    """
    Args:
        input_dim(int): Number of input channels.
        hidden_dim(int): Number of hidden channels.
        kernel_size(tuple): Size of the convolutional kernel.
        num_layers(int): Number of QuanvLSTM layers.
        batch_first(bool): Whether or not dimension 0 is the batch or not.
        bias(bool): Whether to use bias in the convolutional layers.
        return_all_layers(bool): Return the list of computations for all layers.
        Note: Will do same padding.
    
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    output:
        A tuple of two lists of length (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list if the list of last states
                each element of the list is a tuple (h, c) for hidden state and memory
    
    Example:
        >>> x = torch.rand((32, 10, 64, 128, 128))
        >>> quanvlstm = QuanvLSTM(input_dim=64, hidden_dim=16, kernel_size=3, num_layers=1, batch_first=True, bias=True, return_all_layers=False)
        >>> _, last_states = quanvlstm(x)
        >>> h = last_states[0][0] # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim: int, hidden_dim: list, kernel_size: list, n_qubits_list: list, num_layers: int = 1,
                 batch_first: bool = False, return_all_layers: bool = False):
        super(QuanvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.n_qubits_list = n_qubits_list
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            in_channels = input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(QuanvLSTMCell(in_channels, self.hidden_dim[i], self.kernel_size[i], self.n_qubits_list[i]))

        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(self, input_tensor):
        """
        Args:
            input_tensor: 5D Tensor either of shape (B, T, C, H, W) or (T, B, C, H, W)
        
        Returns:
            last_state_list, layer_output
        """
        if not self.batch_first:
            # (T, B, C, H, W) -> (B, T, C, H, W)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        B, _, _, H, W = input_tensor.size()

        hidden_state = self._init_hidden(batch_size=B, image_size=(H, W))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])

                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states
    
    @staticmethod
    def _check_kernel_size_consistence(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples.')
    
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


def test_quanvlstm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.rand((8, 7, 4, 4, 4)).to(device)  # (B, T, C, H, W)
    quanvlstm = QuanvLSTM(input_dim=4, hidden_dim=[16, 16, 1], kernel_size=[(3, 3), (3, 3), (3, 3)], num_layers=3, batch_first=True).to(device)
    layer_output_list, last_states = quanvlstm(input_tensor)

    print("Layer output shape:", layer_output_list[0].shape)  # Should be (B, T, C', H, W)
    print("Last state shape:", last_states[0][0].shape)  # Should be (B, C', H, W)

if __name__ == "__main__":
    test_quanvlstm()
    print("QuanvLSTM model test passed.")