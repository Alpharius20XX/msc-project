import math

import torch
from torch import Tensor, nn


def get_omegas(alpha, dim, **kwargs):
    omega_1 = alpha * torch.logspace(0, 2 / (dim) - 1, (dim // 2), 100, **kwargs)
    omega_2 = omega_1
    if dim % 2 != 0:
        omega_2 = alpha * torch.logspace(0, 2 / (dim) - 1, (dim // 2) + 1, 100, **kwargs)
    return omega_1, omega_2


def pos_enc_symmetric(xs, dim, alpha=1000):
    """Symmetric positional encoding.

    Parameters
    ----------
    xs : torch.Tensor
        Input tensor.
    dim : int
        Dimension of the positional encoding.
    alpha : float, optional
        Scaling factor for the positional encoding, by default 100.

    Returns
    -------
    torch.Tensor
        Symmetric positional encoding.
    """
    xs = xs.unsqueeze(-1)
    kwargs = {"device": xs.device, "dtype": xs.dtype}
    omega_1, omega_2 = get_omegas(alpha, dim, **kwargs)
    p1 = (xs.sin() * omega_1).sin()
    p2 = (xs.cos() * omega_2).sin()
    return torch.cat((p1, p2), dim=-1)


def pos_enc(xs, dim, alpha=1000):
    """Positional encoding.

    Parameters
    ----------
    xs : torch.Tensor
        Input tensor.
    dim : int
        Dimension of the positional encoding.
    alpha : float, optional
        Scaling factor for the positional encoding, by default 100.

    Returns
    -------
    torch.Tensor
        Positional encoding.
    """
    xs = xs.unsqueeze(-1)
    kwargs = {"device": xs.device, "dtype": xs.dtype}
    omega_1, omega_2 = get_omegas(alpha, dim, **kwargs)
    p1 = (xs * omega_1).sin()
    p2 = (xs * omega_2).cos()
    return torch.cat((p1, p2), dim=-1)


class PositionEncoder(nn.Module):
    def __init__(self, input_name: str, fields: list[str], dim: int, sym_fields: list[str] | None = None, alpha=1000):
        """Positional encoder.

        Parameters
        ----------
        input_name : str
            The name of the input object that will be encoded.
        fields : list[str]
            List of fields belonging to the object to apply the positional encoding to.
        fields : list[str]
            List of fields that should use a rotationally symmetric positional encoding.
        dim : int
            Dimension to project the positional encoding into.
        alpha : float
            Scaling factor hyperparamater for the positional encoding.
        """
        super().__init__()

        self.input_name = input_name
        self.fields = fields
        self.sym_fields = sym_fields or []
        self.dim = dim
        self.alpha = alpha

        self.per_input_dim = self.dim // len(self.fields)
        self.remainder_dim = self.dim % len(self.fields)

    def forward(self, inputs: dict):
        """Apply positional encoding to the inputs.

        Parameters
        ----------
        inputs : dict
            Dictionary of inputs.

        Returns
        -------
        torch.Tensor
            Positional encoding of the input variables.
        """
        encodings = []
        for field in self.fields:
            pos_enc_fn = pos_enc_symmetric if field in self.sym_fields else pos_enc
            encodings.append(pos_enc_fn(inputs[f"{self.input_name}_{field}"], self.per_input_dim, self.alpha))
        if self.remainder_dim:
            encodings.append(torch.zeros_like(encodings[0])[..., : self.remainder_dim])
        encodings = torch.cat(encodings, dim=-1)
        return encodings


class FourierPositionEncoder(nn.Module):
    """
    An implementation of Gaussian Fourier positional encoding.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
    see https://arxiv.org/abs/2006.10739
    """

    def __init__(self, input_name: str, dim: int, fields: list[str], scale: float = 1) -> None:
        super().__init__()
        assert scale > 0
        assert dim % 2 == 0, "Dimension must be even"
        self.input_name = input_name
        self.fields = fields
        self.B = torch.nn.parameter.Buffer(scale * torch.randn((len(fields), dim // 2)))
        self.pi = torch.tensor(math.pi)

    def forward(self, xs: dict[str, Tensor]) -> Tensor:
        xs = torch.cat([xs[f"{self.input_name}_{field}"].unsqueeze(-1) for field in self.fields], dim=-1)
        xs = 2 * self.pi * xs
        xs @= self.B
        return torch.cat([torch.sin(xs), torch.cos(xs)], dim=-1)

#####
class RoPE(nn.Module):
    """
    An implementation of Rotary Position Embedding (RoPE).

    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    see https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply RoPE to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., seq_len, dim)

        Returns
        -------
        torch.Tensor
            Tensor with RoPE applied.
        """
        t = torch.arange(x.shape[-2], device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Reshape for broadcasting
        while len(emb.shape) < len(x.shape):
            emb = emb.unsqueeze(0)

        x_cos = emb.cos()
        x_sin = emb.sin()

        # Rotate the even and odd dimensions
        x_rotated = torch.cat([-x[..., 1::2], x[..., ::2]], dim=-1)
        return x * x_cos + x_rotated * x_sin