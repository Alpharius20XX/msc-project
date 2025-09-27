import math

import torch
from torch import Tensor, nn


def get_omegas(alpha, dim, base, **kwargs):
    omega_1 = alpha * torch.logspace(0, 2 / (dim) - 1, (dim // 2), base, **kwargs)
    omega_2 = omega_1
    if dim % 2 != 0:
        omega_2 = alpha * torch.logspace(0, 2 / (dim) - 1, (dim // 2) + 1, base, **kwargs)
    return omega_1, omega_2


def pos_enc_rope_1(xs, dim, alpha=1000):
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

    #omega= torch.logspace(((2*torch.pi)/300), ((2*torch.pi)/0.2), (dim),  device= xs.device, dtype= xs.dtype) #alpha * torch.logspace(0, 2 / (dim) - 1, (dim), 100, device= xs.device, dtype= xs.dtype) #

    omega=torch.logspace(   0,-1,steps=dim,base=10000,device=xs.device,dtype=xs.dtype)

    p1 = (xs * omega).cos()
    p2 = (xs * omega).sin()
    return (p1, p2)


def pos_enc_symmetric(xs, dim, alpha=1000, base=100):
    """Symmetric positional encoding.

    Parameters
    ----------
    xs : torch.Tensor
        Input tensor.
    dim : int
        Dimension of the positional encoding.
    alpha : float, optional
        Scaling factor for the positional encoding, by default 100.

    Returns:
    -------
    torch.Tensor
        Symmetric positional encoding.
    """
    xs = xs.unsqueeze(-1)
    kwargs = {"device": xs.device, "dtype": xs.dtype}
    omega_1, omega_2 = get_omegas(alpha, dim, base, **kwargs)
    p1 = (xs.sin() * omega_1).sin()
    p2 = (xs.cos() * omega_2).sin()
    return torch.cat((p1, p2), dim=-1)


def pos_enc(xs, dim, alpha=1000, base=100):
    """Positional encoding.

    Parameters
    ----------
    xs : torch.Tensor
        Input tensor.
    dim : int
        Dimension of the positional encoding.
    alpha : float, optional
        Scaling factor for the positional encoding, by default 100.

    Returns:
    -------
    torch.Tensor
        Positional encoding.
    """
    xs = xs.unsqueeze(-1)
    kwargs = {"device": xs.device, "dtype": xs.dtype}
    omega_1, omega_2 = get_omegas(alpha, dim, base, **kwargs)
    p1 = (xs * omega_1).sin()
    p2 = (xs * omega_2).cos()
    return torch.cat((p1, p2), dim=-1)


class PositionEncoder(nn.Module):
    def __init__(self, input_name: str, fields: list[str], dim: int, sym_fields: list[str] | None = None, alpha=1000, base=100):
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
        base : float
            Base for the logarithmic scale.
        """
        super().__init__()

        self.input_name = input_name
        self.fields = fields
        self.sym_fields = sym_fields or []
        self.dim = dim
        self.alpha = alpha
        self.base = base

        self.per_input_dim = self.dim // len(self.fields)
        self.remainder_dim = self.dim % len(self.fields)

        

    


    def forward(self, inputs: dict):
        """Apply positional encoding to the inputs.

        Parameters
        ----------
        inputs : dict
            Dictionary of inputs.

        Returns:
        -------
        torch.Tensor
            Positional encoding of the input variables.
        """
        encodings = []
        for field in self.fields:
            pos_enc_fn = pos_enc_symmetric if field in self.sym_fields else pos_enc
            """if (field=="r"):
                encodings.append(pos_enc_fn(inputs[f"{self.input_name}_{field}"]/14, self.per_input_dim, self.alpha, self.base))
            else:"""
            encodings.append(pos_enc_fn(inputs[f"{self.input_name}_{field}"], self.per_input_dim, self.alpha, self.base))
                
        if self.remainder_dim:
            encodings.append(torch.zeros_like(encodings[0])[..., : self.remainder_dim])
        return torch.cat(encodings, dim=-1)


class FourierPositionEncoder(nn.Module):
    """An implementation of Gaussian Fourier positional encoding.

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

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        xs = torch.cat([inputs[f"{self.input_name}_{field}"].unsqueeze(-1) for field in self.fields], dim=-1)
        xs = 2 * self.pi * xs
        xs @= self.B
        return torch.cat([torch.sin(xs), torch.cos(xs)], dim=-1)



class RopeEncoder(nn.Module):
    def __init__(self, input_name: str, fields: list[str], dim: int, sym_fields: list[str] | None = None, alpha=1000):#, dim: int
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
        self.sym_fields = sym_fields or []# ["dtheta", "dphi","phi","theta"] ###added this was  sym_fields or []

        self.alpha = alpha


    def forward(self, inputs: dict, x):#version where rotated in every direction at once
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
        #lenin=len(x[0])
        
        lenin = x.shape[-1]
        # Determine the target device from an input tensor
        target_device = x.device

        

        self.per_input_dim = (lenin // 2)*2
        self.remainder_dim = lenin % 2

        xuse=x[:,:, :self.per_input_dim]

            

        xcos=xuse

        xsin=xuse
        
        for field in self.fields:
            pos_enc_fn = pos_enc_symmetric if field in self.sym_fields else pos_enc_rope_1
            
            (ia,ib)=pos_enc_fn(inputs[f"{self.input_name}_{field}"], self.per_input_dim, self.alpha)

            


            xsin = (xsin*ib)

            xcos= xcos*ia

        
        #xsin = xsin.reshape(-1, self.per_input_dim // 2, 2).flip(-1).reshape(-1, self.per_input_dim)
        
        original_shape = xsin.shape
        # Reshape without flattening the batch and sequence dimensions
        xsin = xsin.reshape(original_shape[0], original_shape[1], -1, 2)
        xsin = xsin.flip(-1)
        # Reshape back to the exact original shape
        xsin = xsin.reshape(original_shape)

        ind1 = torch.arange(self.per_input_dim, device=target_device)

        negation_mask = torch.where(ind1 % 2 == 0, 1., -1.)
        
        xsin = xsin * negation_mask

        #xsin=xsin*keep1

        #xsin = xsin.view(-1, 2).flip(dims=[-1]).view(-1)
        
        xout=xcos+xsin

        if(self.remainder_dim==1):
            last_column = x[:,:, -1:]

            xout = torch.cat([xout, last_column], dim=-1)

        

        
        return xout
    

class RopeEncoderSep(nn.Module):
    def __init__(self, input_name: str, fields: list[str], dim: int, sym_fields: list[str] | None = None, alpha=1000):#, dim: int
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
        self.sym_fields = sym_fields or []# ["dtheta", "dphi","phi","theta"] ###added this was  sym_fields or []

        self.alpha = alpha


    def forward(self, inputs: dict, x):#version where rotated in every direction at once
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
        #lenin=len(x[0])
        
        lenin = x.shape[-1]
        # Determine the target device from an input tensor
        target_device = x.device

        

        self.per_input_dim = (lenin // (2*len(self.fields)))*2
        self.remainder_dim = lenin % (2*len(self.fields))

        xuse=x[:,:, :self.per_input_dim*len(self.fields)]

        #print(self.per_input_dim)

        #print(x.shape)

        #print(inputs)

        xcos=xuse

        xsin=xuse
        
        for i in range(len(self.fields)):

            field=self.fields[i]

            pos_enc_fn = pos_enc_symmetric if field in self.sym_fields else pos_enc_rope_1

            #print(inputs[f"{self.input_name}_{field}"])#dimensions (events, particles)
            
            (ia,ib)=pos_enc_fn(inputs[f"{self.input_name}_{field}"], self.per_input_dim, self.alpha)

            #(ia,ib)=pos_enc_fn(inputs[f"{self.input_name}_{field}"][i*self.per_input_dim:(i+1)*self.per_input_dim], self.per_input_dim, self.alpha)

            #(print(inputs[f"{self.input_name}_{field}"].shape))

            #(print(ib.shape))

            #print(xsin[..., i*self.per_input_dim:(i+1)*self.per_input_dim].shape)

            xsin[..., i*self.per_input_dim:(i+1)*self.per_input_dim] = (xsin[..., i*self.per_input_dim:(i+1)*self.per_input_dim]*ib)

            xcos[..., i*self.per_input_dim:(i+1)*self.per_input_dim] = (xcos[..., i*self.per_input_dim:(i+1)*self.per_input_dim]*ia)

        
        #xsin = xsin.reshape(-1, self.per_input_dim // 2, 2).flip(-1).reshape(-1, self.per_input_dim)
        
        original_shape = xsin.shape
        # Reshape without flattening the batch and sequence dimensions
        xsin = xsin.reshape(original_shape[0], original_shape[1], -1, 2)
        xsin = xsin.flip(-1)
        # Reshape back to the exact original shape
        xsin = xsin.reshape(original_shape)

        ind1 = torch.arange(self.per_input_dim*len(self.fields), device=target_device)

        negation_mask = torch.where(ind1 % 2 == 0, 1., -1.)
        
        xsin = xsin * negation_mask

        #xsin=xsin*keep1

        #xsin = xsin.view(-1, 2).flip(dims=[-1]).view(-1)
        
        xout=xcos+xsin

        if(self.remainder_dim!=0):
            last_column = x[:,:, -self.remainder_dim:]

            xout = torch.cat([xout, last_column], dim=-1)

        

        
        return xout
    


class RopeEncoder4D(nn.Module):
    def __init__(self, input_name: str, fields: list[str], dim: int, sym_fields: list[str] | None = None, alpha=1000):#, dim: int
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
        self.sym_fields = sym_fields or []# ["dtheta", "dphi","phi","theta"] ###added this was  sym_fields or []

        self.alpha = alpha


    def forward(self, inputs: dict, x):#version where rotated in every direction at once
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
        #lenin=len(x[0])
        
        lenin = x.shape[-1]
        # Determine the target device from an input tensor
        target_device = x.device

        

        self.per_input_dim = (lenin // 4)*4
        self.remainder_dim = lenin % 4

        xuse=x[:,:, :self.per_input_dim]

            


        #matrices = torch.zeros((lenin // 4), 4, 4,device=target_device)

        for i in range(3):

            field=self.fields[i]

            pos_enc_fn = pos_enc_symmetric if field in self.sym_fields else pos_enc_rope_1
            
            (cosx,sinx)=pos_enc_fn(inputs[f"{self.input_name}_{field}"], self.per_input_dim//4, self.alpha)
            #cosx 3d: event, hit, frequency
            if i==0:

                matrices = torch.zeros((*cosx.shape, 4, 4),device=target_device)#5d event, hit, frequency, matrix at that freq

                #print(matrices.shape)

                #print(cosx.shape)

                matrices[...,0,0]=cosx

                matrices[...,0,1]=-sinx

                matrices[...,1,0]=sinx

                matrices[...,2,0]=sinx

                matrices[...,1,1]=cosx

                matrices[...,2,1]=cosx

            if i==1:
                matrices[...,3,3]=cosx

                matrices[...,3,2]=sinx

                matrices[...,2,3]=-sinx

                matrices[...,1,3]=sinx

                matrices[...,2,2]=cosx

                matrices[...,1,2]=-cosx

            if i==2:
                matrices[...,1,0]=matrices[...,1,0]*cosx

                matrices[...,2,0]=matrices[...,2,0]*sinx

                matrices[...,1,1]=matrices[...,1,1]*cosx

                matrices[...,1,2]=matrices[...,1,2]*sinx

                matrices[...,1,2]=matrices[...,1,2]*sinx

                matrices[...,2,2]=matrices[...,2,2]*cosx

                matrices[...,1,3]=matrices[...,1,3]*sinx

                matrices[...,2,3]=matrices[...,2,3]*cosx


                

        ##totmat= matrices.reshape(-1, 4)

        ##xout=xuse@totmat
        x_grouped = xuse.view(x.shape[0], x.shape[1], -1, 4)#4d event, hit, dif freqs, features for freq

        #print(x_grouped.shape)

        #print(matrices.shape)

        # Apply the i-th 4x4 matrix to the i-th group of 4 features
        # 'bsdg, dgh -> bsdh' means:
        # For each item in Batch (b) and Sequence (s), multiply the
        # grouped dimension (d) of x with the matrix stack.
        # (B, S, D, 4) @ (D, 4, 4) -> (B, S, D, 4)
        #xout_grouped = torch.einsum('bsdg, dgh -> bsdh', x_grouped, matrices)

        xout_grouped = torch.einsum('abcde, abce -> abcd', matrices,x_grouped)

        #print(xout_grouped.shape)

        # Reshape the output back to the original feature dimension order
        xout = xout_grouped.reshape(x.shape[0], x.shape[1], -1)

        #matrices_comb = matrices.reshape(*matrices.shape[:-3], -1, matrices.shape[-1])

        #xuse@


        if(self.remainder_dim!=0):
            last_column = x[:,:, -self.remainder_dim:]

            xout = torch.cat([xout, last_column], dim=-1)

        

        
        return xout
    