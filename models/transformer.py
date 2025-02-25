import torch

class FunctaTransformer(torch.nn.Module):
    def __init__(self,
        dim_data:list[int],
        dim_window:list[int],
        dim_functa:int,
        dim_embedding:int,
        dim_blocks:list[int],
        dim_hidden:int,
        num_head:int,
        drop_conn:float,
        drop_attn:float,
        drop_ffn:float,
    ) -> None:
        super().__init__()
        self.dim_data = dim_data
        self.dim_window = dim_window
        self.dim_input = len(dim_data)
        self.register_buffer('mask', self.get_mask(dim_data, dim_window))

        if self.dim_input==1:
            self.shallow_extractor = torch.nn.Conv1d(dim_functa, dim_embedding, 3, padding=1)
            self.deep_extractor = torch.nn.Conv1d(dim_embedding, dim_embedding, 3, padding=1)
            self.reconstructor = torch.nn.Conv1d(dim_embedding, dim_functa, 3, padding=1)
        elif self.dim_input==2:
            self.shallow_extractor = torch.nn.Conv2d(dim_functa, dim_embedding, 3, padding=1)
            self.deep_extractor = torch.nn.Conv2d(dim_embedding, dim_embedding, 3, padding=1)
            self.reconstructor = torch.nn.Conv2d(dim_embedding, dim_functa, 3, padding=1)
        elif self.dim_input==3:
            self.shallow_extractor = torch.nn.Conv3d(dim_functa, dim_embedding, 3, padding=1)
            self.deep_extractor = torch.nn.Conv3d(dim_embedding, dim_embedding, 3, padding=1)
            self.reconstructor = torch.nn.Conv3d(dim_embedding, dim_functa, 3, padding=1)
        else:
            raise ValueError(f'dim {self.dim_input} not supported.')
        
        drop_conn = [p.item() for p in torch.linspace(0, drop_conn, sum(dim_blocks))]
        
        self.deep_blocks = torch.nn.ModuleList([
            SwinBlock(
                dim_data,
                dim_window,
                dim_embedding,
                dim_blocks[i],
                dim_hidden,
                num_head,
                drop_conn[sum(dim_blocks[:i]):sum(dim_blocks[:i+1])],
                drop_attn,
                drop_ffn,
            )
            for i in range(len(dim_blocks))
        ])

        with torch.no_grad():
            self.apply(self.init)

    def forward(self,
        x:torch.Tensor,
    ) -> torch.Tensor:
        b = x.shape[0]
        f = x.shape[1]
        d = list(x.shape[2:])

        if d == self.dim_data:
            mask = self.mask
        else:
            mask = self.get_mask(d, self.dim_window).to(x.device)

        x1 = self.shallow_extractor(x)                                                  # (b, e, d~)
        x2 = x1
        for block in self.deep_blocks:
            x2 = block(x2, mask)                                                        # (b, e, d~)
        x2 = self.deep_extractor(x2)                                                    # (b, e, d~)
        x3 = x2 + x1                                                                    # (b, e, d~)
        x3 = self.reconstructor(x3)                                                     # (b, f, d~)
        x = x3 + x
        return x

    def init(self,
        module:torch.nn.Module
    ) -> None:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, mean=0.,std=.02,a=-2,b=2.)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0.)
            torch.nn.init.constant_(module.weight, 1.)
        elif isinstance(module, SwinLayer):
            for param_name, param in module.named_parameters():
                if param_name == 'pos_bias':
                    torch.nn.init.trunc_normal_(param, mean=0.,std=.02,a=-2,b=2.)

    def get_mask(self,
        dim_data:list[int],
        dim_window:list[int]
    ) -> torch.Tensor:
        d = dim_data
        w = dim_window
        n = len(d)
        k = 1
        mask = torch.zeros(*d)
        for i in range(n):
            mask = torch.transpose(mask, 0, i)
            mask[d[i]-w[i]:d[i]] += k
            mask[d[i]-w[i]//2:d[i]] += k
            k *= 3
            mask = torch.transpose(mask, 0, i)

        part = [j for i in range(n) for j in (d[i]//w[i],w[i])]
        perm = [j for i in (range(0,2*n,2), range(1,2*n,2)) for j in i]
        wins = torch.prod(torch.tensor(w)).item()

        mask = torch.reshape(mask, part)
        mask = torch.permute(mask, perm)
        mask = torch.reshape(mask, (-1, wins))
        mask = mask.unsqueeze(1) - mask.unsqueeze(2)
        mask[mask!=0] = -100.
        mask = mask.unsqueeze(1)
        mask = mask.unsqueeze(0)
        return mask.detach()


class SwinBlock(torch.nn.Module):
    def __init__(self,
        dim_data:list[int],
        dim_window:list[int],
        dim_embedding:int,
        dim_layers:int,
        dim_hidden:int,
        num_head:int,
        drop_conn:list[float],
        drop_attn:float,
        drop_ffn:float,
    ) -> None:
        super().__init__()
        self.dim_input = len(dim_data)

        self.layers = torch.nn.ModuleList([
            SwinLayer(
                dim_data,
                dim_window,
                dim_embedding,
                dim_hidden,
                num_head,
                False if (i%2==0) else True,
                drop_conn[i],
                drop_attn,
                drop_ffn,
            )
            for i in range(dim_layers)
        ])

        if self.dim_input==1:
            self.conv = torch.nn.Conv1d(dim_embedding, dim_embedding, 3, padding=1)
        elif self.dim_input==2:
            self.conv = torch.nn.Conv2d(dim_embedding, dim_embedding, 3, padding=1)
        elif self.dim_input==3:
            self.conv = torch.nn.Conv3d(dim_embedding, dim_embedding, 3, padding=1)
        else:
            raise ValueError(f'dim {self.dim_input} not supported.')

    def forward(self,
        x:torch.Tensor,
        mask:torch.Tensor
    ) -> torch.Tensor:
        x1 = x
        for layer in self.layers:
            x1 = layer(x1, mask)                                                        # (b, e, d~)
        x1 = self.conv(x1)                                                              # (b, e, d~)
        x = x1 + x                                                                      # (b, e, d~)
        return x


class SwinLayer(torch.nn.Module):
    def __init__(self,
        dim_data:list[int],
        dim_window:list[int],
        dim_embedding:int,
        dim_hidden:int,
        num_head:int,
        shifted:bool,
        drop_conn:float,
        drop_attn:float,
        drop_ffn:float,
    ) -> None:
        super().__init__()
        self.dim_input = len(dim_data)
        self.dim_data = dim_data
        self.shifted = shifted
        self.dim_window = dim_window
        self.pos_bias = self.get_pos_bias(dim_window, num_head)
        self.norm1 = torch.nn.LayerNorm(dim_embedding)
        self.attention = Attention(
            dim_embedding,
            num_head,
            self.pos_bias,
            drop_attn,
        )
        self.norm2 = torch.nn.LayerNorm(dim_embedding)
        self.mlp = MLP(
            dim_embedding,
            dim_hidden,
            drop_ffn,
        )
        self.drop_conn = StochasticDepth(drop_conn)

        self.e_last = list(range(self.dim_input+2))
        self.e_last.append(self.e_last.pop(1))

        self.e_first = list(range(self.dim_input+2))
        self.e_first.insert(1, self.e_first.pop(-1))

    def forward(self,
        x:torch.Tensor,
        mask:torch.Tensor
    ) -> torch.Tensor:
        b = x.shape[0]
        e = x.shape[1]
        d = x.shape[2:]

        x = torch.permute(x, self.e_last)                                               # (b, d~, e)
        x_norm = self.norm1(x)                                                          # (b, d~, e)

        if self.dim_input==1:
            w1, = self.dim_window
            d1, = d
            if self.shifted:
                x_norm = torch.roll(x_norm, shifts=(-w1//2,), dims=(1,))                    # (b, d1, e)
            x_norm = torch.reshape(x_norm, (b, d1//w1, w1, e))                              # (b, n1, w1, e)

        elif self.dim_input==2:
            w1, w2 = self.dim_window
            d1, d2 = d
            if self.shifted:
                x_norm = torch.roll(x_norm, shifts=(-w1//2, -w2//2), dims=(1,2))            # (b, d1, d2, e)
            x_norm = torch.reshape(x_norm, (b, d1//w1, w1, d2//w2, w2, e))                  # (b, n1, w1, n2, w2, e)
            x_norm = torch.permute(x_norm, (0, 1, 3, 2, 4, 5))                              # (b, n1, n2, w1, w2, e)
            x_norm = torch.reshape(x_norm, (b, -1, w1*w2, e))                               # (b, n1*n2, w1*w2, e)

        elif self.dim_input==3:
            w1, w2, w3 = self.dim_window
            d1, d2, d3 = d
            if self.shifted:
                x_norm = torch.roll(x_norm, shifts=(-w1//2, -w2//2, -w3//2), dims=(1,2,3))
            x_norm = torch.reshape(x_norm, (b, d1//w1, w1, d2//w2, w2, d3//w3, w3, e))      # (b, n1, w1, n2, w2, n3, w3, e)
            x_norm = torch.permute(x_norm, (0, 1, 3, 5, 2, 4, 6, 7))                        # (b, n1, n2, n3, w1, w2, w3, e)
            x_norm = torch.reshape(x_norm, (b, -1, w1*w2*w3, e))                            # (b, n1*n2*n3, w1*w2*w3, e)

        if self.shifted:
            x_norm = self.attention(x_norm, mask)                                           # batch, windows, sequence, embedding
        else:
            x_norm = self.attention(x_norm)                                                 # batch, windows, sequence, embedding


        if self.dim_input==1:
            x_norm = torch.reshape(x_norm, (b, d1, e))                                      # (b, d1, e)
            if self.shifted:
                x_norm = torch.roll(x_norm, shifts=(w1//2,), dims=(1,))                     # (b, d1, e)
        elif self.dim_input==2:
            x_norm = torch.reshape(x_norm, (b, d1//w1, d2//w2, w1, w2, e))                  # (b, n1, n2, w1, w2, e)
            x_norm = torch.permute(x_norm, (0, 1, 3, 2, 4, 5))                              # (b, n1, w1, n2, w2, e)
            x_norm = torch.reshape(x_norm, (b, d1, d2, e))                                  # (b, d1, d2, e)
            if self.shifted:
                x_norm = torch.roll(x_norm, shifts=(w1//2, w2//2), dims=(1,2))              # (b, d1, d2, e)
        elif self.dim_input==3:
            x_norm = torch.reshape(x_norm, (b, d1//w1, d2//w2, d3//w3, w1, w2, w3, e))      # (b, n1, n2, n3, w1, w2, w3, e)
            x_norm = torch.permute(x_norm, (0, 1, 4, 2, 5, 3, 6, 7))                        # (b, n1, w1, n2, w2, n3, w3, e)
            x_norm = torch.reshape(x_norm, (b, d1, d2, d3, e))                              # (b, d1, d2, d3, e)
            if self.shifted:
                x_norm = torch.roll(x_norm, shifts=(w1//2, w2//2, w3//2), dims=(1,2,3))     # (b, d1, d2, d3, e)

        x_norm = self.drop_conn(x_norm)                                                     # stochastic depth 1
        x = x + x_norm                                                                      # residual 1
        x_norm = self.norm2(x)                                                              # (b, d~, e)
        x_norm = self.mlp(x_norm)                                                           # (b, d~, e)
        x_norm = self.drop_conn(x_norm)                                                     # stochastic depth 2
        x = x + x_norm                                                                      # residual 2
        x = torch.permute(x, self.e_first)                                                  # (b, e, d~)
        return x


    def get_pos_bias(self,
        dim_window:list[int],
        num_heads:int
        ) -> torch.nn.Parameter:
        bias_table_size = 1
        for i in range(len(dim_window)):
            bias_table_size *= (2*dim_window[i] - 1)
        bias_table = torch.zeros((bias_table_size, num_heads))
        coords = torch.cartesian_prod(*[torch.arange(d) for d in dim_window])           # (s=d1*d2*d3*..., i)
        if len(dim_window) < 2:
            coords = torch.unsqueeze(coords, -1)
        coords_rel = coords.unsqueeze(1) - coords.unsqueeze(0)                          # (s, s, i)
        coords_rel = torch.permute(coords_rel, (2,0,1))                                 # (i, s, s) ∈ [-(d-1), +(d-1)]
        for i in range(len(dim_window)):
            coords_rel[i] += dim_window[i] - 1                                          # remap to positive only [0, 2d-2]
            scale_factor = 1
            for j in range(i+1, len(dim_window)):
                scale_factor *= (2*dim_window[j] - 1)
            coords_rel[i] *= scale_factor                                               # unique distance in every dim

        coords_rel = torch.sum(coords_rel, 0)                                           # (s, s)
        bias_table = bias_table[coords_rel]                                             # (s, s, h)
        bias_table = torch.permute(bias_table, (2, 0, 1))                               # (h, s, s)
        bias_table = torch.unsqueeze(bias_table, 0)                                     # (w, h, s, s)
        bias_table = torch.unsqueeze(bias_table, 0)                                     # (b, w, h, s, s)
        bias_table = bias_table.detach()
        return torch.nn.Parameter(bias_table)


class Attention(torch.nn.Module):
    def __init__(self,
        dim_embedding:int,
        num_head:int,
        pos_bias:torch.nn.Parameter,
        dropout:float,
    ) -> None:
        super().__init__()
        self.num_head = num_head
        self.dim_head = dim_embedding // num_head
        self.scale = self.dim_head**-0.5
        self.q = torch.nn.Linear(dim_embedding, self.dim_head*num_head)
        self.k = torch.nn.Linear(dim_embedding, self.dim_head*num_head)
        self.v = torch.nn.Linear(dim_embedding, self.dim_head*num_head)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.proj = torch.nn.Linear(self.dim_head*num_head, dim_embedding)
        self.pos_bias = pos_bias
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self,
        x:torch.Tensor,
        mask:torch.Tensor=None
    ) -> torch.Tensor:
        d = self.dim_head
        h = self.num_head
        b, w, s, e = x.shape                                            # batch, windows, sequence, embedding
        q = self.q(x)                                                   # batch, windows, sequence, heads*q_dim
        k = self.k(x)                                                   # batch, windows, sequence, heads*k_dim
        v = self.v(x)                                                   # batch, windows, sequence, heads*v_dim
        q = self.scale * q                                              # batch, windows, sequence, heads*q_dim

        q = torch.reshape(q, (b, w, s, h, d))                           # batch, windows, sequence, heads, q_dim
        q = torch.permute(q, (0, 1, 3, 2, 4))                           # batch, windows, heads, sequence, q_dim
        k = torch.reshape(k, (b, w, s, h, d))                           # batch, windows, sequence, heads, k_dim
        k = torch.permute(k, (0, 1, 3, 2, 4))                           # batch, windows, heads, sequence, k_dim
        v = torch.reshape(v, (b, w, s, h, d))                           # batch, windows, sequence, heads, v_dim
        v = torch.permute(v, (0, 1, 3, 2, 4))                           # batch, windows, heads, sequence, v_dim

        attention = q @ k.transpose(-2,-1)                              # batch, windows, heads, sequence, sequence
        attention = attention + self.pos_bias                           # batch, windows, heads, sequence, sequence
        
        if mask is not None:
            attention = attention + mask                                # batch, windows, heads, sequence, sequence

        attention = self.softmax(attention)                             # batch, windows, heads, sequence, sequence
        attention = self.dropout(attention)                             # attention dropout
        z = attention @ v                                               # batch, windows, heads, sequence, v_dim
        z = torch.transpose(z, 2, 3)                                    # batch, windows, sequence, heads, v_dim
        z = torch.reshape(z, (b, w, s, h*d))                            # batch, windows, sequence, heads*v_dim
        z = self.proj(z)                                                # batch, windows, sequence, embedding
        return z


class MLP(torch.nn.Module):
    def __init__(self, 
        dim_embedding,
        dim_hidden,
        dropout,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(dim_embedding, dim_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(dim_hidden, dim_embedding),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout)
        )

    def forward(self,
        x:torch.Tensor
    ) -> torch.Tensor:
        return self.layers(x)


class StochasticDepth(torch.nn.Module):
    def __init__(self,
        probability:float
    ) -> None:
        super().__init__()
        self.prob = probability
        self.keep_prob = 1 - self.prob

    def forward(self, 
        x:torch.Tensor
    ) -> torch.Tensor:

        if self.prob == 0. or not self.training:
            return x

        shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
        random_tensor = self.keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(self.keep_prob) * random_tensor
        return output