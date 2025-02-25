import torch

class SuperSiren(torch.nn.Module):
    def __init__(self,
        device:torch.device,
        dim_input:int,
        dim_output:int,
        dim_queue:int,
        dim_slice:list[int],
        dim_hidden:int,
        dim_layers:int,
        dim_functa:int,
        inner_steps:int,
        omega:int,
        activation:str,
    ) -> None:
        
        super().__init__()
        self.device = device
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_queue = dim_queue
        self.dim_slice = dim_slice
        self.dim_data = len(dim_slice)
        self.dim_hidden = dim_hidden
        self.dim_layers = dim_layers
        self.dim_functa = dim_functa
        self.inner_steps = inner_steps
        self.omega = omega

        self.lr = torch.nn.Parameter(torch.zeros(dim_functa, dtype=torch.float,device=device))
        self.encoder = torch.nn.Linear(dim_functa, (dim_layers+1)*dim_hidden)
        self.layer_input = torch.nn.Linear(dim_input, dim_hidden)
        self.layer_hidden = torch.nn.ModuleList([torch.nn.Linear(dim_hidden, dim_hidden) for _ in range(dim_layers)])
        self.layer_output = torch.nn.Linear(dim_hidden, dim_output)
        self.init()
        self.to(device)

        if activation == 'spder':
            print('using spder activation')
            self.activation = lambda x: torch.sin(x) * torch.sqrt(torch.abs(x))
        elif activation == 'wire':
            print('using wire activation')
            self.activation = lambda x: torch.cos(x) * torch.exp(-(x**2))
        else:
            print('using siren activation')
            self.activation = lambda x: torch.sin(x)


    def get_lr(self) -> torch.Tensor:
        return torch.clamp(self.lr, 0, 1)
    
    def init(self) -> None:
        with torch.no_grad():
            self.lr.uniform_(0.005,0.1)
            self.layer_input.weight.uniform_(-1/self.dim_input,1/self.dim_input)
            self.layer_input.bias.zero_()
            for layer in self.layer_hidden:
                layer.weight.uniform_(-torch.sqrt(torch.tensor(6/self.dim_hidden))/self.omega, torch.sqrt(torch.tensor(6/self.dim_hidden))/self.omega)
                layer.bias.zero_()
            self.layer_output.weight.uniform_(-torch.sqrt(torch.tensor(6/self.dim_hidden))/self.omega, torch.sqrt(torch.tensor(6/self.dim_hidden))/self.omega)
            self.layer_output.bias.zero_()


    def forward(self, coordinates:torch.Tensor, data:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert data.dim() == 2*self.dim_data+2
        b = data.shape[0]
        s = self.inner_steps + 1
        f = self.dim_functa
        n_ = data.shape[1:self.dim_data+1]
        c = data.shape[self.dim_data+1]
        w_ = data.shape[self.dim_data+2:]

        data = torch.reshape(data, (-1, c, *w_))                                                                # (b*n, c, w_)
        data = torch.flatten(data, start_dim=2)                                                                 # (b*n, c, w)
        k = data.shape[0]
        n = k//b
        w = data.shape[-1]
        coordinates = coordinates[0]                                                                            # (w, c) first from batch
        pred, loss, functa = self.inner_loop(coordinates, data)                                                 # (k, s, c, w), (1,), (k, f)

        pred = torch.reshape(pred, (b, n, s, c, w))                                                             # (b, n, s, c, w)
        pred = torch.permute(pred, (0, 2, 1, 3, 4))                                                             # (b, s, n, c, w)
        pred = torch.reshape(pred, (b, s, *n_, c, *w_))                                                         # (b, s, n_, c, w_)
        functa = torch.reshape(functa, (b, n, f))                                                               # (b, n, f)
        functa = torch.permute(functa, (0, 2, 1))                                                               # (b, f, n)
        functa = torch.reshape(functa, (b, f, *n_))                                                             # (b, f, n_)        
        return pred, loss, functa


    def inner_loop(self, coordinates:torch.Tensor, data:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = data.shape[0]
        f = self.dim_functa
        s = self.inner_steps

        # inner loop
        data_steps = []
        functa = torch.zeros((k, f), requires_grad=True, device=self.device)                                    # (k, f)
        for _ in range(s):
            pred = self.decode(coordinates, functa)                                                             # (k, c, w)
            data_steps.append(pred.detach().clone())
            loss = torch.square(data - pred)                                                                    # (k, c, w)
            loss = torch.flatten(loss, start_dim=1)                                                             # (k, c*w)
            loss = torch.mean(loss,dim=-1)                                                                      # (k,)
            loss = torch.sum(loss)                                                                              # (1,)
            functa_grad = torch.autograd.grad(loss, functa, create_graph=True)[0]                               # (k, f)
            functa = functa - self.get_lr()*functa_grad                                                         # (k, f)
        
        # outer loop
        pred = self.decode(coordinates, functa)                                                                 # (k, c, w)
        data_steps.append(pred.detach())
        data_steps = torch.stack(data_steps, axis=1)                                                            # (k, s, c, w)
        loss = torch.square(data - pred)                                                                        # (k, c, w)
        loss = torch.flatten(loss, start_dim=1)                                                                 # (k, c*w)
        loss = torch.mean(loss,dim=-1)                                                                          # (k,)
        loss = torch.sum(loss)                                                                                  # (1,)

        return data_steps, loss, functa.detach()


    def decode(self, coordinates:torch.Tensor, functa:torch.Tensor) -> torch.Tensor:
        k = functa.shape[0]
        l = self.dim_layers + 1
        u = self.dim_hidden

        shift_modulations = self.encoder(functa)                                                                # (k, l*u)
        shift_modulations = torch.reshape(shift_modulations, (k, l, u))                                         # (k, l, u)
        shift_modulations = torch.unsqueeze(shift_modulations, 1)                                               # (k, 1, l, u)
        coordinates = torch.unsqueeze(coordinates, 0)                                                           # (1, w, i)

        tensor = self.activation(self.omega*(self.layer_input(coordinates) + shift_modulations[:,:,0]))         # (k, w, u)
        for i, layer in enumerate(self.layer_hidden):
            tensor = self.activation(self.omega*(layer(tensor) + shift_modulations[:,:,i+1]))                   # (k, w, u)
        tensor = self.layer_output(tensor)                                                                      # (k, w, c)
        tensor = torch.permute(tensor, (0,2,1))                                                                 # (k, c, w)
        tensor = tensor + 0.5
        return tensor


    def data_to_functa(self, coordinates:torch.Tensor, data:torch.Tensor):
        assert data.dim() == 2*self.dim_data+2
        b = data.shape[0]
        s = self.inner_steps + 1
        f = self.dim_functa
        n_ = data.shape[1:self.dim_data+1]
        c = data.shape[self.dim_data+1]
        w_ = data.shape[self.dim_data+2:]
        q = self.dim_queue

        data = torch.reshape(data, (-1, c, *w_))                                                                # (k, c, w_)
        data = torch.flatten(data, start_dim=2)                                                                 # (k, c, w)
        coordinates = coordinates[0]                                                                            # (w, c) first from batch
        k = data.shape[0]
        w = data.shape[-1]
        n = k//b

        functa = torch.zeros((k, f), device=self.device)                                                        # (k, f)
        pred = torch.zeros((k, s, c, w), device=self.device)                                                    # (k, s, c, w)

        for i in range(0, k, q):
            batch_data = data[i:i+q]
            batch_pred, batch_loss, batch_functa = self.inner_loop(coordinates, batch_data)                     # (q, s, c, w), (1,), (q, f)
            functa[i:i+q] = batch_functa.detach().clone()
            pred[i:i+q] = batch_pred.detach().clone()
            del batch_pred, batch_functa, batch_loss

        functa = torch.reshape(functa, (b, n, f))                                                               # (b, n, f)
        functa = torch.permute(functa, (0, 2, 1))                                                               # (b, f, n)
        functa = torch.reshape(functa, (b, f, *n_))                                                             # (b, f, n_)        

        pred = torch.reshape(pred, (b, n, s, c, w))                                                             # (b, n, s, c, w)
        pred = torch.permute(pred, (0, 2, 1, 3, 4))                                                             # (b, s, n, c, w)
        pred = torch.reshape(pred, (b, s, *n_, c, *w_))                                                         # (b, s, n_, c, w_)
        return pred, functa
      



    def functa_to_data(self, coordinates:torch.Tensor, functa:torch.Tensor) -> torch.Tensor:
        b = functa.shape[0]
        f = functa.shape[1]
        n = functa.shape[2:]
        c = self.dim_output
        w = self.dim_slice

        coordinates = coordinates[0]                                                                            # (w, c) first from batch
        functa = torch.reshape(functa, (b, f, -1))                                                              # (b, f, n)
        functa = torch.permute(functa, (0, 2, 1))                                                               # (b, n, f)
        functa = torch.reshape(functa, (-1, f))                                                                 # (k, f)
        pred = self.decode(coordinates, functa)                                                                 # (k, c, w_)
        pred = torch.reshape(pred, (b, 1, *n, c, *w))                                                           # (b, 1, n~, c, w~)
        return pred.detach()        