#kan in mamba
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", 
                dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                dt_init_floor=1e-4, dropout=0., conv_bias=True, bias=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self._ssm_warned = False
        # if set True from outside, force SSM to run on CPU (safety)
        self.force_cpu = False

    def forward(self, x):
        """
        x: (B, L, D)
        """
        B, L, D = x.shape

        # Input projection
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner)

        # Convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :L]
        x = rearrange(x, 'b d l -> b l d')

        # SSM
        x = F.silu(x)
        y = self.ssm(x)
        
        # Gating
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)
        
        if self.dropout is not None:
            output = self.dropout(output)
            
        return output

    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper
            - run_SSM(A, B, C, u) in The Annotated S4 [2]
        """
        # Ensure dtype/device alignment and contiguous memory to avoid CUDA misalignment
        x_dtype = x.dtype
        x_device = x.device
        A_log = self.A_log.to(dtype=x_dtype, device=x_device).contiguous()
        D_param = self.D.to(dtype=x_dtype, device=x_device).contiguous()
        A = -torch.exp(A_log)  # (d_inner, d_state)
        D = D_param

        deltaBC = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)

        try:
            return self.selective_scan(x, delta, A, B, C, D)
        except RuntimeError as e:
            # Optional safety fallback: run SSM on CPU if CUDA kernel misalignment occurs
            if 'misaligned address' in str(e).lower() or 'device-side assert' in str(e).lower():
                if not self._ssm_warned:
                    print('[Warn][MambaBlock] CUDA SSM failed, falling back to CPU selective_scan once. '
                          'Set CUDA_LAUNCH_BLOCKING=1 for precise trace.')
                    self._ssm_warned = True
                x_cpu = x.detach().to('cpu')
                delta_cpu = delta.detach().to('cpu')
                A_cpu = A.detach().to('cpu')
                B_cpu = B.detach().to('cpu')
                C_cpu = C.detach().to('cpu')
                D_cpu = D.detach().to('cpu')
                y_cpu = self.selective_scan(x_cpu, delta_cpu, A_cpu, B_cpu, C_cpu, D_cpu)
                return y_cpu.to(x_device)
            raise

    def selective_scan(self, u, delta, A, B, C, D):

        B_batch, L, d_inner = u.shape
        d_state = A.shape[1]

        deltaA = torch.exp(delta.unsqueeze(-1) * A)           # (B, L, d_inner, d_state)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)          # (B, L, d_inner, d_state)

        x = torch.zeros(B_batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []

        for i in range(L):
            x = deltaA[:, i] * x + deltaB[:, i] * u[:, i].unsqueeze(-1)
            y = torch.sum(x * C[:, i].unsqueeze(1), dim=-1)   # (B, d_inner)
            ys.append(y)

        y = torch.stack(ys, dim=1)                             # (B, L, d_inner)
        y = y + u * D
        return y
