from typing import NoReturn
import torch.nn as nn
import torch
import plotly.express as px
import pandas as pd

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.contiguous().view(*self.shape)

class CELEBADiscriminator(nn.Module):
    """判别器
    """

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            View(218*178*3),
            nn.Linear(218*178*3, 100),
            nn.LeakyReLU(),
            nn.LayerNorm(100),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

        self.loss_function = nn.BCELoss(reduction='sum')

        self.optimiser = torch.optim.Adam(
            self.parameters(),
            lr=0.0001
        )
        self.temp_loss = 1.
        self.counter = 0
        self.progress = []

    def forward(self, inputs: torch.FloatTensor):
        return self.model(inputs)
    
    def train(
            self,
            inputs: torch.FloatTensor,
            targets: torch.FloatTensor
            ) -> NoReturn:
        outputs = self.forward(inputs)

        loss = self.loss_function(outputs, targets)

        self.temp_loss = loss

        self.counter += 1

        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        if self.counter % 10000 == 0:
            print(f"counter = {self.counter}")

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self) -> NoReturn:
        """绘制过程图
        """
        df = pd.DataFrame({
            "step": [i * 10 for i in range(1, len(self.progress) + 1)],
            "loss": self.progress
        })

        fig = px.line(df, x="step", y="loss")
        fig.show()

class CELEBAGenerator(nn.Module):
    """生成器
    """

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(100, 3*10*10),
            nn.LeakyReLU(),
            nn.LayerNorm(3*10*10),
            nn.Linear(3*10*10, 3*218*178),
            # nn.Sigmoid(),
            View((218, 178, 3)),
            nn.Sigmoid()
        )

        self.optimiser = torch.optim.Adam(
            self.parameters(),
            lr=0.0001
        )
        self.temp_loss = 1.
        self.counter = 0
        self.progress = []


    def forward(self, inputs: torch.FloatTensor):
        return self.model(inputs)

    def train(
            self,
            d: CELEBADiscriminator,
            inputs: torch.FloatTensor,
            targets: torch.FloatTensor
        ):
        g_output = self.forward(inputs)
        d_output = d.forward(g_output)
        
        loss = d.loss_function(d_output, targets)
        self.temp_loss = loss

        self.counter += 1
        
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
        
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self) -> NoReturn:
        """绘制过程图
        """
        df = pd.DataFrame({
            "step": [i * 10 for i in range(1, len(self.progress) + 1)],
            "loss": self.progress
        })

        fig = px.line(df, x="step", y="loss")
        fig.show()

