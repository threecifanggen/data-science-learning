from typing import NoReturn
import torch.nn as nn
import torch
import plotly.express as px
import pandas as pd

class MNISTDiscriminator(nn.Module):
    """判别器
    """

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

        self.loss_function = nn.BCELoss(reduction='sum')

        self.optimiser = torch.optim.Adam(
            self.parameters(),
            lr=0.0001
        )
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

class MNISTGenerator(nn.Module):
    """生成器
    """

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(100, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 784),
            nn.Sigmoid()
        )

        self.optimiser = torch.optim.Adam(
            self.parameters(),
            lr=0.0001
        )

        self.counter = 0
        self.progress = []


    def forward(self, inputs: torch.FloatTensor):
        return self.model(inputs)

    def train(
            self,
            d: MNISTDiscriminator,
            inputs: torch.FloatTensor,
            targets: torch.FloatTensor
        ):
        g_output = self.forward(inputs)
        d_output = d.forward(g_output)
        
        loss = d.loss_function(d_output, targets)

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

