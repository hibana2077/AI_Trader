import torch

#GRU (Regression model)
class Model(torch.nn.Module):
    """
    A class representing a PyTorch model with a GRU layer and a linear layer.

    Attributes:
    -----------
    input_size : int
        The size of the input tensor.
    output_size : int
        The size of the output tensor.

    Methods:
    --------
    forward(x)
        Defines the forward pass of the model.
    """
    def __init__(self,input_size,output_size):
        super(Model,self).__init__()
        self.gru = torch.nn.GRU(input_size, 10, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(10, output_size)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self,x):
        """
        Defines the forward pass of the model.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor.

        Returns:
        --------
        torch.Tensor
            The output tensor.
        """
        x, _ = self.gru(x)
        x = self.linear(x)
        #x = self.relu(x)
        #x = self.sigmoid(x)
        #x = self.tanh(x)
        return x