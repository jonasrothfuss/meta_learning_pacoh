import unittest
from src.models import NeuralNetworkVectorized, NeuralNetwork, LinearVectorized
import torch

class TestNN(unittest.TestCase):

    def testOutputShape(self):
        nn = NeuralNetwork(input_dim=2, output_dim=3, layer_sizes=(4, 5, 6))
        x = torch.normal(mean=torch.zeros(7, 2))
        y = nn(x)
        assert y.shape == (7, 3)

class TestLinearVectorized(unittest.TestCase):

    def testOutputShape(self):
        nn = LinearVectorized(5, 4)
        x = torch.normal(mean=torch.zeros(7, 5))
        y = nn(x)
        assert y.shape == (7, 4)

    def testParam_shape(self):
        nn = LinearVectorized(5, 4)
        for param in nn.parameters():
            assert param.ndim == 1

    def testConsistency(self):
        # linear vectorized
        x = torch.normal(0, 1.0, size=(20,1))
        nn = LinearVectorized(1, 1)

        W = torch.tensor([[1.0], [2.0]])
        b = torch.tensor([[-1.0], [-2.0]])

        nn.weight = torch.nn.Parameter(W)
        nn.bias = torch.nn.Parameter(b)
        y = nn(x)

        # standard Linear
        for i in range(2):
            nn = LinearVectorized(1, 1)
            nn.weight = torch.nn.Parameter(W[i])
            nn.bias = torch.nn.Parameter(b[i])
            y_i = nn(x)
            assert torch.sum(y_i - y[i]).item() < 0.01

    def testfitting(self):
        torch.manual_seed(23)
        x_data = torch.normal(mean=0.0, std=2., size=(50, 3))
        W = torch.diag(torch.tensor([1., 2., 3.])).T
        b = torch.tensor([2.0, 5.0, -2.2])
        y_data = x_data.matmul(W.T) + b

        nn = LinearVectorized(3, 3)
        optim = torch.optim.Adam(params=nn.parameters(), lr=0.1)

        for step in range(500):
            optim.zero_grad()
            loss = torch.mean((nn(x_data) - y_data)**2)
            loss.backward()
            optim.step()
            if step % 100 == 0:
                print('step %i | loss: %.4f'%(step, loss.item()))

        params = dict(nn.named_parameters())
        W_norm = torch.sum((params['weight'].reshape(3,3) - W)**2)
        b_norm = torch.sum((params['bias']- b) ** 2)
        assert W_norm.item() <= 0.1
        assert b_norm.item() <= 0.1

class TestNNVectorized(unittest.TestCase):

    def testOutputShape(self):
        nn = NeuralNetworkVectorized(input_dim=4, output_dim=5, layer_sizes=(8, 12))

        for name, param in nn.named_parameters().items():
            nn.set_parameter(name, param.unsqueeze(0))

        x = torch.normal(mean=torch.zeros(7, 4))
        y = nn(x)
        assert y.shape == (1, 7, 5)

        x = torch.normal(mean=torch.zeros(1, 7, 4))
        y = nn(x)
        assert y.shape == (1, 7, 5)

        def try_unsupported_shape():
            x = torch.normal(mean=torch.zeros(2, 7, 4))
            y = nn(x)

        self.assertRaises(AssertionError, try_unsupported_shape)

    def testForward(self):
        nn = NeuralNetworkVectorized(input_dim=2, output_dim=5, layer_sizes=tuple())

        for name, param in nn.named_parameters().items():
            nn.set_parameter(name, param.unsqueeze(0))

        # set params
        nn.out.weight = torch.nn.Parameter(torch.cat([torch.zeros(1, 5 * 2), torch.ones(1, 5 * 2)]))
        nn.out.bias = torch.nn.Parameter(torch.cat([torch.zeros(1, 5), torch.ones(1, 5)]))

        x = torch.ones(7, 2)
        y = nn(x)

        assert torch.all(y[0] == torch.zeros((7, 5))).item()
        assert torch.all(y[1] == 3.0 * torch.ones((7, 5))).item()
