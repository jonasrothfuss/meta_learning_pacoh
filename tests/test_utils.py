import unittest
from src.models import NeuralNetworkBatched, NeuralNetwork
import torch

class TestNN(unittest.TestCase):

    def testOutputShape(self):
        nn = NeuralNetwork(input_dim=2, output_dim=3, layer_sizes=(4, 5, 6))
        x = torch.normal(mean=torch.zeros(7, 2))
        y = nn(x)
        assert y.shape == (7, 3)

class TestNNBatched(unittest.TestCase):

    def testOutputShape(self):
        nn = NeuralNetworkBatched(input_dim=4, output_dim=5, layer_sizes=(8, 12))
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
        nn = NeuralNetworkBatched(input_dim=2, output_dim=5, layer_sizes=tuple())

        # set params
        nn.out.W = torch.nn.Parameter(torch.cat([torch.zeros(1, 5, 2), torch.ones(1, 5, 2)]))
        nn.out.b = torch.nn.Parameter(torch.cat([torch.zeros(1, 5), torch.ones(1, 5)]))

        x = torch.ones(7, 2)
        y = nn(x)

        assert torch.all(y[0] == torch.zeros((7, 5))).item()
        assert torch.all(y[1] == 3.0 * torch.ones((7, 5))).item()
