import unittest
from src.models import NeuralNetworkVectorized, NeuralNetwork, LinearVectorized, CatDist, EqualWeightedMixtureDist
import torch
import pyro
import numpy as np

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

    def testParamShapes(self):
        nn = NeuralNetworkVectorized(input_dim=2, output_dim=5, layer_sizes=(3,))
        shape_dict = nn.parameter_shapes()
        assert shape_dict['out.bias'] == (5,)
        assert shape_dict['fc_1.weight'] == (2*3,)

    def test_vectorization1(self):
        nn = NeuralNetworkVectorized(input_dim=2, output_dim=5, layer_sizes=(6,))

        x = torch.normal(mean=torch.zeros(22, 2))
        y1 = nn(x)

        params = nn.parameters_as_vector()
        assert params.shape == (2*6 + 6 + 6*5 + 5, )

        nn.set_parameters_as_vector(params)
        y2 = nn(x)

        assert torch.sum(torch.abs(y1 - y2)).item() < 0.001

    def test_vectorization2(self):
        nn = NeuralNetworkVectorized(input_dim=1, output_dim=1, layer_sizes=(3, 6))

        x = torch.normal(mean=torch.zeros(22, 1))
        y1 = nn(x)

        params = 2 * nn.parameters_as_vector()

        nn.set_parameters_as_vector(params)
        y2 = nn(x)

        assert torch.sum(torch.abs(y1 - y2)).item() > 0.001

class TestCatDist(unittest.TestCase):

    def test_sampling1(self):
        torch.manual_seed(22)
        dist1 = pyro.distributions.Normal(torch.ones(7), 0.01 * torch.ones(7,)).to_event(1)
        dist2 = pyro.distributions.Normal(-1 * torch.ones(3), 0.01 * torch.ones(3, )).to_event(1)

        catdist = CatDist([dist1, dist2])
        sample = catdist.sample((100,))
        assert sample.shape == (100, 7+3)

        sample1_mean = sample[:, :7].mean().item()
        sample2_mean = sample[:, 7:].mean().item()
        assert np.abs(sample1_mean - 1) < 0.2
        assert np.abs(sample2_mean + 1) < 0.2

    def test_sampling2(self):
        torch.manual_seed(22)
        dist1 = pyro.distributions.Normal(torch.ones(5), 0.01 * torch.ones(5,)).to_event(1)
        dist2 = pyro.distributions.Normal(-1 * torch.ones(3), 0.01 * torch.ones(3, )).to_event(1)

        catdist = CatDist([dist1, dist2])
        sample = catdist.rsample((100,))
        assert sample.shape == (100, 5+3)

        sample1_mean = sample[:, :5].mean().item()
        sample2_mean = sample[:, 5:].mean().item()
        assert np.abs(sample1_mean - 1) < 0.2
        assert np.abs(sample2_mean + 1) < 0.2

    def test_pdf(self):
        torch.manual_seed(22)
        dist1 = pyro.distributions.Normal(torch.ones(7), 0.01 * torch.ones(7, )).to_event(1)
        dist2 = pyro.distributions.Normal(-1 * torch.ones(3), 0.01 * torch.ones(3, )).to_event(1)
        catdist = CatDist([dist1, dist2])

        x1 = torch.normal(1.0, 0.01, size=(150, 10))
        x2 = torch.normal(-1.0, 0.01, size=(150, 10))
        x3 = torch.cat([x1[:, :7], x2[:, 7:]], dim=-1)

        logp1 = catdist.log_prob(x1)
        logp3 = catdist.log_prob(x3)
        assert torch.mean(logp1).item() < -1000
        assert torch.mean(logp3).item() > 10

    def test_pdf2(self):
        torch.manual_seed(22)
        dist1 = pyro.distributions.Normal(torch.ones(7), 0.01 * torch.ones(7, )).to_event(1)
        dist2 = pyro.distributions.Normal(-1 * torch.ones(3), 0.01 * torch.ones(3, )).to_event(1)
        catdist1 = CatDist([dist1, dist2], reduce_event_dim=False)
        catdist2 = CatDist([dist1, dist2], reduce_event_dim=True)

        x1 = torch.normal(1.0, 0.01, size=(150, 10))
        x2 = torch.normal(-1.0, 0.01, size=(150, 10))
        x3 = torch.cat([x1[:, :7], x2[:, 7:]], dim=-1)

        logp1 = torch.sum(catdist1.log_prob(x3), dim=0).numpy()
        logp2 = catdist2.log_prob(x3).numpy()
        assert np.array_equal(logp1, logp2)

class TestEqualWeightedMixture(unittest.TestCase):

    def setUp(self):
        from pyro.distributions import Normal
        self.mean1 = torch.normal(1., 0.1, size=(8,))
        self.mean2 = torch.normal(-1., 0.1, size=(8,))

        self.scale1 = torch.ones((8,))
        self.scale2 = 2 * torch.ones((8,))

        self.dist1 = Normal(self.mean1, self.scale1).to_event(1)
        self.dist2 = Normal(self.mean2, self.scale2).to_event(1)

        self.dist3 = Normal(torch.stack([self.mean1, self.mean2], dim=0),
                            torch.stack([self.scale1, self.scale2], dim=0)).to_event(1)

        self.mean_mix = (self.mean1 + self.mean2) / 2.0

        var1 = ((self.mean1 - self.mean_mix)**2 + (self.mean2 - self.mean_mix)**2) / 2.0
        var2 = (self.scale1**2 + self.scale2**2) / 2.0
        self.var_mix = var1 + var2

    def test_mean_var(self):
        mixture = EqualWeightedMixtureDist([self.dist1, self.dist2])
        assert np.array_equal(mixture.mean, self.mean_mix)
        assert np.array_equal(mixture.variance, self.var_mix)

        mixture = EqualWeightedMixtureDist(self.dist3, batched=True)
        assert np.array_equal(mixture.mean, self.mean_mix)
        assert np.array_equal(mixture.variance, self.var_mix)

    def test_log_prob(self):
        value = torch.normal(0.0, 2.0, size=(8,))
        mixture1 = EqualWeightedMixtureDist([self.dist1, self.dist2])
        mixture2 = EqualWeightedMixtureDist(self.dist3, batched=True)
        p1 = mixture1.log_prob(value).item()
        p2 = mixture2.log_prob(value).item()
        assert np.array_equal(p1, p2)



