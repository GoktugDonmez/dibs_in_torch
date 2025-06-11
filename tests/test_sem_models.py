import torch
from models.linear_sem import linear_predict
from models.nonlinear_sem import NonlinearSEM


def test_linear_predict():
    x = torch.randn(5, 3)
    theta = torch.eye(3)
    g = torch.eye(3)
    out = linear_predict(x, theta, g)
    assert out.shape == x.shape


def test_nonlinear_sem_forward():
    x = torch.randn(4, 2)
    g = torch.ones(2, 2)
    model = NonlinearSEM(d=2, hidden_dim=4)
    out = model(x, g)
    assert out.shape == x.shape
