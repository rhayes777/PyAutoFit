from autofit.messages import UniformNormalMessage


def test_logpdf_gradient():
    message = UniformNormalMessage(1.0, 0.5)
    x = message.sample()

    res = message.logpdf_gradient(x)
    print(res)
