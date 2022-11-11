from autofit.messages import UniformNormalMessage, NormalMessage
from autofit.messages.transform import phi_transform

OldUniformNormalMessage = NormalMessage.transformed(
    phi_transform, "UniformNormalMessage"
)


def test_logpdf_gradient():
    message = UniformNormalMessage(1.0, 0.5)
    x = message.sample()

    old_message = OldUniformNormalMessage(1.0, 0.5)

    assert message.logpdf_gradient(x) == old_message.logpdf_gradient(x)
