from zennit.attribution import Gradient
from zennit.composites import EpsilonGammaBox, MixedComposite, NameMapComposite
from zennit.torchvision import ResNetCanonizer
from zennit.rules import Gamma, Norm, Flat, Epsilon
from zennit.core import Composite, Hook
from torchvision.transforms import Normalize
import torch

from src.lrp import module_map_resnet
from zennit.core import RemovableHandleList

class GradMock:
    def copy(self):
        return self

    def register(self, module):
        return RemovableHandleList([])

class GradTimesInput(Hook):
    '''Hook for layer-wise gradient times input.'''
    def forward(self, module, input, output):
        '''Remember the input for the backward pass.'''
        self.stored_tensors['input'] = input

    def backward(self, module, grad_input, grad_output):
        '''Modify the gradient by element-wise multiplying the input.'''
        return (self.stored_tensors['input'][0] * grad_input[0],)

canonizer = ResNetCanonizer()
# transform_norm = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
# low, high = transform_norm(torch.tensor([[[[[0.]]] * 3], [[[[1.]]] * 3]]))
# composite = MixedComposite([
#     # EpsilonGammaBox(low=low, high=high),
#     NameMapComposite([
#         ('clf_head', Gamma(0.0))
#     ]),
#     Composite(module_map_resnet, canonizers=[canonizer])
# ])

# transform_norm_1 = Normalize(mean=(0.5,), std=(0.5,))
# low_1, high_1 = transform_norm_1(torch.tensor([[[[[0.]]]], [[[[1.]]]]]))
low, high = -3., 3.

composite = MixedComposite([
    # EpsilonGammaBox(low=low, high=high, epsilon=0.0),
    # NameMapComposite([
    #     ([''], GradTimesInput())
    # ]),
    Composite(module_map_resnet)
], canonizers=[canonizer])

def attr_max(z):
    z[z < z.max()] = 0
    return z

def explain(data, model, target=None):
    with Gradient(model=model, composite=composite, attr_output=attr_max) as attributor:
        if target is None:
            output, attribution = attributor(data)
        else:
            output, attribution = attributor(data, attr_output=target)
    return output, attribution
