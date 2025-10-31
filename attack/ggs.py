import torch
import torch.nn as nn

from attack.utils import *

class GGS():
    """
    GGS Attack

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        num_neighbor (int): the number of neighbors.
        decay (float): the decay factor for momentum calculation.
        zeta (float): the noise scale.
        targeted (bool): targeted/untargeted attack.
        device (torch.device): the device for data. If it is None, the device would be same as model
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, num_neighbor=20, decay=1., 
                zeta = 2, targeted=False, device=None):
        """
        Initialize the hyperparameters

        Arguments:
            model_name (str): the name of surrogate model for attack.
            epsilon (float): the perturbation budget.
            targeted (bool): targeted/untargeted attack.
            device (torch.device): the device for data. If it is None, the device would be same as model
        """
        self.model = load_model(model_name)
        self.model_name = model_name
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.epsilon = epsilon
        self.targeted = targeted
        if isinstance(self.model, EnsembleModel):
            self.device = self.model.device
        else:
            self.device = next(self.model.parameters()).device if device is None else device
        self.loss = nn.CrossEntropyLoss()


        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.zeta = epsilon * zeta
        self.max_iter = num_neighbor

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure
        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        
        # Initialize adversarial perturbation
        delta = torch.zeros_like(data).requires_grad_(True).to(self.device)
        momentum = torch.zeros_like(delta)

        for _ in range(self.epoch): 
            av_grad = torch.zeros_like(delta) 
            
            for i in range(1, self.max_iter + 1): 
                noise = torch.zeros_like(delta).uniform_(-self.zeta, self.zeta).to(self.device) 

                if i != 1: 
                    noise = noise.abs() * grad.detach().sign()
                noise = noise.clamp(-self.zeta, self.zeta)

                x_near = data + delta + noise

                logits = self.model(x_near)
                loss = -self.loss(logits, label) if self.targeted else self.loss(logits, label)
                grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]

                av_grad += grad

            av_grad = av_grad / self.max_iter

            # Calculate the momentum
            momentum = momentum * self.decay + av_grad / (av_grad.abs().mean(dim=(1,2,3), keepdim=True))
            delta = self.update_delta(delta, data, momentum, self.alpha) 

        return delta.detach()

    def update_delta(self, delta, data, grad, alpha):
        delta = torch.clamp(delta + alpha * grad.sign(), -self.epsilon, self.epsilon)
        delta = clamp(delta, img_min-data, img_max-data)
        return delta

    def __call__(self, *input, **kwargs):
        self.model.eval()
        return self.forward(*input, **kwargs)
