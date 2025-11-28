import torch
import torch.optim as optim

__all__ = ['CustomMultiGammaLR']


class CustomMultiGammaLR:
    def __init__(self, optimizer, milestones, gammas):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gammas = gammas
        self.last_epoch = 0

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        for milestone, gamma in zip(self.milestones, self.gammas):
            if epoch == milestone:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= gamma
                print(f"Epoch {epoch}: Learning rate updated with gamma = {gamma}")

    def state_dict(self):
        """Return the state of the scheduler as a `dict`."""
        return {
            'last_epoch': self.last_epoch,
            'milestones': self.milestones,
            'gammas': self.gammas
        }

    def load_state_dict(self, state_dict):
        """Load the scheduler's state."""
        self.last_epoch = state_dict['last_epoch']
        self.milestones = state_dict['milestones']
        self.gammas = state_dict['gammas']
