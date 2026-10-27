from torch import nn, Tensor
class CrossEntropyLoss:
    """
    Custom CrossEntropyLoss that can be used in the model's loss metrics.
    """
    def __init__(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(self, inputs: list, targets: list) -> Tensor:
        """
        Args:
            inputs (list): A list containing the logits.
            targets (list): A list containing the target labels.
            # By convention, inputs is a list with the first element being the logits
        Returns:
            Tensor: The computed cross-entropy loss.
        """
        logits = inputs[0]
        target = targets[0]
        return self.loss_fn(logits, target)

class Accuracy:
    """
    Custom Accuracy metric that can be used in the model's report metrics.
    """
    def __call__(self, inputs: list, targets: list) -> Tensor:
        """
        Args:
            inputs (list): A list containing the logits.
            targets (list): A list containing the target labels.
        Returns:
            Tensor: The computed accuracy.
        """
        logits = inputs[0]
        target = targets[0]
        return (logits.argmax(dim=1) == target).float().mean()
