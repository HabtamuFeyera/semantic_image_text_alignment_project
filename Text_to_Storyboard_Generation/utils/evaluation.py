import torch

def calculate_accuracy(outputs, labels):
    """
    Calculates the accuracy of model predictions.

    Args:
    - outputs (torch.Tensor): Model predictions.
    - labels (torch.Tensor): Ground truth labels.

    Returns:
    - float: Accuracy score.
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

def evaluate_model(model, dataloader, criterion):
    """
    Evaluates a model on a given dataset.

    Args:
    - model (torch.nn.Module): The model to evaluate.
    - dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
    - criterion (torch.nn.Module): Loss function.

    Returns:
    - float: Average loss.
    - float: Accuracy.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, labels)

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)

    return avg_loss, avg_accuracy
