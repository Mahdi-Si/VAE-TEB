import psutil
import GPUtil
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import sys


def log_resource_usage():
    # CPU and RAM usage
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    ram_percent = memory.percent

    # GPU usage - assuming you have NVIDIA GPUs
    gpu_logs = []
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        gpu_name = gpu.name
        gpu_load = f"{gpu.load*100}%"
        gpu_memory_util = f"{gpu.memoryUtil*100}%"
        gpu_logs.append((gpu_name, gpu_load, gpu_memory_util))

    # Construct a log message
    log_message = f"CPU Usage: {cpu_percent}%, RAM Usage: {ram_percent}%, " + ", ".join([f"{name} Load: {load}, Memory Utilization: {memory_util}" for name, load, memory_util in gpu_logs])

    # Print or save log message
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {log_message}\n")


def setup_logging_lightning(log_file_path):
    logger = logging.getLogger('pytorch_lightning')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('- %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

class StreamToLoggerLightning:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


class StreamToLogger:
    """
    Stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger_, log_level=logging.INFO):
        self.logger = logger_
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

    def isatty(self):
        return False


# Setup your logging
def setup_logging(log_file_setup=None):
    # log_file = os.path.join(log_dir, 'log.txt')
    logger_s = logging.getLogger('my_app')
    logger_s.setLevel(logging.INFO)

    # Create handlers for both file and console
    file_handler = logging.FileHandler(log_file_setup, mode='w')
    console_handler = logging.StreamHandler()

    # Optional: add a formatter to include more details
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('- %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add both handlers to the logger
    logger_s.addHandler(file_handler)
    logger_s.addHandler(console_handler)

    return logger_s


# Function to update the learning rate of the optimizer
def update_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def visualize_layer_parameters(model, layer_name, param_type='weight', cmap='viridis'):
    """
    Visualize the weights or biases of a specific layer in a PyTorch model.

    Parameters:
    - model: nn.Module, the model containing the layer.
    - layer_name: str, the name of the layer whose parameters are to be visualized.
    - param_type: str, either 'weight' or 'bias' to specify the type of parameter to visualize.
    - cmap: str, the colormap to be used for heatmap visualization.
    """
    # Ensure the parameter type is either 'weight' or 'bias'
    assert param_type in ['weight', 'bias'], "param_type must be either 'weight' or 'bias'"

    # Find the layer by name and extract the specified parameter
    for name, module in model.named_modules():
        if name == layer_name:
            param = getattr(module, param_type).data.cpu().numpy()
            break
    else:
        raise ValueError(f"Layer named '{layer_name}' not found in the model.")

    # Visualization
    if param.ndim == 2:  # A 2D parameter (e.g., weights of a linear layer)
        # Use a heatmap for 2D data
        plt.figure(figsize=(10, 5))
        plt.imshow(param, cmap=cmap, aspect='auto')
        plt.colorbar()
        plt.title(f'{param_type.capitalize()} of layer {layer_name}')
        plt.xlabel('Features')
        plt.ylabel('Units')
    else:  # For biases or other 1D parameters
        # Use a histogram for 1D data
        plt.figure(figsize=(10, 5))
        plt.hist(param, bins=50)
        plt.title(f'{param_type.capitalize()} distribution of layer {layer_name}')
    plt.show()


def visualize_layer(layer, param_type='weight', cmap='viridis'):
    """
    Visualize the weights or biases of a specific layer.

    Parameters:
    - layer: The PyTorch layer to visualize.
    - param_type: str, either 'weight' or 'bias' to specify the type of parameter to visualize.
    - cmap: str, the colormap to be used for heatmap visualization.
    """
    assert param_type in ['weight', 'bias'], "param_type must be either 'weight' or 'bias'"

    param = getattr(layer, param_type).data.cpu().numpy()

    if param.ndim == 2:  # A 2D parameter (e.g., weights of a linear layer)
        plt.figure(figsize=(10, 5))
        plt.imshow(param, cmap=cmap, aspect='auto')
        plt.colorbar()
        plt.title(f'{param_type.capitalize()} visualization')
        plt.xlabel('Features')
        plt.ylabel('Units')
    else:  # For biases or other 1D parameters
        plt.figure(figsize=(10, 5))
        plt.hist(param, bins=50)
        plt.title(f'{param_type.capitalize()} distribution')
    plt.show()


def visualize_signal(signal, title='Signal Visualization', xlabel='Dimension', ylabel='Value'):
    """
    Visualize a signal (e.g., enc_t values) as a line plot.

    Parameters:
    - signal: torch.Tensor, the signal to visualize. Should be a 1D or 2D tensor.
    - title: str, title of the plot.
    - xlabel: str, label for the x-axis.
    - ylabel: str, label for the y-axis.
    """
    signal = signal.detach().cpu().numpy()  # Ensure it's a CPU NumPy array and no grad

    plt.figure(figsize=(10, 5))
    if signal.ndim == 1:
        plt.plot(signal)
    elif signal.ndim == 2:  # For 2D signals, plot each row as a separate line
        for row in signal:
            plt.plot(row)
    else:
        raise ValueError("Signal dimensionality not supported for visualization.")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.grid(True)
    plt.show()
