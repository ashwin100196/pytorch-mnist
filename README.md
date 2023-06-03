# PyTorch MNIST Classification

This repository contains a simple PyTorch implementation of a convolutional neural network (CNN) for classifying the MNIST handwritten digits dataset. The model is trained using the MNIST dataset, and helper functions for data loading, training, and testing are provided.

## Details on folder structure

The project is split into:
- `model.py`: contains the CNN model
- `utils.py`: contains helper functions for data loading, training, and testing
- `S5.ipynb`: notebook containing the training and testing code

## Model Architecture

The model architecture consists of a convolutional neural network with three convolutional layers followed by two fully connected layers. The details of the architecture are as follows:

1. Convolutional Layer 1:
   - Input: 1 channel (grayscale image)
   - Number of filters: 32
   - Kernel size: 3x3
   - Activation: ReLU

2. Convolutional Layer 2:
   - Number of filters: 64
   - Kernel size: 3x3
   - Activation: ReLU

3. Convolutional Layer 3:
   - Number of filters: 128
   - Kernel size: 3x3
   - Activation: ReLU

4. Convolutional Layer 4:
   - Number of filters: 256
   - Kernel size: 3x3
   - Activation: ReLU

4. Fully Connected Layer 1:
   - Input size: 4096
   - Number of outputs: 50
   - Activation: ReLU

5. Fully Connected Layer 2 (Output layer):
   - Input size: 50
   - Number of neurons: 10 (corresponding to 10 digit classes)
   - Activation: Softmax

## Helper Functions

### Dataloader

The `utils.py` file contains a helper function `gen_data_loader` to load and preprocess the MNIST dataset. The function takes the following arguments:

```python
gen_data_loader(train=True, **kwargs)
```

- `train`: Set to True for train data loader and False for test data loader
**kwargs includes keyword arguments that are sent to torch.utils.data.DataLoader
The ones currently being used are listed below:
- `batch_size`: size of batch to be trained in
- `shuffle`: True shuffles data amongst classes when loaded
- `num_workers`: multiple threads to load data faster; parallelism

The function returns a PyTorch `DataLoader` object; train if the parameters is set to True, else test.

### Training

The `utils.py` file contains a helper function `train` for training the MNIST classification model. The function takes the following arguments:

```python
train(model, device, train_loader, optimizer, criterion, train_acc, train_losses)
```

- `model`: The PyTorch model to be trained.
- `device`: The device to be used for training (e.g. CPU or GPU).
- `train_loader`: The DataLoader object containing the training data.
- `optimizer`: The PyTorch optimizer to be used for training.
- `criterion`: The loss function to be used for training.
- `train_acc`: A list to store the training accuracy for each epoch.
- `train_losses`: A list to store the training loss for each epoch.

The function trains the model using the provided dataloader and prints the training loss and accuracy at each epoch.

### Testing

The `utils.py` file contains a helper function `test` for testing the trained MNIST classification model. The function takes the following arguments:

```python
test(model, device, test_loader, criterion, test_acc, test_losses)
```

- `model`: The trained PyTorch model.
- `device`: The device to be used for testing (e.g. CPU or GPU).
- `test_loader`: The DataLoader object containing the test data.
- `criterion`: The loss function to be used for testing.
- `test_acc`: A list to store the test accuracy for each epoch.
- `test_losses`: A list to store the test loss for each epoch.

The function evaluates the model on the testing data and prints the testing accuracy.

## Usage

To use this code, follow these steps:

1. Clone the repository:

```bash
git clone 
```

2. Install the required dependencies:

```bash
pip install torch torchvision matplotlib torchsummary
```

3. Import the required modules in your Python code:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from utils import gen_data_loader, train, test
from model import Net
```

4. Set the device to be used for training and testing:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

5. Create the model, optimizer, and loss function:

```python
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
```

6. Load the training and testing data:

```python
train_loader = gen_data_loader(train=True, batch_size=64, shuffle=True, num_workers=4)
test_loader = gen_data_loader(train=False, batch_size=64, shuffle=False, num_workers=4)
```

7. Train the model:

```python
train_acc = []
train_losses = []
for epoch in range(10):
    train(model, device, train_loader, optimizer, criterion, train_acc, train_losses)
```

8. Test the model:

```python
test_acc = []
test_losses = []
test(model, device, test_loader, criterion, test_acc, test_losses)
```

## Results

The train and test accuracy and loss plots can be created as shown below by using the utility function `viz_training_graphs` provided in `utils.py`:

```python
from utils import viz_training_graphs
viz_training_graphs(train_acc, train_losses, test_acc, test_losses)
```
