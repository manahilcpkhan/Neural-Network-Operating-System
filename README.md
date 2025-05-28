# Neural Network Operating System
 
# Neural Network Operating System Implementation 🧠⚙️

A multi-core operating system implementation that executes neural network architectures using separate processes and threads. This project demonstrates advanced OS concepts including inter-process communication (IPC), process synchronization, thread management, and memory management through a practical neural network training system.

## 🎯 Project Overview

This system implements a distributed neural network where each layer runs as a separate process and each neuron operates as an individual thread. The architecture leverages multi-core processing capabilities to parallelize neural network computations while using operating system primitives for coordination and communication.

## ✨ Key Features

### 🏗️ System Architecture
- **Process-Based Layers**: Each neural network layer runs as an independent process
- **Thread-Based Neurons**: Individual neurons implemented as separate threads within each layer process
- **Multi-Core Utilization**: Threads distributed across available CPU cores for parallel processing
- **Hierarchical Design**: Clear separation between layer processes and neuron threads

### 🔄 Inter-Process Communication
- **Named/Unnamed Pipes**: Data flow between layer processes using pipe mechanisms
- **Weight & Bias Exchange**: Parameters transmitted through IPC channels
- **Batch Processing**: Input data divided into batches and passed between layers
- **Forward Propagation**: Sequential data flow from input layer to output layer

### 🔐 Process Synchronization
- **Mutex Locks**: Thread-safe access to shared resources
- **Semaphores**: Process coordination and resource management
- **Race Condition Prevention**: Synchronized access to weights and biases
- **Critical Section Management**: Protected access to shared neural network parameters

### 🧮 Neural Network Implementation
- **Forward Propagation**: Input processing through network layers
- **Backpropagation**: Error signal propagation and weight updates
- **Batch Training**: Efficient processing of training data batches
- **Gradient Calculation**: Distributed computation of parameter gradients

## 🛠️ Technical Stack

### Core Technologies
- **C++** - Primary implementation language
- **Linux System Calls** - Direct OS interaction
- **POSIX Threads (pthreads)** - Thread management and synchronization
- **Unix Pipes** - Inter-process communication
- **Ubuntu/Linux** - Target operating system environment

### System Calls & Libraries Used
```cpp
#include <sys/types.h>     // Process types
#include <unistd.h>        // fork(), pipe(), exec()
#include <sys/wait.h>      // wait(), waitpid()
#include <pthread.h>       // Thread creation and management
#include <semaphore.h>     // Semaphore operations
#include <sys/ipc.h>       // Inter-process communication
#include <sys/shm.h>       // Shared memory (for backpropagation)
#include <fcntl.h>         // File control for named pipes
```


## 🚀 Installation & Setup

### Prerequisites
```bash
# Ubuntu/Linux environment
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install g++ gcc
sudo apt-get install pthread-dev

# Verify multi-core processor
nproc --all  # Should show multiple cores
```

### Compilation
```bash
# Clone repository
git clone https://github.com/yourusername/neural-network-os.git
cd neural-network-os

# Compile with threading and IPC support
g++ -o neural_network src/*.cpp -lpthread -std=c++17 -O2

# Or use the provided script
chmod +x scripts/compile.sh
./scripts/compile.sh
```

### Running the System
```bash
# Basic execution
./neural_network

# With custom configuration
./neural_network --config data/network_config.txt --training data/training_data.txt

# Monitor system processes during execution
./neural_network &
ps aux | grep neural_network  # View all related processes
top -H -p $(pgrep neural_network)  # Monitor threads
```

## ⚙️ System Implementation Details

### Forward Propagation Process
1. **Input Layer Process**: Receives training batch via pipe
2. **Neuron Threads**: Each thread processes portion of input in parallel
3. **Layer Output**: Aggregated results sent to next layer via pipe
4. **Multi-Core Utilization**: Threads scheduled across available CPU cores
5. **Synchronization**: Mutex locks ensure thread-safe weight access

### Backpropagation Implementation
1. **Error Calculation**: Output layer computes prediction errors
2. **Shared Memory**: Error gradients stored in shared memory segments
3. **Backward Flow**: Each layer process reads gradients and updates weights
4. **Thread Coordination**: Semaphores coordinate weight update phases
5. **Memory Management**: Dynamic allocation/deallocation of gradient storage

### Inter-Process Communication
```cpp
// Example pipe communication between layers
int pipe_fd[2];
pipe(pipe_fd);  // Create pipe

if (fork() == 0) {
    // Child process (next layer)
    close(pipe_fd[1]);  // Close write end
    read(pipe_fd[0], layer_input, sizeof(layer_input));
    // Process layer computation
} else {
    // Parent process (current layer)
    close(pipe_fd[0]);  // Close read end
    write(pipe_fd[1], layer_output, sizeof(layer_output));
    wait(NULL);  // Wait for child completion
}
```

### Thread Synchronization
```cpp
// Example mutex-protected weight update
pthread_mutex_t weight_mutex = PTHREAD_MUTEX_INITIALIZER;

void update_weights(double* weights, double* gradients, int size) {
    pthread_mutex_lock(&weight_mutex);
    for (int i = 0; i < size; i++) {
        weights[i] -= learning_rate * gradients[i];
    }
    pthread_mutex_unlock(&weight_mutex);
}
```

## 🧠 Neural Network Configuration

### Network Architecture
- **Input Layer**: Configurable number of input neurons
- **Hidden Layers**: Multiple hidden layers with customizable sizes
- **Output Layer**: Classification or regression output neurons
- **Activation Functions**: ReLU, Sigmoid, Tanh implementations
- **Learning Rate**: Adaptive learning rate with momentum

### Training Parameters
```cpp
struct NetworkConfig {
    int input_size = 784;        // Input features (e.g., 28x28 images)
    int hidden_layers[] = {128, 64, 32};  // Hidden layer sizes
    int output_size = 10;        // Output classes
    double learning_rate = 0.001;
    int batch_size = 32;
    int epochs = 100;
    int num_processes = 4;       // Number of layer processes
    int threads_per_layer = 8;   // Threads per layer process
};
```


## 🎓 Educational Value

This project demonstrates practical application of:
- **Operating System Design**: Real-world OS concept implementation
- **Parallel Computing**: Multi-core programming techniques
- **System Programming**: Low-level C++ and system call usage
- **Neural Networks**: Machine learning algorithm implementation
- **Performance Optimization**: System-level performance tuning

---

*This project showcases the intersection of operating systems and machine learning, demonstrating how OS concepts can be leveraged to build efficient, scalable neural network training systems on multi-core processors.*