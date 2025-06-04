# Comprehensive Guide: Distributed PyTorch Training with Docker Swarm

This guide provides step-by-step instructions for setting up distributed training with PyTorch using Docker Swarm and Docker Compose. We'll cover everything from the basic PyTorch distributed training code to orchestrating multi-node training across a Docker Swarm cluster.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Understanding Distributed Training Components](#understanding-distributed-training-components)
3. [PyTorch Distributed Training Code](#pytorch-distributed-training-code)
4. [Docker Setup](#docker-setup)
5. [Docker Swarm Configuration](#docker-swarm-configuration)
6. [Docker Compose Configuration](#docker-compose-configuration)
7. [Running Distributed Training](#running-distributed-training)
8. [Monitoring and Debugging](#monitoring-and-debugging)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- Multiple machines with Docker installed
- NVIDIA Docker (for GPU support)
- Network connectivity between all nodes
- CUDA-compatible GPUs (optional but recommended)

### Software Requirements
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Docker (for GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Understanding Distributed Training Components

### Key Concepts

1. **Process Group**: A collection of processes that can communicate with each other
2. **Backend**: Communication backend (NCCL for GPU, Gloo for CPU)
3. **World Size**: Total number of processes across all nodes
4. **Rank**: Unique identifier for each process (0 to world_size-1)
5. **Local Rank**: Process rank within a single node
6. **Master Node**: Coordinates the distributed training (usually rank 0)

### Communication Patterns
- **All-Reduce**: Combines gradients from all processes
- **Broadcast**: Sends data from one process to all others
- **Gather/Scatter**: Collects/distributes data across processes

## PyTorch Distributed Training Code

### Core Components

#### 1. Dataset Implementation (`datautils.py`)
```python
import torch
from torch.utils.data import Dataset

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        # Generate synthetic data for demonstration
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]
```

#### 2. Main Training Script (`training_code.py`)

The training script includes several critical components:

**Distributed Setup Function:**
```python
def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")
```

**Data Loading with Distribution:**
```python
def prepare_dataloader(dataset, batch_size):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset),  # Ensures data is split across processes
        pin_memory=True,
        shuffle=False
    )
```

**Trainer Class with DDP Wrapper:**
```python
class Trainer:
    def __init__(self, model, train_data, optimizer, save_every, snapshot_path):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        
        # Wrap model with DistributedDataParallel
        self.model = DDP(self.model, device_ids=[self.local_rank])
```

#### 3. Environment Variables Required

- `LOCAL_RANK`: Process rank within the current node
- `RANK`: Global process rank across all nodes
- `WORLD_SIZE`: Total number of processes
- `MASTER_ADDR`: IP address of the master node
- `MASTER_PORT`: Port for communication

## Docker Setup

### Dockerfile Configuration

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Copy your code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Entry is handled via torchrun command
CMD ["bash"]
```

### Dependencies (`requirements.txt`)

The requirements include PyTorch and CUDA-related packages:
```
torch==2.7.0
nvidia-nccl-cu12==2.26.2
# ... other CUDA packages
```

### Building the Docker Image

```bash
# Build the image
docker build -t my-ddp-trainer:latest .

# Push to registry (optional, for multi-node setup)
docker tag my-ddp-trainer:latest your-registry/my-ddp-trainer:latest
docker push your-registry/my-ddp-trainer:latest
```

## Docker Swarm Configuration

### Setting Up Docker Swarm

#### 1. Initialize Swarm on Manager Node
```bash
# On the master node
docker swarm init --advertise-addr <MANAGER_IP>
```

#### 2. Join Worker Nodes
```bash
# On each worker node (use token from swarm init output)
docker swarm join --token <WORKER_TOKEN> <MANAGER_IP>:2377
```

#### 3. Verify Swarm Status
```bash
# Check nodes in the swarm
docker node ls

# Label nodes for placement constraints
docker node update --label-add hostname=node1 <NODE_ID>
docker node update --label-add hostname=node2 <NODE_ID>
docker node update --label-add hostname=node3 <NODE_ID>
```

### Network Configuration

Create an overlay network for container communication:
```bash
docker network create --driver overlay --attachable ddp-net
```

## Docker Compose Configuration

### Complete Docker Compose Setup (`docker-compose.yml`)

```yaml
version: "3.8"

networks:
  ddp-net:
    driver: overlay
    attachable: true

services:
  node0:
    image: tayyib1094/my-ddp-trainer:latest
    command: >
      torchrun --nproc_per_node=1 --nnodes=3 --node_rank=0
      --master_addr=node0 --master_port=12355
      training_code.py 20 5 --batch_size 64 --snapshot_path snapshot.pt
    environment:
      # NCCL Configuration for optimal performance
      - NCCL_SOCKET_IFNAME=eth0
      - NCCL_IB_DISABLE=1
      - NCCL_DEBUG=INFO
      - TORCH_DISTRIBUTED_DEBUG=DETAIL
      - NCCL_BUFFSIZE=8388608        # 8MB buffer
      - NCCL_NTHREADS=64             # Number of threads
      - NCCL_NET_GDR_LEVEL=0         # Disable GPU Direct
      - NCCL_SOCKET_NTHREADS=8       # Socket threads
      - NCCL_NSOCKS_PERTHREAD=2      # Sockets per thread
    networks:
      ddp-net:
        aliases:
          - node0  # Important: provides consistent hostname
    deploy:
      placement:
        constraints:
          - node.hostname == ahsan-pc
      restart_policy:
        condition: none

  node1:
    image: tayyib1094/my-ddp-trainer:latest
    command: >
      torchrun --nproc_per_node=1 --nnodes=3 --node_rank=1
      --master_addr=node0 --master_port=12355
      training_code.py 20 5 --batch_size 64 --snapshot_path snapshot.pt
    environment:
      - NCCL_SOCKET_IFNAME=eth0
      - NCCL_IB_DISABLE=1
      - NCCL_DEBUG=INFO
      - TORCH_DISTRIBUTED_DEBUG=DETAIL
      - NCCL_BUFFSIZE=8388608
      - NCCL_NTHREADS=64
      - NCCL_NET_GDR_LEVEL=0
      - NCCL_SOCKET_NTHREADS=8
      - NCCL_NSOCKS_PERTHREAD=2
    networks:
      - ddp-net
    deploy:
      placement:
        constraints:
          - node.hostname == stallion-01
      restart_policy:
        condition: none
    depends_on:
      - node0

  node2:
    image: tayyib1094/my-ddp-trainer:latest
    command: >
      torchrun --nproc_per_node=1 --nnodes=3 --node_rank=2
      --master_addr=node0 --master_port=12355
      training_code.py 20 5 --batch_size 64 --snapshot_path snapshot.pt
    environment:
      - NCCL_SOCKET_IFNAME=eth0
      - NCCL_IB_DISABLE=1
      - NCCL_DEBUG=INFO
      - TORCH_DISTRIBUTED_DEBUG=DETAIL
      - NCCL_BUFFSIZE=8388608
      - NCCL_NTHREADS=64
      - NCCL_NET_GDR_LEVEL=0
      - NCCL_SOCKET_NTHREADS=8
      - NCCL_NSOCKS_PERTHREAD=2
    networks:
      - ddp-net
    deploy:
      placement:
        constraints:
          - node.hostname == sama-stallion-02
      restart_policy:
        condition: none
    depends_on:
      - node0
```

### Key Configuration Elements

#### TorchRun Parameters
- `--nproc_per_node=1`: Number of processes per node (usually 1 per GPU)
- `--nnodes=3`: Total number of nodes
- `--node_rank=X`: Rank of this node (0, 1, 2, ...)
- `--master_addr=node0`: Hostname of master node
- `--master_port=12355`: Communication port

#### NCCL Environment Variables
- `NCCL_SOCKET_IFNAME=eth0`: Network interface for communication
- `NCCL_IB_DISABLE=1`: Disable InfiniBand if not available
- `NCCL_DEBUG=INFO`: Enable debugging output
- `NCCL_BUFFSIZE`: Buffer size for communication
- `NCCL_NTHREADS`: Number of NCCL threads

#### Deployment Constraints
- `node.hostname == <hostname>`: Ensures each service runs on specific node
- `restart_policy: condition: none`: Prevents automatic restarts

## Running Distributed Training

### Step-by-Step Execution

#### 1. Prepare the Environment
```bash
# Ensure all nodes are part of the swarm
docker node ls

# Create overlay network (if not exists)
docker network create --driver overlay --attachable ddp-net
```

#### 2. Deploy the Stack
```bash
# Deploy the distributed training stack
docker stack deploy -c docker-compose.yml distributed-training

# Check service status
docker service ls
```

#### 3. Monitor Progress
```bash
# Check logs from all services
docker service logs distributed-training_node0
docker service logs distributed-training_node1
docker service logs distributed-training_node2

# Follow logs in real-time
docker service logs -f distributed-training_node0
```

#### 4. Scale Services (if needed)
```bash
# Scale a specific service
docker service scale distributed-training_node1=2

# Update service configuration
docker service update --env-add NEW_VAR=value distributed-training_node0
```

### Alternative: Manual Container Deployment

For testing or development, you can run containers manually:

```bash
# Run master node
docker run --rm --network ddp-net --name node0 \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_DISABLE=1 \
  my-ddp-trainer:latest \
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
  --master_addr=node0 --master_port=12355 \
  training_code.py 10 2 --batch_size 32

# Run worker node
docker run --rm --network ddp-net --name node1 \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_DISABLE=1 \
  my-ddp-trainer:latest \
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
  --master_addr=node0 --master_port=12355 \
  training_code.py 10 2 --batch_size 32
```

## Monitoring and Debugging

### Log Analysis

#### Key Log Indicators
1. **Successful Initialization**: Look for "Process group initialized"
2. **NCCL Communication**: Check for NCCL backend setup messages
3. **Training Progress**: Monitor epoch and batch processing
4. **Error Messages**: Watch for timeout or communication errors

#### Debugging Commands
```bash
# Get detailed service information
docker service inspect distributed-training_node0

# Check node placement
docker service ps distributed-training_node0

# Access container for debugging
docker exec -it $(docker ps -q -f name=distributed-training_node0) bash
```

### Performance Monitoring

#### System Resource Usage
```bash
# Monitor GPU usage
nvidia-smi

# Monitor CPU and memory
htop

# Monitor network traffic
iftop -i eth0
```

#### Training Metrics
- Monitor loss convergence across nodes
- Check batch processing times
- Verify gradient synchronization

## Best Practices

### 1. Network Configuration
- Use dedicated network interfaces for training communication
- Configure appropriate MTU sizes for your network
- Use overlay networks for Docker Swarm

### 2. Resource Management
- Allocate sufficient memory and GPU resources
- Use CPU affinity for optimal performance
- Configure swap settings appropriately

### 3. Error Handling
- Implement checkpointing for fault tolerance
- Use proper exception handling in training code
- Set up automatic restart policies carefully

### 4. Security
- Use secure Docker registries
- Implement proper network segmentation
- Use secrets management for sensitive data

### 5. Scalability
- Design for horizontal scaling
- Use load balancing where appropriate
- Implement efficient data loading strategies

## Troubleshooting

### Common Issues and Solutions

#### 1. NCCL Timeout Errors
```bash
# Symptoms: "NCCL timeout" or "Process group initialization failed"
# Solutions:
- Increase timeout values
- Check network connectivity between nodes
- Verify firewall settings
- Ensure consistent NCCL configuration
```

#### 2. Container Communication Issues
```bash
# Symptoms: Cannot resolve hostnames or connect to master
# Solutions:
- Verify overlay network configuration
- Check DNS resolution within containers
- Ensure proper service dependencies
- Verify port availability
```

#### 3. GPU Access Problems
```bash
# Symptoms: CUDA device not available
# Solutions:
- Install NVIDIA Docker runtime
- Add device mappings to compose file
- Check GPU driver compatibility
- Verify CUDA version compatibility
```

#### 4. Performance Issues
```bash
# Symptoms: Slow training or poor scaling
# Solutions:
- Tune NCCL parameters
- Optimize batch sizes
- Check network bandwidth
- Profile GPU utilization
```

### Debug Configuration

For intensive debugging, add these environment variables:

```yaml
environment:
  - NCCL_DEBUG=TRACE
  - TORCH_DISTRIBUTED_DEBUG=DETAIL
  - TORCH_SHOW_CPP_STACKTRACES=1
  - CUDA_LAUNCH_BLOCKING=1
```

### Health Checks

Add health checks to your services:

```yaml
services:
  node0:
    # ... other configuration
    healthcheck:
      test: ["CMD", "python", "-c", "import torch; print(torch.cuda.is_available())"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Advanced Configuration

### GPU-Specific Setup

For multi-GPU setups:

```yaml
services:
  node0:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Data Persistence

For persistent data storage:

```yaml
services:
  node0:
    volumes:
      - data-volume:/data
      - ./checkpoints:/workspace/checkpoints

volumes:
  data-volume:
    driver: local
```

### Service Discovery

For dynamic service discovery:

```yaml
services:
  node0:
    deploy:
      labels:
        - "traefik.enable=true"
        - "traefik.http.routers.training.rule=Host(`training.local`)"
```

## Conclusion

This comprehensive guide covers the complete setup of distributed PyTorch training using Docker Swarm and Docker Compose. The key success factors are:

1. Proper network configuration for inter-node communication
2. Correct NCCL settings for optimal performance
3. Appropriate service placement and scaling strategies
4. Robust error handling and monitoring

Start with a simple 2-node setup to validate your configuration, then scale to larger clusters as needed. Always test thoroughly in a development environment before deploying to production.

For production deployments, consider additional factors such as:
- Data backup and recovery strategies
- Monitoring and alerting systems
- Automated deployment pipelines
- Performance optimization and tuning 