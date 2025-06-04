# Simple 2-Node Distributed Training Example

This example demonstrates setting up distributed training with just 2 nodes - perfect for getting started or testing your setup.

## Prerequisites

- 2 machines with Docker installed
- Network connectivity between machines
- Both machines should be able to reach each other

## Step 1: Prepare Both Machines

On both machines:

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (optional, requires logout/login)
sudo usermod -aG docker $USER
```

## Step 2: Set Up Docker Swarm

On **Machine 1** (Manager):
```bash
# Get the IP address of this machine
ip addr show

# Initialize swarm (replace with your actual IP)
docker swarm init --advertise-addr 192.168.1.10

# Note the join token from the output
```

On **Machine 2** (Worker):
```bash
# Join the swarm (use the token from Machine 1)
docker swarm join --token SWMTKN-1-xxxxx 192.168.1.10:2377
```

Verify on **Machine 1**:
```bash
docker node ls
```

## Step 3: Create Network and Build Image

On **Machine 1**:
```bash
# Create overlay network
docker network create --driver overlay --attachable ddp-net

# Build the training image
cd /path/to/distributed_training
docker build -t my-ddp-trainer:latest .
```

On **Machine 2**:
```bash
# Since we're not using a registry, build the same image
cd /path/to/distributed_training
docker build -t my-ddp-trainer:latest .
```

## Step 4: Create Simple Docker Compose

Create `docker-compose-2node.yml`:

```yaml
version: "3.8"

networks:
  ddp-net:
    driver: overlay
    attachable: true

services:
  node0:
    image: my-ddp-trainer:latest
    command: >
      torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0
      --master_addr=node0 --master_port=12355
      training_code.py 10 2 --batch_size 32 --snapshot_path snapshot.pt
    environment:
      - NCCL_SOCKET_IFNAME=eth0
      - NCCL_IB_DISABLE=1
      - NCCL_DEBUG=INFO
    networks:
      ddp-net:
        aliases:
          - node0
    deploy:
      placement:
        constraints:
          - node.role == manager
      restart_policy:
        condition: none

  node1:
    image: my-ddp-trainer:latest
    command: >
      torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1
      --master_addr=node0 --master_port=12355
      training_code.py 10 2 --batch_size 32 --snapshot_path snapshot.pt
    environment:
      - NCCL_SOCKET_IFNAME=eth0
      - NCCL_IB_DISABLE=1
      - NCCL_DEBUG=INFO
    networks:
      - ddp-net
    deploy:
      placement:
        constraints:
          - node.role == worker
      restart_policy:
        condition: none
    depends_on:
      - node0
```

## Step 5: Deploy and Monitor

On **Machine 1**:
```bash
# Deploy the stack
docker stack deploy -c docker-compose-2node.yml simple-training

# Check services
docker service ls

# Monitor logs
docker service logs -f simple-training_node0
docker service logs -f simple-training_node1
```

## Step 6: Verify Training

You should see logs showing:
- Both nodes initializing
- Process group setup
- Training progress on both nodes
- Gradient synchronization

Example output:
```
[GPU0] epoch 0 | batch_size: 32 | steps: 64
[GPU1] epoch 0 | batch_size: 32 | steps: 64
Epoch 0 | train snapshot saved at snapshot.pt
```

## Cleanup

```bash
# Remove the stack
docker stack rm simple-training

# Leave swarm (on worker node)
docker swarm leave

# Remove swarm (on manager node)
docker swarm leave --force
```

## Troubleshooting

### Issue: Containers can't communicate
```bash
# Check network
docker network ls
docker network inspect ddp-net

# Verify connectivity
docker run --rm --network ddp-net alpine ping node0
```

### Issue: Image not found
```bash
# Build on both machines or use a registry
docker build -t my-ddp-trainer:latest .
```

### Issue: Permission denied
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Logout and login again
```

## Next Steps

Once this works:
1. Try with GPU support by adding device mappings
2. Scale to 3+ nodes
3. Experiment with different batch sizes and configurations
4. Add persistent volume mounts for data 