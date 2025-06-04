# Quick Start Guide: Distributed PyTorch Training

This is a condensed version of the complete setup guide. Follow these steps to quickly set up distributed training with PyTorch using Docker Swarm.

## Prerequisites

- 2+ machines with Docker installed
- Network connectivity between machines
- NVIDIA Docker (for GPU support)

## Step 1: Install Dependencies

```bash
# Install Docker (on all nodes)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Docker (for GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Step 2: Set Up Docker Swarm

```bash
# On manager node (replace with your manager IP)
docker swarm init --advertise-addr <MANAGER_IP>

# On worker nodes (use token from previous command)
docker swarm join --token <WORKER_TOKEN> <MANAGER_IP>:2377

# Verify swarm setup
docker node ls
```

## Step 3: Create Overlay Network

```bash
# Run on manager node
docker network create --driver overlay --attachable ddp-net
```

## Step 4: Build Docker Image

```bash
# Clone this repository and build the image
git clone <your-repo>
cd distributed_training
docker build -t my-ddp-trainer:latest .

# Optional: Push to registry for multi-node access
docker tag my-ddp-trainer:latest your-registry/my-ddp-trainer:latest
docker push your-registry/my-ddp-trainer:latest
```

## Step 5: Update Docker Compose

Edit `docker-compose.yml` to match your node hostnames:

```yaml
# Update these constraints to match your actual node hostnames
deploy:
  placement:
    constraints:
      - node.hostname == your-node-hostname
```

## Step 6: Deploy Training

```bash
# Deploy the stack
docker stack deploy -c docker-compose.yml distributed-training

# Monitor progress
docker service ls
docker service logs -f distributed-training_node0
```

## Step 7: Check Results

```bash
# View logs from all nodes
docker service logs distributed-training_node0
docker service logs distributed-training_node1
docker service logs distributed-training_node2

# Check service status
docker service ps distributed-training_node0
```

## Quick Testing (2-Node Setup)

For a simple 2-node test without Docker Compose:

```bash
# On manager node - run master container
docker run --rm --network ddp-net --name node0 \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_DISABLE=1 \
  my-ddp-trainer:latest \
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
  --master_addr=node0 --master_port=12355 \
  training_code.py 10 2 --batch_size 32

# On worker node - run worker container
docker run --rm --network ddp-net --name node1 \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_DISABLE=1 \
  my-ddp-trainer:latest \
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
  --master_addr=node0 --master_port=12355 \
  training_code.py 10 2 --batch_size 32
```

## Troubleshooting Quick Fixes

### Common Issues:

1. **Connection timeouts**: Check firewall and network connectivity
2. **Image not found**: Ensure image is available on all nodes or use registry
3. **NCCL errors**: Add `NCCL_IB_DISABLE=1` environment variable
4. **Container placement**: Verify node hostnames in constraints

### Debug Commands:

```bash
# Check node labels
docker node inspect <node-id>

# View service details
docker service inspect distributed-training_node0

# Access container shell
docker exec -it $(docker ps -q -f name=distributed-training_node0) bash
```

## Scaling Up

To add more nodes:

1. Join additional nodes to the swarm
2. Add new services to `docker-compose.yml`
3. Update `--nnodes` parameter
4. Redeploy with `docker stack deploy`

## Key Parameters to Adjust

- `--nnodes`: Total number of nodes
- `--nproc_per_node`: Processes per node (usually 1 per GPU)
- `--batch_size`: Batch size per process
- Node hostnames in placement constraints

For detailed explanations and advanced configurations, see the complete documentation in `DISTRIBUTED_TRAINING_SETUP.md`. 