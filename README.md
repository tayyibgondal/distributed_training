# Distributed PyTorch Training with Docker Swarm

This repository provides a complete setup for distributed PyTorch training using Docker Swarm and Docker Compose. Train your models across multiple nodes with ease and scalability.

## 🚀 Quick Start

1. **Set up Docker Swarm** across your nodes
2. **Build the training image** with your code
3. **Deploy with Docker Compose** using our configuration
4. **Monitor and scale** your distributed training

See [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) for step-by-step instructions.

## 📁 Repository Structure

```
distributed_training/
├── README.md                           # This file
├── DISTRIBUTED_TRAINING_SETUP.md      # Complete documentation
├── QUICK_START_GUIDE.md               # Quick start instructions
├── docker-compose.yml                 # Multi-node training configuration
├── Dockerfile                         # Training container definition
├── requirements.txt                   # Python dependencies
├── training_code.py                   # Distributed training implementation
└── datautils.py                       # Dataset utilities
```

## 🎯 Features

- **Multi-node distributed training** with PyTorch DDP
- **Docker Swarm orchestration** for easy scaling
- **NCCL optimization** for high-performance GPU communication
- **Fault tolerance** with checkpointing and restart policies
- **Monitoring and debugging** tools and configurations
- **Production-ready** setup with best practices

## 📋 Prerequisites

- Multiple machines with Docker installed
- NVIDIA Docker runtime (for GPU support)
- Network connectivity between all nodes
- CUDA-compatible GPUs 

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Manager Node  │    │  Worker Node 1  │    │  Worker Node 2  │
│    (node0)      │    │    (node1)      │    │    (node2)      │
│                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │ Trainer 0 │  │    │  │ Trainer 1 │  │    │  │ Trainer 2 │  │
│  │  Rank 0   │  │    │  │  Rank 1   │  │    │  │  Rank 2   │  │
│  │ (Master)  │  │    │  │ (Worker)  │  │    │  │ (Worker)  │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                           NCCL Communication
                        (Gradient Synchronization)
```

## 🛠️ Setup Instructions

### Option 1: Quick Start (Recommended for beginners)
Follow the [Quick Start Guide](QUICK_START_GUIDE.md) for a streamlined setup process.

### Option 2: Detailed Setup (Recommended for production)
Follow the [Complete Documentation](DISTRIBUTED_TRAINING_SETUP.md) for comprehensive understanding and production deployment.

## 🏃‍♂️ Running Your First Distributed Training

1. **Clone and build:**
   ```bash
   git clone <this-repo>
   cd distributed_training
   docker build -t my-ddp-trainer:latest .
   ```

2. **Set up Docker Swarm:**
   ```bash
   # On manager node
   docker swarm init --advertise-addr <MANAGER_IP>
   
   # On worker nodes
   docker swarm join --token <TOKEN> <MANAGER_IP>:2377
   ```

3. **Deploy training:**
   ```bash
   docker stack deploy -c docker-compose.yml distributed-training
   ```

4. **Monitor progress:**
   ```bash
   docker service logs -f distributed-training_node0
   ```

## 📊 Configuration Options

### Training Parameters
- `--nnodes`: Number of nodes (machines)
- `--nproc_per_node`: Processes per node (usually 1 per GPU)
- `--batch_size`: Batch size per process
- `total_epochs`: Number of training epochs
- `save_every`: Checkpoint frequency

### NCCL Optimization
- `NCCL_SOCKET_IFNAME`: Network interface
- `NCCL_BUFFSIZE`: Communication buffer size
- `NCCL_NTHREADS`: Number of NCCL threads
- `NCCL_DEBUG`: Debug level (INFO, TRACE)

### Resource Allocation
- GPU memory and compute allocation
- CPU affinity and thread management
- Network bandwidth optimization

## 🔍 Monitoring and Debugging

### Real-time Monitoring
```bash
# Service status
docker service ls

# Container logs
docker service logs -f distributed-training_node0

# Resource usage
docker stats
```

### Debug Tools
```bash
# Container inspection
docker service inspect distributed-training_node0

# Network debugging
docker network inspect ddp-net

# Node information
docker node ls
```

## 📈 Scaling Your Training

### Adding More Nodes
1. Join new nodes to the swarm
2. Update `docker-compose.yml` with new services
3. Adjust `--nnodes` parameter
4. Redeploy the stack

### Performance Optimization
- Tune NCCL parameters for your network
- Optimize batch sizes for your GPU memory
- Use appropriate data loading strategies
- Monitor network bandwidth utilization

## 🐛 Troubleshooting

### Common Issues
- **NCCL timeouts**: Check network connectivity and firewall settings
- **Container placement**: Verify node hostname constraints
- **GPU access**: Ensure NVIDIA Docker runtime is installed
- **Image availability**: Use Docker registry for multi-node deployments

### Debug Environment Variables
```bash
export NCCL_DEBUG=TRACE
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1
```

## 🏭 Production Considerations

### Security
- Use secure Docker registries
- Implement proper network segmentation
- Use secrets management for sensitive data
- Configure firewall rules appropriately

### High Availability
- Implement proper health checks
- Use restart policies and fault tolerance
- Set up monitoring and alerting
- Plan for disaster recovery

### Performance
- Use dedicated network interfaces for training
- Optimize NCCL settings for your hardware
- Implement efficient data loading pipelines
- Monitor and profile resource usage

## 📚 Documentation

- [Complete Setup Guide](DISTRIBUTED_TRAINING_SETUP.md) - Comprehensive documentation
- [Quick Start Guide](QUICK_START_GUIDE.md) - Fast setup instructions
- [Docker Compose Reference](docker-compose.yml) - Service configuration
- [Training Code](training_code.py) - PyTorch DDP implementation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the distributed training framework
- Docker team for containerization and orchestration tools
- NVIDIA for NCCL and GPU acceleration support

## 📞 Support

- Create an issue for bug reports or feature requests
- Check the troubleshooting section in the documentation
- Review existing issues and discussions

---

**Happy Distributed Training! 🚀** 
