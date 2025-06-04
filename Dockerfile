FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Copy your code
COPY . .

# Install Python dependencies (if any)
RUN pip install --no-cache-dir -r requirements.txt

# Entry is handled via torchrun command passed from docker-compose
CMD ["bash"]
