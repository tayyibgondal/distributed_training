import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

def prepare_dataloader(dataset, batch_size):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset),
        pin_memory=True,
        shuflle=False
    )

class Trainer:
    def __inti__(
            self,
            model,
            train_data,
            optimizer,
            save_every,
            snapshot_path
    ):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snashot_path = snapshot_path

        if os.path.exists(self.snashot_path):
            print('Loading snapshot')
            self._load_snapshot()

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(self.snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot['MODEL_STATE'])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f'resume training at epoch {self.epochs_run}')

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": self.epochs_run
        }
        torch.save(snapshot, self.snashot_path)
        print(f"Epoch {epoch} | train snapshot saved at {self.snashot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
    
    def _run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] epoch {epoch} | batch_size: {batch_size} | steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch(source, targets)

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()



def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: int):
    ddp_setup()

    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)

    destroy_process_group()