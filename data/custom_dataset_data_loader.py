### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
### This script is modified based on the pix2pixHD official implementation (see license above)
### https://github.com/NVIDIA/pix2pixHD
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'Paired/aligned dataloader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        
        # Check if DDP is enabled
        use_ddp = hasattr(opt, 'world_size') and opt.world_size > 1
        
        if use_ddp:
            # Use DistributedSampler for DDP
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=opt.world_size,
                rank=opt.rank,
                shuffle=True
            )
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=False,  # DistributedSampler handles shuffling
                sampler=self.sampler,
                num_workers=int(opt.nThreads),
                pin_memory=True,
                drop_last=True  # Recommended for DDP to avoid uneven batches
            )
        else:
            # Single GPU / CPU mode
            self.sampler = None
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=True,
                num_workers=int(opt.nThreads),
                pin_memory=True
            )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
