from importlib import import_module
from torch.utils.data import DataLoader
import numpy as np
import random

class Data(object):
    def __init__(self, args):
        self.args = args
        Data.seed = args.seed
        loader = import_module("data.data_loader_%s" % args.dataset)


        train_set_task, val_set_task, = loader.get_data_loader(args.root_path)

        self.train_loader_task1 = DataLoader(train_set_task,
                                       batch_size=args.batch_size,
                                       num_workers=args.workers,
                                       shuffle=True,
                                       pin_memory=True,
                                       worker_init_fn=self.worker_init_fn)

        self.test_loader_task1 = DataLoader(val_set_task,
                                      batch_size=args.batch_size,
                                      num_workers=args.workers,
                                      shuffle=False,
                                      pin_memory=True,
                                      worker_init_fn=self.worker_init_fn)

    @staticmethod
    def worker_init_fn(worker_id):
        """确保每个 worker 使用不同的种子"""
        worker_seed = 0
        np.random.seed(worker_seed + worker_id)
        random.seed(worker_seed + worker_id)


