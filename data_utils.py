# this class wraps a torch.utils.data.DataLoader into an iterator for batch by batch fetching
import torch

class DataIterator(object):
    def __init__(self, dataloader, nonstop=True):
        assert isinstance(dataloader, torch.utils.data.DataLoader), 'Wrong loader type'
        self.loader = dataloader
        self.iterator = iter(self.loader)
        self.nonstop = nonstop

    def __next__(self):
        try:
            tup = next(self.iterator)
        except StopIteration:
            if not self.nonstop:
                raise StopIteration()
            
            self.iterator = iter(self.loader)
            tup = next(self.iterator)

        return tup
