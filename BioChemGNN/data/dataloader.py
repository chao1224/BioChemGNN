import torch
from torch._six import container_abcs, string_classes, int_classes
from BioChemGNN import data


def graph_collate(batch):
    elem = batch[0]

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, int_classes):
        return torch.LongTensor(batch)
    elif isinstance(elem, float):
        return torch.FloatTensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, data.MoleculeGraph):
        return elem.pack(batch)
    elif isinstance(elem, container_abcs.Mapping):
        return {key: graph_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, container_abcs.Sequence) and (isinstance(elem[0], float) or isinstance(elem[0], int_classes)):
        return torch.stack([graph_collate(samples) for samples in zip(*batch)], dim=1)
    elif isinstance(elem, container_abcs.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('Each element in list of batch should be of equal size')
        return [graph_collate(samples) for samples in zip(*batch)]

    raise TypeError("Can't collate data with type `%s`" % type(elem))


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, graph_collate=graph_collate, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=graph_collate, **kwargs)
