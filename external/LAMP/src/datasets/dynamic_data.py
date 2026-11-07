from torch_geometric.data import Data


class DynamicData:
    def __init__(self, static_keys=(), **storage):
        self._static_keys = static_keys
        self._storage = storage

    def __getitem__(self, t):
        return Data(
            **{
                key: value if key in self._static_keys else value[t]
                for key, value in self._storage.items()
            }
        )

    def __len__(self):
        return len(next(v for k, v in self._storage.items() if k not in self._static_keys))
