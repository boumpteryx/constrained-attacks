import numpy as np
import torch

from constrained_attacks.constraints.constraints_executor import (
    PytorchConstraintsExecutor,
)
from constrained_attacks.constraints.relation_constraint import AndConstraint
from constrained_attacks.datasets import load_dataset


def test_lcld():
    N_SAMPLES = 100

    ds = load_dataset("lcld_v2_time")
    x, _, _ = ds.get_x_y_t()
    constraints = ds.get_constraints()

    executor = PytorchConstraintsExecutor(
        AndConstraint(constraints.relation_constraints), x.columns.to_numpy()
    )
    x_test = torch.Tensor(x.iloc[:N_SAMPLES].values)
    out = executor.execute(x_test)
    out = out.numpy()

    assert not np.isnan(out).any()
