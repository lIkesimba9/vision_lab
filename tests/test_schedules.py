import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vision_lab.core.optim import param_groups
from vision_lab.core.schedules import (
    ConstantSchedule,
    CosineSchedule,
    LinearSchedule,
    LinearWarmupConstant,
    ScheduleDriver,
    build_warmup_cosine,
)


def test_cosine_schedule_endpoints_and_midpoint():
    s = CosineSchedule(start=0.992, end=1.0)
    assert s(0, 100) == pytest.approx(0.992)
    assert s(100, 100) == pytest.approx(1.0)
    assert s(50, 100) == pytest.approx((0.992 + 1.0) / 2)
    assert s(200, 100) == pytest.approx(1.0)  # клампится за горизонтом


def test_linear_and_constant():
    assert LinearSchedule(0.0, 1.0)(25, 100) == pytest.approx(0.25)
    assert ConstantSchedule(0.3)(7, 10) == 0.3


def test_warmup_constant():
    s = LinearWarmupConstant(start=0.04, value=0.07, warmup_steps=30)
    assert s(0, 1000) == pytest.approx(0.04)
    assert s(15, 1000) == pytest.approx(0.055)
    assert s(30, 1000) == pytest.approx(0.07)
    assert s(999, 1000) == pytest.approx(0.07)


class FakeMethod(nn.Module):
    def __init__(self):
        super().__init__()
        self.current_tau = 0.99


class FakeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.method = FakeMethod()


def make_trainer(module, total=100, step=0):
    opt = torch.optim.SGD(
        param_groups({"method": module.method}, base_lr=0.1, weight_decay=1e-4)
    )
    return SimpleNamespace(estimated_stepping_batches=total, global_step=step, optimizers=[opt])


def test_driver_assigns_attr_and_param_group():
    module = FakeModule()
    module.method.fc = nn.Linear(2, 2)  # чтобы были decay-параметры
    driver = ScheduleDriver(
        {
            "method.current_tau": CosineSchedule(0.9, 1.0),
            "optimizer/method.decay.weight_decay": LinearSchedule(0.04, 0.4),
        },
        log_values=False,
    )
    trainer = make_trainer(module)
    driver.on_fit_start(trainer, module)

    trainer.global_step = 50
    driver.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
    assert module.method.current_tau == pytest.approx(0.95)
    group = next(g for g in trainer.optimizers[0].param_groups if g["name"] == "method.decay")
    assert group["weight_decay"] == pytest.approx(0.04 + (0.4 - 0.04) * 0.5)


def test_driver_validates_paths_on_fit_start():
    module = FakeModule()
    module.method.fc = nn.Linear(2, 2)
    trainer = make_trainer(module)

    with pytest.raises(AttributeError):
        ScheduleDriver({"method.currnet_tau": ConstantSchedule(1.0)}, log_values=False) \
            .on_fit_start(trainer, module)
    with pytest.raises(KeyError):
        ScheduleDriver({"optimizer/nope.weight_decay": ConstantSchedule(0.1)}, log_values=False) \
            .on_fit_start(trainer, module)


def test_build_warmup_cosine_reaches_base_lr_after_warmup():
    net = nn.Linear(2, 2)
    opt = torch.optim.SGD(net.parameters(), lr=1.0)
    sched = build_warmup_cosine(opt, warmup_steps=10, total_steps=100)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)  # start_factor
    for _ in range(10):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0, rel=1e-3)
    for _ in range(90):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-3)  # честный отжиг до ~0


def test_build_warmup_cosine_zero_warmup():
    net = nn.Linear(2, 2)
    opt = torch.optim.SGD(net.parameters(), lr=1.0)
    sched = build_warmup_cosine(opt, warmup_steps=0, total_steps=10)
    assert math.isclose(opt.param_groups[0]["lr"], 1.0)
    assert sched is not None
