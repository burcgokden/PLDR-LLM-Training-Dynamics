"""Direct numerical verification of the decoupled AdamW decay ledger."""

import math

import torch


@torch.no_grad()
def adamw_step_ledger_residual(optimizer, params, previous, lr_applied):
    """Return the relative residual of the realized AdamW update identity.

    The predicted loss update uses the post-step bias-corrected moments that
    generated the realized step.  The decoupled decay term uses the pre-step
    parameter exactly once.  The function returns ``None`` when the required
    optimizer state is absent.
    """
    if len(params) != len(previous):
        raise ValueError("parameter and previous-state lengths differ")
    by_parameter = {}
    for group in optimizer.param_groups:
        if group.get("amsgrad", False) or group.get("maximize", False):
            raise ValueError("ordered ledger supports standard AdamW only")
        for parameter in group["params"]:
            by_parameter[id(parameter)] = group
    err2 = actual2 = predicted2 = 0.0
    for param, before in zip(params, previous):
        group = by_parameter.get(id(param))
        state = optimizer.state.get(param)
        if group is None or state is None or "exp_avg" not in state:
            return None
        beta1, beta2 = group["betas"]
        step = state["step"]
        step = step.item() if torch.is_tensor(step) else step
        mhat = state["exp_avg"] / (1.0 - beta1 ** step)
        vhat = state["exp_avg_sq"] / (1.0 - beta2 ** step)
        predicted_after = before.clone()
        predicted_after.mul_(
            1.0
            - float(lr_applied)
            * float(group.get("weight_decay", 0.0))
        )
        predicted_after.addcdiv_(
            mhat,
            vhat.sqrt().add(float(group["eps"])),
            value=-float(lr_applied),
        )
        predicted = predicted_after - before
        actual = param - before
        err2 += float(((actual - predicted) ** 2).sum())
        actual2 += float((actual ** 2).sum())
        predicted2 += float((predicted ** 2).sum())
    scale = max(math.sqrt(actual2), math.sqrt(predicted2), 1e-30)
    return math.sqrt(err2) / scale
