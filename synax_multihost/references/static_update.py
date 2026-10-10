# Extracted reference functions; do not run as a training script.
from __future__ import annotations
import copy
import math
import torch
from torch import nn


def train_synax_ANN(node, epoch, round, g_t, beta=1, gamma=1.0):
    node.model.to(node.device).train()
    train_loader = node.train_data
    loss_fn = nn.CrossEntropyLoss()
    loss_m = AverageMeter()
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "Node{:d}: loss={:.4f} acc={:.2f}%"

    # Ensure g_t is on the correct device
    g_t_device = {k: v.to(node.device) for k, v in g_t.items()}

    # Calculate scaling factor for g_t to match gradient scale
    # We assume effective step size is roughly lr.
    # To be consistent with the formula: g_correct = beta * grad + (1-beta) * gamma * g_t

    with tqdm(train_loader) as epochs:
        for batch_idx, (data, target) in enumerate(epochs):
            epochs.set_description(description.format(node.num, avg_loss, acc))
            data = data.float().to(node.device)
            if target.dim() > 1 and target.size(1) == 1:
                target = target.long().to(node.device).squeeze(dim=1)
            else:
                target = target.long().to(node.device)
            if node.args.dataset.lower() == 'chestxray':
                target = (target.sum(dim=1) > 0).long()

            node.optimizer.zero_grad()
            output = node.model(data)
            output = output[0] # Handle model output tuple
            loss = loss_fn(output, target)
            loss.backward()

            # --- SyNAX: Gradient Correction ---
            with torch.no_grad():
                for name, param in node.model.named_parameters():
                    if param.grad is not None and name in g_t_device:
                        # g_{k,t}^{(j)} <- beta * grad + (1-beta) * gamma * g_t
                        # Note: param.grad is updated in place
                        param.grad.data = beta * param.grad.data + (1 - beta) * gamma * g_t_device[name]

            node.optimizer.step()

            loss_m.update(loss.item(), n=data.size(0))
            total_loss += loss
            avg_loss = total_loss / (batch_idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100

    return avg_loss


def update_global_models(self, avg_ann_update, avg_alpha, eta_ann=1.0, eta_snn=1.0):
        """
        w_{t+1}^{ANN} = w_t^{ANN} + eta_ann * Delta_w_A - eta_snn * alpha_bar * (g_t / ||g_t||)
        """
        new_ann_state = copy.deepcopy(self.model.state_dict())

        # Normalize once across all trainable parameters. State buffers such as
        # BatchNorm running statistics must not change the update direction.
        parameter_names = set(dict(self.model.named_parameters()))
        valid_g = {
            k: v for k, v in self.g_t.items()
            if k in parameter_names and v.is_floating_point()
        }
        if valid_g:
            g_t_norm = torch.sqrt(sum(
                torch.sum(v.float() ** 2) for v in valid_g.values()
            ))
        else:
            g_t_norm = torch.tensor(0.0, device=next(self.model.parameters()).device)
        correction_step = eta_snn * abs(float(avg_alpha)) if g_t_norm > 1e-12 else 0.0
        print(
            f"Global g_t norm: {g_t_norm.item():.6f}, "
            f"normalized correction step: {correction_step:.6f}"
        )

        # 1. Update ANN
        for k in new_ann_state:
            is_float = new_ann_state[k].is_floating_point()
            # Ensure update_term is float for calculation to avoid type errors with LongTensors (e.g. num_batches_tracked)
            update_term = torch.zeros_like(new_ann_state[k], dtype=torch.float)

            if k in avg_ann_update:
                update_term += eta_ann * avg_ann_update[k]

            if k in valid_g and g_t_norm > 1e-12 and avg_alpha != 0:
                update_term -= eta_snn * avg_alpha * (valid_g[k] / g_t_norm)

            if is_float:
                new_ann_state[k] += update_term
            else:
                # Cast to float for update then back to original type (e.g. Long)
                new_ann_state[k] = (new_ann_state[k].float() + update_term).long()

        # Update self.model (w_{t+1}^{ANN})
        self.w_t_ann = copy.deepcopy(self.model.state_dict()) # Save old w_t before updating
        self.model.load_state_dict(new_ann_state)

        # 2. Convert to SNN (w_{t+1}^{SNN})
        self.w_t_snn = copy.deepcopy(self.snn_model.state_dict()) # Save old w_t SNN
        self.snn_model = ann2snn(self.model, self.test_data, args=self.args)

        # 3. Update Guidance for next round
        # g_{t+1} = w_t^{ANN} - w_{t+1}^{ANN}
        current_ann = self.model.state_dict()
        for k in self.g_t:
            self.g_t[k] = self.w_t_ann[k] - current_ann[k]

        # G_{t+1} = w_t^{SNN} - w_{t+1}^{SNN}
        current_snn = self.snn_model.state_dict()
        for k in self.G_t:
             # Ensure keys match (SNN structure might change slightly or keys match exactly)
             if k in current_snn and k in self.w_t_snn:
                 self.G_t[k] = self.w_t_snn[k] - current_snn[k]
             else:
                 self.G_t[k] = torch.zeros_like(self.G_t[k])
