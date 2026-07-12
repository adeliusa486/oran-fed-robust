"""Federated training orchestration.

Each round:
  1. sample participating clients,
  2. each client trains locally (malicious data-level clients flip labels),
  3. compute per-client update = local_params - global_params,
  4. malicious update-level clients perturb their update,
  5. aggregate updates with the configured rule,
  6. apply aggregated update to the global model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..aggregation import Aggregator, AggregationContext
from ..attacks import apply_update_attack, is_data_level_attack, poison_labels
from ..data import ClientDataset
from ..evaluation import evaluate_model
from ..models import SoftmaxClassifier


@dataclass
class RoundResult:
    round: int
    accuracy: float
    macro_f1: float


class FederatedTrainer:
    def __init__(
        self,
        clients: List[ClientDataset],
        test_set,
        root_set,
        aggregator: Aggregator,
        n_features: int,
        n_classes: int,
        *,
        attack_name: str = "none",
        compromise_fraction: float = 0.0,
        attack_scale: float = 5.0,
        participants_per_round: int = 20,
        local_epochs: int = 2,
        lr: float = 0.05,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.clients = clients
        self.x_test, self.y_test = test_set
        self.x_root, self.y_root = root_set
        self.aggregator = aggregator
        self.n_features = n_features
        self.n_classes = n_classes
        self.attack_name = attack_name
        self.attack_scale = attack_scale
        self.participants = min(participants_per_round, len(clients))
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        self.global_model = SoftmaxClassifier(n_features, n_classes, seed=seed)
        self.global_params = self.global_model.get_params()

        # designate malicious clients (stable across rounds)
        n_mal = int(round(compromise_fraction * len(clients)))
        mal = set(self.rng.choice(len(clients), size=n_mal, replace=False).tolist()) if n_mal else set()
        self.malicious_ids = mal

    # ------------------------------------------------------------------
    def _local_update(self, client: ClientDataset) -> np.ndarray:
        model = SoftmaxClassifier(self.n_features, self.n_classes)
        model.set_params(self.global_params)
        x, y = client.x, client.y
        if client.client_id in self.malicious_ids and is_data_level_attack(self.attack_name):
            y = poison_labels(y, self.n_classes)
        new_params = model.local_train(
            x, y, self.local_epochs, self.lr, self.batch_size, self.rng
        )
        update = new_params - self.global_params
        if client.client_id in self.malicious_ids and not is_data_level_attack(self.attack_name):
            update = apply_update_attack(update, self.attack_name, self.attack_scale, self.rng)
        return update

    def _server_update(self) -> np.ndarray:
        """Trusted root update for FLTrust."""
        model = SoftmaxClassifier(self.n_features, self.n_classes)
        model.set_params(self.global_params)
        new_params = model.local_train(
            self.x_root, self.y_root, self.local_epochs, self.lr, self.batch_size, self.rng
        )
        return new_params - self.global_params

    # ------------------------------------------------------------------
    def run_round(self, round_idx: int) -> RoundResult:
        ids = self.rng.choice(len(self.clients), size=self.participants, replace=False)
        updates = np.array([self._local_update(self.clients[i]) for i in ids])

        n_byz = sum(1 for i in ids if i in self.malicious_ids)
        ctx = AggregationContext(
            client_ids=[int(i) for i in ids],
            server_update=self._server_update() if self.aggregator.name == "fltrust" else None,
            n_byzantine=max(1, n_byz),
        )
        agg_update = self.aggregator.aggregate(updates, ctx)
        self.global_params = self.global_params + agg_update
        self.global_model.set_params(self.global_params)

        metrics = evaluate_model(self.global_model, self.x_test, self.y_test)
        return RoundResult(round_idx, metrics["accuracy"], metrics["macro_f1"])

    def fit(self, rounds: int, log_every: int = 0) -> List[RoundResult]:
        history: List[RoundResult] = []
        for r in range(rounds):
            res = self.run_round(r)
            history.append(res)
            if log_every and (r % log_every == 0 or r == rounds - 1):
                print(f"round {r:3d} | acc={res.accuracy:.3f} | macroF1={res.macro_f1:.3f}")
        return history
