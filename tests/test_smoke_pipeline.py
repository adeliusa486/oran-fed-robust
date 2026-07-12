"""End-to-end smoke test: tiny federated run converges and defenses help."""
import numpy as np

from oran_fed_robust.aggregation import build_aggregator
from oran_fed_robust.data import generate_federated_dataset
from oran_fed_robust.training import FederatedTrainer


def _make(alpha=0.5):
    return generate_federated_dataset(
        n_clients=12, n_features=10, n_classes=4,
        samples_per_client=150, dirichlet_alpha=alpha, seed=7,
    )


def test_no_attack_run_learns():
    clients, test_set, root_set = _make()
    trainer = FederatedTrainer(
        clients, test_set, root_set, build_aggregator("fedavg"),
        n_features=10, n_classes=4, attack_name="none",
        participants_per_round=6, local_epochs=2, lr=0.1, seed=0,
    )
    history = trainer.fit(rounds=10)
    assert history[-1].accuracy > 0.4  # well above random (0.25)


def test_reputation_defends_against_fabricated_attack():
    clients, test_set, root_set = _make()
    rep = FederatedTrainer(
        clients, test_set, root_set, build_aggregator("reputation", beta=0.8),
        n_features=10, n_classes=4, attack_name="fabricated",
        compromise_fraction=0.25, attack_scale=8.0,
        participants_per_round=8, local_epochs=2, lr=0.1, seed=0,
    ).fit(rounds=12)[-1]

    nodef = FederatedTrainer(
        clients, test_set, root_set, build_aggregator("fedavg"),
        n_features=10, n_classes=4, attack_name="fabricated",
        compromise_fraction=0.25, attack_scale=8.0,
        participants_per_round=8, local_epochs=2, lr=0.1, seed=0,
    ).fit(rounds=12)[-1]

    # the defense should not be worse than no-defense under attack
    assert rep.accuracy >= nodef.accuracy
