"""DAgger rounds: the training data of each round. ``positronic.dagger.rounds`` runs the rounds.

The training data of DAgger round ``k`` holds the DAgger reader's output for the records of rounds ``0`` to ``k``. The
round driver writes it as a positronic dataset::

    labels = (keys.TARGET_EE_POSE, keys.TARGET_GRIP)
    # record_paths[k]: the records of DAgger round k
    rule = LicenseRule({'apache-2.0'})
    rounds = [DaggerDataset(LocalDataset(path), embodiment.observations, labels, rule) for path in record_paths]
    write_training_data(rounds, output)
    steps_without_shadow = [round_data.steps_without_shadow for round_data in rounds]
"""

from positronic.dagger.training_data import META_RECORD_UID, META_ROUND, write_training_data

__all__ = ['META_RECORD_UID', 'META_ROUND', 'write_training_data']
