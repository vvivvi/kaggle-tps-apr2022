import pandas as pd
import numpy as np
import os

DATADIR = "../data"


def ingest_sensor_data(csv_name: str, csv_name_labels: str = None):
    labels = None
    sensor_data = None
    sequence_ids = None

    df = pd.read_csv(os.path.join(DATADIR, csv_name))

    n_seq = len(np.unique(df["sequence"]))
    n_steps = np.max(df["step"]) + 1

    sensor_columns = [c for c in df.columns if "sensor" in c]
    n_sensors = len(sensor_columns)

    sensor_data = df[sensor_columns].to_numpy()
    sensor_data = np.reshape(sensor_data, newshape=(-1, n_steps, n_sensors))

    # permute dimensions -> (sequence, sensor, step)
    # sensor_data = np.transpose(sensor_data, (0, 2, 1))

    # do not permute -> dimensions (sequence, timestep, sensor)
    print(sensor_data.shape, n_seq)
    sequence_ids = df["sequence"].to_numpy()
    # remove duplicate elements
    sequence_ids = list(dict.fromkeys(list(sequence_ids)))

    if csv_name_labels is not None:
        df_labels = pd.read_csv(os.path.join(DATADIR, csv_name_labels))
        sequence_to_state = {}
        for _, row in df_labels.iterrows():
            sequence_to_state[row["sequence"]] = row["state"]
        labels = [sequence_to_state[seq_id] for seq_id in sequence_ids]

    return sensor_data, sequence_ids, labels


def validation_split(*, training_fraction: float, sensor_data, sequence_ids, labels):
    n_total = sensor_data.shape[0]
    n_train = int(training_fraction * n_total)

    sensor_data_train = sensor_data[:n_train]
    sensor_data_val = sensor_data[n_train:]

    sequence_ids_train = sequence_ids[:n_train]
    sequence_ids_val = sequence_ids[n_train:]

    labels_train = labels[:n_train]
    labels_val = labels[n_train:]

    return (
        sensor_data_train,
        sequence_ids_train,
        labels_train,
        sensor_data_val,
        sequence_ids_val,
        labels_val,
    )


def preprocess_sensor_data(data_in):
    clip_min = -3
    clip_max = 3
    data_out = np.clip(data_in, clip_min, clip_max) / (clip_max - clip_min)
    return data_out


def generate_submission(scores, sequence_ids, submission_id):
    seq_ids = np.array(sequence_ids).reshape((-1, 1))
    df = pd.DataFrame(data=seq_ids, columns=["sequence"])
    df["state"] = scores
    df.to_csv(f"submission-{submission_id}.csv", index=False)
