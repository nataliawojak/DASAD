import argparse
import logging
import yaml
import pandas as pd
import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from DASAD.engine import StreamEngine
from DASAD.detectors import DriftDetector
from DASAD.models import DANNPredictor
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score
from DASAD.networks import build_encoder, build_task, build_discriminator
from DASAD.utils import make_json_serializable


#python Python 3.11.7 

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path="configs/default.yaml"):

    config = load_config(config_path)
    logging.basicConfig(
        level=getattr(
            logging,
            config.get("logging", {}).get("level", "INFO").upper(),
        ),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    # Load Data
    df = pd.read_csv(config["data"]["path"])

    X = df[config["x_cols"]]
    y = df[config["y_col"]]
    source_size = config["engine"]["source_size"]
    dataset_name = config["data"]["name"]
    if not 0 < source_size < len(X):
        raise ValueError("engine.source_size must be between 1 and len(dataset) - 1")

    latent_dims = {
        int(config["network"][component]["latent_dim"])
        for component in ("encoder", "task", "discriminator")
    }
    if len(latent_dims) != 1:
        raise ValueError(
            "network encoder/task/discriminator must use the same latent_dim"
        )

    # S0 is the only scaler fitted offline. StreamEngine receives raw data and
    # fits each later domain scaler on its observed target adaptation window.
    scaler = StandardScaler()
    scaler.fit(X.iloc[:source_size].to_numpy())
    logger.info(
        "DATA LOADED | path=%s | samples=%d | features=%d | source=%d | "
        "stream_after_source=%d",
        config["data"]["path"],
        len(X),
        X.shape[1],
        source_size,
        len(X) - source_size,
    )
    
    input_shape = (len(config["x_cols"]),)
    encoder = build_encoder(input_shape, config["network"]["encoder"])
    task = build_task(config["network"]["task"])
    discriminator = build_discriminator(config["network"]["discriminator"])

    logger.info(
        "NETWORK CONFIG | latent_dim=%d | task_learning_rate=%g | "
        "encoder_learning_rate=%g | discriminator_learning_rate=%g | "
        "gamma=%g | alpha=%g | lambda=%g | batch_size=%d | epochs=%d",
        config["network"]["encoder"]["latent_dim"],
        config["network"]["task"]["learning_rate"],
        config["network"]["encoder"]["learning_rate"],
        config["network"]["discriminator"]["discriminator_learning_rate"],
        config["predictor"]["gamma"],
        config["predictor"]["alpha"],
        config["predictor"].get(
            "lambda", config["predictor"].get("adv_weight", 0.001)
        ),
        config["predictor"]["batch_size"],
        config["predictor"]["epochs"],
    )

 
    detector = DriftDetector(
        reference_size=config["detector"]["reference_size"],
        window_size=config["detector"]["window_size"],
        min_instances=config["detector"]["min_instances"],
        delta=config["detector"]["delta"], 
        threshold=config["detector"]["threshold"], 
        alpha=config["detector"]["alpha"]
    )
    

    predictor = DANNPredictor(
        encoder = encoder, 
        task = task, 
        discriminator = discriminator, 
        adv_weight=config["predictor"].get(
            "lambda", config["predictor"].get("adv_weight", 0.001)
        ),
        learning_rate=config["predictor"]["learning_rate"],
        gamma=config["predictor"]["gamma"],
        class_balance=config["predictor"]["class_balance"],
        alpha=config["predictor"]["alpha"],
        epochs=config["predictor"]["epochs"],
        batch_size=config["predictor"]["batch_size"],
        threshold=config["predictor"]["threshold"],
        pretrain_epochs=config["predictor"].get(
            "pretrain_epochs", config["predictor"]["epochs"]
        ),
        pretrain_validation_split=config["predictor"].get(
            "pretrain_validation_split", 0.2
        ),
        pretrain_patience=config["predictor"].get("pretrain_patience", 20),
    )


    engine = StreamEngine(
        detector,
        predictor,
        source_data=X.iloc[:source_size].to_numpy(),
        source_labels=y.iloc[:source_size].to_numpy(),
        source_size=source_size,
        window_size=config["engine"]["window_size"],
        source_scaler=scaler,
        feature_names=config["x_cols"],
    )

    results = engine.run(X, y)
    
    output_directory = Path(config.get("results_directory", "results"))
    output_directory.mkdir(parents=True, exist_ok=True)

    target_window_size = config["engine"]["window_size"]
    
    output_data = {
        "dataset_name": dataset_name,
        "dataset_path": str(config["data"]["path"]),
        "source_size": source_size,
        "target_window_size": target_window_size,
        "total_samples": len(X),
        "evaluated_samples": len(results["true"]),
        "feature_names": list(config["x_cols"]),
        "results": results,
    }
    
    output_path = output_directory / (
        f"{dataset_name}"
        f"_window_{target_window_size}.json"
    )
    
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            make_json_serializable(output_data),
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    
    logger.info(
        "RESULTS SAVED | dataset=%s | path=%s",
        dataset_name,
        output_path,
    )

    print(f"F1 score: {f1_score(results['true'], results['predictions']):.3f}")
    print(f"G-mean: {geometric_mean_score(results['true'], results['predictions']):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_ds1.yaml")

    args = parser.parse_args()

    main(args.config)
