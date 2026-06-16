import argparse
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from DDDA.engine import StreamEngine
from DDDA.detectors import DriftDetector
from DDDA.models import DANNPredictor
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score
from DDDA.networks import build_encoder, build_task, build_discriminator

#python Python 3.11.7 

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path="configs/default.yaml"):

    config = load_config(config_path)
    # Load Data
    df = pd.read_csv(config["data"]["path"])

    X = df[config["x_cols"]]
    y = df[config["y_col"]]
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=config["x_cols"])
    
    input_shape = (len(config["x_cols"]),)
    encoder = build_encoder(input_shape, config["network"]["encoder"])
    task = build_task(config["network"]["task"])
    discriminator = build_discriminator(config["network"]["discriminator"])

 
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
        adv_weight=config["predictor"]["adv_weight"], 
        learning_rate=config["predictor"]["learning_rate"],
        gamma=config["predictor"]["gamma"],
        class_balance=config["predictor"]["class_balance"],
        alpha=config["predictor"]["alpha"],
        epochs=config["predictor"]["epochs"],
        batch_size=config["predictor"]["batch_size"],
        threshold=config["predictor"]["threshold"]
    )


    engine = StreamEngine(
        detector,
        predictor,
        window_size=config["engine"]["window_size"]
    )

    results = engine.run(X_scaled, y)

    print(f"F1 score: {f1_score(results['true'], results['predictions']):.3f}")
    print(f"G-mean: {geometric_mean_score(results['true'], results['predictions']):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_ds1.yaml")

    args = parser.parse_args()

    main(args.config)