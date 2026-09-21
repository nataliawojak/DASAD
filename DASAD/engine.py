import logging
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class StreamEngine:
    """Streaming engine for drift detection, adaptive training, and prediction.

    This engine processes data sequentially, detects drift using a detector,
    collects target samples, retrains the predictor, and generates predictions
    after retraining.
    """
    
    def __init__(
        self,
        detector,
        predictor,
        source_data,
        source_labels,
        source_size,
        window_size=100,
        source_scaler=None,
        feature_names=None,
    ):
        """Initialize stream engine.

        Args:
            detector: Drift detector implementing `update()` method.
            predictor: Model implementing `fit()` and `predict()` methods.
            source_data: Fixed labeled source features in their raw scale.
            source_labels: Labels corresponding to `source_data`.
            source_size (int): Number of samples in the fixed source prefix.
            window_size (int, optional): Number of target samples collected
                after drift before retraining. Defaults to 100.
            source_scaler: Scaler already fitted only on the source prefix. If
                omitted, a StandardScaler is fitted internally on source_data.
            feature_names: Optional names used in DEBUG scaler logging.
        """
        self.detector = detector
        self.predictor = predictor
        self.window_size = window_size
        self.source_size = source_size
        self.feature_names = list(feature_names) if feature_names is not None else None

        if source_size <= 0:
            raise ValueError("source_size must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if len(source_data) != source_size or len(source_labels) != source_size:
            raise ValueError(
                "source_data and source_labels must contain exactly source_size samples"
            )

        # Source is fixed before stream processing. It must never be extended
        # at a detected drift because that would leak post-change labels.
        # Raw values are retained so that each newly observed domain can fit a
        # separate scaler without using future samples.
        self.source_data_raw = np.asarray(source_data).copy()
        self.source_labels = np.asarray(source_labels).copy()
        self.source_scaler = source_scaler or StandardScaler()
        if source_scaler is None:
            self.source_scaler.fit(self.source_data_raw)
        self.detector_scaler = self.source_scaler
        self.active_scaler = self.source_scaler
        self.active_scaler_version = 0
        self.domain_scalers = {0: self.source_scaler}
        self.source_data = self.source_scaler.transform(self.source_data_raw)

        logger.info(
            "ENGINE INITIALIZED | source_raw=%s | source_scaled=%s | "
            "source_size=%d | target_window_size=%d | features=%d",
            self.source_data_raw.shape,
            self.source_data.shape,
            self.source_size,
            self.window_size,
            self.source_data.shape[1],
        )
        self._log_scaler(self.source_scaler, "S0 source scaler", self.source_data_raw)
        pretraining_details = self.predictor.pretrain_source(
            self.source_data,
            self.source_labels,
        )

        self.history = {
            "predictions": [],
            "drift_points": [],
            "true": [],
            "idx": [],
            "model_trainings": (
                [dict(pretraining_details)]
                if pretraining_details is not None
                else []
            ),
            "scaler_updates": [
                self._scaler_details(
                    self.source_scaler,
                    scaler_version=0,
                    fitted_samples=len(self.source_data_raw),
                    fitted_start=0,
                    fitted_end=self.source_size - 1,
                )
            ],
        }

        self.target_buffer = []
        self.target_indices = []
        self.collecting = False
        self.current_drift_index = None

    @staticmethod
    def _scaler_details(
        scaler, scaler_version, fitted_samples, fitted_start, fitted_end
    ):
        details = {
            "scaler": f"S{scaler_version}",
            "scaler_class": type(scaler).__name__,
            "fitted_samples": int(fitted_samples),
            "fitted_start": int(fitted_start),
            "fitted_end": int(fitted_end),
        }
        for attribute in ("data_min_", "data_max_", "mean_", "scale_"):
            value = getattr(scaler, attribute, None)
            if value is not None:
                details[attribute.rstrip("_")] = np.asarray(value).tolist()
        return details

    def _log_scaler(self, scaler, name, raw_data):
        raw_data = np.asarray(raw_data)
        ranges = np.ptp(raw_data, axis=0)
        logger.info(
            "SCALER FITTED | name=%s | class=%s | samples=%d | features=%d | "
            "raw_min=%g | raw_max=%g | constant_features=%d | "
            "mean_feature_range=%g",
            name,
            type(scaler).__name__,
            raw_data.shape[0],
            raw_data.shape[1],
            float(np.min(raw_data)),
            float(np.max(raw_data)),
            int(np.sum(ranges == 0)),
            float(np.mean(ranges)),
        )
        if self.feature_names is not None:
            for feature, minimum, maximum, feature_range in zip(
                self.feature_names,
                np.min(raw_data, axis=0),
                np.max(raw_data, axis=0),
                ranges,
            ):
                logger.debug(
                    "SCALER FEATURE | name=%s | feature=%s | min=%g | "
                    "max=%g | range=%g",
                    name,
                    feature,
                    minimum,
                    maximum,
                    feature_range,
                )


    def _handle_drift(self, i):
        """Handle drift detection event.

        Args:
            i (int): Current index in stream.
        """
        logger.warning("Drift detected at index %d", i)
    
        self.history["drift_points"].append(i)
    
        # Always restart unlabeled target collection. Source remains fixed.
        self.collecting = True
        self.target_buffer = []
        self.target_indices = []
        self.current_drift_index = i

    def _collect_target(self, i, x_i):
        """Collect target samples after drift.

        Args:
            x_i (np.ndarray): Incoming sample.
        """
        if self.collecting:
            self.target_buffer.append(x_i)
            self.target_indices.append(i)

    def _train_if_ready(self):
        """Train predictor when enough target samples are collected."""
        if len(self.target_buffer) == self.window_size:
            Xt_raw = np.array(self.target_buffer)
            previous_version = getattr(self.predictor, "model_version", None)
            next_scaler_version = (
                int(previous_version) + 1
                if previous_version is not None
                else self.active_scaler_version + 1
            )
            logger.info(
                "TARGET BATCH READY | drift_index=%d | target_start=%d | "
                "target_end=%d | target_samples=%d | source_samples=%d | "
                "current_model=%s",
                self.current_drift_index,
                self.target_indices[0],
                self.target_indices[-1],
                len(self.target_indices),
                len(self.source_data),
                f"M{previous_version}" if previous_version is not None else "unknown",
            )

            target_scaler = clone(self.source_scaler)
            target_scaler.fit(Xt_raw)
            Xt = target_scaler.transform(Xt_raw)
            self._log_scaler(
                target_scaler,
                f"S{next_scaler_version} target scaler",
                Xt_raw,
            )
            logger.info(
                "DOMAIN-SPECIFIC SCALING | source=S0 fitted_samples=%d | "
                "target=S%d fitted_samples=%d | target_scaled_min=%g | "
                "target_scaled_max=%g",
                len(self.source_data_raw),
                next_scaler_version,
                len(Xt_raw),
                float(np.min(Xt)),
                float(np.max(Xt)),
            )
            old_min = getattr(self.active_scaler, "data_min_", None)
            new_min = getattr(target_scaler, "data_min_", None)
            old_max = getattr(self.active_scaler, "data_max_", None)
            new_max = getattr(target_scaler, "data_max_", None)
            if all(value is not None for value in (old_min, new_min, old_max, new_max)):
                logger.info(
                    "SCALER CHANGE S%d -> S%d | mean_abs_min_change=%g | "
                    "mean_abs_max_change=%g | max_abs_bound_change=%g",
                    self.active_scaler_version,
                    next_scaler_version,
                    float(np.mean(np.abs(new_min - old_min))),
                    float(np.mean(np.abs(new_max - old_max))),
                    float(
                        max(
                            np.max(np.abs(new_min - old_min)),
                            np.max(np.abs(new_max - old_max)),
                        )
                    ),
                )

            training_details = self.predictor.fit(
                self.source_data,
                self.source_labels,
                Xt,
            )
            if training_details is not None:
                training_details = dict(training_details)
                training_details.update(
                    {
                        "drift_index": self.current_drift_index,
                        "target_start": self.target_indices[0],
                        "target_end": self.target_indices[-1],
                        "source_scaler": "S0",
                        "target_scaler": f"S{next_scaler_version}",
                    }
                )
                self.history["model_trainings"].append(training_details)

            self.active_scaler = target_scaler
            self.active_scaler_version = next_scaler_version
            self.domain_scalers[next_scaler_version] = target_scaler
            self.history["scaler_updates"].append(
                self._scaler_details(
                    target_scaler,
                    scaler_version=next_scaler_version,
                    fitted_samples=len(Xt_raw),
                    fitted_start=self.target_indices[0],
                    fitted_end=self.target_indices[-1],
                )
            )

            self.collecting = False
            self.target_buffer = []
            self.target_indices = []
            logger.info(
                "NEW MODEL ACTIVATED | active_model=M%s | active_scaler=S%d | "
                "scaler_window=%d..%d (%d samples)",
                getattr(self.predictor, "model_version", "unknown"),
                self.active_scaler_version,
                self.current_drift_index,
                self.current_drift_index + self.window_size - 1,
                self.window_size,
            )

    def _predict(self, i, x_i, y_i):
        """Generate a prequential prediction with the currently active model.

        Args:
            i (int): Current index.
            x_i (np.ndarray): Input features.
            y_i: True label.
        """
        x_scaled = self.active_scaler.transform(
            np.asarray(x_i).reshape(1, -1)
        )[0]
        y_pred = self.predictor.predict(x_scaled)

        self.history["predictions"].append(y_pred)
        self.history["true"].append(y_i)
        self.history["idx"].append(i)

    def _process_sample(self, X, y, i):
        """Process a single sample from the data stream.
        This method performs drift detection, target collection, model training,
        and prediction for one incoming sample in the stream.
        
        Args:
            X (pd.DataFrame): Feature dataset.
            y (pd.Series): Target labels.
            i (int): Index of the current sample in the stream.

        Returns:
            None
        """
        x_i = X.iloc[i].to_numpy()
        y_i = y.iloc[i]

        # Predict first. This prevents a sample from being adapted on before it
        # is evaluated, and keeps predictions available while target is being
        # collected. The labeled source prefix itself is not test data.
        if i >= self.source_size:
            self._predict(i, x_i, y_i)

        # Drift detection stays in the fixed S0 coordinate system; changing
        # scaling would invalidate the detector's historical reference.
        x_detector = self.detector_scaler.transform(
            x_i.reshape(1, -1)
        )[0]
        drift = self.detector.update(x_detector)
        # The labeled prefix initializes the detector but belongs to the
        # offline source-training phase, so alarms there must not trigger DA.
        if drift and i >= self.source_size:
            self._handle_drift(i)

        self._collect_target(i, x_i)
        self._train_if_ready()

    def run(self, X, y):
        """Run streaming pipeline.

        Args:
            X (pd.DataFrame): Feature dataset.
            y (pd.Series): Target labels.

        Returns:
            dict: Dictionary containing predictions, true labels,
                drift points, and prediction indices.
        """
        logger.info(
            "STARTING STREAM | total_samples=%d | source_samples=%d | "
            "evaluated_samples=%d | initial_model=M0 | initial_scaler=S0",
            len(X),
            self.source_size,
            len(X) - self.source_size,
        )
        for i in range(len(X)):
            self._process_sample(X, y, i)

        logger.info(
            "STREAM FINISHED | predictions=%d | drifts=%d | adaptations=%d | "
            "final_model=M%s | final_scaler=S%d",
            len(self.history["predictions"]),
            len(self.history["drift_points"]),
            len(self.history["model_trainings"]) - 1,
            getattr(self.predictor, "model_version", "unknown"),
            self.active_scaler_version,
        )

        return self.history
