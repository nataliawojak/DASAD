import logging
import numpy as np

logger = logging.getLogger(__name__)


class StreamEngine:
    """Streaming engine for drift detection, adaptive training, and prediction.

    This engine processes data sequentially, detects drift using a detector,
    collects target samples, retrains the predictor, and generates predictions
    after retraining.
    """
    
    def __init__(self, detector, predictor, window_size=100):
        """Initialize stream engine.

        Args:
            detector: Drift detector implementing `update()` method.
            predictor: Model implementing `fit()` and `predict()` methods.
            window_size (int, optional): Number of target samples collected
                after drift before retraining. Defaults to 100.
        """
        self.detector = detector
        self.predictor = predictor
        self.window_size = window_size

        self.history = {
            "predictions": [],
            "drift_points": [],
            "true": [],
            "idx": [],
        }

        self.target_buffer = []
        self.collecting = False

        self.source_data = None
        self.source_labels = None
        
    def _handle_drift(self, i, X, y, x_i):
        """Handle drift detection event.

        Args:
            i (int): Current index in stream.
            X (pd.DataFrame): Feature dataset.
            y (pd.Series): Target labels.
            x_i (np.ndarray): Current sample.
        """
        logger.warning("Drift detected at index %d", i)
    
        self.history["drift_points"].append(i)
    
        #Only set source ONCE
        if self.source_data is None:
            logger.info("Setting source data using first drift at index %d", i)
    
            self.source_data = X[:i]
            self.source_labels = y[:i]
    
        #Always restart target collection
        self.collecting = True
        self.target_buffer = []

    def _collect_target(self, x_i):
        """Collect target samples after drift.

        Args:
            x_i (np.ndarray): Incoming sample.
        """
        if self.collecting:
            self.target_buffer.append(x_i)

    def _train_if_ready(self):
        """Train predictor when enough target samples are collected."""
        if len(self.target_buffer) == self.window_size:
            logger.info("Collected %d samples. Starting training...", self.window_size)
            

            Xt = np.array(self.target_buffer)


            self.predictor.fit(
                self.source_data,
                self.source_labels.values,
                Xt,
            )

            logger.info("Model training completed")

    def _predict_if_ready(self, i, x_i, y_i):
        """Generate prediction after model is trained.

        Args:
            i (int): Current index.
            x_i (np.ndarray): Input features.
            y_i: True label.
        """
        if len(self.target_buffer) > self.window_size:
            y_pred = self.predictor.predict(x_i)

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
        drift = self.detector.update(x_i)
        if drift:
            self._handle_drift(i, X, y, x_i)

        self._collect_target(x_i)
        self._train_if_ready()
        self._predict_if_ready(i, x_i, y_i)

    def run(self, X, y):
        """Run streaming pipeline.

        Args:
            X (pd.DataFrame): Feature dataset.
            y (pd.Series): Target labels.

        Returns:
            dict: Dictionary containing predictions, true labels,
                drift points, and prediction indices.
        """
        logger.info("Starting stream processing...")
        for i in range(len(X)):
            self._process_sample(X, y, i)

        logger.info("Stream processing finished")

        return self.history