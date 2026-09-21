import logging
import time
import numpy as np
from adapt.feature_based import DANN
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.optimizers.legacy import Adam

logger = logging.getLogger(__name__)


class BasePredictor:
    """Abstract base class for predictors.

    Defines the interface for a predictor with `fit` and `predict` methods.
    """ 
    def predict(self, x):
        """Predict the label for a given input.

        Args:
            x (np.ndarray): Input features.

        Returns:
            Prediction label(s).

        Raises:
            NotImplementedError: Must be implemented in subclasses.
        """
        raise NotImplementedError

    def fit(self, source_data, source_labels, target_data):
        """Train the predictor on source and target data.

        Args:
            source_data (np.ndarray): Source domain features.
            source_labels (np.ndarray): Source domain labels.
            target_data (np.ndarray): Target domain features.

        Raises:
            NotImplementedError: Must be implemented in subclasses.
        """
        raise NotImplementedError

    def pretrain_source(self, source_data, source_labels):
        """Train the initial anomaly detector using labeled source data only."""
        raise NotImplementedError


class DANNPredictor(BasePredictor):
    """Domain-Adversarial Neural Network (DANN) predictor.

    This predictor adapts features from a source domain to a target domain
    using adversarial training, combining an encoder, task-specific head, and
    domain discriminator.

    """
    def __init__(
        self,
        encoder,
        task,
        discriminator,
        adv_weight=0.001,
        learning_rate=0.001,
        gamma=1,
        class_balance=True,
        alpha=0.1,
        epochs=500,
        batch_size=128,
        threshold=0.5,
        pretrain_epochs=500,
        pretrain_validation_split=0.2,
        pretrain_patience=20):
        
        """Initialize DANN predictor.

        Args:
            encoder: Feature extractor network.
            task: Task-specific output head.
            discriminator: Domain discriminator network.
            adv_weight (float, optional): Adversarial loss weight.
                Defaults to 0.001.
            learning_rate (float, optional): Optimizer learning rate.
                Defaults to 0.001.
            gamma (float, optional): Focal loss gamma parameter.
                Defaults to 1.
            class_balance (bool, optional): Apply class balancing.
                Defaults to True.
            alpha (float, optional): Focal loss alpha parameter.
                Defaults to 0.1.
            epochs (int, optional): Training epochs. Defaults to 500.
            batch_size (int, optional): Batch size. Defaults to 128.
            threshold (float, optional): Classification threshold.
                Defaults to 0.5.
        """
        self.encoder = encoder
        self.task = task
        self.discriminator = discriminator

        self.adv_weight = adv_weight
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.class_balance = class_balance
        self.alpha = alpha

        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_validation_split = pretrain_validation_split
        self.pretrain_patience = pretrain_patience

        self.model = None
        self.source_model = None
        self.model_version = 0
        self.adaptation_round = 0
        self.training_history = []

    @staticmethod
    def _parameter_count(model):
        """Return parameter count when a Keras component is already built."""
        try:
            return int(model.count_params())
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _log_model_summary(model, name):
        """Write a Keras model summary through the module logger."""
        logger.info("=" * 70)
        logger.info("MODEL SUMMARY: %s", name)
        logger.info("=" * 70)
        try:
            model.summary(print_fn=lambda line: logger.info(line))
        except (ValueError, AttributeError):
            logger.info("Summary unavailable: model component is not built yet")

    def _task_loss(self):
        return BinaryFocalCrossentropy(
            gamma=self.gamma,
            apply_class_balancing=self.class_balance,
            alpha=self.alpha,
        )

    @staticmethod
    def _flat_weights(model):
        weights = model.get_weights()
        if not weights:
            return np.array([], dtype=float)
        return np.concatenate([np.asarray(weight).ravel() for weight in weights])

    def pretrain_source(self, source_data, source_labels):
        """Pretrain encoder and anomaly head on the fixed labeled source.

        No target observations or target anomaly labels are used here.  The
        trained encoder and task weights are subsequently reused by DANN when
        an unlabeled target batch becomes available.
        """
        source_data = np.asarray(source_data)
        source_labels = np.asarray(source_labels)
        if len(source_data) == 0:
            raise ValueError("Source data is empty")
        if len(source_data) != len(source_labels):
            raise ValueError("Source data and labels have different lengths")

        inputs = Input(shape=(source_data.shape[1],), name="source_features")
        outputs = self.task(self.encoder(inputs))
        self.source_model = Model(inputs=inputs, outputs=outputs)
        self.source_model.compile(
            optimizer=Adam(
                learning_rate=self.learning_rate,
                clipvalue=0.5,
            ),
            loss=self._task_loss(),
            metrics=["accuracy"],
        )
        self._log_model_summary(
            self.source_model,
            "M0 - initial source anomaly detector",
        )

        callbacks = []
        if self.pretrain_validation_split > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.pretrain_patience,
                    restore_best_weights=True,
                )
            )

        normal_count = int(np.sum(source_labels == 0))
        anomaly_count = int(np.sum(source_labels == 1))
        logger.info(
            "SOURCE PRETRAINING | model=M0 | source=%s | normal=%d | "
            "anomaly=%d | epochs=%d | batch_size=%d | learning_rate=%g | "
            "validation_split=%.2f | parameters=%s",
            source_data.shape,
            normal_count,
            anomaly_count,
            self.pretrain_epochs,
            self.batch_size,
            self.learning_rate,
            self.pretrain_validation_split,
            self._parameter_count(self.source_model),
        )
        started_at = time.perf_counter()
        history = self.source_model.fit(
            source_data,
            source_labels,
            epochs=self.pretrain_epochs,
            batch_size=self.batch_size,
            validation_split=self.pretrain_validation_split,
            shuffle=False,
            callbacks=callbacks,
            verbose=0,
        )
        elapsed_seconds = time.perf_counter() - started_at
        # Before the first drift, predictions are produced by the supervised
        # source model. DANN replaces this active model after adaptation.
        self.model = self.source_model
        epochs_completed = len(history.history.get("loss", []))
        final_loss = history.history.get("loss", [None])[-1]
        best_val_loss = None
        if history.history.get("val_loss"):
            best_val_loss = min(history.history["val_loss"])
        details = {
            "model": "M0",
            "training_type": "source_pretraining",
            "source_samples": len(source_data),
            "target_samples": 0,
            "epochs_completed": epochs_completed,
            "elapsed_seconds": elapsed_seconds,
            "final_loss": final_loss,
            "best_val_loss": best_val_loss,
        }
        self.training_history.append(details)
        logger.info(
            "SOURCE PRETRAINING FINISHED | model=M0 | epochs_completed=%d | "
            "elapsed_seconds=%.3f | final_loss=%s | best_val_loss=%s",
            epochs_completed,
            elapsed_seconds,
            final_loss,
            best_val_loss,
        )
        return details

    def _build_model(self):
        """Constructs the DANN model with specified encoder, task, and discriminator.

        Returns:
            DANN: A compiled DANN model ready for training.
        """
        logger.debug("Building DANN model")

        return DANN(
            encoder=self.encoder,
            task=self.task,
            discriminator=self.discriminator,
            metrics=["acc"],
            lambda_=self.adv_weight,
            random_state=0,
            loss=self._task_loss(),
            optimizer=Adam(
                learning_rate=self.learning_rate,
                clipvalue=0.5,
            ),
            copy=False,
        )

    def fit(self, source_data, source_labels, target_data):
        """Train the DANN model using source and target domain data.

        Args:
            source_data (np.ndarray): Source domain features.
            source_labels (np.ndarray): Source domain labels.
            target_data (np.ndarray): Target domain features.

        Raises:
            ValueError: If either source or target data is empty.
        """
        if len(source_data) == 0 or len(target_data) == 0:
            raise ValueError("Source or target data is empty")

        self.adaptation_round += 1
        previous_version = self.model_version
        next_version = previous_version + 1

        logger.info("=" * 70)
        logger.info(
            "STARTING DANN ADAPTATION | round=%d | M%d -> M%d",
            self.adaptation_round,
            previous_version,
            next_version,
        )
        logger.info(
            "TRAINING DATA | source=%s | target=%s",
            source_data.shape,
            target_data.shape,
        )
        logger.info(
            "TRAINING DATA STATISTICS | source_mean=%g | source_std=%g | "
            "source_min=%g | source_max=%g | target_mean=%g | "
            "target_std=%g | target_min=%g | target_max=%g",
            float(np.mean(source_data)),
            float(np.std(source_data)),
            float(np.min(source_data)),
            float(np.max(source_data)),
            float(np.mean(target_data)),
            float(np.std(target_data)),
            float(np.min(target_data)),
            float(np.max(target_data)),
        )
        logger.info(
            "TRAINING PARAMETERS | epochs=%d | batch_size=%d | "
            "learning_rate=%g | adv_weight=%g | gamma=%g | alpha=%g | "
            "threshold=%g",
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.adv_weight,
            self.gamma,
            self.alpha,
            self.threshold,
        )
        self._log_model_summary(self.encoder, f"M{next_version} - encoder")
        self._log_model_summary(self.task, f"M{next_version} - anomaly classifier")
        self._log_model_summary(
            self.discriminator,
            f"M{next_version} - domain discriminator",
        )

        encoder_before = self._flat_weights(self.encoder)
        task_before = self._flat_weights(self.task)
        discriminator_before = self._flat_weights(self.discriminator)

        dann_model = self._build_model()
        dann_model.compile()

        started_at = time.perf_counter()
        dann_model.fit(
            X=source_data,
            y=source_labels,
            Xt=target_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )
        elapsed_seconds = time.perf_counter() - started_at

        encoder_after = self._flat_weights(self.encoder)
        task_after = self._flat_weights(self.task)
        discriminator_after = self._flat_weights(self.discriminator)

        def weight_change(before, after):
            if before.size == 0 or before.shape != after.shape:
                return None, None
            absolute = float(np.linalg.norm(after - before))
            relative = absolute / (float(np.linalg.norm(before)) + 1e-12)
            return absolute, relative

        encoder_delta, encoder_relative_delta = weight_change(
            encoder_before, encoder_after
        )
        task_delta, task_relative_delta = weight_change(task_before, task_after)
        discriminator_delta, discriminator_relative_delta = weight_change(
            discriminator_before, discriminator_after
        )

        self.model = dann_model
        self.model_version = next_version

        details = {
            "model": f"M{next_version}",
            "previous_model": f"M{previous_version}",
            "training_type": "dann_adaptation",
            "adaptation_round": self.adaptation_round,
            "source_samples": len(source_data),
            "target_samples": len(target_data),
            "epochs_requested": self.epochs,
            "elapsed_seconds": elapsed_seconds,
            "encoder_parameters": self._parameter_count(self.encoder),
            "task_parameters": self._parameter_count(self.task),
            "discriminator_parameters": self._parameter_count(self.discriminator),
            "encoder_weight_delta_l2": encoder_delta,
            "encoder_weight_relative_delta": encoder_relative_delta,
            "task_weight_delta_l2": task_delta,
            "task_weight_relative_delta": task_relative_delta,
            "discriminator_weight_delta_l2": discriminator_delta,
            "discriminator_weight_relative_delta": discriminator_relative_delta,
        }
        self.training_history.append(details)

        logger.info(
            "DANN ADAPTATION FINISHED | round=%d | active_model=M%d | "
            "elapsed_seconds=%.3f | encoder_relative_change=%s | "
            "task_relative_change=%s | discriminator_relative_change=%s",
            self.adaptation_round,
            self.model_version,
            elapsed_seconds,
            encoder_relative_delta,
            task_relative_delta,
            discriminator_relative_delta,
        )
        return details

    def predict(self, x):
        """Predict the label for a single input sample.

        Args:
            x (np.ndarray): Input feature vector.

        Returns:
            int: Predicted class label (0 or 1).

        Raises:
            RuntimeError: If model has not been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        preds_score = self.model.predict(x.reshape(1, -1), verbose=0)[:, 0]
        preds_labels = (preds_score > self.threshold).astype(int)[0]

        logger.debug("Prediction: score=%.4f label=%d", preds_score[0], preds_labels)

        return preds_labels
