import logging
import numpy as np
from adapt.feature_based import DANN
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
        threshold=0.5):
        
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

        self.model = None  

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
            loss=BinaryFocalCrossentropy(
                gamma=self.gamma,
                apply_class_balancing=self.class_balance,
                alpha=self.alpha,
            ),
            optimizer=Adam(
                learning_rate=self.learning_rate,
                clipvalue=0.5,
            ),
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

        logger.info(
            "Training model | source=%s target=%s",
            source_data.shape,
            target_data.shape,
        )

        self.model = self._build_model()
        self.model.compile()

        self.model.fit(
            X=source_data,
            y=source_labels,
            Xt=target_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )

        logger.info("Training finished")

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

        preds_score = self.model.predict(x.reshape(1, -1))[:, 0]
        preds_labels = (preds_score > self.threshold).astype(int)[0]

        logger.debug("Prediction: score=%.4f label=%d", preds_score[0], preds_labels)

        return preds_labels



