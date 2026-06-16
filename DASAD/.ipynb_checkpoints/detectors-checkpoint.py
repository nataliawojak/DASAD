import numpy as np
from river.drift import PageHinkley
from DDDA.utils import kl_divergence

class BaseDriftDetector:
    """Abstract base class for drift detector.

    Defines the interface for a drift detection with `update` method.
    """ 
    def update(self, x):
        """Update detector with new observation.

        Args:
            x (np.ndarray): New incoming sample.

        Returns:
            bool: True if drift is detected, False otherwise.
        """
        raise NotImplementedError

class DriftDetector(BaseDriftDetector):
    """KL-divergence based drift detector using Page-Hinkley test.

    This detector builds a reference window from initial samples and
    monitors distribution changes using KL divergence. The Page-Hinkley
    test is applied to detect statistically significant drift.
    """


    def __init__(self,
                 reference_size=100,
                 window_size=1,
                 min_instances=10,
                 delta=0.05,
                 threshold=4,
                 alpha=0.9999):
        """Initialize drift detector.

        Args:
            reference_size (int, optional): Number of samples used to build
                initial reference distribution. Defaults to 100.

            window_size (int, optional): Number of incoming samples used
                to compute KL divergence. Defaults to 1.

            min_instances (int, optional): Minimum number of observations
                before Page-Hinkley starts detecting drift. Defaults to 10.

            delta (float, optional): Page-Hinkley sensitivity parameter.

            threshold (float, optional): Page-Hinkley drift detection threshold.
                Lower values detect drift earlier.

            alpha (float, optional): Forgetting factor for Page-Hinkley.
              
        """


        self.reference_size = reference_size
        self.window_size = window_size

        self.PageHinkley = PageHinkley(
            min_instances=min_instances,
            delta=delta,
            threshold=threshold,
            alpha=alpha
        )

        self.reference_window = []
        self.buffer = []

        self.initialized = False

    def update(self, x):
        """Update detector with new observation.

        Args:
            x (np.ndarray): New incoming sample.

        Returns:
            bool: True if drift is detected, False otherwise.
        """
        
        #Build initial reference
        if not self.initialized:
            self.reference_window.append(x)

            if len(self.reference_window) >= self.reference_size:
                self.initialized = True

            return False

      
        #Collect sample batch
        self.buffer.append(x)

        if len(self.buffer) < self.window_size:
            return False
            
        reference = np.array(self.reference_window)
        sample = np.array(self.buffer)

      
        #KL divergence 
        kl = kl_divergence(reference, sample)
        self.PageHinkley.update(kl)
        drift = self.PageHinkley.drift_detected


        #Update reference
        if drift:
            self.reference_window = list(sample)  
            
        else:
            self.reference_window.extend(sample)  

        #reset buffer
        self.buffer = []

        return drift