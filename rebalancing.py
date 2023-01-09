# Provided as open source under the MIT License. Originally based on 
# thresholder.py from sklego.meta from the sklego package, also 
# distrib. under the MIT License. Depends on sklego which is in PyPi.
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
)
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from sklego.base import ProbabilisticClassifier

class BenefitRebalancingClassifier(BaseEstimator, ClassifierMixin):
    """
    Meta-estimator that takes an original base model (probabilistic classifier, multi-class
    or binary) and returns a variant of it whose predict method will predict the class
    having the greatest expected benefit rather than merely the greatest probability, 
    without needing to refit the base model if already fitted.  The predict_proba is unaffected.
    
    Does not support multi-label classification. Does not yet implement the score method, 
    but when it does it will probably
    use the balanced accuracy for scoring rather then the ordinary accuracy.
    Assumes that every class does indeed occur at least once in the training dataset.
    Does not yet take sample_weight nor class_weight into account, so avoid using them;
    anyway this meta-estimator is intended to be used instead of such weights, not
    in addition to them.
    
    :param model: 
        The base classifier model
    :param benefit_per_class: array-like (list is okay) with shape (n_classes,) or 'balanced' (default) or None
        The total assumed relative* potential** benefit PER CLASS of correct classification.
        The total is over all samples of a given class in the training dataset.
        If an explicit sequence is provided, its element 0 applies to class 0... 
        and element n_classes-1 applies to class (n_classes-1).
        
        Default value of 'balanced' means use a sequence whose elements all have equal
        value, corresponding to targeting the balanced accuracy rather than accuracy.  
        
        Value of None means simply use the uniform benefit PER SAMPLE (not per
        class!) just as the original base model by itself would do implicitly, corresponding
        to targeting the ordinary classification accuracy.
        
        *By "relative" we mean that you can include any overall scale factor applying to
        all the elements of this array and the effect will be the same.  For example
        for four-class, the value [1, 1, 1, 1] is equivalent to [37.2, 37.2, 37.2, 37.2]
        and both are equivalent to the default 'balanced';
        or for three-class, [4, 10, 6] is equivalent to [2, 5, 3].
        
        **We call this benefit "potential" because the benefit is only actually obtained
        for a given sample when the predicted class correctly matches the true class.
    :param refit: 
        If True, we will always refit the base model even if it is already fitted.
        If False we will only fit the base model if it isn't already fitted.
    """

    def __init__(self, model, *, benefit_per_class='balanced', refit=False):
        self.model = model
        self.benefit_per_class = benefit_per_class
        self.refit = refit

    def _handle_refit(self, X, y, sample_weight=None):
        """Only refit when we need to, unless refit=True is present."""
        if self.refit:
            self.estimator_ = clone(self.model)
            self.estimator_.fit(X, y, sample_weight=sample_weight)
        else:
            try:
                _ = self.estimator_.predict(X[:1])
            except NotFittedError:
                self.estimator_.fit(X, y, sample_weight=sample_weight)

    def fit(self, X, y, sample_weight=None):
        """
        Fit the data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :param sample_weight: array-like, shape=(n_samples) Individual weights for each sample.
        :return: Returns self
        """
        self.estimator_ = self.model
        if not isinstance(self.estimator_, ProbabilisticClassifier):
            raise ValueError(
                "The {self.name} meta model only works on classification models with .predict_proba."
            )
        self._handle_refit(X, y, sample_weight)
        self.training_prevalence = pd.Series(y).value_counts().sort_index() # but what if some classes not found in train set??
        n_classes = len(self.training_prevalence)
        if str(self.benefit_per_class) == 'balanced':
            self.benefit_per_class = self.training_prevalence*0 + 1.0/n_classes
        elif self.benefit_per_class is None:
            self.benefit_per_class = self.training_prevalence
        self.benefit_per_class = np.array(self.benefit_per_class) # convert list or series to array
        self.benefit_per_sample = self.benefit_per_class / self.training_prevalence # length n_classes
        self.benefit_per_sample = np.array(self.benefit_per_sample) # cnvt list or series to array
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["classes_", "estimator_"])
        return self.estimator_.predict_proba(X) # unmodified probability preds of base classifier

    def predict(self, X):
        """
        Predict new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        check_is_fitted(self, ["classes_", "estimator_"])
        proba = self.predict_proba(X)
        # Correct pred'd prob distrib on each row from training_prevalence to pred'n prevalence:
        eb = proba * self.benefit_per_sample # expected benefit: mult each of the n_samples rows by same benefit
        #eb = np.transpose(np.transpose(eb) / np.sum(eb, axis=1)) # normalization unnecessary
        return self.classes_[np.argmax(eb, axis=1)]

    #def score(self, X, y):    # should override so it uses self.predict() , not self.estimator_.predict()
    #    return self.estimator_.score(X, y)   # also default scoring should be by balanced accur
