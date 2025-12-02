#!/usr/bin/env python

import logging
import os

import joblib

try:
    from importlib.resources import files
except ImportError:
    # Fallback for Python < 3.9
    from importlib_resources import files

from .utils import REPO_BASE_URL, download_file


class SoftNewsModel:
    MODELFN = None
    VECTFN = None

    @classmethod
    def load_model_data(cls, latest: bool = False):
        if cls.MODELFN:
            model_fn = str(files("notnews").joinpath(cls.MODELFN))
            path = os.path.dirname(model_fn)
            if not os.path.exists(path):
                os.makedirs(path)
            if not os.path.exists(model_fn) or latest:
                logging.info(f"Downloading model data from the server ({model_fn})...")
                if not download_file(REPO_BASE_URL + cls.MODELFN, model_fn):
                    logging.info("ERROR: Cannot download model data file")
                    return None, None
            else:
                print(f"Using model data from {model_fn}...")
        if cls.VECTFN:
            vect_fn = str(files("notnews").joinpath(cls.VECTFN))
            path = os.path.dirname(vect_fn)
            if not os.path.exists(path):
                os.makedirs(path)
            if not os.path.exists(vect_fn) or latest:
                logging.info(
                    f"Downloading vectorizer data from the server ({vect_fn})..."
                )
                if not download_file(REPO_BASE_URL + cls.VECTFN, vect_fn):
                    logging.info("ERROR: Cannot download vectorizer data file")
                    return None, None
            else:
                logging.info(f"Using vectorizer data from {vect_fn}...")

        logging.info("Loading the model and vectorizer data file...")

        model = joblib.load(model_fn)
        vect = joblib.load(vect_fn)

        # Fix vectorizer compatibility issues
        _fix_vectorizer_compatibility(vect)

        # sklearn compatibility fix: models saved with sklearn 0.22 use base_estimator
        # but sklearn 1.3+ expects estimator attribute
        try:
            from sklearn.calibration import CalibratedClassifierCV

            if isinstance(model, CalibratedClassifierCV):
                # Fix the main CalibratedClassifierCV object
                if hasattr(model, "base_estimator") and not hasattr(model, "estimator"):
                    model.estimator = model.base_estimator
                    logging.debug(
                        "Applied sklearn compatibility fix for CalibratedClassifierCV"
                    )

                # Fix each individual _CalibratedClassifier in calibrated_classifiers_
                if hasattr(model, "calibrated_classifiers_"):
                    for calibrated_clf in model.calibrated_classifiers_:
                        # Fix estimator attribute
                        if hasattr(calibrated_clf, "base_estimator") and not hasattr(
                            calibrated_clf, "estimator"
                        ):
                            calibrated_clf.estimator = calibrated_clf.base_estimator
                            logging.debug(
                                "Applied sklearn compatibility fix for _CalibratedClassifier estimator"
                            )

                        # Fix calibrators attribute (calibrators -> calibrators_)
                        if hasattr(calibrated_clf, "calibrators_") and not hasattr(
                            calibrated_clf, "calibrators"
                        ):
                            calibrated_clf.calibrators = calibrated_clf.calibrators_
                            logging.debug(
                                "Applied sklearn compatibility fix for _CalibratedClassifier calibrators"
                            )

                        # Fix IsotonicRegression compatibility for models trained with sklearn 0.22
                        if hasattr(calibrated_clf, "calibrators"):
                            for calibrator in calibrated_clf.calibrators:
                                _fix_isotonic_regression_compatibility(calibrator)
        except (ImportError, AttributeError):
            # If sklearn is not available or missing attributes, continue without patching
            pass

        return model, vect


def _fix_isotonic_regression_compatibility(obj):
    """Fix IsotonicRegression objects for compatibility with newer sklearn versions.

    Models trained with sklearn 0.22 used _necessary_X_ and _necessary_y_ attributes,
    while sklearn 0.24+ uses X_thresholds_ and y_thresholds_. This function handles
    the attribute name change and rebuilds the f_ interpolation function if missing.
    """
    try:
        from sklearn.isotonic import IsotonicRegression

        if isinstance(obj, IsotonicRegression):
            # Handle attribute name change: _necessary_X_ -> X_thresholds_
            if hasattr(obj, "_necessary_X_") and not hasattr(obj, "X_thresholds_"):
                obj.X_thresholds_ = obj._necessary_X_
                logging.debug(
                    "Applied sklearn compatibility fix: _necessary_X_ -> X_thresholds_"
                )

            # Handle attribute name change: _necessary_y_ -> y_thresholds_
            if hasattr(obj, "_necessary_y_") and not hasattr(obj, "y_thresholds_"):
                obj.y_thresholds_ = obj._necessary_y_
                logging.debug(
                    "Applied sklearn compatibility fix: _necessary_y_ -> y_thresholds_"
                )

            # Rebuild f_ interpolation function if missing but data is available
            if (
                not hasattr(obj, "f_")
                and hasattr(obj, "X_thresholds_")
                and hasattr(obj, "y_thresholds_")
            ):
                obj._build_f(obj.X_thresholds_, obj.y_thresholds_)
                logging.debug(
                    "Applied sklearn compatibility fix: rebuilt f_ interpolation function"
                )

    except (ImportError, AttributeError):
        # If sklearn is not available or missing attributes, continue without patching
        pass


def _fix_vectorizer_compatibility(vect):
    """Fix TfidfVectorizer objects for compatibility with newer sklearn versions.

    This function handles compatibility issues with TfidfVectorizer objects
    that may arise when loading models trained with older sklearn versions.
    """
    try:
        from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

        if isinstance(vect, TfidfVectorizer):
            # Check if the internal TfidfTransformer is missing fitted attributes
            if hasattr(vect, "_tfidf") and isinstance(vect._tfidf, TfidfTransformer):
                transformer = vect._tfidf

                # Copy fitted attributes from vectorizer to transformer for compatibility
                # This is needed because sklearn 0.22 -> 1.5+ unpickling breaks the internal state
                if hasattr(vect, "idf_") and not hasattr(transformer, "idf_"):
                    transformer.idf_ = vect.idf_
                    logging.debug(
                        "Applied sklearn compatibility fix: copied idf_ to TfidfTransformer"
                    )

                # Copy other essential fitted attributes
                if hasattr(vect, "vocabulary_") and not hasattr(
                    transformer, "vocabulary_"
                ):
                    transformer.vocabulary_ = vect.vocabulary_
                    logging.debug(
                        "Applied sklearn compatibility fix: copied vocabulary_ to TfidfTransformer"
                    )

                # Set norm and sublinear_tf attributes that may be missing
                if hasattr(vect, "norm") and not hasattr(transformer, "norm"):
                    transformer.norm = vect.norm

                if hasattr(vect, "sublinear_tf") and not hasattr(
                    transformer, "sublinear_tf"
                ):
                    transformer.sublinear_tf = vect.sublinear_tf

    except (ImportError, AttributeError):
        # If sklearn is not available or missing attributes, continue without patching
        pass
