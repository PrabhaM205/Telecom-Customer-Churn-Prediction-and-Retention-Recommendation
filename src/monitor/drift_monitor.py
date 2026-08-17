import numpy as np
import pandas as pd


class DriftMonitor:
    """
    Detects distribution drift between reference/training data
    and incoming/new data using Population Stability Index (PSI).
    """

    def __init__(self, threshold=0.20):
        """
        Parameters
        ----------
        threshold : float
            PSI threshold above which drift is considered significant.

            PSI < 0.10
                Little or no drift

            0.10 <= PSI < 0.20
                Moderate drift

            PSI >= 0.20
                Significant drift
        """

        self.threshold = threshold

    # ============================================================
    # PSI CALCULATION
    # ============================================================

    def calculate_psi(
        self,
        reference,
        current,
        bins=10
    ):
        """
        Calculate Population Stability Index (PSI).

        Handles:
        - numeric values
        - numeric strings
        - blank strings
        - whitespace
        - NaN values
        - invalid numeric values

        Invalid numeric values are converted to NaN
        and removed before calculation.
        """

        # --------------------------------------------------------
        # Convert reference data to numeric
        # --------------------------------------------------------

        reference = pd.to_numeric(
            pd.Series(reference),
            errors="coerce"
        )

        # --------------------------------------------------------
        # Convert current data to numeric
        # --------------------------------------------------------

        current = pd.to_numeric(
            pd.Series(current),
            errors="coerce"
        )

        # --------------------------------------------------------
        # Remove invalid / missing values
        # --------------------------------------------------------

        reference = reference.dropna()
        current = current.dropna()

        # --------------------------------------------------------
        # Validate data
        # --------------------------------------------------------

        if len(reference) == 0:
            raise ValueError(
                "Reference data cannot be empty "
                "after numeric conversion."
            )

        if len(current) == 0:
            raise ValueError(
                "Current data cannot be empty "
                "after numeric conversion."
            )

        # --------------------------------------------------------
        # Create bins from reference distribution
        # --------------------------------------------------------

        breakpoints = np.percentile(
            reference,
            np.linspace(
                0,
                100,
                bins + 1
            )
        )

        # --------------------------------------------------------
        # Remove duplicate breakpoints
        # --------------------------------------------------------

        breakpoints = np.unique(
            breakpoints
        )

        # --------------------------------------------------------
        # Handle features with too few unique values
        # --------------------------------------------------------

        if len(breakpoints) < 3:
            return 0.0

        # --------------------------------------------------------
        # Expand first and last boundaries
        # --------------------------------------------------------

        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        # --------------------------------------------------------
        # Create reference bins
        # --------------------------------------------------------

        reference_bins = pd.cut(
            reference,
            bins=breakpoints,
            include_lowest=True
        )

        # --------------------------------------------------------
        # Create current bins
        # --------------------------------------------------------

        current_bins = pd.cut(
            current,
            bins=breakpoints,
            include_lowest=True
        )

        # --------------------------------------------------------
        # Count values in each bin
        # --------------------------------------------------------

        reference_counts = (
            reference_bins
            .value_counts(sort=False)
        )

        current_counts = (
            current_bins
            .value_counts(sort=False)
        )

        # --------------------------------------------------------
        # Convert counts to percentages
        # --------------------------------------------------------

        reference_percent = (
            reference_counts
            / len(reference)
        )

        current_percent = (
            current_counts
            / len(current)
        )

        # --------------------------------------------------------
        # Avoid division by zero
        # --------------------------------------------------------

        epsilon = 1e-6

        reference_percent = (
            reference_percent
            .clip(lower=epsilon)
        )

        current_percent = (
            current_percent
            .clip(lower=epsilon)
        )

        # --------------------------------------------------------
        # Calculate PSI
        # --------------------------------------------------------

        psi = np.sum(
            (
                current_percent
                - reference_percent
            )
            *
            np.log(
                current_percent
                / reference_percent
            )
        )

        return float(psi)

    # ============================================================
    # FEATURE DRIFT
    # ============================================================

    def check_feature_drift(
        self,
        reference_df,
        current_df,
        features
    ):
        """
        Calculate PSI for multiple features.

        Returns
        -------
        dict
            Contains PSI and drift status for every feature.
        """

        results = {}

        # --------------------------------------------------------
        # Check each feature
        # --------------------------------------------------------

        for feature in features:

            # ----------------------------------------------------
            # Reference feature missing
            # ----------------------------------------------------

            if feature not in reference_df.columns:

                results[feature] = {
                    "psi": None,
                    "drift": False,
                    "status": "missing_reference_feature"
                }

                continue

            # ----------------------------------------------------
            # Current feature missing
            # ----------------------------------------------------

            if feature not in current_df.columns:

                results[feature] = {
                    "psi": None,
                    "drift": False,
                    "status": "missing_current_feature"
                }

                continue

            # ----------------------------------------------------
            # Calculate PSI
            # ----------------------------------------------------

            try:

                psi = self.calculate_psi(
                    reference_df[feature],
                    current_df[feature]
                )

                results[feature] = {
                    "psi": round(
                        psi,
                        4
                    ),
                    "drift": psi >= self.threshold,
                    "status": (
                        "drift_detected"
                        if psi >= self.threshold
                        else "stable"
                    )
                }

            except ValueError as error:

                results[feature] = {
                    "psi": None,
                    "drift": False,
                    "status": str(error)
                }

        return results

    # ============================================================
    # OVERALL DRIFT CHECK
    # ============================================================

    def has_drift(
        self,
        reference_df,
        current_df,
        features
    ):
        """
        Returns True if at least one feature
        has significant drift.
        """

        results = self.check_feature_drift(
            reference_df,
            current_df,
            features
        )

        return any(
            result["drift"]
            for result in results.values()
        )


# ================================================================
# DIRECT TEST
# ================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DRIFT MONITOR MODULE TEST")
    print("=" * 60)

    # ------------------------------------------------------------
    # Example reference data
    # ------------------------------------------------------------

    reference_data = pd.Series([
        10,
        12,
        15,
        18,
        20,
        22,
        25,
        27,
        30,
        32
    ])

    # ------------------------------------------------------------
    # Example current data
    # ------------------------------------------------------------

    current_data = pd.Series([
        11,
        13,
        16,
        19,
        21,
        23,
        26,
        28,
        31,
        33
    ])

    # ------------------------------------------------------------
    # Create monitor
    # ------------------------------------------------------------

    monitor = DriftMonitor(
        threshold=0.20
    )

    # ------------------------------------------------------------
    # Calculate PSI
    # ------------------------------------------------------------

    psi = monitor.calculate_psi(
        reference_data,
        current_data
    )

    print("\nReference data:")
    print(reference_data.tolist())

    print("\nCurrent data:")
    print(current_data.tolist())

    print(
        f"\nPSI: {psi:.4f}"
    )

    print(
        f"Drift detected: "
        f"{psi >= monitor.threshold}"
    )

    print("\n" + "=" * 60)