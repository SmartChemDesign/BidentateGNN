"""
Applicability domain (AD) of the GCNN-FCNN ensemble in its latent space.

Latent space: the fused 272-dimensional representation after the pooling of the
molecular and the solvent branch (the input of the post-fusion FCNN). Every fold model
has its own latent space, so the AD is fitted per fold on the latent vectors of the
experimental records the fold model was trained on: the dimensions are standardised,
reduced by PCA keeping PCA_VARIANCE of the variance, and a Gaussian kernel density
estimate (KDE) is fitted in the PCA space with the bandwidth chosen by 5-fold
cross-validated likelihood. The AD score of a record is its negative log-density.
Thresholds are not set from the training scores (the training records define the
density) but from the validation records of the fold, which the fold model has not
seen: a record gets, per fold, the percentile of its score in the validation
distribution; the fold-averaged percentile is the final AD score, and the record is
inside the AD when it is below AD_COVERAGE.

The AD is validated on the test set: RMSE inside and outside the AD for the thresholds
0.95 and 0.90, RMSE of the records inside the AD of all fold models against the rest,
and RMSE in quartiles of the KDE percentile and of the ensemble spread (standard
deviation of the fold predictions). Query molecules (QUERY_CSV, columns smiles, solvent
(name or SMILES), redox_type and optionally E_exp) are scored in the same way.

Inputs: a train.py run (models, split_assignments.csv, scaler) and the dataset.
Writes to OUTPUT_DIR:
  ad_fit_summary.csv    per fold: sizes, PCA components, KDE bandwidth, threshold,
                        test inlier rate
  ad_test_scores.csv    per test record: score and percentile per fold, averaged
                        percentile, AD flags, ensemble mean, spread and error
  ad_summary.csv        test RMSE/MAE inside and outside the AD and by quartiles
  ad_query_scores.csv   the same scores for the query molecules
"""

import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

sys.path.append(os.path.abspath("."))

from rdkit import Chem

from Source.data import load_assignments, record_id
from Source.GCNN_FCNN.featurizers import featurize_sdf_with_solvent_all, load_solvent_db
from Source.GCNN_FCNN.model_redox import GCNN_FCNN
from Source.GCNN_FCNN.old_featurizer import ConvMolFeaturizer
from Source.trainer_redox import ModelShell


class LatentAD:
    """KDE in the standardised, PCA-reduced latent space, with validation-based percentiles."""

    def __init__(self, pca_variance=0.95, bandwidths=np.logspace(-1, 1, 25), seed=0):
        self.pca_variance = pca_variance
        self.bandwidths = bandwidths
        self.seed = seed

    def fit(self, x_train, x_val):
        self.scaler = StandardScaler().fit(x_train)
        p = self._project(x_train, fit_pca=True)
        search = GridSearchCV(KernelDensity(kernel="gaussian"), {"bandwidth": self.bandwidths},
                              cv=KFold(5, shuffle=True, random_state=self.seed)).fit(p)
        self.kde = search.best_estimator_
        self.bandwidth = search.best_params_["bandwidth"]
        self.n_components = self.pca.n_components_
        # Out-of-sample reference distribution of the score: the validation records
        self.reference = np.sort(self.scores(x_val))
        return self

    def _project(self, x, fit_pca=False):
        z = self.scaler.transform(x)
        if fit_pca:
            self.pca = PCA(n_components=self.pca_variance, svd_solver="full", random_state=self.seed).fit(z)
        return self.pca.transform(z)

    def scores(self, x):
        """Negative log-density of the records x (larger = further from the training data)."""
        return -self.kde.score_samples(self._project(x))

    def percentiles(self, x):
        """Percentile of the score in the validation distribution (0: typical, 1: beyond every validation record)."""
        return np.searchsorted(self.reference, self.scores(x), side="right") / len(self.reference)

    def threshold(self, coverage):
        return float(np.quantile(self.reference, coverage))


def latent_vectors(model, records, device, batch_size=256):
    """Fused latent vectors and predictions of a fold model for a list of records."""
    loader = DataLoader(records, batch_size=batch_size, shuffle=False)
    latents, predictions = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            solvent_x = batch.solvent_x.squeeze(1) if batch.solvent_x.dim() == 3 else batch.solvent_x
            solvent = model.solvent_fc_sequential(model._drop_data_type(solvent_x))
            latents.append(model.global_pooling(model.graph_sequential(batch), solvent).cpu().numpy())
            predictions.append(output["E"].reshape(-1).cpu().numpy())
    return np.vstack(latents), np.concatenate(predictions)


def read_queries(path):
    """Query table with the columns smiles, solvent, redox_type and (optional) E_exp; ',' or ';' separated."""
    with open(path, encoding="utf-8") as f:
        header = f.readline()
    table = pd.read_csv(path, sep=";" if header.count(";") > header.count(",") else ",")
    table.columns = [c.strip().lower() for c in table.columns]
    table = table.rename(columns={"e": "e_exp"})
    missing = [name for name in ("smiles", "solvent", "redox_type") if name not in table.columns]
    if missing:
        raise ValueError(f"{path}: columns {missing} not found; columns present: {list(table.columns)}")
    return table


def featurize_queries(query_table, solvent_db_path, scaler, mol_featurizer):
    """Records of the query molecules in the format of the training data (experimental flag set)."""
    solvent_db, name_to_smiles = load_solvent_db(solvent_db_path)
    name_to_smiles = {name.lower(): smiles for name, smiles in name_to_smiles.items()}
    records = []
    for _, row in query_table.iterrows():
        solvent = str(row["solvent"]).strip()
        solvent_smiles = name_to_smiles.get(solvent.lower()) or Chem.MolToSmiles(Chem.MolFromSmiles(solvent))
        redox_type = "ox" if str(row["redox_type"]).lower().startswith("ox") else "red"
        descriptors = scaler.transform([solvent_db[solvent_smiles]])[0].tolist()
        data = mol_featurizer.featurize(Chem.MolFromSmiles(row["smiles"]))
        data.solvent_x = torch.tensor(descriptors + [1 if redox_type == "ox" else 0, 1], dtype=torch.float32).unsqueeze(0)
        data.y = {"E": torch.tensor([[float(row.get("e_exp", np.nan))]])}
        data.smiles = Chem.MolToSmiles(Chem.MolFromSmiles(row["smiles"]))
        data.solvent_smiles = solvent_smiles
        data.redox_type = redox_type
        data.data_type = "exp"
        records.append(data)
    return records


def error_stats(name, errors):
    return {"subset": name, "n": int(len(errors)), "rmse": np.sqrt(np.mean(errors ** 2)) if len(errors) else np.nan,
            "mae": np.mean(np.abs(errors)) if len(errors) else np.nan}


def ad_summary(table, coverages, n_folds):
    """Test errors inside and outside the AD and in quartiles of the two reliability indicators."""
    rows = []
    for coverage in coverages:
        inside = table["kde_percentile"] <= coverage
        rows.append(error_stats(f"inside AD, threshold {coverage}", table.loc[inside, "error"].values))
        rows.append(error_stats(f"outside AD, threshold {coverage}", table.loc[~inside, "error"].values))
    all_folds = table["kde_inlier_folds"] == n_folds
    rows.append(error_stats("inside AD of all fold models", table.loc[all_folds, "error"].values))
    rows.append(error_stats("outside AD of at least one fold model", table.loc[~all_folds, "error"].values))
    for indicator in ("kde_percentile", "ensemble_sd"):
        quartile = pd.qcut(table[indicator].rank(method="first"), 4, labels=False)
        for q in range(4):
            rows.append(error_stats(f"{indicator}, quartile {q + 1}", table.loc[quartile == q, "error"].values))
    return pd.DataFrame(rows)


def main():
    path_to_sdf = "Data/redox/dataset/calc_exp/exp_calc_dataset.sdf"
    solvent_db_path = "Data/redox/additional/solvent_properties.csv"
    run_dir = Path("Output/redox/GCNN_FCNN/exp_calc_multifidelity_molsplit_seed42")
    output_dir = run_dir / "applicability_domain"
    query_csv = "Data/redox/additional/ad_query_molecules.csv"
    pca_variance = 0.95
    ad_coverage = 0.95
    coverages = (0.95, 0.90)
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "ad.log", mode="w")])

    all_data = featurize_sdf_with_solvent_all(path_to_sdf=path_to_sdf, mol_featurizer=ConvMolFeaturizer(),
                                              solvent_db_path=solvent_db_path, scale_solvent=True, seed=seed,
                                              shuffle=False)
    assignments = load_assignments(run_dir / "split_assignments.csv")
    reference = [dp for dp in all_data if dp.data_type == "exp" and assignments[record_id(dp)].startswith("fold_")]
    test = [dp for dp in all_data if dp.data_type == "exp" and assignments[record_id(dp)] == "test"]
    shell = ModelShell(GCNN_FCNN, run_dir, device=device)
    targets = np.array([float(dp.y["E"].reshape(-1)[0]) for dp in test])
    query_table = read_queries(query_csv) if Path(query_csv).exists() else None
    queries = (featurize_queries(query_table, solvent_db_path, joblib.load(run_dir / "scaler.joblib"), ConvMolFeaturizer())
               if query_table is not None else [])

    fit_rows, test_percentiles, test_predictions, query_percentiles, query_predictions = [], [], [], [], []
    table = pd.DataFrame({"smiles": [dp.smiles for dp in test], "solvent_smiles": [dp.solvent_smiles for dp in test],
                          "redox_type": [dp.redox_type for dp in test], "target": targets})
    for k, model in enumerate(shell.models, start=1):
        train = [dp for dp in reference if assignments[record_id(dp)] != f"fold_{k}"]
        val = [dp for dp in reference if assignments[record_id(dp)] == f"fold_{k}"]
        x_train, _ = latent_vectors(model, train, device)
        x_val, _ = latent_vectors(model, val, device)
        x_test, prediction = latent_vectors(model, test, device)
        fit = LatentAD(pca_variance, seed=seed).fit(x_train, x_val)

        table[f"kde_score_fold{k}"] = fit.scores(x_test)
        table[f"kde_percentile_fold{k}"] = fit.percentiles(x_test)
        test_percentiles.append(table[f"kde_percentile_fold{k}"].values)
        test_predictions.append(prediction)
        if queries:
            x_query, query_prediction = latent_vectors(model, queries, device)
            query_percentiles.append(fit.percentiles(x_query))
            query_predictions.append(query_prediction)
        row = {"fold": k, "n_train": len(train), "n_val": len(val), "latent_dim": x_train.shape[1],
               "pca_components": fit.n_components, "kde_bandwidth": fit.bandwidth,
               "threshold": fit.threshold(ad_coverage),
               "test_inlier_rate": float(np.mean(table[f"kde_percentile_fold{k}"] <= ad_coverage))}
        fit_rows.append(row)
        logging.info("Fold %d: %s", k, row)

    test_predictions = np.array(test_predictions)
    table["prediction"] = test_predictions.mean(axis=0)
    table["ensemble_sd"] = test_predictions.std(axis=0, ddof=1)
    table["error"] = table["prediction"] - table["target"]
    table["kde_percentile"] = np.mean(test_percentiles, axis=0)
    table["kde_inlier"] = table["kde_percentile"] <= ad_coverage
    table["kde_inlier_folds"] = np.sum([p <= ad_coverage for p in test_percentiles], axis=0)
    table.to_csv(output_dir / "ad_test_scores.csv", index=False, float_format="%.5f")
    pd.DataFrame(fit_rows).to_csv(output_dir / "ad_fit_summary.csv", index=False, float_format="%.5f")

    summary = ad_summary(table, coverages, len(shell.models))
    summary.to_csv(output_dir / "ad_summary.csv", index=False, float_format="%.4f")
    logging.info("Test errors by AD:\n%s", summary.to_string(index=False))

    if queries:
        query_predictions = np.array(query_predictions)
        query_out = query_table.copy()
        query_out["canonical_smiles"] = [dp.smiles for dp in queries]
        query_out["prediction"] = query_predictions.mean(axis=0)
        query_out["ensemble_sd"] = query_predictions.std(axis=0, ddof=1)
        if "e_exp" in query_out.columns:
            query_out["error"] = query_out["prediction"] - query_out["e_exp"]
        query_out["kde_percentile"] = np.mean(query_percentiles, axis=0)
        query_out["kde_inlier"] = query_out["kde_percentile"] <= ad_coverage
        query_out["kde_inlier_folds"] = np.sum([p <= ad_coverage for p in query_percentiles], axis=0)
        query_out.to_csv(output_dir / "ad_query_scores.csv", index=False, float_format="%.4f")
        logging.info("Query molecules:\n%s", query_out.to_string())


if __name__ == "__main__":
    main()