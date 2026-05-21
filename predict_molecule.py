import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(SCRIPT_DIR, "molecule_data_with_embeddings.csv")
GNN_MODEL_PATH = os.path.join(SCRIPT_DIR, "gnn_embedding_model.pt")
PREDICTOR_PATH = os.path.join(SCRIPT_DIR, "property_predictor.joblib")
SCALER_X_PATH = os.path.join(SCRIPT_DIR, "scaler_X.joblib")
SCALER_Y_PATH = os.path.join(SCRIPT_DIR, "scaler_y.joblib")

EMBEDDING_DIM = 32
EMBEDDING_COLS = [f"emb_{i}" for i in range(EMBEDDING_DIM)]

TARGET_COLS = [
    "dipole_moment",
    "polarizability",
    "homo_energy",
    "lumo_energy",
    "homo_lumo_gap",
    "spatial_extent",
    "zero_point_energy",
    "internal_energy_0K",
    "heat_capacity",
    "internal_energy_298K",
    "enthalpy_298K",
    "free_energy_298K",
]

TARGET_UNITS = {
    "dipole_moment": "Debye",
    "polarizability": "Bohr^3",
    "homo_energy": "Hartree",
    "lumo_energy": "Hartree",
    "homo_lumo_gap": "Hartree",
    "spatial_extent": "Bohr^2",
    "zero_point_energy": "Hartree",
    "internal_energy_0K": "Hartree",
    "heat_capacity": "cal/mol*K",
    "internal_energy_298K": "Hartree",
    "enthalpy_298K": "Hartree",
    "free_energy_298K": "Hartree",
}



BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 1.0,
    Chem.rdchem.BondType.DOUBLE: 2.0,
    Chem.rdchem.BondType.TRIPLE: 3.0,
    Chem.rdchem.BondType.AROMATIC: 1.5,
}


def atom_to_features(atom):
    hybridization_map = {
        Chem.rdchem.HybridizationType.SP: 1.0,
        Chem.rdchem.HybridizationType.SP2: 2.0,
        Chem.rdchem.HybridizationType.SP3: 3.0,
    }
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        1.0 if atom.GetIsAromatic() else 0.0,
        hybridization_map.get(atom.GetHybridization(), 0.0),
        float(atom.GetMass()),
    ]


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    node_features = [atom_to_features(a) for a in mol.GetAtoms()]
    if not node_features:
        return None

    x = torch.tensor(node_features, dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = BOND_TYPE_MAP.get(bond.GetBondType(), 0.0)
        edge_index += [[i, j], [j, i]]
        edge_attr += [[bt], [bt]]

    if not edge_index:
        return None

    return Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )


class MolecularGNN(torch.nn.Module):

    def __init__(self, input_dim=7, hidden_dim=128, embedding_dim=32,
                 num_targets=12, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, 64)
        self.dropout = torch.nn.Dropout(dropout)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.embed_layer = torch.nn.Linear(64, embedding_dim)
        self.prediction_head = torch.nn.Linear(embedding_dim, num_targets)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)
        embedding = F.relu(self.embed_layer(x))
        prediction = self.prediction_head(embedding)
        return prediction, embedding


def load_gnn_model():
    model = MolecularGNN(
        input_dim=7, hidden_dim=128,
        embedding_dim=EMBEDDING_DIM, num_targets=len(TARGET_COLS),
    )
    state = torch.load(GNN_MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def get_embedding(model, smiles):
    graph = smiles_to_graph(smiles)
    if graph is None:
        return None

    batch = Batch.from_data_list([graph])
    with torch.no_grad():
        _, emb = model(batch.x, batch.edge_index, batch.batch)
    return emb[0].cpu().numpy()


def train_predictor():
    df = pd.read_csv(DATASET_PATH)
    print(f"   Dataset: {df.shape[0]} molecules, {df.shape[1]} columns")
    missing = [c for c in EMBEDDING_COLS if c not in df.columns]
    if missing:
        print(f"   ERROR: Missing embedding columns: {missing}")
        return None

    missing_t = [c for c in TARGET_COLS if c not in df.columns]
    if missing_t:
        print(f"   ERROR: Missing target columns: {missing_t}")
        return None
    X = df[EMBEDDING_COLS].values
    y = df[TARGET_COLS].values

    print(f"   X shape: {X.shape}  (embeddings)")
    print(f"   y shape: {y.shape}  (targets)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[SPLIT] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    model = MLPRegressor(
        hidden_layer_sizes=(256, 256, 128),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=True,
        batch_size=512,
    )

    model.fit(X_train_s, y_train_s)

    y_pred_s = model.predict(X_test_s)
    y_pred = scaler_y.inverse_transform(y_pred_s)

    print(f"\n{'Property':<25s} {'MAE':>10s} {'R2':>10s}")
    print("-" * 47)
    for i, col in enumerate(TARGET_COLS):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        unit = TARGET_UNITS.get(col, "")
        print(f"   {col:<23s} {mae:10.4f} {r2:10.4f}  ({unit})")

    overall_r2 = r2_score(y_test, y_pred, multioutput="uniform_average")
    print(f"\n   Overall R2 (avg): {overall_r2:.4f}")

    joblib.dump(model, PREDICTOR_PATH)
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    print(f"   predictor : {PREDICTOR_PATH}")
    print(f"   Scaler X  : {SCALER_X_PATH}")
    print(f"   Scaler y  : {SCALER_Y_PATH}")

    return model, scaler_X, scaler_y


def load_predictor():
    if not all(os.path.exists(p) for p in [PREDICTOR_PATH, SCALER_X_PATH, SCALER_Y_PATH]):
        result = train_predictor()
        if result is None:
            raise RuntimeError("Failed to train predictor.")
        return result

    model = joblib.load(PREDICTOR_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    return model, scaler_X, scaler_y


def predict_properties(smiles, gnn_model=None, predictor=None,
                       scaler_X=None, scaler_y=None):
    if gnn_model is None:
        gnn_model = load_gnn_model()
    if predictor is None or scaler_X is None or scaler_y is None:
        predictor, scaler_X, scaler_y = load_predictor()

    embedding = get_embedding(gnn_model, smiles)
    if embedding is None:
        print(f"[ERROR] Invalid SMILES: '{smiles}'")
        return None

    emb_scaled = scaler_X.transform(embedding.reshape(1, -1))
    pred_scaled = predictor.predict(emb_scaled)
    pred_values = scaler_y.inverse_transform(pred_scaled)[0]

    results = {}
    for i, col in enumerate(TARGET_COLS):
        results[col] = float(pred_values[i])

    return results


def display_prediction(smiles, results):
    mol = Chem.MolFromSmiles(smiles)
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol) if mol else "Unknown"
    print(f"  SMILES:   {smiles}")
    print(f"  Formula:  {formula}")
    print(f"  {'Property':<25s} {'Value':>12s}  {'Unit':<12s}")

    for prop, value in results.items():
        unit = TARGET_UNITS.get(prop, "")
        print(f"  {prop:<25s} {value:>12.6f}  {unit:<12s}")



def main():
    parser = argparse.ArgumentParser(
        description="Predict molecular properties from SMILES"
    )
    parser.add_argument(
        "--smiles", type=str, default=None,
        help="SMILES string to predict (e.g. 'CCO' for ethanol)"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Retrain the property predictor model"
    )
    args = parser.parse_args()

    if args.train:
        train_predictor()
        return


    gnn_model = load_gnn_model()
    predictor, scaler_X, scaler_y = load_predictor()
    if args.smiles:
        results = predict_properties(
            args.smiles, gnn_model, predictor, scaler_X, scaler_y
        )
        if results:
            display_prediction(args.smiles, results)
        return

    while True:
        try:
            smiles = input("\n> Enter SMILES: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not smiles or smiles.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        results = predict_properties(
            smiles, gnn_model, predictor, scaler_X, scaler_y
        )
        if results:
            display_prediction(smiles, results)


if __name__ == "__main__":
    main()