import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool



ATOM_FEATURES = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53,
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

    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(atom_to_features(atom))

    if len(node_features) == 0:
        return None

    x = torch.tensor(node_features, dtype=torch.float)

    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BOND_TYPE_MAP.get(bond.GetBondType(), 0.0)

        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append([bond_type])
        edge_attr.append([bond_type])

    if len(edge_index) == 0:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class MolecularGNN(torch.nn.Module):

    def __init__(self, input_dim=7, hidden_dim=128, embedding_dim=32, num_targets=12, dropout=0.2):
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
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        embedding = self.embed_layer(x)
        embedding = F.relu(embedding)

        prediction = self.prediction_head(embedding)

        return prediction, embedding

    def get_embedding(self, x, edge_index, batch):
        with torch.no_grad():
            _, embedding = self.forward(x, edge_index, batch)
        return embedding

def prepare_dataset(df, target_columns):
    dataset = []
    failed = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building graphs"):
        graph = smiles_to_graph(row['smiles'])

        if graph is None:
            failed.append(idx)
            continue

        targets = [float(row[col]) for col in target_columns]
        graph.y = torch.tensor([targets], dtype=torch.float)
        graph.idx = idx 

        dataset.append(graph)

    return dataset, failed

def train_model(model, dataset, epochs=30, batch_size=256, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    model.train()

    n = len(dataset)
    history = []

    for epoch in range(epochs):
        indices = torch.randperm(n).tolist()
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            batch_data = [dataset[i] for i in batch_indices]
            batch = Batch.from_data_list(batch_data)

            optimizer.zero_grad()
            pred, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step(avg_loss)
        history.append(avg_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")

    return history

def extract_embeddings(model, dataset, batch_size=512):
    model.eval()
    embeddings = {}

    with torch.no_grad():
        for start in tqdm(range(0, len(dataset), batch_size), desc="Extracting embeddings"):
            end = min(start + batch_size, len(dataset))
            batch_data = dataset[start:end]
            batch = Batch.from_data_list(batch_data)

            _, emb = model(batch.x, batch.edge_index, batch.batch)

            for i, data_obj in enumerate(batch_data):
                embeddings[data_obj.idx] = emb[i].cpu().numpy().tolist()

    return embeddings


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_CSV = os.path.join(SCRIPT_DIR, "molecule_data.csv")
    OUTPUT_CSV = os.path.join(SCRIPT_DIR, "molecule_data_with_embeddings.csv")
    MODEL_PATH = os.path.join(SCRIPT_DIR, "gnn_embedding_model.pt")

    EMBEDDING_DIM = 32      
    EPOCHS = 30             
    BATCH_SIZE = 256        
    LEARNING_RATE = 0.001

    print(f"\nLoading data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"   Loaded {len(df)} molecules with {df.shape[1]} features")
    print(f"   Columns: {list(df.columns)}")

    target_columns = [
        'dipole_moment', 'polarizability', 'homo_energy', 'lumo_energy',
        'homo_lumo_gap', 'spatial_extent', 'zero_point_energy',
        'internal_energy_0K', 'heat_capacity', 'internal_energy_298K',
        'enthalpy_298K', 'free_energy_298K'
    ]
    num_targets = len(target_columns)

    dataset, failed = prepare_dataset(df, target_columns)
    print(f"    Successfully converted: {len(dataset)}")
    print(f"    Failed/skipped: {len(failed)}")

    if len(dataset) == 0:
        print("No graphs were created. Check your SMILES data!")
        return

    sample = dataset[0]
    print(f"\n   Sample graph: {sample.x.shape[0]} atoms, "
          f"{sample.edge_index.shape[1]} edges, "
          f"{sample.x.shape[1]} atom features")

    input_dim = sample.x.shape[1]
    print(f"   Input dim: {input_dim} | Hidden: 128 | Embedding: {EMBEDDING_DIM}")
    print(f"   Targets: {num_targets} properties")

    model = MolecularGNN(
        input_dim=input_dim,
        hidden_dim=128,
        embedding_dim=EMBEDDING_DIM,
        num_targets=num_targets,
        dropout=0.2
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    print(f"\nTraining GNN ({EPOCHS} epochs, batch_size={BATCH_SIZE})...")
    history = train_model(
        model, dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE
    )
    print(f"   Final loss: {history[-1]:.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    print(f"\nExtracting {EMBEDDING_DIM}-dim embeddings for all molecules")
    embeddings = extract_embeddings(model, dataset, batch_size=512)
    print(f"   Extracted embeddings for {len(embeddings)} molecules")

    emb_columns = [f"emb_{i}" for i in range(EMBEDDING_DIM)]
    emb_df = pd.DataFrame(
        index=df.index,
        columns=emb_columns,
        dtype=float
    )

    for idx, emb_vector in embeddings.items():
        emb_df.loc[idx] = emb_vector

    df_final = pd.concat([df, emb_df], axis=1)

    rows_before = len(df_final)
    df_final = df_final.dropna(subset=emb_columns)
    rows_after = len(df_final)

    print(f"   Rows with embeddings: {rows_after} / {rows_before}")

    df_final.to_csv(OUTPUT_CSV, index=False)

    print(f"   Output: {OUTPUT_CSV}")
    print(f"   Shape:  {df_final.shape[0]} rows × {df_final.shape[1]} columns")
    print(f"   Features: {len(df.columns)} original + {EMBEDDING_DIM} embedding dims")

    for i, col in enumerate(df_final.columns):
        marker = "" if col.startswith("emb_") else ""
        print(f"   {marker} {col}")

if __name__ == "__main__":
    main()