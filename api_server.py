import os
import sys
import time
import argparse
import warnings
import glob

import numpy as np
import joblib
import torch
import torch.nn.functional as F

from flask import Flask, request, jsonify
from flask_cors import CORS

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GNN_MODEL_PATH = os.path.join(SCRIPT_DIR, "gnn_embedding_model.pt")
SAVED_MODELS_DIR = os.path.join(SCRIPT_DIR, "saved_models")

EMBEDDING_DIM = 32

TARGET_COLS = [
    "dipole_moment", "polarizability", "homo_energy", "lumo_energy",
    "homo_lumo_gap", "spatial_extent", "zero_point_energy",
    "internal_energy_0K", "heat_capacity",
]

TARGET_UNITS = {
    "dipole_moment": "Debye", "polarizability": "Bohr^3",
    "homo_energy": "Hartree", "lumo_energy": "Hartree",
    "homo_lumo_gap": "Hartree", "spatial_extent": "Bohr^2",
    "zero_point_energy": "Hartree", "internal_energy_0K": "Hartree",
    "heat_capacity": "cal/mol*K",
}

TARGET_DESCRIPTIONS = {
    "dipole_moment": "Electric dipole moment",
    "polarizability": "Isotropic polarizability",
    "homo_energy": "Highest occupied molecular orbital energy",
    "lumo_energy": "Lowest unoccupied molecular orbital energy",
    "homo_lumo_gap": "HOMO-LUMO energy gap",
    "spatial_extent": "Electronic spatial extent",
    "zero_point_energy": "Zero-point vibrational energy",
    "internal_energy_0K": "Internal energy at 0 Kelvin",
    "heat_capacity": "Heat capacity at 298.15 Kelvin",
}

FILE_TO_NAME = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "mlp_neural_network": "MLP Neural Network",
    "svr_linear": "SVR (Linear)",
    "k-nearest_neighbors": "K-Nearest Neighbors",
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


def load_gnn():
    model = MolecularGNN(
        input_dim=7, hidden_dim=128,
        embedding_dim=EMBEDDING_DIM, num_targets=12,
    )
    state = torch.load(GNN_MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def get_embedding(gnn_model, smiles):
    graph = smiles_to_graph(smiles)
    if graph is None:
        return None
    batch = Batch.from_data_list([graph])
    with torch.no_grad():
        _, emb = gnn_model(batch.x, batch.edge_index, batch.batch)
    return emb[0].cpu().numpy()


def get_embeddings_batch(gnn_model, smiles_list):
    graphs = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        g = smiles_to_graph(smi)
        if g is not None:
            graphs.append(g)
            valid_indices.append(i)

    if not graphs:
        return {}, []

    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        _, embs = gnn_model(batch.x, batch.edge_index, batch.batch)

    results = {}
    for idx, vi in enumerate(valid_indices):
        results[vi] = embs[idx].cpu().numpy()

    return results, valid_indices


def list_saved_models():
    models = {}
    pattern = os.path.join(SAVED_MODELS_DIR, "*.joblib")
    for path in sorted(glob.glob(pattern)):
        basename = os.path.splitext(os.path.basename(path))[0]
        if basename.startswith("scaler"):
            continue
        display = FILE_TO_NAME.get(basename, basename)
        models[basename] = display
    return models


def load_prediction_model(model_key):
    model_path = os.path.join(SAVED_MODELS_DIR, f"{model_key}.joblib")
    scaler_x_path = os.path.join(SAVED_MODELS_DIR, "scaler_X.joblib")
    scaler_y_path = os.path.join(SAVED_MODELS_DIR, "scaler_y.joblib")

    for p in [model_path, scaler_x_path, scaler_y_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    model = joblib.load(model_path)
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    return model, scaler_X, scaler_y


def get_molecule_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "smiles": smiles,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "num_bonds": mol.GetNumBonds(),
    }




app = Flask(__name__)
CORS(app)   

_gnn_model = None
_pred_model = None
_scaler_X = None
_scaler_y = None
_active_model_name = None
_uses_scaled_y = False


def _predict_single(smiles):
    emb = get_embedding(_gnn_model, smiles)
    if emb is None:
        return None

    emb_scaled = _scaler_X.transform(emb.reshape(1, -1))
    pred_raw = _pred_model.predict(emb_scaled)

    if _uses_scaled_y:
        pred_values = _scaler_y.inverse_transform(pred_raw)[0]
    else:
        pred_values = pred_raw[0]

    mol_info = get_molecule_info(smiles)
    properties = []
    for i, col in enumerate(TARGET_COLS):
        properties.append({
            "name": col,
            "value": round(float(pred_values[i]), 6),
            "unit": TARGET_UNITS.get(col, ""),
            "description": TARGET_DESCRIPTIONS.get(col, ""),
        })

    return {
        "molecule": mol_info,
        "properties": properties,
        "embedding_dim": EMBEDDING_DIM,
        "model_used": _active_model_name,
    }



@app.route("/")
def index():
    """Serve an interactive landing page for the API."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Molecular Property Prediction API</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0a0e1a;--surface:#111827;--surface2:#1a2235;--border:#1e293b;--accent:#6366f1;--accent2:#818cf8;--accent-glow:rgba(99,102,241,.15);--text:#e2e8f0;--text2:#94a3b8;--success:#22c55e;--warn:#f59e0b;--err:#ef4444;--radius:12px}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}
.bg-grid{position:fixed;inset:0;background-image:radial-gradient(rgba(99,102,241,.08) 1px,transparent 1px);background-size:32px 32px;pointer-events:none;z-index:0}
.container{max-width:960px;margin:0 auto;padding:2rem 1.5rem;position:relative;z-index:1}
header{text-align:center;padding:3rem 0 2rem}
header h1{font-size:2.2rem;font-weight:700;background:linear-gradient(135deg,#818cf8,#6366f1,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.5rem}
header p{color:var(--text2);font-size:1.05rem;font-weight:300}
.badge{display:inline-block;background:var(--accent-glow);color:var(--accent2);border:1px solid rgba(99,102,241,.25);padding:.25rem .75rem;border-radius:20px;font-size:.8rem;font-weight:500;margin-top:.75rem}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.75rem;margin-bottom:1.5rem;transition:border-color .2s}
.card:hover{border-color:rgba(99,102,241,.3)}
.card h2{font-size:1.15rem;font-weight:600;margin-bottom:1rem;display:flex;align-items:center;gap:.5rem}
.card h2 span{font-size:1.2rem}
.input-group{display:flex;gap:.75rem}
#smiles-input{flex:1;background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:.85rem 1rem;color:var(--text);font-size:1rem;font-family:'Inter',monospace;outline:none;transition:border-color .2s}
#smiles-input:focus{border-color:var(--accent)}
#smiles-input::placeholder{color:var(--text2)}
.btn{background:linear-gradient(135deg,#6366f1,#4f46e5);color:#fff;border:none;border-radius:8px;padding:.85rem 1.75rem;font-size:.95rem;font-weight:600;cursor:pointer;transition:all .2s;white-space:nowrap}
.btn:hover{transform:translateY(-1px);box-shadow:0 8px 24px rgba(99,102,241,.3)}
.btn:active{transform:translateY(0)}
.btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
#smiles-dropdown{width:100%;background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:.85rem 1rem;color:var(--text);font-size:1rem;font-family:'Inter',sans-serif;outline:none;cursor:pointer;transition:border-color .2s}
#smiles-dropdown:focus{border-color:var(--accent)}
#smiles-dropdown:hover{border-color:rgba(99,102,241,.3)}
#smiles-dropdown option{background:var(--surface);color:var(--text);padding:.5rem}
.dropdown-label{display:block;margin-top:.75rem;margin-bottom:.5rem;font-size:.85rem;font-weight:500;color:var(--text2)}
.examples{display:flex;gap:.5rem;margin-top:.75rem;flex-wrap:wrap}
.ex-btn{background:var(--surface2);border:1px solid var(--border);color:var(--text2);border-radius:6px;padding:.35rem .7rem;font-size:.8rem;cursor:pointer;transition:all .15s;font-family:'Inter',monospace}
.ex-btn:hover{border-color:var(--accent);color:var(--accent2);background:var(--accent-glow)}
#result-area{display:none}
#error-area{display:none;background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);border-radius:8px;padding:1rem;color:var(--err);margin-bottom:1rem}
.mol-info{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.75rem;margin-bottom:1.25rem}
.mol-chip{background:var(--surface2);border-radius:8px;padding:.75rem;text-align:center}
.mol-chip .label{font-size:.7rem;color:var(--text2);text-transform:uppercase;letter-spacing:.05em;margin-bottom:.25rem}
.mol-chip .value{font-size:1.1rem;font-weight:600;color:var(--accent2)}
.props-table{width:100%;border-collapse:separate;border-spacing:0}
.props-table th{text-align:left;font-size:.75rem;color:var(--text2);text-transform:uppercase;letter-spacing:.05em;padding:.6rem .75rem;border-bottom:1px solid var(--border)}
.props-table td{padding:.7rem .75rem;border-bottom:1px solid rgba(30,41,59,.5);font-size:.9rem;transition:background .15s}
.props-table tr:hover td{background:rgba(99,102,241,.04)}
.props-table .prop-name{font-weight:500;color:var(--text)}
.props-table .prop-val{font-family:'Inter',monospace;font-weight:600;color:var(--accent2);text-align:right}
.props-table .prop-unit{color:var(--text2);font-size:.8rem;text-align:right}
.timing{display:inline-block;background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.2);color:var(--success);padding:.3rem .7rem;border-radius:6px;font-size:.8rem;font-weight:500;margin-top:.75rem}
.endpoints{display:grid;gap:.5rem}
.ep{display:flex;align-items:center;gap:.75rem;padding:.6rem .75rem;border-radius:8px;background:var(--surface2);font-size:.85rem}
.ep .method{font-weight:700;font-size:.75rem;padding:.2rem .5rem;border-radius:4px;min-width:3.5rem;text-align:center}
.ep .method.get{background:rgba(34,197,94,.12);color:#22c55e}
.ep .method.post{background:rgba(99,102,241,.12);color:#818cf8}
.ep .path{font-family:monospace;color:var(--text)}
.ep .desc{color:var(--text2);margin-left:auto;font-size:.8rem}
.spinner{display:inline-block;width:18px;height:18px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .6s linear infinite;vertical-align:middle;margin-right:.5rem}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.fade-in{animation:fadeIn .3s ease-out}
footer{text-align:center;color:var(--text2);font-size:.8rem;padding:2rem 0 1rem;opacity:.6}
@media(max-width:600px){.input-group{flex-direction:column}.ep .desc{display:none}header h1{font-size:1.6rem}}
</style>
</head>
<body>
<div class="bg-grid"></div>
<div class="container">
<header>
  <h1>Molecular Property Prediction</h1>
  <p>Predict quantum-mechanical properties from molecular SMILES using GNN embeddings</p>
  <div class="badge">""" + f"Model: {_active_model_name or 'Not loaded'}" + """</div>
</header>

<div class="card">
  <h2><span></span> Predict Properties</h2>
  <div class="input-group">
    <input type="text" id="smiles-input" placeholder="Enter SMILES string" autocomplete="off" spellcheck="false">
    <button class="btn" id="predict-btn" onclick="predict()">Predict</button>
  </div>
  <label for="smiles-dropdown" class="dropdown-label">Select Example Molecule:</label>
  <select id="smiles-dropdown" onchange="fillFromDropdown()">
    <option value="">-- Choose a molecule --</option>
    <option value="CCO">CCO (Ethanol)</option>
    <option value="c1ccccc1">c1ccccc1 (Benzene)</option>
    <option value="CC(=O)O">CC(=O)O (Acetic acid)</option>
    <option value="C(=O)O">C(=O)O (Formic acid)</option>
    <option value="CC(C)O">CC(C)O (Isopropanol)</option>
    <option value="C1CCCC1">C1CCCC1 (Cyclopentane)</option>
    <option value="CC(C)C(C)(C)O">CC(C)C(C)(C)O (tert-Butanol)</option>
    <option value="c1ccccc1c2ccccc2">c1ccccc1c2ccccc2 (Biphenyl)</option>
    <option value="CC(C)C(=O)O">CC(C)C(=O)O (Isobutyric acid)</option>
    <option value="C1CCCCC1">C1CCCCC1 (Cyclohexane)</option>
    <option value="CC(=O)c1ccccc1">CC(=O)c1ccccc1 (Acetophenone)</option>
    <option value="c1cc(O)ccc1">c1cc(O)ccc1 (Phenol)</option>
    <option value="CCN(CC)CC">CCN(CC)CC (Triethylamine)</option>
    <option value="c1ccc2c(c1)cccc2">c1ccc2c(c1)cccc2 (Naphthalene)</option>
    <option value="CC(=C)C">CC(=C)C (2-Methylpropene)</option>
    <option value="CC#CC">CC#CC (2-Butyne)</option>
    <option value="C1CCOCC1">C1CCOCC1 (Tetrahydropyran)</option>
    <option value="c1cnccn1">c1cnccn1 (Pyrimidine)</option>
    <option value="CC(C)Br">CC(C)Br (2-Bromopropane)</option>
    <option value="CC(C)C(=O)N">CC(C)C(=O)N (Isobutyramide)</option>
  </select>
</div>

<div id="error-area"></div>

<div id="result-area" class="card fade-in">
  <h2><span></span> Prediction Results</h2>
  <div class="mol-info" id="mol-info"></div>
  <table class="props-table">
    <thead><tr><th>Property</th><th style="text-align:right">Value</th><th style="text-align:right">Unit</th></tr></thead>
    <tbody id="props-body"></tbody>
  </table>
  <div class="timing" id="timing"></div>
</div>

<div class="card">
  <h2><span></span> API Endpoints</h2>
  <div class="endpoints">
    <div class="ep"><span class="method post">POST</span><span class="path">/api/predict</span><span class="desc">Single molecule prediction</span></div>
    <div class="ep"><span class="method post">POST</span><span class="path">/api/predict/batch</span><span class="desc">Batch prediction (up to 100)</span></div>
    <div class="ep"><span class="method post">POST</span><span class="path">/api/embedding</span><span class="desc">Get GNN embedding vector</span></div>
    <div class="ep"><span class="method get">GET</span><span class="path">/api/models</span><span class="desc">List available models</span></div>
    <div class="ep"><span class="method get">GET</span><span class="path">/api/properties</span><span class="desc">Property definitions</span></div>
    <div class="ep"><span class="method get">GET</span><span class="path">/api/health</span><span class="desc">Health check</span></div>
  </div>
</div>

<footer>Molecular Property Prediction API &middot; GNN + ML Pipeline</footer>
</div>

<script>
const input=document.getElementById('smiles-input');
const btn=document.getElementById('predict-btn');
const resultArea=document.getElementById('result-area');
const errorArea=document.getElementById('error-area');

function fill(s){input.value=s;input.focus();predict()}

function fillFromDropdown(){
  const dropdown=document.getElementById('smiles-dropdown');
  const smiles=dropdown.value;
  if(smiles){
    input.value=smiles;
    input.focus();
    predict();
    dropdown.value='';
  }
}

input.addEventListener('keydown',e=>{if(e.key==='Enter')predict()});

async function predict(){
  const smiles=input.value.trim();
  if(!smiles)return;
  btn.disabled=true;
  btn.innerHTML='<span class="spinner"></span>Predicting...';
  errorArea.style.display='none';
  resultArea.style.display='none';
  try{
    const res=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({smiles})});
    const data=await res.json();
    if(!res.ok){errorArea.textContent=data.error+(data.hint?' — '+data.hint:'');errorArea.style.display='block';return}
    const mol=data.molecule||{};
    document.getElementById('mol-info').innerHTML=
      chip('Formula',mol.formula||'-')+chip('Atoms',mol.num_atoms||'-')+chip('Heavy Atoms',mol.num_heavy_atoms||'-')+chip('Bonds',mol.num_bonds||'-')+chip('SMILES',mol.smiles||smiles);
    const tbody=document.getElementById('props-body');
    tbody.innerHTML='';
    (data.properties||[]).forEach(p=>{
      const tr=document.createElement('tr');
      tr.innerHTML=`<td class="prop-name">${fmtName(p.name)}</td><td class="prop-val">${p.value.toFixed(6)}</td><td class="prop-unit">${p.unit}</td>`;
      tbody.appendChild(tr)});
    document.getElementById('timing').textContent=`Inference: ${data.inference_time_ms} ms · Model: ${data.model_used||'-'}`;
    resultArea.style.display='block';resultArea.classList.remove('fade-in');void resultArea.offsetWidth;resultArea.classList.add('fade-in');
  }catch(e){errorArea.textContent='Network error: '+e.message;errorArea.style.display='block'}
  finally{btn.disabled=false;btn.textContent='Predict'}
}
function chip(l,v){return `<div class="mol-chip"><div class="label">${l}</div><div class="value">${v}</div></div>`}
function fmtName(n){return n.replace(/_/g,' ').replace(/\\b\\w/g,c=>c.toUpperCase())}
</script>
</body>
</html>"""


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": _active_model_name,
        "gnn_loaded": _gnn_model is not None,
        "target_properties": len(TARGET_COLS),
        "embedding_dim": EMBEDDING_DIM,
    })


@app.route("/api/models", methods=["GET"])
def models_list():
    available = list_saved_models()
    return jsonify({
        "active_model": _active_model_name,
        "available_models": [
            {"key": k, "name": v} for k, v in available.items()
        ],
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "smiles" not in data:
        return jsonify({"error": "Missing 'smiles' field in JSON body"}), 400

    smiles = data["smiles"].strip()
    if not smiles:
        return jsonify({"error": "SMILES string is empty"}), 400

    t0 = time.perf_counter()
    result = _predict_single(smiles)
    elapsed = time.perf_counter() - t0

    if result is None:
        return jsonify({
            "error": f"Invalid SMILES: '{smiles}'",
            "hint": "Check that the SMILES string represents a valid molecule",
        }), 422

    result["inference_time_ms"] = round(elapsed * 1000, 2)
    return jsonify(result)


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)
    if not data or "smiles_list" not in data:
        return jsonify({
            "error": "Missing 'smiles_list' field in JSON body",
            "example": {"smiles_list": ["CCO", "c1ccccc1", "CC(=O)O"]},
        }), 400

    smiles_list = data["smiles_list"]
    if not isinstance(smiles_list, list) or len(smiles_list) == 0:
        return jsonify({"error": "'smiles_list' must be a non-empty array"}), 400

    if len(smiles_list) > 100:
        return jsonify({"error": "Maximum 100 molecules per batch"}), 400

    t0 = time.perf_counter()
    results = []
    for smi in smiles_list:
        r = _predict_single(smi.strip())
        if r is None:
            results.append({"smiles": smi, "error": "Invalid SMILES"})
        else:
            results.append(r)
    elapsed = time.perf_counter() - t0

    return jsonify({
        "count": len(results),
        "total_time_ms": round(elapsed * 1000, 2),
        "avg_time_ms": round(elapsed / len(smiles_list) * 1000, 2),
        "results": results,
    })


@app.route("/api/embedding", methods=["POST"])
def embedding():
    data = request.get_json(silent=True)
    if not data or "smiles" not in data:
        return jsonify({"error": "Missing 'smiles' field in JSON body"}), 400

    smiles = data["smiles"].strip()
    t0 = time.perf_counter()
    emb = get_embedding(_gnn_model, smiles)
    elapsed = time.perf_counter() - t0

    if emb is None:
        return jsonify({"error": f"Invalid SMILES: '{smiles}'"}), 422

    mol_info = get_molecule_info(smiles)
    return jsonify({
        "molecule": mol_info,
        "embedding": emb.tolist(),
        "embedding_dim": len(emb),
        "inference_time_ms": round(elapsed * 1000, 2),
    })


@app.route("/api/properties", methods=["GET"])
def properties_info():
    props = []
    for col in TARGET_COLS:
        props.append({
            "name": col,
            "unit": TARGET_UNITS.get(col, ""),
            "description": TARGET_DESCRIPTIONS.get(col, ""),
        })
    return jsonify({"properties": props, "count": len(props)})



def main():
    global _gnn_model, _pred_model, _scaler_X, _scaler_y
    global _active_model_name, _uses_scaled_y

    parser = argparse.ArgumentParser(description="Molecular Prediction API")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run the API server (default: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model key to use, e.g. 'random_forest'. "
                             "Run with --list to see available models.")
    parser.add_argument("--list", action="store_true",
                        help="List available models and exit")
    parser.add_argument("--debug", action="store_true",
                        help="Run Flask in debug mode")
    args = parser.parse_args()

    if args.list:
        available = list_saved_models()
        if not available:
            print("[ERROR] No models found in saved_models/. Run model_comparison.py first.")
        else:
            print("\nAvailable models:")
            for k, v in available.items():
                print(f"  --model {k:<25s}  ({v})")
        return

    if not os.path.exists(GNN_MODEL_PATH):
        print(f"[ERROR] GNN model not found: {GNN_MODEL_PATH}")
        print("        Run generate_embeddings.py first.")
        return
    _gnn_model = load_gnn()

    available = list_saved_models()
    if not available:
        print("[ERROR] No prediction models found. Run model_comparison.py first.")
        return

    model_key = args.model
    if model_key is None:
        model_key = list(available.keys())[0]
        print(f"\n[INFO] No --model specified, using: {model_key}")

    if model_key not in available:
        print(f"[ERROR] Model '{model_key}' not found.")
        print(f"        Available: {list(available.keys())}")
        return

    print(f"\n[LOAD] Loading prediction model: {available[model_key]}...")
    _pred_model, _scaler_X, _scaler_y = load_prediction_model(model_key)
    _active_model_name = available[model_key]
    _uses_scaled_y = "mlp" in model_key or "svr" in model_key
    print(f"   Model loaded: {_active_model_name}")
    print(f"   Uses scaled y: {_uses_scaled_y}")
    print(f"  Model:  {_active_model_name}")
    print(f"  Server: http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()