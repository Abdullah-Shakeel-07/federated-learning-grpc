# Minimal Federated Learning Framework with gRPC

A simplified Federated Learning platform (inspired by [Flower](https://flower.ai)).
A **Coordinator** sends training tasks to **Participants**, which train on their own
private data and return only a model parameter. The coordinator averages the
results (FedAvg) into a global model.

Two versions:
- **Monolithic**: everything in one Python process (`monolith/`)
- **Distributed**: independent processes communicating through **gRPC**
  - Coordinator: Python (gRPC server)
  - CLI client: Python
  - Participants: Node.js

## Project structure

```
proto/fl.proto           gRPC contract (services + messages)
monolith/main.py         monolithic version
coordinator/server.py    coordinator gRPC server
client/                  CLI client (in progress)
participant/             Node.js participant (in progress)
scripts/                 helper scripts
tests/                   test scripts
docs/                    analysis, diagrams, AI prompt log
```

## Setup

### Option 1: GitHub Codespaces (easiest)
On the repo page: **Code → Codespaces → Create codespace on main**.
Python, Node.js and the dependencies are installed automatically.

### Option 2: Local machine
Requirements: Python 3.10+, Node.js 18+

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r coordinator/requirements.txt
```

### Generate the gRPC code (both options)
Required once after cloning, and again whenever `proto/fl.proto` changes:

```bash
./scripts/generate_proto.sh
```

## Run

### Monolithic version
```bash
python3 monolith/main.py
```
Expected result: `global weight = 20.0`

### Distributed version
**Terminal 1: start the coordinator**
```bash
cd coordinator
python3 server.py            # listens on port 50051 (change with --port)
```

**Terminal 2: test the coordinator** (simulates the CLI and two participants)
```bash
python3 tests/test_coordinator.py
```
Expected result: `TEST PASSED`

> The CLI client and Node.js participants are in progress. Run instructions will be added here.
