# AtreyaQ: Quantum Drug Discovery Platform

## Overview

**AtreyaQ** is an open-source quantum-enhanced drug discovery platform. It brings together principles of **quantum chemistry**, **molecular simulations**, and **machine learning** into a unified, user-friendly interface that helps researchers, pharmaceutical scientists, and enthusiasts explore the potential of molecules in a highly efficient and intuitive way.

AtreyaQ leverages tools like **RDKit** for molecular modeling, and quantum computing frameworks (e.g., PennyLane) for energy simulations — aiming to push the boundaries of early-stage drug discovery.

---

## Motivation

Traditional drug discovery is expensive, time-consuming, and computationally intensive. With the rise of quantum computing, new algorithms can simulate molecular behavior more precisely than classical counterparts. **AtreyaQ** provides a gateway to explore such capabilities while maintaining compatibility with classical computational chemistry methods.

---

## Features

* ✨ **Quantum Molecular Energy Estimation**
* 🔬 **SMILES-based Molecule Input**
* 🔎 **Molecule Visualization (2D)**
* 🔢 **Classical Descriptors Calculation**
* 📊 **Binding Affinity Predictions (Coming Soon)**
* 📅 **Report Generation & Export (CSV)**
* ⚖️ **Two Modes**:

  * **Demo Mode**: Try predefined molecules
  * **Pro Mode**: Upload your own molecules

---

## Chemistry Background

At its core, AtreyaQ is built on **molecular descriptors** and **quantum chemistry principles**:

### Molecular Descriptors:

These are numerical values that describe the properties of molecules, such as:

* Molecular weight
* Number of hydrogen bond donors/acceptors
* Topological polar surface area (TPSA)
* LogP (lipophilicity)
* Rotatable bonds, rings, and aromaticity

These descriptors feed into ML models or filtering processes during drug discovery.

### Quantum Chemistry:

Using **variational quantum eigensolvers (VQE)** and Hamiltonians (e.g., Pauli operators), we can estimate the ground-state energy of a molecule. Lower energy often correlates with higher stability and drug-likeness.

AtreyaQ will soon incorporate actual quantum hardware (or simulators) to perform these tasks.

---

## Quantum Computing Principles

### Variational Quantum Eigensolver (VQE):

A hybrid quantum-classical algorithm to solve for the smallest eigenvalue (i.e., ground state energy) of a molecule’s Hamiltonian. The quantum computer prepares parameterized quantum states, and a classical optimizer tunes them to minimize the expected energy.

### Qubit Mapping:

The molecule is encoded into a qubit Hamiltonian using transformations such as Jordan-Wigner or Bravyi-Kitaev. This allows quantum computers to simulate fermionic systems.

---

## Math Behind the Scenes

### Hamiltonian Construction:

For a given molecule:

```
H = c0*I + c1*Z0 + c2*Z1 + c3*Z0Z1 + ...
```

Where:

* `Z0`, `Z1` are Pauli-Z operators on different qubits
* `c0, c1...` are coefficients from molecular integrals

### Energy Estimation:

Energy is computed as:

```
E = <ψ(θ)| H | ψ(θ)>
```

Where `ψ(θ)` is a parameterized quantum state.

### Optimization:

Minimize `E(θ)` using gradient descent or other optimizers. Classical computers update the parameters based on quantum outputs.

---

## How to Use

### Run Locally

1. Clone the repo:

   ```bash
   git clone https://github.com/SujayKulkarni-2211/atreyaq
   cd atreyaq
   ```
2. Create and activate virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:

   ```bash
   streamlit run app.py
   ```

### Docker (Coming Soon)

You will be able to build and run this app in a fully encapsulated Docker container.

---

## Benefits to Researchers

* 🧬 **Faster prototyping** of drug-like molecules
* ⚖️ **Accurate energy profiling** using quantum methods
* 📈 **Molecular descriptor evaluation** for ML pipelines
* ⚙️ **Integrated visualization** for intuitive understanding
* 🌟 **Future-proof**: Quantum-ready framework

---

## Limitations & Roadmap

* ⚠️ No backend currently (for login/state management)
* ⚠️ Binding affinity is in prototype stage
* ⏳ Quantum simulations run in classical mode for now

### Future Plans:

* [ ] Integrate PennyLane for real quantum simulation
* [ ] Launch DockerHub-based container
* [ ] OAuth2 login with dashboard features
* [ ] REST API for inference-as-a-service
* [ ] PubChem integration for fetching molecule data

---

## Credits

* ✨ Developed by **Sujay Kulkarni**
* 🔗 Powered by **RDKit**, **Streamlit**, **Quantum Frameworks (Coming)**

---

## License

MIT License. See `LICENSE` file for details.

---

## Support / Contact

Feel free to raise issues on [GitHub Issues](https://github.com/SujayKulkarni-2211/atreyaq/issues) or connect with the developer for collaborations and queries.
Special Invitation to Mr. Aditya Maller to join me on this endeavour. 
If there are any chemistry peeps who want to help me out by explaining more stuff please do contact me on sujayvk.btech23@rvu.edu.in

---

> "The quantum revolution in pharma is not tomorrow — it's already begun. Be a part of it with AtreyaQ."
