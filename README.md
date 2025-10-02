# QRL: Quantum Reinforcement Learning for Eye Disease Classification

Exploring quantum advantage through RL-optimized variational quantum circuits

- An advanced research project combining quantum machine learning (QML) with reinforcement learning (RL) to adaptively optimize feature encoding and variational quantum classifier (VQC) parameters for classifying eye diseases (normal, cataract, diabetic retinopathy, glaucoma) from retinal images. Uses a hybrid approach: quantum policy network with classical RL training via PPO/SAC.

## Features

- **Dataset Handling**: Loads and processes the Eye Diseases Classification dataset (~4217 images across 4 classes) with pretrained ResNet50 feature extraction (2048D features)
- **Dimensionality Reduction**: Applies PCA to reduce features to 128D intermediate and 16D final for quantum input
- **Quantum Encoding**: Implements ZZFeatureMap and RealAmplitudes ansatz for VQC-based feature encoding (4 qubits, 2 layers)
- **Noise Simulation**: Optional depolarizing noise model for realistic quantum hardware simulation
- **RL Environment**: Custom Gymnasium env for actions like feature selection, circuit rotation, PCA adjustment, and noise adaptation; rewards based on accuracy, improvement, cost, and stability
- **RL Training**: Uses Stable Baselines3 (PPO or SAC) with vectorized environments and callbacks for monitoring quantum-specific metrics
- **Evaluation**: Compares RL-optimized VQC against classical baselines (SVM, RandomForest); includes multi-run statistics and per-class analysis
- **Visualizations**: Training progress, performance boxplots, improvement histograms, confusion matrices, per-class accuracies, and VQC parameter evolution

## Tech Stack

- **Core Language**: Python 3.10+ (with Jupyter Notebook)
- **Quantum Libraries**: Qiskit (circuits, VQC, noise models, AerSimulator)
- **Deep Learning**: PyTorch (ResNet feature extraction, optimizers)
- **Reinforcement Learning**: Stable Baselines3 (PPO/SAC), Gymnasium (custom envs)
- **Classical ML**: Scikit-learn (SVM, RandomForest, PCA, metrics, preprocessing)
- **Visualization**: Matplotlib, Seaborn (plots, heatmaps)
- **Data Handling**: NumPy, Pandas, TQDM (progress), Pillow (image processing)
- **Environment**: Supports CPU/CUDA; quantum simulation via Aer (no real hardware required)
- **Dataset Source**: Local dataset path (assumes prior download from Kaggle)

## Getting Started

1. Clone the repository
2. imports: qiskit, qiskit-aer, torch, torchvision, stable-baselines3, gymnasium, scikit-learn, numpy, matplotlib, seaborn, tqdm, pillow)
3. Prepare the dataset: Ensure the Eye Diseases Dataset is in `./data/eye_diseases_dataset/` with subfolders for each class (download from Kaggle if needed)
4. Run the Jupyter Notebook: `jupyter notebook QRL.ipynb`
5. Execute cells sequentially: Loads dataset, extracts features, sets up quantum/RL components, trains agent, evaluates, generates visualizations
6. Optional: Adjust CONFIG in Cell 2 for custom experiments (e.g., enable noise, change RL algorithm); training may take time due to quantum simulations
