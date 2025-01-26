# PyTorch Installation Guide for Medical Imaging Research  
*By Dominic Velasco, MD ([@domvmd](https://github.com/domvmd))*  

This guide helps nuclear medicine researchers set up PyTorch with GPU support for deep learning in medical imaging (e.g., PET/CT, SPECT reconstruction).

### 1. Why PyTorch for Medical Imaging?
Key Benefits:

GPU acceleration for large 3D medical datasets (e.g., 4D dynamic PET scans).

Flexible data pipelines (custom DICOM/NIfTI loaders).

Pretrained models for tasks like tumor segmentation (e.g., MONAI framework).

### 2. Prerequisites
Hardware:
NVIDIA GPU (e.g., RTX 3070/3090, A100) with ≥12GB VRAM (for 3D volumes).
Driver ≥535.86.10 (Download).

Software:
Windows 10/11 or Linux (recommended for GPU support).
Miniconda (Python 3.9).

### 3. Step-by-step Installation
Here’s a streamlined, **medical-researcher-friendly** integrating all requested components (Miniconda, environment setup, Jupyter integration, and PyTorch installation):

---

### **Step-by-Step Installation Guide**  
**Target Audience**: Nuclear Medicine Residents with Minimal Coding Experience  

---

#### **1. Install Miniconda**  
**Purpose**: Isolate dependencies and simplify package management.  
**Steps**:  
1. Download [Miniconda for Windows](https://docs.conda.io/en/latest/miniconda.html#windows-installers) (or [Linux](https://docs.conda.io/en/latest/miniconda.html#linux-installers)).  
2. Run the installer. At the "Advanced Options" screen:  
   - Check ✅ "Add Miniconda3 to my PATH environment variable" (avoids manual setup).  
   - Check ✅ "Register Miniconda3 as my default Python".  

---

#### **2. Create a Conda Environment**  
**Purpose**: Avoid conflicts with existing Python projects.  
```bash
# Open Anaconda Prompt (Windows) or Terminal (Linux)
conda create -n medical_pytorch python=3.9
conda activate medical_pytorch
```

---

#### **3. Install Jupyter Notebook & Kernel**  
**Purpose**: Run code interactively for medical imaging experiments.  
```bash
# Install Jupyter Notebook and ipykernel inside the environment
conda install jupyter ipykernel

# Link your environment to Jupyter
python -m ipykernel install --user --name=medical_pytorch --display-name="PyTorch (Medical)"
```

---

#### **4. Install PyTorch with CUDA 12.4**  
**Purpose**: Leverage GPU acceleration for large medical datasets (e.g., 3D PET/CT scans).  
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

---

#### **5. Verify Installation in Jupyter**  
1. Launch Jupyter:  
   ```bash
   jupyter notebook
   ```  
2. Create a new notebook and select the `PyTorch (Medical)` kernel.  
3. Run this code:  
   ```python
   import torch

   # GPU check (critical for medical imaging workflows)
   print(f"GPU Available: {torch.cuda.is_available()}")  # Must return "True"
   print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

   # Test tensor operations on GPU
   pet_scan_simulated = torch.randn(128, 128, 128).cuda()  # Simulate 3D PET scan
   print(f"Tensor on {pet_scan_simulated.device}")
   ```

**Expected Output**:  
```
GPU Available: True  
GPU Memory: 8.0 GB  # RTX 3070 has 8GB VRAM  
Tensor on cuda:0
```

---

### **Troubleshooting for Medical Researchers**  
- **Jupyter Kernel Not Found**:  
  - Ensure you ran `python -m ipykernel install` **after activating** `medical_pytorch`.  
- **CUDA Not Detected**:  
  - Update NVIDIA drivers [here](https://www.nvidia.com/Download/index.aspx).  
  - Reinstall PyTorch:  
    ```bash
    conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
    ```

---

### **Next Steps for Medical Imaging**  
Include in your repo’s `examples/` folder:  
- Loading DICOM/NIfTI files with `SimpleITK`.  
- Training a tumor segmentation model with `MONAI`.  

---

### **Sample GitHub README Snippet**  
```markdown
## Quick Start  
1. Install Miniconda.  
2. Copy-paste these commands into Anaconda Prompt:  
   ```bash
   conda create -n medical_pytorch python=3.9
   conda activate medical_pytorch
   conda install jupyter ipykernel
   python -m ipykernel install --user --name=medical_pytorch --display-name="PyTorch (Medical)"
   conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
   jupyter notebook
   ```
3. Verify GPU support using the provided test code.  
```

This structure ensures **clarity** and **reproducibility** for clinicians new to deep learning. Let me know if you’d like a full repo template!
