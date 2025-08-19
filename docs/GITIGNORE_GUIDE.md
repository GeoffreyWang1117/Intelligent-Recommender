# 📄 .gitignore Configuration Guide
## Git Ignore Rules for Intelligent Recommender System

**Author**: Geoffrey Wang  
**Repository**: https://github.com/GeoffreyWang1117/Intelligent-Recommender  
**Purpose**: Define which files should be tracked by Git and which should be ignored

---

## 🎯 Overview

This `.gitignore` file is specifically designed for the **Intelligent Recommender System** project, which combines traditional machine learning algorithms with large language models. The configuration ensures that only essential source code, documentation, and configuration files are tracked in the repository.

---

## 📋 File Categories

### ✅ **Files Included in Repository**

#### Core Application Files
- `app.py` - Main application entry point
- `config.py` - System configuration
- `requirements.txt` - Python dependencies
- `Dockerfile`, `docker-compose.yml` - Container configuration
- `Makefile` - Build automation

#### Source Code
- `models/` - Algorithm implementations
- `services/` - Microservices and APIs
- `teachers/` - Dual-teacher system code
- `utils/` - Utility functions
- `tests/` - Test suites
- `scripts/` - Automation scripts

#### Documentation
- `README.md`, `README_CN.md` - Main documentation
- `LICENSE` - Project license
- `ARCHITECTURE.md` - System architecture
- `docs/` - Technical documentation
- Phase summary documents

#### Sample Data
- `data/sample/` - Small demo datasets (movies.csv, ratings.csv, users.csv)
- `experiment_results/*.json` - Experiment configuration and summary

### ❌ **Files Excluded from Repository**

#### Python Runtime Files
- `__pycache__/` - Python bytecode cache
- `*.pyc`, `*.pyo` - Compiled Python files
- `*.so` - Shared object files

#### Machine Learning Artifacts
- `models/saved/` - Trained model files
- `*.pkl`, `*.pth`, `*.h5` - Model weights and pickled objects
- `*.model`, `*.joblib` - Serialized models
- `*.faiss`, `*.index` - FAISS search indices

#### Large Datasets
- `data/raw/`, `data/processed/` - Large data directories
- `data/movieLens*/`, `data/ml-*/` - Full MovieLens datasets
- `*.csv.gz`, `*.parquet` - Compressed data files

#### Training Artifacts
- `training_logs/` - Training session logs
- `tensorboard_logs/` - TensorBoard visualizations
- `wandb/`, `mlruns/` - Experiment tracking
- `**/evaluation_results/` - Cached evaluation results

#### Environment Configuration
- `.env*` - Environment variables (contains sensitive data)
- `venv/`, `.venv/` - Virtual environments
- `conda-meta/` - Conda environment metadata

#### Development Tools
- `.vscode/`, `.idea/` - IDE configuration
- `*.swp`, `*.swo` - Editor temporary files
- `.DS_Store` - macOS system files
- `Thumbs.db` - Windows thumbnail cache

#### Temporary Files
- `logs/`, `*.log` - Log files
- `tmp/`, `temp/` - Temporary directories
- `*.bak`, `*.backup` - Backup files
- Temporary report files

---

## 🔧 Special Rules

### Force Include Important Files
```gitignore
!.gitignore
!requirements.txt
!README.md
!LICENSE
```

### Project-Specific Exclusions
```gitignore
# Remove temporary documentation
**/TEMP_*.md
PROJECT_SIMPLIFICATION_REPORT.md

# Ignore old README versions
README_OLD.md
README_EN.md
```

### Keep Sample Data for Demo
```gitignore
# Exclude large datasets
data/raw/
data/processed/

# But keep small demo files
!data/sample/
!data/sample/*.csv
```

---

## 📊 File Size Considerations

### Small Files (Included)
- **Source Code**: `*.py` files (usually < 100KB)
- **Configuration**: `*.yml`, `*.json`, `*.txt` (< 10KB)
- **Documentation**: `*.md` files (< 1MB)
- **Sample Data**: Demo CSV files (< 100KB each)

### Large Files (Excluded)
- **Trained Models**: `*.pkl`, `*.pth` files (can be 100MB+)
- **Full Datasets**: MovieLens full datasets (25MB+)
- **Evaluation Cache**: Results from multiple experiments (10MB+)
- **Log Files**: Training and system logs (can grow indefinitely)

---

## 🎯 Benefits of This Configuration

### 🚀 **Repository Performance**
- **Faster cloning**: Excludes large binary files
- **Smaller repository size**: Focuses on source code
- **Quick checkout**: No unnecessary cache files

### 🔒 **Security**
- **No secrets**: `.env` files with API keys excluded
- **No credentials**: Database connections and tokens ignored
- **Clean history**: No accidentally committed sensitive data

### 👥 **Team Collaboration**
- **Consistent environment**: Everyone excludes the same files
- **No conflicts**: Personal IDE settings not shared
- **Focus on code**: Only relevant changes tracked

### 🧹 **Maintenance**
- **Clean repository**: No clutter from temporary files
- **Easy navigation**: Directory structure remains clear
- **Reduced merge conflicts**: Fewer irrelevant file changes

---

## 🛠️ Usage Instructions

### For New Contributors
```bash
# Clone the repository
git clone https://github.com/GeoffreyWang1117/Intelligent-Recommender.git
cd Intelligent-Recommender

# The .gitignore is already configured
# Install dependencies and start developing
pip install -r requirements.txt
```

### For Maintainers
```bash
# Check what files are being ignored
git status --ignored

# Test ignore rules for specific files
git check-ignore filename.txt

# Add new ignore patterns if needed
echo "new_pattern/" >> .gitignore
```

### For Model Training
```bash
# Train models (files will be automatically ignored)
python scripts/train_model.py

# Models saved to models/saved/ (ignored by Git)
# Only code changes will be tracked
```

---

## 📝 Maintenance Notes

### When to Update .gitignore
1. **New file types**: When introducing new binary formats
2. **New tools**: When adding development tools that create cache
3. **Data changes**: When dataset structure changes
4. **Security updates**: When new sensitive files are introduced

### Testing Changes
```bash
# Before committing .gitignore changes
git add .gitignore
git status --porcelain

# Verify expected files are tracked/ignored
```

---

## 📞 Support

For questions about the .gitignore configuration:
- **Repository Issues**: [GitHub Issues](https://github.com/GeoffreyWang1117/Intelligent-Recommender/issues)
- **Documentation**: See main [README.md](README.md)
- **Author**: Geoffrey Wang

---

*This .gitignore configuration is optimized for the Intelligent Recommender System project combining traditional ML and LLM approaches.*
