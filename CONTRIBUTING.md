# 🤝 Contributing to AI & Machine Learning Course

First of all — **thank you** for taking the time to contribute! 🎉  
Whether you're fixing a typo, improving a notebook, or adding a whole new module, your contribution matters.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Notebook Standards](#notebook-standards)
- [Commit Message Convention](#commit-message-convention)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)

---

## 📜 Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).  
By participating, you agree to uphold this code. Please report unacceptable behavior to the maintainers.

---

## 💡 How Can I Contribute?

### 🐛 Report Bugs
Found an error in a notebook (wrong output, broken code, incorrect formula)?

1. **Check existing issues** first to avoid duplicates.
2. Open a new issue using the **Bug Report** template.
3. Include the notebook path, Python version, and a minimal reproducible example.

### ✨ Suggest Enhancements
Have an idea for a new topic, better explanation, or additional dataset?

1. Open a **Feature Request** issue.
2. Describe the improvement and why it would be useful.
3. If applicable, share references or papers.

### 📓 Add or Improve Notebooks
Want to contribute a notebook?

- Fix a bug or improve an explanation in an existing notebook.
- Add a new notebook covering a topic that fits the curriculum.
- Improve comments, markdown cells, or visualizations.

### 📊 Contribute Datasets
- Add small, publicly available datasets (< 50 MB preferred).
- Always include the **data source/license** in the notebook or a companion `README.md` inside the module folder.
- Never commit private or proprietary data.

---

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repo via the GitHub UI, then:
git clone https://github.com/<YOUR_USERNAME>/AI-ML.git
cd AI-ML
```

### 2. Set Up Your Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Create a Branch

Use a descriptive branch name:

```bash
git checkout -b feature/add-transformer-notebook
# or
git checkout -b fix/svm-module-typo
# or
git checkout -b docs/improve-readme
```

### 4. Make Your Changes

Follow the [Notebook Standards](#notebook-standards) below.

### 5. Commit and Push

```bash
git add .
git commit -m "feat: Add Transformer architecture notebook (Module X)"
git push origin feature/add-transformer-notebook
```

### 6. Open a Pull Request

Go to the original repository and click **"New Pull Request"**.  
Fill in the PR template and link any related issues.

---

## 📓 Notebook Standards

All Jupyter Notebooks must follow these standards:

### Structure

Every notebook should have:

```
1. Title Cell (Markdown H1)
2. Overview / Learning Objectives
3. Table of Contents (for notebooks > 100 cells)
4. Prerequisites / Imports
5. Main Content (with clear section headers)
6. Summary / Key Takeaways
7. References / Further Reading
```

### Code Quality

- ✅ All code cells must **run without errors** top-to-bottom (`Kernel → Restart & Run All`)
- ✅ Use **meaningful variable names** — not `x`, `y`, `df1` without context
- ✅ Add **inline comments** for non-obvious logic
- ✅ Use **markdown cells** to explain concepts before code
- ✅ **Clear all outputs** before committing (to keep diffs clean): `Kernel → Restart & Clear Output`
- ❌ Do NOT hardcode absolute file paths — use relative paths
- ❌ Do NOT commit notebooks with large embedded outputs (images/base64 blobs)

### Datasets

- Place datasets in the same module folder as the notebook.
- For large datasets (> 10 MB), link to the original source instead of committing the file.
- Update `.gitignore` for large files.

---

## 📝 Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new notebook, module, or feature |
| `fix` | A bug fix in existing code/notebook |
| `docs` | Documentation changes only |
| `style` | Formatting, whitespace, no logic change |
| `refactor` | Code restructuring without feature change |
| `data` | Dataset additions or updates |
| `chore` | Build process, dependencies, config |

### Examples

```bash
feat: Add KNN classification notebook (Module 17)
fix: Correct SVM kernel parameter in Module 15
docs: Improve README quick start section
data: Add Heart Disease dataset to Module 9
chore: Update .gitignore for large CSV files
```

---

## 🔄 Pull Request Process

1. **Link related issues** in your PR description (`Closes #42`)
2. **Describe your changes** clearly — what problem does it solve?
3. **Run the notebook** end-to-end before submitting
4. **Clear notebook outputs** (sensitive or very large outputs)
5. **Update the README** if you've added a new module or changed the structure
6. **Request a review** from a maintainer
7. PRs require at least **1 approving review** before merging
8. The maintainer may request changes — please respond promptly

---

## 🎨 Style Guide

### Python Code

```python
# ✅ Good
import numpy as np
import pandas as pd

# Load and inspect the dataset
df = pd.read_csv("heart.csv")
print(df.shape)
print(df.head())

# ❌ Avoid
import numpy as np, pandas as pd
d=pd.read_csv("heart.csv")
print(d.shape)
```

### Markdown Cells

- Use **H2 (`##`)** for major sections
- Use **H3 (`###`)** for subsections
- Use **bold** for key terms on first mention
- Add LaTeX math where applicable: `$y = wx + b$`

---

## ❓ Questions?

If you're unsure about anything, open a **Discussion** in the GitHub Discussions tab or create an issue labeled `question`.

We're happy to help! 😊

---

*Thank you for making this learning resource better for everyone.* 🙏
