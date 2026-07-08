# 🔒 Security Policy

## Overview

The **AI & Machine Learning Course** repository is primarily an educational resource containing Jupyter Notebooks, datasets, and learning materials. While this project does not expose network services or handle sensitive user data, we take security seriously and encourage responsible disclosure of any security concerns.

---

## ✅ Supported Versions

The following versions / states of the repository are actively maintained:

| Branch / State | Supported |
|----------------|-----------|
| `main` (latest) | ✅ Yes |
| Older tags / releases | ❌ No |

---

## 🚨 Reporting a Vulnerability

If you discover a security vulnerability in this repository, **please do NOT open a public GitHub issue** for it.

Instead, please report it responsibly using one of the following methods:

### Option 1: GitHub Private Security Advisory (Preferred)

1. Go to the repository's **Security** tab on GitHub.
2. Click **"Report a vulnerability"**.
3. Fill in the details of the vulnerability.
4. Submit — this creates a private, encrypted communication channel.

### Option 2: Email

Send an email to the maintainer at:

> 📧 **[atik203@github.com]** *(replace with your actual contact email)*

**Please include in your report:**

- A clear description of the vulnerability
- Steps to reproduce the issue
- Potential impact or severity assessment
- Any suggested fix (optional but appreciated)

---

## ⏱️ Response Timeline

| Action | Timeline |
|--------|----------|
| Initial acknowledgement | Within **48 hours** |
| Severity assessment | Within **5 business days** |
| Fix / Mitigation | Within **14 days** for critical issues |
| Public disclosure | After fix is merged and users are notified |

We follow a **coordinated disclosure** policy — we will work with the reporter before making any vulnerability public.

---

## 🔐 Scope — What We Consider a Security Issue

### In Scope

- **Malicious code** embedded in notebooks (e.g., notebooks that execute harmful shell commands)
- **Exposed secrets or credentials** accidentally committed (API keys, tokens, passwords)
- **Malicious datasets** (files crafted to exploit parsers such as pickle exploits in `.pkl` model files)
- **Dependency vulnerabilities** — critical CVEs in packages listed in `requirements.txt`
- **Pickle deserialization risks** in saved model files (`.pkl`)

### Out of Scope

- Theoretical vulnerabilities with no practical impact on this repository
- Issues in third-party libraries that are already publicly known (report to the library maintainers directly)
- Broken links or typos (please open a regular issue)
- Notebooks that produce warnings (not errors) during execution

---

## ⚠️ Important Security Notes for Users

### 🧪 Jupyter Notebook Safety

Jupyter Notebooks can execute **arbitrary code**. Before running any notebook from this or any repository:

1. **Review the code** — read through cells before running them.
2. **Use a virtual environment** — never run untrusted notebooks in your base Python environment.
3. **Use a sandboxed environment** — consider using Docker, Google Colab, or Kaggle Kernels for untrusted notebooks.
4. **Disable auto-run** — ensure you run notebooks cell-by-cell for unfamiliar code.

### 🧠 Pickle Model Files

Model files (`*.pkl`) in this repository are generated from our own training code. However:

- **Never unpickle files from untrusted sources** — pickle can execute arbitrary code on deserialization.
- Only load `.pkl` files if you trust the source and have reviewed the training code that generated them.

### 📦 Dependency Safety

Keep your dependencies up to date:

```bash
pip install --upgrade pip
pip install -r requirements.txt --upgrade
```

Check for known vulnerabilities:

```bash
pip install pip-audit
pip-audit
```

---

## 🙏 Acknowledgements

We thank security researchers who responsibly disclose vulnerabilities to us. With your permission, we will acknowledge your contribution in this file or in the repository's release notes.

---

## 📚 References

- [GitHub Security Advisories](https://docs.github.com/en/code-security/security-advisories)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Jupyter Security Guidelines](https://jupyter-notebook.readthedocs.io/en/stable/security.html)
- [Python Pickle Security](https://docs.python.org/3/library/pickle.html#restricting-globals)

---

*Last updated: July 2026*
