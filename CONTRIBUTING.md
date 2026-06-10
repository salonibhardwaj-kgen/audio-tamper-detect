# Contributing

Thank you for your interest in contributing to the Audio Tampering Detection project.

## How to contribute

### Reporting issues

Before opening a new issue, please search existing issues to avoid duplicates.

Use the appropriate issue template:
- **Bug report** — something is not working as described
- **Feature request** — a new capability or improvement

Include as much detail as possible: Python version, OS, steps to reproduce, and any error output.

### Submitting a pull request

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Set up your environment:**
   ```bash
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Make your changes.** Follow the code style in the existing files.

4. **Test your changes** on at least one WAV file with `run_new_pipeline.py`.

5. **Commit** with a descriptive message following [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat: add Tamil noise addition training data
   fix: handle audio shorter than 30s gracefully
   docs: update FLEURS download instructions
   ```

6. **Open a pull request** using the PR template.

### Development priorities

The highest-impact areas for contribution:

| Area | Description |
|---|---|
| CNN v4 | Add FLEURS genuine training data to fix phone-recording false positives |
| New languages | Generate noise addition spectrograms for Tamil, Telugu, Marathi, Gujarati |
| Report output | Build highlighted spectrogram PNG with annotated manipulation regions |
| VLM validation | Systematic noise type accuracy testing with paid Gemini tier |
| Documentation | Installation guides, video walkthroughs, worked examples |

## Code style

- Python 3.9+
- No type annotations required (project is research-grade)
- Keep BASE path resolution via `Path(__file__).parent` — never hardcode absolute paths
- Environment variables for all API keys — never commit credentials
- All new scripts should use `os.environ.get()` for `GEMINI_API_KEY` and `HF_TOKEN`

## Dataset and model files

Do **not** commit:
- Audio files (`.wav`, `.mp3`, `.flac`)
- Spectrogram images in `datasets/`
- Model checkpoints (`.pt` files)

These are covered by `.gitignore` and should be placed in releases or downloaded from HuggingFace Hub.

## Questions

Open a GitHub Discussion or an issue labelled `question`.
