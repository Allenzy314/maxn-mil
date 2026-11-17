"""Check release contents, not Git history or a remotely hosted mirror."""

import argparse
from pathlib import Path
import re
import subprocess
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "personal_path": re.compile(r"/(?:home|Users|data)/[\w.-]+(?:/|\b)|[A-Za-z]:[\\/]Users[\\/][^\\/\s]+", re.I),
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "credential": re.compile(r"(?:gh[pousr]_|github_pat_|olp_|hf_)[A-Za-z0-9_]{20,}"),
    "embedded_auth": re.compile(r"https?://[^/\s]+:[^/\s]+@"),
    "private_key": re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    "orcid": re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b"),
}
REVIEW_REQUIRED = {".pdf", ".png", ".jpg", ".jpeg", ".svg", ".tif", ".tiff", ".ipynb",
                   ".zip", ".gz", ".tar", ".pt", ".pth", ".ckpt", ".h5", ".npy", ".npz"}


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def scan_files(root, paths, terms=()):
    terms = [normalized(term) for term in terms if term.strip()]
    findings = []
    for relative in paths:
        path = Path(root) / relative
        if path.is_symlink() or not path.is_file():
            findings.append((relative, 0, "non_regular_file"))
            continue
        if any(term in normalized(relative) for term in terms):
            findings.append((relative, 0, "identity_in_filename"))
        if Path(relative).suffix.lower() in REVIEW_REQUIRED:
            findings.append((relative, 0, "artifact_requires_manual_review"))
            continue
        try:
            content = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            findings.append((relative, 0, "binary_requires_manual_review"))
            continue
        for number, line in enumerate(content.splitlines(), 1):
            if any(term in normalized(line) for term in terms):
                findings.append((relative, number, "identity_term"))
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append((relative, number, name))
        if "\x00" in content:
            findings.append((relative, 0, "binary_requires_manual_review"))
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms-file", type=Path,
                        help="Private, untracked list outside this repository; one literal term per line")
    args = parser.parse_args()
    terms = []
    if args.terms_file:
        if args.terms_file.resolve().is_relative_to(ROOT):
            parser.error("Keep the identity list outside the released repository")
        terms = args.terms_file.read_text(encoding="utf-8").splitlines()
        if not any(term.strip() for term in terms):
            parser.error("The identity list must not be empty")
    raw = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT)
    paths = sorted(set(raw.decode("utf-8").rstrip("\0").split("\0")))
    findings = scan_files(ROOT, paths, terms)
    for path, line, reason in findings:
        print("%s:%d: %s" % (path, line, reason))
    if findings:
        raise SystemExit("FAIL: review the findings before release")
    print("PASS: checked %d release files; private identity list: %s" % (len(paths), bool(terms)))
    print("Git history and the hosted anonymous mirror require separate checks.")


if __name__ == "__main__":
    main()
