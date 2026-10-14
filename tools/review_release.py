"""Review release contents and dependency licenses before tagging a release.

Scans the blobs in the Git index (what `git ls-files` tracks), not the working tree,
so results do not depend on line-ending conversion or untracked local files. Uses
only the standard library and importlib.metadata; nothing is downloaded.

Default: write the JSON report and exit 1 if any error-severity finding exists.
--check: run the same review without writing the report; exit 1 on errors.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import io
import json
import os
import platform
import re
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "bench/results/release-review.json"
MAX_BYTES = 5 * 1024 * 1024

FORBIDDEN_SUFFIXES = {
    ".gguf": "model weights",
    ".ggml": "model weights",
    ".safetensors": "model weights",
    ".bin": "model weights or opaque binary",
    ".pt": "model weights",
    ".pth": "model weights",
    ".ckpt": "model weights",
    ".onnx": "model weights",
    ".h5": "model weights",
    ".tflite": "model weights",
    ".pkl": "pickled data",
    ".pickle": "pickled data",
    ".npy": "array data",
    ".npz": "array data",
    ".parquet": "dataset",
    ".arrow": "dataset",
    ".pem": "key material",
    ".key": "key material",
    ".p12": "key material",
    ".pfx": "key material",
}
FORBIDDEN_NAMES = {
    ".env": "environment secrets",
    ".netrc": "credentials",
    ".pypirc": "credentials",
    "tokenizer.json": "upstream tokenizer artifact",
    "tokenizer.model": "upstream tokenizer artifact",
}
FORBIDDEN_PREFIXES = {
    ".env.": "environment secrets",
    "id_rsa": "SSH key",
    "id_dsa": "SSH key",
    "id_ecdsa": "SSH key",
    "id_ed25519": "SSH key",
}
SECRET_PATTERNS = {
    "github-token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}"),
    "github-fine-grained-token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}"),
    "huggingface-token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "sk-api-key": re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{19,}"),
    "aws-access-key-id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "slack-token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "private-key-block": re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"),
}
FOREIGN_COPYRIGHT = re.compile(r"Copyright\s*(?:\([cC]\)|©|[0-9]{4})")
OWNER = "guru-bharadwaj20"
OWNER_NOTICE = "Copyright (c) 2026 guru-bharadwaj20"

# Binary or data files that are reviewed individually. Each entry is checked, not ignored.
ALLOWLIST = {
    "tests/fixtures/reference-logits.npz": {
        "max_bytes": 2_000_000,
        "members": r"^(tokens|logits)_[0-9]+\.npy$",
        "reason": (
            "Small test fixture: token IDs and float32 output logits captured from the "
            "random-weight hf-internal-testing/tiny-random-LlamaForCausalLM oracle "
            "(tests/fixtures/reference-model.json). Contains no weights or tokenizer files."
        ),
    },
}

THIRD_PARTY = {
    "bench/third_party/stream.c": {
        "component": "STREAM 5.10 (John D. McCalpin, Joe R. Zagar)",
        "license": "STREAM license (embedded in file header)",
        "upstream": "https://www.cs.virginia.edu/stream/FTP/Code/stream.c",
        # SHA-256 of the LF-normalized upstream file, compared on 2026-10-14.
        "upstream_sha256_lf": "a52bae5e175bea3f7832112af9c085adab47117f7d2ce219165379849231692b",
        "notice_doc": "docs/third-party.md",
        "required_header": [
            "Revision: $Id: stream.c,v 5.10",
            "1991-2013: John D. McCalpin",
            "License:",
            "You are free to use this program and/or to redistribute",
            "Results based on modified source code or on runs not in",
            "labelled whenever they are published",
            "Absolutely no warranty is expressed or implied.",
        ],
    },
}

MIT = "MIT (LICENSE)"
CATEGORIES = [
    ("license-text", lambda p: p == "LICENSE", "MIT license text"),
    ("third-party", lambda p: "third_party/" in p or "vendor/" in p, "see third_party table"),
    ("test-fixtures", lambda p: p.startswith("tests/fixtures/"), MIT),
    ("benchmark-results", lambda p: p.startswith("bench/results/"), MIT),
    (
        "native-source",
        lambda p: p.startswith(("core/", "bench/")) or p in ("CMakeLists.txt", "CMakePresets.json"),
        MIT,
    ),
    ("python-package", lambda p: p.startswith("python/"), MIT),
    ("tests", lambda p: p.startswith("tests/"), MIT),
    ("tools", lambda p: p.startswith("tools/"), MIT),
    ("ci", lambda p: p.startswith(".github/workflows/"), MIT),
    ("documentation", lambda p: p.endswith(".md"), MIT),
    ("build-config", lambda p: p in ("pyproject.toml", "ruff.toml", ".gitignore"), MIT),
]
REQUIRED_IGNORES = ["/models/", "/build/", "/.venv/", "*.gguf", "*.safetensors", ".env"]
COPYLEFT = re.compile(
    r"\b(?:A?GPL|LGPL|MPL|EPL|EUPL|CDDL|OSL)\b|GNU |Mozilla Public|Eclipse Public|Affero",
    re.IGNORECASE,
)
GROUP_ORDER = ["runtime", "dev", "oracle", "server"]
NOT_INSTALLED = "not installed - license from upstream must be checked"


def git(*args, data=None):
    return subprocess.run(
        ["git", *args], cwd=ROOT, input=data, capture_output=True, check=True
    ).stdout


def tracked_blobs():
    entries = []
    for record in git("ls-files", "-s", "-z").split(b"\0"):
        if not record:
            continue
        info, path = record.split(b"\t", 1)
        mode, oid, _stage = info.decode().split()
        entries.append((path.decode("utf8"), mode, oid))
    entries.sort()
    request = "".join(oid + "\n" for _, mode, oid in entries if mode != "160000").encode()
    raw = git("cat-file", "--batch", data=request)
    contents, offset = {}, 0
    while offset < len(raw):
        end = raw.index(b"\n", offset)
        oid, _kind, size = raw[offset:end].decode().split()
        start = end + 1
        contents[oid] = raw[start : start + int(size)]
        offset = start + int(size) + 1
    return [(path, mode, contents.get(oid)) for path, mode, oid in entries]


def is_binary(data):
    return b"\0" in data[:8192]


def finding(findings, severity, check, subject, message):
    findings.append({"severity": severity, "check": check, "subject": subject, "message": message})


def review_npz(path, data, rule, findings):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = sorted(archive.namelist())
            for name in names:
                if not re.match(rule["members"], name):
                    finding(findings, "error", "allowlist", path, f"unexpected member {name}")
                    continue
                header = archive.read(name)[:512]
                if not header.startswith(b"\x93NUMPY"):
                    finding(findings, "error", "allowlist", path, f"{name} is not .npy data")
                elif b"'descr': '|O'" in header or b"'descr': 'O'" in header:
                    finding(findings, "error", "allowlist", path, f"{name} holds pickled objects")
    except zipfile.BadZipFile:
        finding(findings, "error", "allowlist", path, "not a valid npz archive")
        return []
    return names


def review_files(blobs, findings):
    categories, allowlisted, third_party = {}, [], []
    total = 0
    for path, mode, data in blobs:
        if mode == "160000":
            finding(findings, "error", "contents", path, "submodule content is not reviewed")
            continue
        if mode == "120000":
            finding(findings, "warning", "contents", path, "symbolic link")
        size = len(data)
        total += size
        name = path.rsplit("/", 1)[-1]
        lower = name.lower()
        category = next((c for c, match, _ in CATEGORIES if match(path)), "unclassified")
        bucket = categories.setdefault(category, {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += size
        if category == "unclassified":
            finding(findings, "error", "license-coverage", path, "no license category matches")
        if category == "third-party" and path not in THIRD_PARTY:
            finding(findings, "error", "license-coverage", path, "third-party file not listed")

        reasons = []
        suffix = "." + lower.rsplit(".", 1)[-1] if "." in lower[1:] else ""
        if suffix in FORBIDDEN_SUFFIXES:
            reasons.append(FORBIDDEN_SUFFIXES[suffix] + f" ({suffix})")
        if lower in FORBIDDEN_NAMES:
            reasons.append(FORBIDDEN_NAMES[lower])
        reasons += [why for prefix, why in FORBIDDEN_PREFIXES.items() if lower.startswith(prefix)]
        if size > MAX_BYTES:
            reasons.append(f"larger than {MAX_BYTES} bytes")
        binary = is_binary(data)
        if binary and not reasons:
            reasons.append("unreviewed binary file")
        rule = ALLOWLIST.get(path)
        if rule:
            if size > rule["max_bytes"]:
                finding(findings, "error", "allowlist", path, "exceeds allowlisted size")
            members = review_npz(path, data, rule, findings) if suffix == ".npz" else []
            allowlisted.append(
                {
                    "path": path,
                    "bytes": size,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "members": members,
                    "matched_rules": reasons,
                    "reason": rule["reason"],
                }
            )
        elif reasons:
            finding(findings, "error", "artifact", path, "; ".join(reasons))

        if binary:
            continue
        text = data.decode("utf8", errors="replace")
        for label, pattern in SECRET_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                finding(
                    findings, "error", "secret", f"{path}:{line}", f"{label} pattern (redacted)"
                )
        if path != "LICENSE" and path not in THIRD_PARTY:
            for match in FOREIGN_COPYRIGHT.finditer(text):
                line_text = text[match.start() : text.find("\n", match.start())]
                if OWNER not in line_text:
                    line = text.count("\n", 0, match.start()) + 1
                    finding(
                        findings,
                        "warning",
                        "foreign-copyright",
                        f"{path}:{line}",
                        "copyright notice in original-material file; confirm provenance",
                    )

    for path, info in sorted(THIRD_PARTY.items()):
        third_party.append(review_third_party(path, info, blobs, findings))
    for bucket_name, bucket in categories.items():
        license_name = next((lic for c, _, lic in CATEGORIES if c == bucket_name), "none")
        bucket["license"] = license_name
    return dict(sorted(categories.items())), allowlisted, third_party, total


def review_third_party(path, info, blobs, findings):
    by_path = {p: d for p, _, d in blobs}
    record = {"path": path, **{k: v for k, v in info.items() if k != "required_header"}}
    data = by_path.get(path)
    if data is None:
        finding(findings, "error", "third-party", path, "listed third-party file is not tracked")
        return record
    text = data.decode("utf8", errors="replace").replace("\r\n", "\n")
    sha = hashlib.sha256(text.encode("utf8")).hexdigest()
    header = text[:4000]
    missing = [phrase for phrase in info["required_header"] if phrase not in header]
    record.update(
        {
            "sha256_lf": sha,
            "matches_upstream": sha == info["upstream_sha256_lf"],
            "header_license_present": not missing,
        }
    )
    if missing:
        finding(findings, "error", "third-party", path, f"license header missing: {missing}")
    if sha != info["upstream_sha256_lf"]:
        finding(
            findings,
            "error",
            "third-party",
            path,
            "differs from pinned upstream; docs claim unmodified and STREAM 3b labels apply",
        )
    notice = by_path.get(info["notice_doc"])
    if notice is None or path.encode() not in notice:
        finding(findings, "error", "third-party", path, f"not described in {info['notice_doc']}")
    return record


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[([^\]]*)\])?[^;]*(?:;(.*))?$")
MARKER = re.compile(
    r"([A-Za-z_]+)\s*(===|==|!=|<=|>=|~=|<|>|not\s+in|in)\s*(\"[^\"]*\"|'[^']*')"
    r"|(\"[^\"]*\"|'[^']*')\s*(===|==|!=|<=|>=|~=|<|>|not\s+in|in)\s*([A-Za-z_]+)"
)
VERSION_VARS = {"python_version", "python_full_version", "implementation_version"}


def marker_environment():
    version = sys.implementation.version
    return {
        "python_version": "%d.%d" % sys.version_info[:2],
        "python_full_version": platform.python_version(),
        "implementation_version": f"{version.major}.{version.minor}.{version.micro}",
        "implementation_name": sys.implementation.name,
        "platform_python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_release": platform.release(),
        "sys_platform": sys.platform,
        "os_name": os.name,
    }


def version_key(text):
    return tuple(int(part) for part in re.findall(r"[0-9]+", text)[:4])


def compare(left, op, right, versioned):
    op = " ".join(op.split())
    if op in ("in", "not in"):
        return (left in right) == (op == "in")
    if versioned and op not in ("===",):
        if right.endswith(".*"):
            prefix = version_key(right[:-2])
            result = version_key(left)[: len(prefix)] == prefix
            return result if op == "==" else not result
        a, b = version_key(left), version_key(right)
        if op == "~=":
            return a >= b and a[: len(b) - 1] == b[: len(b) - 1]
        return {"==": a == b, "!=": a != b, "<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
    return {"==": left == right, "===": left == right, "!=": left != right}.get(op, False)


def marker_true(marker, env, extra):
    values = dict(env, extra=extra)

    def replace(match):
        if match.group(1):
            var, op, literal, reverse = match.group(1), match.group(2), match.group(3), False
        else:
            literal, op, var, reverse = match.group(4), match.group(5), match.group(6), True
        left, right = values.get(var, ""), literal[1:-1]
        if var == "extra":
            left, right = norm(left), norm(right)
        if reverse:
            left, right = right, left
        return " True " if compare(left, op, right, var in VERSION_VARS) else " False "

    expression = MARKER.sub(replace, marker)
    if not re.fullmatch(r"[\s()]*(?:(?:True|False|and|or|not)[\s()]*)*", expression):
        return None
    return bool(eval(expression, {"__builtins__": {}}, {}))  # tokens validated above


def license_of(meta):
    classifiers = sorted(
        c.split(" :: ", 1)[1]
        for c in meta.get_all("Classifier") or []
        if c.startswith("License ::")
    )
    expression = (meta.get("License-Expression") or "").strip()
    text = (meta.get("License") or "").strip()
    if expression:
        return expression, "License-Expression", classifiers
    if text and "\n" not in text and len(text) <= 80 and text.upper() != "UNKNOWN":
        return text, "License", classifiers
    if classifiers:
        return "; ".join(c.rsplit(" :: ", 1)[-1] for c in classifiers), "classifier", classifiers
    if text:
        return text.splitlines()[0][:80], "License (first line of text)", classifiers
    return "UNKNOWN", "none", classifiers


def review_dependencies(pyproject, findings):
    project = pyproject["project"]
    roots = {"runtime": project.get("dependencies", [])}
    roots.update(project.get("optional-dependencies", {}))
    env = marker_environment()
    packages, unevaluated = {}, []
    for group in sorted(roots, key=lambda g: (GROUP_ORDER + [g]).index(g)):
        queue = [(req, f"pyproject:{group}") for req in roots[group]]
        seen = set()
        while queue:
            requirement, parent = queue.pop(0)
            match = REQUIREMENT.match(requirement)
            if not match:
                finding(findings, "warning", "dependency", requirement, "unparsed requirement")
                continue
            key = norm(match.group(1))
            extras = sorted(norm(e) for e in (match.group(2) or "").split(",") if e.strip())
            entry = packages.setdefault(
                key, {"name": match.group(1), "groups": set(), "required_by": set()}
            )
            entry["groups"].add(group)
            entry["required_by"].add(parent)
            if (key, tuple(extras)) in seen:
                continue
            seen.add((key, tuple(extras)))
            try:
                dist = metadata.distribution(match.group(1))
            except metadata.PackageNotFoundError:
                entry.update(installed=False, version=None, license=NOT_INSTALLED)
                entry.update(license_source="none", license_classifiers=[])
                continue
            meta = dist.metadata
            license_name, source, classifiers = license_of(meta)
            entry.update(name=meta["Name"], installed=True, version=meta["Version"])
            entry.update(license=license_name, license_source=source)
            entry["license_classifiers"] = classifiers
            for child in meta.get_all("Requires-Dist") or []:
                child_match = REQUIREMENT.match(child)
                marker = (child_match.group(3) or "").strip() if child_match else ""
                if marker:
                    results = [marker_true(marker, env, extra) for extra in [""] + extras]
                    if None in results:
                        unevaluated.append(f"{meta['Name']}: {child}")
                    if not any(results) and None not in results:
                        continue
                queue.append((child, norm(meta["Name"])))

    table = []
    for key, entry in packages.items():
        groups = sorted(entry["groups"], key=lambda g: (GROUP_ORDER + [g]).index(g))
        licenses = " ".join([entry["license"], *entry.get("license_classifiers", [])])
        copyleft = bool(entry["installed"] and COPYLEFT.search(licenses))
        row = {
            "name": entry["name"],
            "version": entry["version"],
            "installed": entry["installed"],
            "license": entry["license"],
            "license_source": entry["license_source"],
            "license_classifiers": entry.get("license_classifiers", []),
            "groups": groups,
            "direct": any(p.startswith("pyproject:") for p in entry["required_by"]),
            "required_by": sorted(entry["required_by"]),
            "copyleft": copyleft,
        }
        table.append(row)
        if not entry["installed"]:
            finding(findings, "warning", "dependency-license", entry["name"], NOT_INSTALLED)
        elif entry["license"] == "UNKNOWN":
            finding(findings, "warning", "dependency-license", entry["name"], "license unknown")
        if copyleft:
            optional = "runtime" not in groups
            scope = "optional extra only" if optional else "runtime dependency"
            finding(
                findings,
                "info" if optional else "warning",
                "copyleft",
                entry["name"],
                f"{row['license']} ({scope}: {', '.join(groups)}); installed from PyPI by the "
                "user, not vendored or redistributed in the repository",
            )
    table.sort(key=lambda r: (GROUP_ORDER.index(r["groups"][0]), not r["direct"], norm(r["name"])))
    for item in sorted(set(unevaluated)):
        finding(findings, "info", "dependency", item, "marker not evaluated; included")
    return table, env


def review_imports(blobs, dependencies, findings):
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    local = {"tvara"} | {
        Path(p).stem
        for p, _, _ in blobs
        if p.endswith(".py") and p.startswith(("tools/", "tests/"))
    }
    modules = {}
    for path, _, data in blobs:
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(data.decode("utf8"), filename=path)
        except SyntaxError as error:
            finding(findings, "warning", "imports", path, f"not parsed: {error.msg}")
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            elif (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", getattr(node.func, "id", "")) == "importorskip"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                names = [node.args[0].value]
            for name in names:
                top = name.split(".")[0]
                if top not in stdlib and top not in local:
                    modules.setdefault(top, set()).add(path)
    mapping = metadata.packages_distributions()
    known = {norm(row["name"]): row for row in dependencies}
    rows = []
    for module in sorted(modules):
        dists = sorted({norm(d) for d in mapping.get(module, [])}) or [norm(module)]
        row = next((known[d] for d in dists if d in known), None)
        files = sorted(modules[module])
        if row is None:
            status = "undeclared"
            finding(
                findings,
                "warning",
                "imports",
                module,
                f"imported by {', '.join(files)} but not declared in pyproject.toml or "
                "resolved from an installed extra; not installed - license must be checked "
                "upstream",
            )
        elif row["direct"]:
            status = "declared"
        else:
            status = "transitive"
            finding(
                findings,
                "info",
                "imports",
                module,
                f"imported directly but declared only transitively via {row['required_by']}",
            )
        rows.append({"module": module, "status": status, "distributions": dists, "files": files})
    return rows


def review_repository(blobs, findings):
    by_path = {p: d for p, _, d in blobs}
    license_text = by_path.get("LICENSE", b"").decode("utf8", errors="replace")
    if "MIT License" not in license_text or OWNER_NOTICE not in license_text:
        finding(findings, "error", "license", "LICENSE", "MIT text or copyright line missing")
    ignore = by_path.get(".gitignore", b"").decode("utf8").splitlines()
    patterns = [line.strip() for line in ignore if line.strip() and not line.startswith("#")]
    for required in REQUIRED_IGNORES:
        if required not in patterns:
            finding(findings, "warning", "gitignore", required, "expected ignore pattern missing")
    actions = set()
    for path, data in by_path.items():
        if path.startswith(".github/workflows/"):
            text = data.decode("utf8", errors="replace")
            actions.update(re.findall(r"^\s*-?\s*uses:\s*([^\s#]+)", text, re.MULTILINE))
    return patterns, sorted(actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="report path (repo-relative)")
    parser.add_argument("--check", action="store_true", help="do not write the report")
    args = parser.parse_args()

    findings = []
    blobs = tracked_blobs()
    categories, allowlisted, third_party, total = review_files(blobs, findings)
    pyproject = tomllib.loads(dict((p, d) for p, _, d in blobs)["pyproject.toml"].decode("utf8"))
    project = pyproject["project"]
    if project.get("license") != {"file": "LICENSE"}:
        finding(findings, "error", "license", "pyproject.toml", "license is not LICENSE file")
    dependencies, env = review_dependencies(pyproject, findings)
    imports = review_imports(blobs, dependencies, findings)
    ignores, actions = review_repository(blobs, findings)
    index_matches_head = (
        subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0
    )
    order = {"error": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: (order[f["severity"]], f["check"], f["subject"], f["message"]))
    status = "fail" if any(f["severity"] == "error" for f in findings) else "pass"
    report = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "tool": "tools/review_release.py",
        "project": {"name": project["name"], "version": project["version"]},
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "marker_environment": env,
        "git": {
            "head": git("rev-parse", "HEAD").decode().strip(),
            "source": "git index blobs (git ls-files -s), not the working tree",
            "index_matches_head": index_matches_head,
        },
        "tracked_files": len(blobs),
        "total_bytes": total,
        "categories": categories,
        "third_party": third_party,
        "allowlisted_artifacts": allowlisted,
        "excluded_by_gitignore": ignores,
        "dependencies": dependencies,
        "imports": imports,
        "ci_actions": {
            "uses": actions,
            "note": "Executed by GitHub-hosted CI only; not redistributed with the release",
        },
        "limits": (
            "Dependency licenses come from installed metadata in this environment only; "
            "markers were evaluated for this interpreter and platform. Packages that are not "
            "installed are not resolved transitively. Secret scanning is pattern-based."
        ),
        "findings": findings,
    }
    counts = {s: sum(f["severity"] == s for f in findings) for s in order}
    for item in findings:
        if item["severity"] != "info":
            print(
                f"{item['severity'].upper()}: [{item['check']}] {item['subject']}: {item['message']}"
            )
    print(
        f"{status}: {len(blobs)} tracked files, {total} bytes, {len(dependencies)} dependencies, "
        f"{counts['error']} errors, {counts['warning']} warnings, {counts['info']} info"
    )
    if not args.check:
        target = ROOT / args.output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
        print(f"wrote {args.output}")
    return 1 if status == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
