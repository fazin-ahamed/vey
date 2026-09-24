# Security Policy

## Supported release

| Version | Supported |
|---|---|
| 1.x | Yes |

## Reporting a vulnerability

**Do not file security-sensitive issues as public GitHub Issues.**

Use GitHub's private vulnerability reporting:

<https://github.com/fazin-ahamed/vey/security/advisories/new>

Please include a description, the affected version, and a minimal reproduction
if you have one. You can expect an acknowledgement within a few days.

## What to report

We are especially interested in:

- **Unsafe action compilation.** Any way to make the action compiler emit a
  value that has no evidence in the input state, emit an argument that violates
  the declared schema, or bypass the ASK_FOR_INFO path. This is the property
  Vey claims to guarantee, so a counterexample is a high-priority report.
- **Trust policy bypasses.** A way to obtain `ACT` where the deterministic
  policy should have produced `ASK_FOR_INFO`, `ABSTAIN`, or `ESCALATE`.
- **Dependency vulnerabilities** in the runtime, especially in
  `transformers`, `torch`, or `huggingface_hub` deserialization paths.
- **Unsafe model or dataset loading.** Path traversal, arbitrary code execution
  during model load, or unsafe deserialization (`pickle`, arbitrary
  `trust_remote_code`, unsafe archive extraction).
- **Path traversal or file handling** in recording, replay, or export paths.
- **Credential exposure.** Any committed key, token, or secret, including in
  test fixtures, recorded outputs, or example configuration.

## What is out of scope

- Accuracy or calibration findings. Those are research issues, not
  vulnerabilities; please file them as Issues with your reproduction.
- Denial of service through deliberately large user-supplied inputs that do not
  cross a trust boundary.
- Vulnerabilities in upstream dependencies that have no reachable path through
  Vey's code and a fix available upstream.
