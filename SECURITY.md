# Security

Workflow code is untrusted by design: the platform runs it in a sandbox, and `qsolve lint` flags
secrets and imports the sandbox does not allow. If you find a way for a workflow to escape the
sandbox, read another run's data, forge a measured number (circuits, shots, fidelity, cost) or abuse
the lint and validation gates, please report it privately instead of opening a public issue.

Use GitHub's private vulnerability reporting on this repository (Security tab, "Report a
vulnerability") with what you did, what you expected and what happened, and allow reasonable time
for a fix before disclosing it.
