# Security

Report vulnerabilities privately through GitHub security advisories on this repository rather than
public issues.

Scope notes for v0:

* Deterministic alternatives, verifiers and step bodies run **in-process, without a sandbox**. Only run
  pipelines and benchmark suites you trust.
* Steps declared `side_effects="external"` are executed at most once per call and never retried by
  any policy; replay never re-executes actions.
* The ledger stores step inputs and outputs in plain text (SQLite). Do not put secrets in step inputs;
  keep the ledger file private.
