# License history

Vey's public lineage has three licenses, each anchored to explicit
revisions. Grants already made are not revoked by later changes: a snapshot
received under a license keeps that license.

| Period | License | Anchors |
|---|---|---|
| Early releases | Apache-2.0 | tags `v1.0.0`, `v2.0.0-rc1` |
| Research period | AGPL-3.0-only | commits `e52bcf2` through `12e87ce` |
| Current | FSL-1.1-MIT | tag `v2.0.0-rc3` and the rewritten `main` forward |

## Current license

FSL-1.1-MIT permits use, modification, and redistribution, including
commercially, with one restriction during a two-year protection period: do
not offer a competing product or service whose value derives substantially
from Vey's functionality. Every version becomes plain MIT two years after
its distribution date. See [LICENSE](../LICENSE) and
[COMMERCIAL-LICENSE.md](../COMMERCIAL-LICENSE.md).

Vey is Fair Source during the protection period, not OSI open source; a
version is open source once it converts to MIT.

## Rewritten history

The public `main` branch was rebuilt from a clean root in September 2026 so
the repository presents a coherent engineering history rather than the
experimental sequence that produced it. The complete original history,
including every retraction and failed mechanism, is preserved on the public
`research-history` branch and in a private bundle, and summarized in
[docs/research/](research/STEF_HISTORY.md) and
[docs/research/NEGATIVE_RESULTS.md](research/NEGATIVE_RESULTS.md).

The rewrite does not change any license grant. Snapshots obtained before the
rewrite keep the license they were distributed under, identified by their
content and any surviving tags, regardless of whether those commits remain
reachable from the rewritten `main`.
