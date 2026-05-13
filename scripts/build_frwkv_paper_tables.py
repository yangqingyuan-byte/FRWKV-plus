#!/usr/bin/env python3
"""
Backward-compatible entrypoint.

The old FRWKV paper-table script encoded an outdated CrossBranchGate-centric
story. Keep this filename as a compatibility shim, but delegate to the current
KBS-oriented table builder.
"""

from build_kbs_phasegate_tables import main


if __name__ == "__main__":
    main()
