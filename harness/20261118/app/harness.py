"""Copy this file to /app/harness.py. The platform appends describe, train or predict."""
import sys

from contest_harness.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
