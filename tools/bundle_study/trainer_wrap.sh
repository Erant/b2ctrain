#!/bin/bash
# b2ctrain with $B2C_EXTRA appended to training invocations (not to --help / --version probes)
T=${B2C_TRAINER:-$(dirname "$0")/../../build/b2ctrain}
if [ $# -gt 1 ] && [ -n "$B2C_EXTRA" ]; then exec $T "$@" $B2C_EXTRA; else exec $T "$@"; fi
