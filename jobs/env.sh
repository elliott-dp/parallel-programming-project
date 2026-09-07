# jobs/env.sh -- sourced by every PBS script. Loads the compiler and MPI.
#
# The architecture guide warns that the MPI module is not named like the gcc
# one, and that you must run `module avail` to find it. Rather than leave a
# placeholder in six files for you to forget, this discovers it.
#
# Override either name explicitly if the guess is wrong:
#     GCC_MODULE=gcc91 MPI_MODULE=mpich-3.2.1--gcc-9.1.0 qsub jobs/strong.pbs
#
# (PBS passes the submitting environment through by default; if your site
#  disables that, use `qsub -v GCC_MODULE,MPI_MODULE ...`.)

set -u

_have_module() { command -v module >/dev/null 2>&1 || [ -n "${MODULESHOME:-}" ]; }

if _have_module; then
    module load "${GCC_MODULE:-gcc91}" 2>/dev/null \
        || echo "note: could not load ${GCC_MODULE:-gcc91}; using the default compiler"

    if [ -n "${MPI_MODULE:-}" ]; then
        module load "$MPI_MODULE" || { echo "ERROR: MPI_MODULE=$MPI_MODULE failed to load"; exit 1; }
    elif ! command -v mpicc >/dev/null 2>&1; then
        # Prefer an MPI built against the same gcc, then any MPI at all.
        _av=$(module avail 2>&1 | tr ' ' '\n' | grep -iE '^(openmpi|mpich|mvapich|intel-mpi|impi)' | grep -v '^$')
        _pick=$(printf '%s\n' "$_av" | grep -i "gcc" | head -1)
        [ -z "$_pick" ] && _pick=$(printf '%s\n' "$_av" | head -1)
        if [ -n "$_pick" ]; then
            echo "using MPI module: $_pick"
            module load "$_pick" || { echo "ERROR: failed to load $_pick"; exit 1; }
        fi
    fi
fi

if ! command -v mpicc >/dev/null 2>&1; then
    echo "ERROR: no mpicc on PATH after module loading."
    echo "Run 'module avail' on this cluster, find the MPI module, and resubmit with:"
    echo "    MPI_MODULE=<its-exact-name> qsub \$0"
    exit 1
fi

echo "# toolchain: $(mpicc -show 2>/dev/null | head -1)"
echo "# mpicc:     $(command -v mpicc)"
mpicc --version 2>/dev/null | head -1
