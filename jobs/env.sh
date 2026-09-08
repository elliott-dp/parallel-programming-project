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

# `module` is a shell function, and a PBS batch shell does not always inherit
# it. Source the init script if it is missing, trying both Lmod and the classic
# environment-modules locations, before deciding it is unavailable.
if ! command -v module >/dev/null 2>&1; then
    for _init in /usr/share/lmod/lmod/init/bash \
                 /usr/share/Modules/init/bash \
                 /etc/profile.d/modules.sh \
                 /etc/profile.d/lmod.sh \
                 "${MODULESHOME:-/nonexistent}/init/bash"; do
        [ -r "$_init" ] && . "$_init" 2>/dev/null && break
    done
fi

_have_module() { command -v module >/dev/null 2>&1; }

# Defaults for the UniTN cluster (EasyBuild module tree). OpenMPI 4.1.6 is
# chosen because it is the version the code was verified against; the toolchain
# module pulls in its matching GCC automatically. OpenMPI/5.0.3-GCC-13.3.0 is
# also available and gives MPI-4 -- switch with MPI_MODULE= if you want it.
: "${MPI_MODULE:=OpenMPI/4.1.6-GCC-13.2.0}"
: "${GCC_MODULE:=GCC/13.2.0}"

if _have_module; then
    module load "$GCC_MODULE" 2>/dev/null \
        || echo "note: could not load $GCC_MODULE; relying on the MPI toolchain"

    if [ -n "${MPI_MODULE:-}" ]; then
        module load "$MPI_MODULE" 2>/dev/null \
            || echo "note: $MPI_MODULE not available here; falling back to discovery"
    fi

    if ! command -v mpicc >/dev/null 2>&1; then
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

# --- provenance -----------------------------------------------------------
# This cluster interconnects some nodes with 10 GbE and others with Omni-Path.
# A comparison across runs that landed on different fabrics is meaningless, so
# record what each job actually got. Guide S11 asks for node IDs to be noted.
if [ -n "${PBS_NODEFILE:-}" ] && [ -r "$PBS_NODEFILE" ]; then
    _tag="${PBS_JOBNAME:-job}.${PBS_JOBID:-nojobid}"
    mkdir -p results
    { echo "# jobid=${PBS_JOBID:-?} name=${PBS_JOBNAME:-?} queue=${PBS_QUEUE:-?}"
      echo "# unique nodes:"; sort -u "$PBS_NODEFILE"
      echo "# slots: $(wc -l < "$PBS_NODEFILE")"
    } > "results/${_tag}.nodes"
    echo "# nodes recorded in results/${_tag}.nodes"

    # Assert the allocation is actually spread over the nodes the experiment
    # needs. PBS defaults to place=free, and this cluster's nodes average 76
    # cores, so four 32-core chunks can legally land on ONE node. A job that
    # believes it is multi-node but is not will report that the shared-memory
    # broadcast makes no difference and that scaling is perfect -- both wrong,
    # and both indistinguishable from a real result. Fail loudly instead.
    _got=$(sort -u "$PBS_NODEFILE" | wc -l)
    echo "# distinct nodes in this allocation: $_got"
    if [ -n "${EXPECT_NODES:-}" ] && [ "$_got" -ne "${EXPECT_NODES}" ]; then
        echo "ERROR: expected ${EXPECT_NODES} distinct nodes, got $_got."
        echo "       The results from this allocation would not mean what the"
        echo "       experiment claims. Check '#PBS -l place=scatter:excl' is"
        echo "       present and that the queue honours it."
        exit 1
    fi
fi
