#!/bin/bash
# cluster_probe.sh -- survey the cluster from a head node.
#
# Every command here is a READ-ONLY scheduler query (pbsnodes / qstat). None of
# them execute anything on a compute node, so this does not run a job and does
# not touch policy 6.2. Run it on the head node after logging in:
#
#     ./scripts/cluster_probe.sh | tee results/cluster_survey.txt
#
# The point is to answer three questions before you submit anything:
#   1. which queues can you use, and what are their walltime / size limits
#   2. how many nodes exist, with how many cores and how much memory each
#   3. which nodes sit on the fast fabric (Omni-Path) versus 10 GbE -- because
#      a scaling curve assembled across both is not a valid comparison
set -u
command -v pbsnodes >/dev/null || { echo "no pbsnodes on PATH -- are you on a head node?"; exit 1; }
h(){ printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

h "server"
qstat -B 2>/dev/null || echo "(qstat -B unavailable)"

h "queues you can submit to"
qstat -Q 2>/dev/null

h "limits on the queues (walltime, ncpus, per-user caps)"
for q in $(qstat -Q 2>/dev/null | awk 'NR>2 {print $1}'); do
    echo "--- $q"
    qstat -Qf "$q" 2>/dev/null | grep -iE \
      "resources_max|resources_default|max_run|max_queued|enabled|started|acl_" \
      | sed 's/^ */  /'
done

h "node inventory: one line per node (state, cores used/total, jobs)"
pbsnodes -aSj 2>/dev/null | head -60
echo "  ... ($(pbsnodes -a 2>/dev/null | grep -c '^[^ ]' ) node entries in total)"

h "how many nodes of each shape (cores x memory)"
pbsnodes -a 2>/dev/null | awk '
  /^[^ \t]/                          {node=$1}
  /resources_available.ncpus/        {c=$3}
  /resources_available.mem/          {m=$3}
  /^$/ && c!="" {print c" cores, "m; c=""; m=""}
' | sort | uniq -c | sort -rn

h "distinct node properties / resources -- look here for the fabric tag"
# Sites tag the fast-interconnect nodes with a custom resource. It is usually
# something like opa, omnipath, infiniband, ib, or a switch/group name. Whatever
# it is called here, it will show up in this list.
pbsnodes -a 2>/dev/null \
  | grep -oE "resources_available\.[a-zA-Z_]+" \
  | sort -u | sed 's/^/  /'

h "values of the non-numeric resources (candidate fabric / partition tags)"
for r in $(pbsnodes -a 2>/dev/null | grep -oE "resources_available\.[a-zA-Z_]+" \
           | sort -u | sed 's/resources_available\.//' \
           | grep -viE "^(ncpus|mem|vmem|vnode|host|arch|file|naccelerators)$"); do
    vals=$(pbsnodes -a 2>/dev/null | grep "resources_available.$r " \
           | awk -F= '{print $2}' | tr -d ' ' | sort -u | head -8 | paste -sd, -)
    [ -n "$vals" ] && printf "  %-24s %s\n" "$r" "$vals"
done

h "what is free right now"
pbsnodes -aSj 2>/dev/null | awk 'NR>2 && $2 ~ /free/' | wc -l | \
  xargs -I{} echo "  {} nodes reporting free"

h "your queued and running jobs"
qstat -u "${USER:-$(whoami)}" 2>/dev/null || echo "  (none)"

cat <<'NOTE'

Next:
  * Pick a queue whose resources_max.walltime covers the job (strong.pbs asks
    for 2h) and whose size limits allow the select statement.
  * If a fabric tag showed up above, pin it in the select so every run of a
    sweep lands on the same interconnect, e.g.
        #PBS -l select=4:ncpus=32:mpiprocs=32:mem=64gb:<tag>=<value>
    and keep it identical across strong.pbs / shm_ablation.pbs, or the network
    becomes an uncontrolled variable.
  * Adjust #PBS -l select= in jobs/*.pbs to match what this queue actually
    grants. They currently request 4 nodes x 32 cores = 128 cores.
NOTE
