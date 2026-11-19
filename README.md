# hpcmon

[![PyPI](https://img.shields.io/pypi/v/hpcmon)](https://pypi.org/project/hpcmon/)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.14-blue)](https://pypi.org/project/hpcmon/)
[![Tests](https://github.com/dkarageo/hpcmon/actions/workflows/tests.yml/badge.svg)](https://github.com/dkarageo/hpcmon/actions/workflows/tests.yml)
[![Coverage: 100%](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/dkarageo/hpcmon/actions/workflows/tests.yml)
[![License](https://img.shields.io/pypi/l/hpcmon)](https://github.com/dkarageo/hpcmon/blob/main/LICENSE)

`hpcmon` is a command-line tool that shows, in one view, your available resources on multiple EuroHPC clusters:

- **compute budgets**: what remains until the end of each project, and until the end of the
  month for clusters that split budgets per month;
- **storage**: used and available space (and file counts) of your home and project areas;
- **jobs**: your running and pending Slurm jobs;
- **GPUs**: the GPUs of the nodes of every cluster, and every budget that can be spent on them
  converted to an estimate in **GPU-hours**, so that the budgets of different centres (core-hours,
  node-hours, ...) can be compared.

It connects to every cluster with your own ssh config, runs the site's accounting tools and presents the
results in the same format for every cluster.

Supported systems: **LUMI**, **JUPITER**, **Leonardo** and **MeluXina**. Any other Slurm cluster
can be added to show its jobs only.

## Installation

Install it with pip:

```bash
pip install hpcmon
```

You can also install it in an environment of its own, so that
its dependencies never conflict with those of your other Python packages, with
[pipx](https://pipx.pypa.io/) (`pipx install hpcmon`) or [uv](https://docs.astral.sh/uv/)
(`uv tool install hpcmon`).

To install the development version, directly from GitHub:

```bash
pip install git+https://github.com/dkarageo/hpcmon.git
```

or from a clone of this repository:

```bash
git clone https://github.com/dkarageo/hpcmon.git
cd hpcmon
pip install .
```

`hpcmon` requires Python 3.9 or later on your local machine (Linux or macOS; the automated tests
run on Linux). Nothing needs to be installed on the clusters.

## Prerequisites: SSH access to clusters

`hpcmon` runs locally and connects to each cluster through SSH. By default it runs
`ssh lumi`, `ssh jupiter`, `ssh leonardo` and `ssh meluxina`, so the simplest setup is a host alias
for every cluster in `~/.ssh/config`:

```sshconfig
Host lumi
    HostName lumi.csc.fi
    User <your LUMI username>
    IdentityFile ~/.ssh/id_lumi

Host leonardo
    HostName login.leonardo.cineca.it
    User <your CINECA username>
```

If you use other host names or need extra SSH options, set them in the configuration file (see
*Configuration* below).

Several centres require multi-factor authentication or short-lived certificates (e.g. JUPITER's
OTP, or Leonardo's `step` certificates). Make sure that `ssh <cluster>` works without prompting
before you run `hpcmon`. An SSH *ControlMaster* is the easiest way to do this: you authenticate
once, and the connection is reused for hours:

```sshconfig
Host lumi jupiter leonardo meluxina
    ControlMaster auto
    ControlPath ~/.ssh/control-%h-%p-%r
    ControlPersist 12h
```

If a cluster does prompt for a password or OTP, run `hpcmon --sequential` (or set
`parallel = false`). The clusters are then queried one after the other, without the progress
spinner, so the SSH prompts are visible and don't overlap. If you would rather have `hpcmon` fail
fast than prompt, add `-o BatchMode=yes` to the SSH command of that cluster.

## Usage

```bash
hpcmon                      # all enabled clusters, all sections
hpcmon lumi leonardo        # only these clusters
hpcmon --jobs               # only the jobs (fast)
hpcmon leonardo --compute   # options and cluster names can be mixed freely
hpcmon --json | jq .        # machine-readable output
```

Example output (illustrative data):

```text
LUMI · GPU nodes: 4× AMD MI250X 128 GB (2 GCDs each) ─────────────────────────────────────────────────────────
Compute allocations

  Project             Resource           Total        Remaining    Ends (days left)   Notes
 ─────────────────────────────────────────────────────────────────────────────────────────────────────
  project_465000123   CPU [core-h]      50,000    50,000 (100%)   2027-04-04 (137d)
  project_465000123   GPU [GPU-h]       12,000    12,000 (100%)   2027-04-04 (137d)
  project_465000123   Storage [TB-h]   200,000   200,000 (100%)   2027-04-04 (137d)   storage billing

Storage

  Area                             Used       Quota        Free    Use   Files / quota
 ──────────────────────────────────────────────────────────────────────────────────────
  /users/jdoe1234              20.0 KiB    25.0 GiB    25.0 GiB   0.0%        5 / 150k
  /project/project_465000123    4.0 KiB   100.0 GiB   100.0 GiB   0.0%        1 / 200k
  /scratch/project_465000123    4.0 KiB    20.0 TiB    20.0 TiB   0.0%          1 / 4M
  /flash/project_465000123      4.0 KiB     1.0 TiB     1.0 TiB   0.0%          1 / 3M

Jobs: 2 running, 1 pending

  Job ID    Name               Partition    State                Time / limit   Nodes   Node list / reason
 ──────────────────────────────────────────────────────────────────────────────────────────────────────────
  1234567   train|model [v2]   standard-g   RUNNING      1:02:03 / 2-00:00:00       4   nid[005000-005003]
  1234570   cleanup            dev-g        COMPLETING          10:00 / 30:00       1   nid005100
  1234568   eval               small-g      PENDING            0:00 / 1:00:00       1   (Priority)


Leonardo · GPU nodes: 4× NVIDIA A100 64 GB ───────────────────────────────────────────────────────────────────
Compute allocations

                                                              Month          Month
  Project         Resource            Total       Remaining   total      remaining    Ends (days left)
 ──────────────────────────────────────────────────────────────────────────────────────────────────────
  EUHPC_D00_001   Booster [core-h]   96,000    92,813 (97%)   8,250   8,250 (100%)   2027-06-30 (224d)
                                              ~11,602 GPU-h           ~1,031 GPU-h

Storage

  Area                                                Used      Quota       Free    Use   Files / quota
 ───────────────────────────────────────────────────────────────────────────────────────────────────────
  /leonardo_work/EUHPC_D00_001                    87.0 GiB    2.0 TiB    1.9 TiB   4.2%               -
  /leonardo_scratch/fast/EUHPC_D00_001             4.0 KiB    2.0 TiB    2.0 TiB   0.0%               -
  /leonardo_scratch/large/userexternal/jdoe5678    4.0 KiB   no quota          -      -               -
  /leonardo/pub/userexternal/jdoe5678              4.0 KiB   60.0 GiB   60.0 GiB   0.0%               -
  /leonardo/home/userexternal/jdoe5678             3.4 GiB   60.0 GiB   56.6 GiB   5.7%               -

Note: as of 3 hours ago
No jobs in the queue.

MeluXina · GPU nodes: 4× NVIDIA A100 40 GB ───────────────────────────────────────────────────────────────────
Compute allocations

                                               Month        Month
  Project   Resource       Total   Remaining   total    remaining   Ends (days left)
 ────────────────────────────────────────────────────────────────────────────────────
  p200001   GPU [node-h]       -           -     210    183 (87%)                  -
                                                       ~732 GPU-h

Note: current month only
...

All clusters ─────────────────────────────────────────────────────────────────────────────────────────────────
GPU-hours left (estimated)

                                                    Month
 Cluster   Project                Remaining     remaining   Ends (days left)
 ───────────────────────────────────────────────────────────────────────────
 LUMI      project_465000123  12,000 (100%)             -  2027-04-04 (137d)
 Leonardo  EUHPC_D00_001       11,602 (97%)  1,031 (100%)  2027-06-30 (224d)
 MeluXina  p200001                        -     732 (87%)                  -

Note: GPU-hours (of the GPUs named next to each cluster) are estimates: they
assume that jobs use all the GPUs of the nodes they are charged for (on
shared nodes, no more than the cores and memory that come with their GPUs).
```

*Remaining* is what is left until the end of the project. *Month remaining* is what is left until
the end of the current month, for clusters that split budgets per month. Below the amounts of the
budgets that can be spent on GPU nodes, `~N GPU-h` is the estimate of what they buy in GPU-hours
(see *GPUs and GPU-hours* below). When several clusters have such budgets, a summary of the
GPU-hours left on all of them is printed last.

### Options

| Option | Description |
| --- | --- |
| `CLUSTER ...` | Names of the clusters to query (default: all enabled clusters). A disabled cluster is queried when it is named explicitly. |
| `--compute`, `--storage`, `--jobs` | Show only these sections (any combination). By default all sections are shown. |
| `--json` | Print the results as JSON instead of tables. |
| `-c PATH`, `--config PATH` | Use this configuration file. |
| `-t SECONDS`, `--timeout SECONDS` | Time limit for each cluster (overrides the configuration). |
| `--sequential` | Query the clusters one after the other (and without the progress spinner). |
| `--no-color` | Disable colours. |
| `--debug` | Log the commands run on the clusters and their raw output (useful to report parsing problems). |
| `--init-config [--force]` | Write a commented default configuration file and exit (`--force` overwrites an existing file). |
| `--show-config` | Show the configured clusters and exit. |
| `-V`, `--version` | Show the version and exit. |

Exit status:

- `0`: every cluster was queried successfully. Warnings don't count.
- `1`: the data of at least one cluster is incomplete. The problem is shown under that cluster.
- `2`: usage or configuration error.
- `130`: interrupted with Ctrl-C.

## What is shown for each cluster

| Cluster | Compute budgets | Per month | Project end | Storage | Commands used |
| --- | --- | --- | --- | --- | --- |
| LUMI | CPU core-hours, GPU-hours and storage TB-hours per project | no (LUMI does not split budgets per month) | yes ("days of compute left") | home, `/project`, `/scratch`, `/flash` (space and files) | `lumi-allocations`, `lumi-ldap-projectinfo`, `lumi-workspaces` |
| JUPITER | core-hours per budget account on the system | for budgets that JSC does not account in "fixed" mode, as far as `jutil` reports them | yes (end of the accounting period) | home, project, scratch and fast-scratch directories (space and inodes) | `jutil user cpuquota`, `jutil user dataquota`, `jutil project dataquota` |
| Leonardo | local core-hours per account, Booster and DCGP partitions | yes (`monthTotal`/`monthConsumed`) | yes | home, public, work and scratch areas | `saldo -b`, `saldo -b --dcgp`, `cindata` |
| MeluXina | node-hours of the current month (CPU, GPU, FPGA, large memory) | yes: this is all that MeluXina reports | no (it can be set in the configuration) | home and project directories (space and files) | `myquota` |
| any Slurm cluster (`type = "slurm"`) | – | – | – | – | – |

Jobs are listed with `squeue` on all clusters.

Notes and limitations:

- Amounts are shown in each centre's own unit (core-hours, GPU-hours, node-hours, ...). The budgets
  that can be spent on GPU nodes are also estimated in GPU-hours (see below).
- **LUMI**: the storage billing budget (TB-hours) is listed with the compute budgets, as LUMI accounts
  for it the same way. It is marked "storage billing". The end of a project is computed from the
  "days of compute left" reported by LUMI.
- **JUPITER**: budgets are reported for the current system only (`jutil ... --currsys`). *Remaining*
  is the quota minus the consumption until the end of the accounting period. For accounting modes
  other than "fixed", the accounting mode and the amount available right now (when it differs) are
  shown as notes. Only the "fixed" mode could be checked on a real system. If you are not allowed to
  read the quotas of a project, your own usage in it is shown instead.
- **Leonardo**: `cindata` takes around 30–40 seconds, so the storage section of Leonardo is the
  slowest part of a full query. Use `hpcmon leonardo --compute --jobs` for a quick look. For project
  areas (work, fast scratch), the usage and size of the whole area are shown. For personal areas
  (paths under `userexternal`/`userinternal`), your own usage and quota are shown. `cindata`
  refreshes its figures periodically, so their age is shown as a note.
- **MeluXina**: `myquota` reports neither the total budget of a project nor its end date, so only
  the current month is shown. You can set the end dates of your projects with `project_end` in the
  configuration. `myquota` also refuses to run twice within 10 seconds. If you run `hpcmon` again too
  quickly, MeluXina asks you to retry later.
- Storage quotas show the *soft* limit, i.e. the quota you are expected to stay below. "no quota"
  means that the cluster reports no limit for that area.

## GPUs and GPU-hours

Each cluster's name is followed by the GPUs of its GPU nodes. The budgets that can be spent on these
nodes are converted to **GPU-hours**: hours of one of these GPUs. GPUs are taken to be the primary
resource, so the estimates assume that jobs use the GPU nodes as efficiently as the accounting
allows:

- on clusters that allocate **whole nodes** (JUPITER, MeluXina, LUMI's `standard-g`), jobs use all
  the GPUs of their nodes. A job that uses 1 GPU of a 4-GPU node still pays for 4 GPU-hours per hour;
- on clusters that **share nodes** (Leonardo's Booster, LUMI's `small-g` and `dev-g`), jobs request
  no more than the cores and memory that come with their GPUs (e.g. 25% of the node for 1 of its 4
  GPUs). Asking for more than the cores or memory billed with a GPU (last column) makes the same
  GPU-hour cost more.

| Cluster | GPU nodes | Budget converted | GPU-hours per unit | Allocation | Billed with each GPU |
| --- | --- | --- | --- | --- | --- |
| LUMI | 4× AMD MI250X 128 GB, 2 GCDs each (Slurm counts 8 GPUs of 64 GB) | GPU [GPU-h] | 1 (a LUMI GPU-hour is one MI250X, i.e. 2 GCDs, for an hour) | shared (`small-g`, `dev-g`) or whole nodes (`standard-g`) | 16 cores and 128 GB per MI250X: LUMI bills each GCD half a GPU-hour per hour, or more if a job takes more than 8 cores or 64 GB per GCD (only 56 cores of a node, 7 per GCD, are usable) |
| JUPITER | 4× NVIDIA GH200 96 GB (288 Grace cores per node) | jupiter_booster [core-h] | 1/72 (a node-hour is charged 288 core-hours) | whole nodes | – |
| Leonardo | 4× NVIDIA A100 64 GB (32 cores per node) | Booster [core-h] | 1/8 (a node-hour is charged 32 core-hours) | shared | 8 cores and 1/4 of the node memory (the default memory follows the cores) |
| MeluXina | 4× NVIDIA A100 40 GB | GPU [node-h] | 4 | whole nodes | – |

The other budgets cannot be spent on GPU nodes and get no estimate: LUMI's CPU core-hours (LUMI-C
and the visualisation nodes) and storage TB-hours, Leonardo's DCGP core-hours (CPU-only nodes), and
MeluXina's CPU, FPGA and large-memory node-hours.

The rates come from the centres' accounting rules, and were checked against the accounting of real
jobs: the core-hours charged by JUPITER and Leonardo, and the node-hours charged by MeluXina, are the
node-hours of the jobs times 288, 32 and 1. GPUs of different clusters are of course not equally
fast: the GPU-hours tell how long you can use the GPUs of each cluster, not how much work they do.

The rates can be changed, or added for other budgets of the supported systems, with the
`gpu_hours_per_unit` setting, and the GPU description with `gpus` (see *Configuration*). For generic
Slurm clusters (`type = "slurm"`), which show jobs only, `gpus` describes their GPU nodes. The estimates are also in the JSON output
(`gpu_hours_per_unit`, `gpu_hours_total`, `gpu_hours_remaining`, `month_gpu_hours_total`,
`month_gpu_hours_remaining` of every budget, and `gpus` of every cluster).

## Configuration

`hpcmon` works without any configuration and queries the four clusters with `ssh <name>`. To change
this, create a configuration file:

```bash
hpcmon --init-config      # writes ~/.config/hpcmon/config.toml
hpcmon --show-config      # shows what hpcmon will use
```

The configuration is read from the file given with `--config`, or else from the file named by the
`HPCMON_CONFIG` environment variable. With either of these, the file must exist. Otherwise it is
read from `$XDG_CONFIG_HOME/hpcmon/config.toml` (by default `~/.config/hpcmon/config.toml`), if that
file exists. It is a [TOML](https://toml.io) file:

```toml
[settings]
timeout = 180        # time limit for each cluster, in seconds
parallel = true      # query all clusters at the same time
login_shell = true   # run remote commands in a login shell (loads the site's PATH)

[clusters.lumi]
type = "lumi"                      # lumi, jupiter, leonardo, meluxina or slurm
ssh = "ssh lumi"                   # how to connect: a string or a list of arguments
projects = ["project_465000123"]   # only show these projects (default: all)

[clusters.jupiter]
ssh = ["ssh", "-o", "ConnectTimeout=20", "jupiter"]   # `type` can be omitted when the name is a type

[clusters.leonardo]
ssh = "ssh -l jdoe0000 login.leonardo.cineca.it"
timeout = 240                      # per-cluster override

[clusters.meluxina]
ssh = "ssh meluxina"
enabled = false                    # skipped unless named on the command line: `hpcmon meluxina`

[clusters.meluxina.project_end]    # end dates for clusters that don't report them
p200001 = 2027-06-30

[clusters.my-uni-cluster]          # any other Slurm cluster: shows the jobs only
type = "slurm"
ssh = "ssh hpc.example.org"
gpus = "8× NVIDIA H100 80 GB"      # describes its GPU nodes
```

When the file has a `[clusters]` section, only the clusters listed in it are used. A cluster table
accepts these keys:

| Key | Default | Description |
| --- | --- | --- |
| `type` | the table name | The kind of system, selecting how it is queried: `lumi`, `jupiter`, `leonardo`, `meluxina` or `slurm`. |
| `ssh` | `"ssh <name>"` | The command used to connect. `hpcmon` appends `bash -l -s` to it and sends the commands on standard input. |
| `enabled` | `true` | Whether the cluster is queried when no cluster is named on the command line. |
| `timeout` | `settings.timeout` (180) | Time limit, in seconds, for the commands run on this cluster. Another 15 seconds are allowed for connecting. |
| `login_shell` | `settings.login_shell` (true) | Whether to run the commands in a login shell. |
| `projects` | all projects | Only show these projects (case-insensitive): their compute budgets, their storage areas and the jobs charged to them. Personal storage is always shown. A warning tells how many jobs are hidden, and names any configured project the cluster does not know (to catch typos). |
| `project_end` | – | A table of `project = date` giving project end dates, used when the cluster does not report them. |
| `gpus` | built in for the supported systems | Description of the GPUs of a GPU node, shown next to the cluster name. `""` hides it. |
| `gpu_hours_per_unit` | built in (see *GPUs and GPU-hours*) | A table of `resource = rate`: the GPU-hours bought by one unit of the budgets of that resource, replacing the built-in rate. The resource is the *Resource* column without the `[unit]` (e.g. `GPU`, `Booster` or `jupiter_booster`; case-insensitive). `0` disables the estimate. A resource that matches no budget of the cluster is reported as a warning. Example: `[clusters.leonardo.gpu_hours_per_unit]` with `Booster = 0.125`. |

You can use any SSH setup: jump hosts (`ssh -J gateway lumi`), a different user
(`ssh -l user host`), or a wrapper script. The command only has to run its last argument
(`bash -l -s`) on the cluster with the standard input attached.

## How it works

For every cluster, `hpcmon` opens **one** SSH session. In it, a small bash script runs all the
commands needed for the requested sections, one after the other. The fast ones (jobs, compute) go
first, and the slow storage tools go last. Each command's output is enclosed in markers, so any
banner printed by the login shell is ignored, and a failing command affects only its own section.
The commands share the time limit of the cluster: a command that exceeds what is left of it is
stopped, and the results of the commands that completed are still shown. Each command's standard
input comes from `/dev/null`. All clusters are queried in parallel, and the results are printed in
the order of the configuration as soon as they are available.

Each supported system has a class in `hpcmon/clusters/` that derives from `hpcmon.clusters.Cluster`
(or `SlurmCluster`). The class declares the remote commands needed for each section, and parses
their output into the common model of `hpcmon/models.py`: `ComputeBudget`, `StorageQuota` and `Job`.
To support another cluster:

1. Create `hpcmon/clusters/<name>.py` with a subclass of `SlurmCluster`. Set `type_name` and
   `display_name`, and implement `compute_commands()`, `storage_commands()`, `parse_compute()` and
   `parse_storage()`. Set `gpus` (the GPUs of a node) and `gpu_rates` (GPU-hours per unit of each
   `(resource, unit)` of budget that can be spent on GPUs).
2. Add the class to `CLUSTER_TYPES` in `hpcmon/clusters/__init__.py`.
3. Add the real output of the tools to `tests/fixtures/<name>/` (anonymised) and tests in
   `tests/test_<name>.py`.

`hpcmon --debug` prints the raw output of every remote command, which is the easiest way to capture
fixtures.

## Development

From a clone of the repository:

```bash
pip install -r requirements.txt
pip install -e .

pytest                    # runs the test suite with coverage (it requires 100%)
```

The tests never connect to a cluster. They use mock outputs of the cluster tools, stored in
`tests/fixtures/`.

The tests also run on GitHub Actions (`.github/workflows/tests.yml`) for every push and pull
request, on all the supported Python versions and with the oldest supported dependencies.

## PyPI Releases

Releases are published on PyPI by GitHub Actions (`.github/workflows/release.yml`) when a release
is published on GitHub. The workflow runs the tests, checks that the release tag is the package
version, builds the sdist and the wheel, and uploads them with PyPI's *trusted publishing*, so no
API token is stored anywhere.

To release a version:

1. Set `__version__` in `hpcmon/__init__.py` (e.g. `"0.2.0"`) and add the changes to `CHANGELOG.md`.
2. Commit and push.
3. Create a release on GitHub with a new tag named after the version (`v0.2.0` or `0.2.0`).

## License

MIT. See [LICENSE](https://github.com/dkarageo/hpcmon/blob/main/LICENSE).
