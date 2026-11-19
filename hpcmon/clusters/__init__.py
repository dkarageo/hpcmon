"""Cluster-specific interfaces and the registry of supported cluster types."""

from __future__ import annotations

from typing import Dict, Optional, Type

from hpcmon.clusters.base import Cluster, CollectionError
from hpcmon.clusters.jupiter import JupiterCluster
from hpcmon.clusters.leonardo import LeonardoCluster
from hpcmon.clusters.lumi import LumiCluster
from hpcmon.clusters.meluxina import MeluxinaCluster
from hpcmon.clusters.slurm import GenericSlurmCluster, SlurmCluster
from hpcmon.config import ClusterConfig
from hpcmon.runner import SSHRunner

#: All supported cluster types, by the ``type`` used in the configuration.
CLUSTER_TYPES: Dict[str, Type[Cluster]] = {
    cls.type_name: cls
    for cls in (LumiCluster, JupiterCluster, LeonardoCluster, MeluxinaCluster, GenericSlurmCluster)
}


def create_cluster(config: ClusterConfig, runner: Optional[SSHRunner] = None) -> Cluster:
    """Instantiate the interface for the cluster described by ``config``.

    :raises KeyError: if the type of the cluster is not supported.
    """
    try:
        cls = CLUSTER_TYPES[config.type]
    except KeyError:
        raise KeyError(f"unsupported cluster type: {config.type!r}") from None
    return cls(config, runner=runner)


__all__ = [
    "CLUSTER_TYPES",
    "Cluster",
    "CollectionError",
    "GenericSlurmCluster",
    "JupiterCluster",
    "LeonardoCluster",
    "LumiCluster",
    "MeluxinaCluster",
    "SlurmCluster",
    "create_cluster",
]
