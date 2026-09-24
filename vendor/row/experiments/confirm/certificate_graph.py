"""Typed, content-addressed dependency graph for primary certificates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


GRAPH_SCHEMA = "pldr-certificate-graph-v1"
DERIVATION_KINDS = {
    "root_artifact",
    "exact_recurrence",
    "rational_interval_evaluation",
    "induced_norm_inequality",
    "exact_ldl_witness",
    "positive_comparison",
    "finite_horizon",
    "diagnostic_float",
    "empirical_estimator",
}
PRIMARY_KINDS = DERIVATION_KINDS - {"diagnostic_float"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _text(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _sha256(value, label):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _parent_names(value):
    if not isinstance(value, (list, tuple)):
        raise ValueError("certificate parents must be an array")
    if any(not isinstance(name, str) or not name for name in value):
        raise ValueError("certificate parent names must be nonempty strings")
    if len(set(value)) != len(value):
        raise ValueError("certificate parents contain duplicates")
    return tuple(value)


def canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def digest_object(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CertificateNode:
    name: str
    derivation_kind: str
    value: object
    unit: str
    domain: str
    constructor: str
    constructor_sha256: str
    parents: tuple[str, ...]
    metadata: dict

    def payload(self):
        return {
            "name": self.name,
            "derivation_kind": self.derivation_kind,
            "value": self.value,
            "unit": self.unit,
            "domain": self.domain,
            "constructor": self.constructor,
            "constructor_sha256": self.constructor_sha256,
            "parents": list(self.parents),
            "metadata": self.metadata,
        }

    @property
    def digest(self):
        return digest_object(self.payload())

    def to_object(self):
        return {**self.payload(), "sha256": self.digest}


class CertificateGraph:
    def __init__(self):
        self._nodes = {}

    def add(self, node):
        if not isinstance(node, CertificateNode):
            raise TypeError("certificate graph accepts CertificateNode objects")
        name = _text(node.name, "certificate node name")
        _text(node.unit, "certificate node unit")
        _text(node.domain, "certificate node domain")
        _text(node.constructor, "certificate node constructor")
        _sha256(node.constructor_sha256, "constructor digest")
        if not isinstance(node.metadata, dict):
            raise ValueError("certificate node metadata must be an object")
        parents = _parent_names(node.parents)
        if node.derivation_kind not in DERIVATION_KINDS:
            raise ValueError("unknown certificate derivation kind")
        if name in self._nodes:
            raise ValueError(f"duplicate certificate node {name}")
        missing = [parent for parent in parents if parent not in self._nodes]
        if missing:
            raise ValueError("certificate parents must precede their child: "
                             + ", ".join(missing))
        if node.derivation_kind == "root_artifact" and parents:
            raise ValueError("root certificate node cannot have parents")
        if node.derivation_kind != "root_artifact" and not parents:
            raise ValueError("derived certificate node has no parents")
        canonical_json(node.value)
        canonical_json(node.metadata)
        self._nodes[name] = node
        return node.digest

    def node(self, name):
        return self._nodes[name]

    def to_object(self, *, outputs):
        outputs = _parent_names(outputs)
        if not outputs:
            raise ValueError("certificate graph must declare at least one output")
        missing = [name for name in outputs if name not in self._nodes]
        if missing:
            raise ValueError("graph output nodes are missing: " + ", ".join(missing))
        payload = {
            "schema_version": GRAPH_SCHEMA,
            "nodes": [node.to_object() for node in self._nodes.values()],
            "outputs": list(outputs),
        }
        return {**payload, "graph_sha256": digest_object(payload)}


def validate_graph(graph, *, require_primary_outputs=True):
    if not isinstance(graph, dict):
        raise ValueError("certificate graph must be an object")
    if set(graph) != {"schema_version", "nodes", "outputs", "graph_sha256"}:
        raise ValueError("certificate graph has missing or unknown fields")
    if graph["schema_version"] != GRAPH_SCHEMA:
        raise ValueError("certificate graph schema is stale")
    if not isinstance(graph["nodes"], list) or not graph["nodes"]:
        raise ValueError("certificate graph nodes must be a nonempty array")
    outputs = _parent_names(graph["outputs"])
    if not outputs:
        raise ValueError("certificate graph outputs must be nonempty")
    _sha256(graph["graph_sha256"], "certificate graph digest")
    payload = {name: graph[name] for name in ("schema_version", "nodes", "outputs")}
    if digest_object(payload) != graph["graph_sha256"]:
        raise ValueError("certificate graph digest mismatch")
    realized = {}
    for raw in graph["nodes"]:
        if not isinstance(raw, dict) or set(raw) != {
            "name", "derivation_kind", "value", "unit", "domain",
            "constructor", "constructor_sha256", "parents", "metadata", "sha256",
        }:
            raise ValueError("certificate node has missing or unknown fields")
        name = _text(raw["name"], "certificate node name")
        _text(raw["unit"], "certificate node unit")
        _text(raw["domain"], "certificate node domain")
        _text(raw["constructor"], "certificate node constructor")
        _sha256(raw["constructor_sha256"], "constructor digest")
        _sha256(raw["sha256"], "certificate node digest")
        parents = _parent_names(raw["parents"])
        if not isinstance(raw["metadata"], dict):
            raise ValueError("certificate node metadata must be an object")
        node_payload = {key: raw[key] for key in raw if key != "sha256"}
        if digest_object(node_payload) != raw["sha256"]:
            raise ValueError(f"certificate node digest mismatch: {name}")
        if name in realized:
            raise ValueError(f"duplicate certificate node {name}")
        if raw["derivation_kind"] not in DERIVATION_KINDS:
            raise ValueError("unknown certificate derivation kind")
        missing = [parent for parent in parents if parent not in realized]
        if missing:
            raise ValueError("certificate graph is not topologically ordered")
        if raw["derivation_kind"] == "root_artifact" and parents:
            raise ValueError("root certificate node cannot have parents")
        if raw["derivation_kind"] != "root_artifact" and not parents:
            raise ValueError("derived certificate node has no parents")
        realized[name] = raw
    for name in outputs:
        if name not in realized:
            raise ValueError(f"unknown certificate output node {name}")
        if (require_primary_outputs
                and realized[name]["derivation_kind"] not in PRIMARY_KINDS):
            raise ValueError(f"diagnostic node cannot be a primary output: {name}")
    return realized


def load_graph(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        graph = json.load(stream)
    validate_graph(graph)
    return graph
