"""The unified model interface: ``load`` a model and run capabilities on it.

This is the only module that ties the others together. ``plms.load(name)``
returns a :class:`Model` whose ``embed``/``likelihood`` methods validate the
request against the model's manifest, stage inputs, drive the runner, and parse
the outputs into Python objects.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from plms.contract import (
    Capability,
    Manifest,
    Result,
    check_contract_compatibility,
    parse_container_error,
)
from plms.exceptions import (
    CapabilityNotSupportedError,
    ContainerExecutionError,
    InvalidRequestError,
)
from plms.io import (
    check_csv_has_columns,
    load_per_residue_embeddings,
    load_pooled_embeddings,
    read_fasta,
    read_likelihoods,
    read_result,
    read_variant_scores,
    stage_file,
    stage_inputs,
)
from plms.registry import ModelEntry, Registry
from plms.runner import Runner, RunSpec, SubprocessDockerRunner

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from plms.io import FastaRecord

logger = logging.getLogger(__name__)

_STDERR_TAIL_LINES = 50


@dataclass
class EmbeddingResult:
    """Handle to the outputs of an ``embed`` run (arrays loaded lazily)."""

    result: Result
    output_dir: Path
    pooling: str
    _keepalive: tempfile.TemporaryDirectory | None = field(default=None, repr=False)

    def pooled(self) -> dict[str, np.ndarray]:
        """Return pooled embeddings keyed by record id (shape ``(embedding_dim,)``)."""
        return load_pooled_embeddings(self.output_dir, self.result)

    def per_residue(self) -> dict[str, np.ndarray]:
        """Return per-residue embeddings keyed by record id (shape ``(L, embedding_dim)``)."""
        return load_per_residue_embeddings(self.output_dir, self.result)


@dataclass
class LikelihoodResult:
    """Handle to the outputs of a ``likelihood`` run (CSV parsed lazily)."""

    result: Result
    output_dir: Path
    _keepalive: tempfile.TemporaryDirectory | None = field(default=None, repr=False)

    def rows(self) -> list[dict[str, str | int | float | None]]:
        """Return one row per record with likelihood/perplexity columns."""
        return read_likelihoods(self.output_dir, self.result)


@dataclass
class ScoreResult:
    """Handle to the outputs of a ``score`` run (CSV parsed lazily)."""

    result: Result
    output_dir: Path
    method: str
    _keepalive: tempfile.TemporaryDirectory | None = field(default=None, repr=False)

    def rows(self) -> list[dict[str, str | int | float | None]]:
        """Return one row per variant: variant_id, mutant, n_mutations, score."""
        return read_variant_scores(self.output_dir, self.result)


class Model:
    """A loaded model exposing the unified inference capabilities."""

    def __init__(self, entry: ModelEntry, runner: Runner, manifest: Manifest) -> None:
        self._entry = entry
        self._runner = runner
        self._manifest = manifest

    @property
    def manifest(self) -> Manifest:
        """The model's declared manifest."""
        return self._manifest

    @property
    def name(self) -> str:
        """The model's canonical name."""
        return self._manifest.name

    def embed(
        self,
        fasta: str | Path,
        *,
        pooling: str = "mean",
        layers: Sequence[int] = (-1,),
        output_dir: Path | None = None,
        use_gpu: bool = False,
        batch_size: int | None = None,
    ) -> EmbeddingResult:
        """Compute embeddings for the sequences in ``fasta``.

        Args:
            fasta: Path to the input FASTA file.
            pooling: One of the model's supported pooling modes (``mean``/``cls``
                produce pooled vectors; ``none`` produces per-residue arrays).
            layers: Transformer layer indices to read (negative indexing allowed).
            output_dir: Where to write outputs; a temporary directory if ``None``.
            use_gpu: Request all GPUs for the container run.
            batch_size: Override the model's default batch size.

        Raises:
            CapabilityNotSupportedError: If the model does not support embedding.
            InvalidRequestError: If ``pooling`` is unsupported or the input is empty.
            ContainerExecutionError: If the container run fails.
        """
        self._require_capability(Capability.EMBED)
        self._require_pooling(pooling)
        records = self._read_records(fasta)
        extra = ["--pooling", pooling, "--layers", ",".join(str(x) for x in layers)]
        if batch_size is not None:
            extra += ["--batch-size", str(batch_size)]
        result, out_dir, keep = self._run(
            Capability.EMBED, stage_inputs(records), extra, output_dir, use_gpu
        )
        return EmbeddingResult(result=result, output_dir=out_dir, pooling=pooling, _keepalive=keep)

    def likelihood(
        self,
        fasta: str | Path,
        *,
        output_dir: Path | None = None,
        use_gpu: bool = False,
        batch_size: int | None = None,
    ) -> LikelihoodResult:
        """Compute pseudo-log-likelihoods for the sequences in ``fasta``.

        Raises:
            CapabilityNotSupportedError: If the model does not support likelihoods.
            InvalidRequestError: If the input is empty.
            ContainerExecutionError: If the container run fails.
        """
        self._require_capability(Capability.LIKELIHOOD)
        records = self._read_records(fasta)
        extra = ["--batch-size", str(batch_size)] if batch_size is not None else []
        result, out_dir, keep = self._run(
            Capability.LIKELIHOOD, stage_inputs(records), extra, output_dir, use_gpu
        )
        return LikelihoodResult(result=result, output_dir=out_dir, _keepalive=keep)

    def score(
        self,
        variants_csv: str | Path,
        *,
        method: str = "masked-marginal",
        output_dir: Path | None = None,
        use_gpu: bool = False,
        batch_size: int | None = None,
    ) -> ScoreResult:
        """Score sequence variants for effect.

        Args:
            variants_csv: CSV with columns ``variant_id, wt_sequence, mutant``.
            method: ``"masked-marginal"`` (default) or ``"wt-marginal"``.
            output_dir: Where to write outputs; a temporary directory if ``None``.
            use_gpu: Request all GPUs for the container run.
            batch_size: Override the model's default batch size.

        Raises:
            CapabilityNotSupportedError: If the model does not support scoring.
            InvalidRequestError: If ``method`` is invalid or the CSV lacks columns.
            ContainerExecutionError: If the container run fails.
        """
        self._require_capability(Capability.SCORE)
        if method not in ("masked-marginal", "wt-marginal"):
            raise InvalidRequestError(
                f"unsupported scoring method {method!r}; choose 'masked-marginal' or 'wt-marginal'"
            )
        path = Path(variants_csv)
        check_csv_has_columns(path, ("variant_id", "wt_sequence", "mutant"))
        extra = ["--method", method]
        if batch_size is not None:
            extra += ["--batch-size", str(batch_size)]
        result, out_dir, keep = self._run(
            Capability.SCORE, stage_file(path, "variants.csv"), extra, output_dir, use_gpu
        )
        return ScoreResult(result=result, output_dir=out_dir, method=method, _keepalive=keep)

    # --- internals ---------------------------------------------------------

    def _require_capability(self, capability: Capability) -> None:
        if capability not in self._manifest.capabilities:
            raise CapabilityNotSupportedError(
                f"model {self._manifest.name!r} does not support {capability.value!r}; "
                f"supported: {[c.value for c in self._manifest.capabilities]}"
            )

    def _require_pooling(self, pooling: str) -> None:
        supported = {m.value for m in self._manifest.pooling_modes}
        if pooling not in supported:
            raise InvalidRequestError(
                f"unsupported pooling {pooling!r}; model supports {sorted(supported)}"
            )

    def _read_records(self, fasta: str | Path) -> list[FastaRecord]:
        records = read_fasta(Path(fasta))
        if not records:
            raise InvalidRequestError(f"input FASTA {fasta} contains no records")
        too_long = [r.id for r in records if len(r.sequence) > self._manifest.max_sequence_length]
        if too_long:
            logger.warning(
                "%d sequence(s) exceed max_sequence_length=%d and will be truncated by the "
                "container: %s",
                len(too_long),
                self._manifest.max_sequence_length,
                too_long[:5],
            )
        return records

    def _run(
        self,
        capability: Capability,
        staging,  # contextmanager[StagedInput]
        extra_args: list[str],
        output_dir: Path | None,
        use_gpu: bool,
    ) -> tuple[Result, Path, tempfile.TemporaryDirectory | None]:
        out_dir, keep = self._resolve_output_dir(output_dir)
        with staging as staged:
            command = [
                capability.value,
                "--input",
                staged.container_input_path,
                "--output",
                "/out",
                *extra_args,
            ]
            spec = RunSpec(
                image=self._entry.image,
                command=command,
                input_dir=staged.input_dir,
                output_dir=out_dir,
                use_gpu=use_gpu,
            )
            run_result = self._runner.run(spec)
        if run_result.exit_code != 0:
            self._raise_container_error(run_result)
        return read_result(out_dir), out_dir, keep

    @staticmethod
    def _resolve_output_dir(
        output_dir: Path | None,
    ) -> tuple[Path, tempfile.TemporaryDirectory | None]:
        if output_dir is not None:
            out = Path(output_dir).resolve()
            out.mkdir(parents=True, exist_ok=True)
            return out, None
        keep = tempfile.TemporaryDirectory(prefix="plms-out-")
        return Path(keep.name).resolve(), keep

    @staticmethod
    def _raise_container_error(run_result) -> None:  # noqa: ANN001
        tail = "\n".join(run_result.stderr.splitlines()[-_STDERR_TAIL_LINES:])
        err = parse_container_error(run_result.stderr)
        if err is not None:
            raise ContainerExecutionError(
                err.message,
                error_type=err.error_type,
                details=err.details,
                exit_code=run_result.exit_code,
                stderr_tail=tail,
            )
        raise ContainerExecutionError(
            f"container exited with code {run_result.exit_code}",
            exit_code=run_result.exit_code,
            stderr_tail=tail,
        )


def load(
    name: str,
    *,
    runner: Runner | None = None,
    registry: Registry | None = None,
) -> Model:
    """Resolve a model name and return a ready-to-use :class:`Model`.

    Resolves the name against the registry, reads the image's manifest, checks
    contract compatibility, and constructs the model.

    Args:
        name: A model name or alias known to the registry.
        runner: The container runner (defaults to a local docker subprocess runner).
        registry: The model registry (defaults to the packaged registry).

    Raises:
        ModelNotFoundError: If the name is unknown.
        ImageNotFoundError: If the image is not available locally.
        ContractVersionError: If the image's contract major version mismatches.
    """
    runner = runner or SubprocessDockerRunner()
    registry = registry or Registry.load()
    entry = registry.resolve(name)
    manifest = Manifest.model_validate_json(runner.manifest(entry.image))
    check_contract_compatibility(manifest.contract_version)
    return Model(entry, runner, manifest)
