"""The unified model interface: ``load`` a model and run capabilities on it.

This is the only module that ties the others together. ``plms.load(name)``
returns a :class:`Model` whose ``embed``/``likelihood`` methods validate the
request against the model's manifest, stage inputs, drive the runner, and parse
the outputs into Python objects.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from plms.chunking import run_chunked
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
    read_generated,
    read_likelihoods,
    read_result,
    read_variant_scores,
    stage_file,
    stage_inputs,
)
from plms.registry import ModelEntry, Registry
from plms.runner import Runner, RunSpec, SubprocessDockerRunner, ensure_image

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


@dataclass
class GenerationResult:
    """Handle to the outputs of a ``generate`` run (FASTA parsed lazily)."""

    result: Result
    output_dir: Path
    _keepalive: tempfile.TemporaryDirectory | None = field(default=None, repr=False)

    def sequences(self) -> list[FastaRecord]:
        """Return the generated sequences (headers ``{prompt_id}__sample{k}``)."""
        return read_generated(self.output_dir, self.result)


class Model:
    """A loaded model exposing the unified inference capabilities."""

    def __init__(
        self, entry: ModelEntry, runner: Runner, manifest: Manifest, image_ref: str
    ) -> None:
        self._entry = entry
        self._runner = runner
        self._manifest = manifest
        self._image_ref = image_ref

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
        chunk_size: int | None = None,
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
            chunk_size: If set, split the input into runs of at most this many records and
                merge the outputs (resumable into a persistent output_dir).

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
        if chunk_size is not None:
            result, out_dir, keep = self._run_chunked(
                Capability.EMBED, records, extra, output_dir, use_gpu, chunk_size
            )
        else:
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
        chunk_size: int | None = None,
    ) -> LikelihoodResult:
        """Compute per-sequence log-likelihoods.

        The scoring method depends on the model: masked-marginal for masked LMs,
        causal (left-to-right) for autoregressive LMs. The method used is recorded
        in ``result.json`` under ``params.likelihood_method``.

        Args:
            fasta: Path to the input FASTA file.
            output_dir: Where to write outputs; a temporary directory if ``None``.
            use_gpu: Request all GPUs for the container run.
            batch_size: Override the model's default batch size.
            chunk_size: If set, split the input into runs of at most this many records and
                merge the outputs (resumable into a persistent output_dir).

        Raises:
            CapabilityNotSupportedError: If the model does not support likelihoods.
            InvalidRequestError: If the input is empty.
            ContainerExecutionError: If the container run fails.
        """
        self._require_capability(Capability.LIKELIHOOD)
        records = self._read_records(fasta)
        extra = ["--batch-size", str(batch_size)] if batch_size is not None else []
        if chunk_size is not None:
            result, out_dir, keep = self._run_chunked(
                Capability.LIKELIHOOD, records, extra, output_dir, use_gpu, chunk_size
            )
        else:
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

    def generate(
        self,
        prompts_fasta: str | Path,
        *,
        num_samples: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_length: int | None = None,
        seed: int | None = None,
        output_dir: Path | None = None,
        use_gpu: bool = False,
        batch_size: int | None = None,
        chunk_size: int | None = None,
    ) -> GenerationResult:
        """Sample sequences from an autoregressive model.

        Args:
            prompts_fasta: FASTA of prompt prefixes; an empty sequence means
                unconditional sampling. At least one record is required.
            num_samples: Samples to draw per prompt.
            temperature: Sampling temperature.
            top_p: Nucleus-sampling probability mass.
            max_length: Maximum sequence length (model default if ``None``).
            seed: Random seed for reproducible sampling.
            output_dir: Where to write outputs; a temporary directory if ``None``.
            use_gpu: Request all GPUs for the container run.
            batch_size: Override the model's default batch size.
            chunk_size: If set, split the input into runs of at most this many records and
                merge the outputs (resumable into a persistent output_dir).

        Raises:
            CapabilityNotSupportedError: If the model does not support generation.
            InvalidRequestError: If the prompts file contains no records.
            ContainerExecutionError: If the container run fails.
        """
        self._require_capability(Capability.GENERATE)
        records = self._read_records(prompts_fasta)
        extra = [
            "--num-samples",
            str(num_samples),
            "--temperature",
            str(temperature),
            "--top-p",
            str(top_p),
        ]
        if max_length is not None:
            extra += ["--max-length", str(max_length)]
        if seed is not None:
            extra += ["--seed", str(seed)]
        if batch_size is not None:
            extra += ["--batch-size", str(batch_size)]
        if chunk_size is not None:
            result, out_dir, keep = self._run_chunked(
                Capability.GENERATE, records, extra, output_dir, use_gpu, chunk_size
            )
        else:
            result, out_dir, keep = self._run(
                Capability.GENERATE, stage_inputs(records), extra, output_dir, use_gpu
            )
        return GenerationResult(result=result, output_dir=out_dir, _keepalive=keep)

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

    def _run_into_dir(
        self,
        capability: Capability,
        staging,  # contextmanager[StagedInput]
        extra_args: list[str],
        out_dir: Path,
        use_gpu: bool,
    ) -> Result:
        """Run one container job into ``out_dir`` and return its parsed Result."""
        out_dir.mkdir(parents=True, exist_ok=True)
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
                image=self._image_ref,
                command=command,
                input_dir=staged.input_dir,
                output_dir=out_dir,
                use_gpu=use_gpu,
            )
            run_result = self._runner.run(spec)
        if run_result.exit_code != 0:
            self._raise_container_error(run_result)
        return read_result(out_dir)

    def _run(
        self,
        capability: Capability,
        staging,  # contextmanager[StagedInput]
        extra_args: list[str],
        output_dir: Path | None,
        use_gpu: bool,
    ) -> tuple[Result, Path, tempfile.TemporaryDirectory | None]:
        out_dir, keep = self._resolve_output_dir(output_dir)
        result = self._run_into_dir(capability, staging, extra_args, out_dir, use_gpu)
        return result, out_dir, keep

    def _run_chunked(
        self,
        capability: Capability,
        records: list[FastaRecord],
        extra_args: list[str],
        output_dir: Path | None,
        use_gpu: bool,
        chunk_size: int,
    ) -> tuple[Result, Path, tempfile.TemporaryDirectory | None]:
        out_dir, keep = self._resolve_output_dir(output_dir)

        def run_chunk(chunk: list[FastaRecord], chunk_dir: Path) -> Result:
            return self._run_into_dir(
                capability, stage_inputs(chunk), extra_args, chunk_dir, use_gpu
            )

        result = run_chunked(
            capability=capability.value,
            records=records,
            chunk_size=chunk_size,
            output_dir=out_dir,
            run_chunk=run_chunk,
        )
        return result, out_dir, keep

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


def _resolve_allow_pull(allow_pull: bool | None) -> bool:
    """Resolve pull policy: explicit arg wins, else consult ``PLMS_NO_PULL``."""
    if allow_pull is not None:
        return allow_pull
    no_pull = os.environ.get("PLMS_NO_PULL", "").strip().lower() in {"1", "true", "yes"}
    return not no_pull


def load(
    name: str,
    *,
    runner: Runner | None = None,
    registry: Registry | None = None,
    allow_pull: bool | None = None,
) -> Model:
    """Resolve a model name and return a ready-to-use :class:`Model`.

    Resolves the name against the registry, ensures the pinned image is present
    locally (pulling it when permitted), reads the image's manifest, checks
    contract compatibility, and constructs the model.

    Args:
        name: A model name or alias known to the registry.
        runner: The container runner (defaults to a local docker subprocess runner).
        registry: The model registry (defaults to the packaged registry).
        allow_pull: Whether to pull a missing image. ``None`` (default) consults
            the ``PLMS_NO_PULL`` environment variable.

    Raises:
        ModelNotFoundError: If the name is unknown.
        ImageNotFoundError: If the image is absent and pulling is disabled.
        ImagePullError: If the image must be pulled and the pull fails.
        ContractVersionError: If the image's contract major version mismatches.
    """
    runner = runner or SubprocessDockerRunner()
    registry = registry or Registry.load()
    entry = registry.resolve(name)
    ref = entry.pinned_ref()
    ensure_image(runner, ref, allow_pull=_resolve_allow_pull(allow_pull), model_name=name)
    manifest = Manifest.model_validate_json(runner.manifest(ref))
    check_contract_compatibility(manifest.contract_version)
    return Model(entry, runner, manifest, image_ref=ref)
