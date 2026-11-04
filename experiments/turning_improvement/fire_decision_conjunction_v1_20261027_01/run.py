"""One explicitly authorized OFFLINE supervised fit; never constructs a PPO/env."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from . import metrics
from .model import Conjunction, SCHEMA, fit, probability


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VERIFIED = ROOT / "experiments/turning_improvement/fire_head_provenance_loader_v1_20261027_01"
HISTORICAL = ROOT / "experiments/turning_improvement/fire_head_matched_adapter_20261027_01"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def exclusive(output):
    target = Path(output).resolve()
    if target.parent != (ROOT / "experiments/turning_improvement").resolve() or target.exists():
        raise FileExistsError("New exclusive direct experiment directory required")
    return target


def reviewed_code():
    # Execute no historical runner: retain its frozen dependency contract by
    # reading the literal list and checking hashes with standard-library code.
    adapter = HISTORICAL / "adapter.py"
    tree = ast.parse(adapter.read_bytes().decode("utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reviewed_code_hashes")
    assignment = next(n for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "dependencies" for t in n.targets))
    dependencies = ast.literal_eval(assignment.value)
    record = HISTORICAL / "protected_before.json"
    pins = json.loads(record.read_bytes().decode("utf-8"))
    for name in dependencies:
        if sha(ROOT / name) != pins.get(name):
            raise ValueError("Frozen dependency differs: " + name)
    paths = [ROOT / name for name in dependencies] + [record, adapter, HISTORICAL / "data.py"]
    paths += [VERIFIED / "loader.py", VERIFIED / "entrypoint.py"]
    paths += [HERE / name for name in ("model.py", "metrics.py", "run.py")]
    return {str(path.resolve()): sha(path) for path in paths}


def actual_loader():
    from experiments.turning_improvement.fire_head_provenance_loader_v1_20261027_01.entrypoint import runtime_load
    return runtime_load()  # Pure data/label/layout contracts, not its run() entry.


def guard(pins):
    if any(sha(path) != value for path, value in pins.items()):
        raise ValueError("Input/code provenance changed")


def execute(output, data_loader, code_pins):
    """Testable execution body. Caller must already have explicit authorization."""
    output = exclusive(output)
    output.mkdir(exist_ok=False)
    try:
        guard(code_pins)
        prepared = data_loader()
        inputs = dict(prepared.snapshot_hashes)
        guard(inputs); guard(code_pins)
        write(output / "plan.json", dict(schema="direct-observation-conjunction-experiment-v1",
              algorithm="minimum positive-containing rectangle; reject contradictory training labels",
              source_initialization="Candidate A flight policy unchanged; no source actor/head loaded",
              input_space="existing frozen normalized 19-observation vector", used_features=[10, 11],
              deterministic=True, fit_passes=1, optimizer_updates=0, threshold=.5,
              source_checkpoint=prepared.metadata["source_bundle"], inputs_sha256=inputs,
              code_sha256=code_pins, dataset_lineage=prepared.lineage,
              qualification=dict(minimum_eligible_windows=1, max_false_positive_rate=.005, max_false_request_windows=5),
              validation_usage="one held-out monitoring/qualification pass; never boundary fitting"))
        # Only this split reaches fit(); no selected policy actions or gate
        # fields are used by the model. No tuning/search after validation.
        model, fit_record = fit(prepared.training)
        def paired(rows):
            return [{**row, "pair_id": prepared.pair_registry[row["episode_id"]][1]} for row in rows]
        train_rows, validation_rows = paired(prepared.training), paired(prepared.validation)
        train_metrics, train_windows = metrics.monitor(model, train_rows)
        validation, validation_windows = metrics.monitor(model, validation_rows)
        guard(inputs); guard(code_pins)
        eligible = metrics.qualifies(validation)
        parameters = model.to_dict()
        artifact = dict(schema="direct-observation-fire-artifact-v1", model=parameters,
                        qualified=eligible, source_pair_hashes=prepared.metadata["source_pair_hashes"],
                        dataset_sha256=prepared.lineage["merged_dataset_sha256"],
                        input_sha256=inputs, code_sha256=code_pins, fit=fit_record)
        # Always retain diagnostic parameters in result.json. Only a qualified
        # model gets a loadable qualified_model.json; never a PPO checkpoint.
        if eligible:
            write(output / "qualified_model.json", artifact)
        write(output / "result.json", dict(status="fit_completed", qualified=eligible,
              model_schema=SCHEMA, model_parameters=parameters, fit=fit_record,
              training=train_metrics, validation=validation,
              saved_model="qualified_model.json" if eligible else None,
              input_sha256=inputs, code_sha256=code_pins,
              environment_steps=0, PPO_updates=0, optimizer_updates=0,
              combat_performance_established=False))
        write(output / "windows.json", dict(training=train_windows, validation=validation_windows))
        with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
            for row in train_rows + validation_rows:
                score = model.logit(row["normalized_observation"])
                stream.write(json.dumps({key: row[key] for key in ("split", "episode_id", "pair_id", "step", "label", "fit_mask")}
                      | dict(logit=score, probability=probability(score), fire_request=probability(score) >= .5),
                      sort_keys=True, allow_nan=False) + "\n")
        hashes = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        guard(inputs); guard(code_pins)
        write(output / "integrity.json", dict(protected_inputs_unchanged=True, code_unchanged=True, output_sha256=hashes))
        return output
    except BaseException as error:
        write(output / "error.json", dict(type=type(error).__name__, message=str(error), automatic_retry=False))
        raise


def load_qualified(path, expected_file_sha256, expected_source_pair):
    path = Path(path)
    if path.name != "qualified_model.json" or (path.parent / "error.json").exists():
        raise ValueError("Only a successfully completed qualified artifact may load")
    integrity = json.loads((path.parent / "integrity.json").read_bytes().decode("utf-8"))
    result_raw = (path.parent / "result.json").read_bytes()
    if hashlib.sha256(result_raw).hexdigest() != integrity["output_sha256"]["result.json"]:
        raise ValueError("Completion record integrity mismatch")
    result = json.loads(result_raw.decode("utf-8"))
    if result.get("status") != "fit_completed" or result.get("qualified") is not True:
        raise ValueError("No qualified completed fit")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise ValueError("Model artifact hash mismatch")
    if integrity["output_sha256"][path.name] != expected_file_sha256:
        raise ValueError("Artifact differs from completed-fit inventory")
    data = json.loads(raw.decode("utf-8"))
    if (data.get("schema") != "direct-observation-fire-artifact-v1" or data.get("qualified") is not True
            or data.get("source_pair_hashes") != expected_source_pair):
        raise ValueError("Unqualified model or mismatched source/normalization lineage")
    return Conjunction.from_dict(data["model"])


def run(output, authorize_fit=False):
    if authorize_fit is not True:
        raise PermissionError("Explicit offline-fitting authorization required")
    exclusive(output)  # Before source imports/data access.
    return execute(output, actual_loader, reviewed_code())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorize-fit", action="store_true")
    args = parser.parse_args(argv)
    run(args.output, args.authorize_fit)


if __name__ == "__main__":
    main()
