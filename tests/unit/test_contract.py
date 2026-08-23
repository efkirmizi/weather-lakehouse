"""The window and channel geometry every job in this image has to agree on.

Runs inside dag-pytorch-model-training (PYTHONPATH=/app).
"""
import ast

import torch

import batch_inference
import evaluate_and_promote
import lakehouse
from models import ConvLSTMWeatherForecaster, TimeSeriesTransformer


def test_the_contract_declares_the_geometry():
    assert lakehouse.SEQ_LEN == 72
    assert lakehouse.PRED_LEN == 24
    assert lakehouse.INPUT_CHANNELS == 16
    assert lakehouse.OUTPUT_CHANNELS == 4
    assert lakehouse.TEMPERATURE_CHANNEL == 0


def test_only_the_output_channels_are_forecast():
    """Calendar channels are deterministic and the exogenous ones are inputs;
    forecasting them would make the task harder for nothing."""
    assert lakehouse.OUTPUT_CHANNELS < lakehouse.INPUT_CHANNELS


def test_every_consumer_reads_the_same_window_geometry():
    """These used to be literals in three files. A mismatch does not raise - it
    slices a different window and reports a confident wrong number."""
    assert batch_inference.SEQ_LEN == lakehouse.SEQ_LEN
    assert batch_inference.PRED_LEN == lakehouse.PRED_LEN
    assert evaluate_and_promote.SEQ_LEN == lakehouse.SEQ_LEN
    assert evaluate_and_promote.PRED_LEN == lakehouse.PRED_LEN


def _batch():
    return torch.zeros(2, lakehouse.SEQ_LEN, lakehouse.INPUT_CHANNELS)


def test_both_architectures_map_the_input_width_to_the_output_width():
    """input_dim and output_dim used to be the same number passed twice. They are
    genuinely different now, and passing the input width as the output width would
    have the model emitting calendar predictions nobody reads.

    Built from the shipped CONFIG dicts rather than from literals. The capacity in
    those dicts is expected to change - the hyperparameters were sized when the input
    was four channels wide - and a copy of them here would quietly stop testing the
    architecture this pipeline actually trains. It also means a capacity change that
    does not construct at all, such as an n_heads that no longer divides d_model,
    fails here rather than an hour into a GPU run."""
    import train_lstm
    import train_transformer

    expected = (2, lakehouse.PRED_LEN, lakehouse.OUTPUT_CHANNELS)
    lstm_config, transformer_config = train_lstm.CONFIG, train_transformer.CONFIG

    lstm = ConvLSTMWeatherForecaster(
        lstm_config["input_dim"], lstm_config["hidden_dim"], lstm_config["output_dim"],
        lstm_config["pred_len"], lstm_config["dropout"]
    )
    transformer = TimeSeriesTransformer(
        transformer_config["input_dim"], transformer_config["d_model"],
        transformer_config["n_heads"], transformer_config["num_layers"],
        transformer_config["dim_feedforward"], transformer_config["output_dim"],
        transformer_config["pred_len"], transformer_config["dropout"]
    )

    assert lstm(_batch()).shape == expected
    assert transformer(_batch()).shape == expected


def test_the_training_configs_agree_with_the_contract():
    import train_lstm
    import train_transformer

    for config in (train_lstm.CONFIG, train_transformer.CONFIG):
        assert config["input_dim"] == lakehouse.INPUT_CHANNELS
        assert config["output_dim"] == lakehouse.OUTPUT_CHANNELS
        assert config["seq_len"] == lakehouse.SEQ_LEN
        assert config["pred_len"] == lakehouse.PRED_LEN


def _config_keys_referenced_in_source(module):
    """Every string literal used as a CONFIG[...] subscript key anywhere in the
    module's source, found by parsing rather than importing. Importing only runs
    the lines that execute at module load; the call sites that matter here live
    inside main(), behind the __main__ guard."""
    with open(module.__file__) as f:
        tree = ast.parse(f.read())
    return {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name) and node.value.id == "CONFIG"
        and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)
    }


def test_every_config_key_read_in_source_is_defined_in_the_dict():
    """CONFIG["feature_dim"] was deleted from both training scripts' CONFIG dicts,
    but the model-construction call sites kept reading it. Those call sites live
    inside main(), behind the __main__ guard, so importing the module for its
    CONFIG dict - which is all test_the_training_configs_agree_with_the_contract
    above does - never reaches them; nothing caught the dead key until a real
    training run hit the KeyError. Parsing the source instead of executing it is
    what reaches those lines."""
    import train_lstm
    import train_transformer

    for module in (train_lstm, train_transformer):
        missing = _config_keys_referenced_in_source(module) - module.CONFIG.keys()
        assert not missing, f"{module.__name__} reads undefined CONFIG keys: {missing}"


def _call_line_in_main(module, func_name):
    """Line of the first call to func_name inside main(), found by parsing.

    Same reason as _config_keys_referenced_in_source above: the calls that matter
    are inside main(), which importing the module never runs."""
    with open(module.__file__) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for call in ast.walk(node):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id == func_name):
                    return call.lineno
    return None


def test_the_warm_start_decision_precedes_everything_that_reads_it():
    """warm_start() returns the *effective* incremental flag: False means the
    champion's weights did not fit the architecture, so this run is a scratch run.

    get_dataloaders and resolve_epochs branch on that same flag - the data window, the
    epoch budget and the learning rate all come from it. Put either of them above the
    warm_start call and a declined warm start still trains a randomly initialised
    model for two epochs at 1e-4 on the recent-window split: not a scratch run, not an
    incremental one, and nothing in the logs or the registry would look wrong. It
    would simply be a bad model, promoted or rejected on its merits."""
    import train_lstm
    import train_transformer

    for module in (train_lstm, train_transformer):
        decision = _call_line_in_main(module, "warm_start")
        assert decision is not None, f"{module.__name__}.main() never calls warm_start"

        for dependent in ("get_dataloaders", "resolve_epochs"):
            line = _call_line_in_main(module, dependent)
            assert line is not None and line > decision, (
                f"{module.__name__}.main(): {dependent} on line {line} reads the "
                f"incremental flag before warm_start settles it on line {decision}"
            )
