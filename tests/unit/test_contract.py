"""The window and channel geometry every job in this image has to agree on.

Runs inside dag-pytorch-model-training (PYTHONPATH=/app).
"""
import batch_inference
import evaluate_and_promote
import lakehouse


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


import torch

from models import ConvLSTMWeatherForecaster, TimeSeriesTransformer


def _batch():
    return torch.zeros(2, lakehouse.SEQ_LEN, lakehouse.INPUT_CHANNELS)


def test_both_architectures_map_the_input_width_to_the_output_width():
    """input_dim and output_dim used to be the same number passed twice. They are
    genuinely different now, and passing the input width as the output width would
    have the model emitting calendar predictions nobody reads."""
    expected = (2, lakehouse.PRED_LEN, lakehouse.OUTPUT_CHANNELS)

    lstm = ConvLSTMWeatherForecaster(
        lakehouse.INPUT_CHANNELS, 64, lakehouse.OUTPUT_CHANNELS, lakehouse.PRED_LEN, 0.2
    )
    transformer = TimeSeriesTransformer(
        lakehouse.INPUT_CHANNELS, 32, 2, 4, 128,
        lakehouse.OUTPUT_CHANNELS, lakehouse.PRED_LEN, 0.1
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


import ast


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
