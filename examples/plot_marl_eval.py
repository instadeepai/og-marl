# Install MARL eval in your virtual environment
# pip install "git+https://github.com/instadeepai/marl-eval.git"
import json

# Relevant imports
from marl_eval.plotting_tools.plotting import performance_profiles, sample_efficiency_curves
from marl_eval.utils.data_processing_utils import create_matrices_for_rliable, data_process_pipeline

# Specify any metrics that should be normalised
METRICS_TO_NORMALIZE = ["episode_return"]

# Read in and process data
with open("logs/metrics.json", "r") as f:
    raw_data = json.load(f)

processed_data = data_process_pipeline(raw_data=raw_data, metrics_to_normalize=METRICS_TO_NORMALIZE)

environment_comparison_matrix, sample_effeciency_matrix = create_matrices_for_rliable(
    data_dictionary=processed_data,
    environment_name="smac_v1",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
)

# Generate performance profile plot
fig = performance_profiles(
    environment_comparison_matrix,
    metric_name="episode_return",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
)
fig.figure.savefig("performance_profile.png", bbox_inches="tight")

fig, _, _ = sample_efficiency_curves(
    sample_effeciency_matrix,
    metric_name="episode_return",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
    xlabel="Training Steps",
)

fig.figure.savefig("sample_efficiency.png", bbox_inches="tight")
