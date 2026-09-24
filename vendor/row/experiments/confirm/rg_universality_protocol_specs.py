"""Frozen specifications for the separate RG and universality campaign."""

from __future__ import annotations


BLOCK_SIZES = (1, 2, 4, 8, 16, 32)

THRESHOLDS = {
    "exact_semigroup_relative": 1e-12,
    "empirical_kernel_semigroup_relative": 0.35,
    "projection_prediction_defect": 0.25,
    "projection_innovation_sliced_wasserstein": 0.25,
    "stationarity_standardized_mean_defect": 0.20,
    "stationarity_relative_covariance_defect": 0.30,
    "anchor_kernel_relative_defect": 0.40,
    "innovation_cross_covariance_frobenius": 0.20,
    "innovation_conditional_sliced_wasserstein": 0.20,
    "embedding_reconstruction_relative": 1e-6,
    "final_maximum_absolute_skewness": 0.40,
    "final_maximum_absolute_excess_kurtosis": 0.80,
    "final_sliced_wasserstein_to_gaussian": 0.25,
    "cross_system_final_sliced_wasserstein": 0.25,
}


def _stage(stage_id, title, dependencies, tier, execution, target,
           resource_cap, checks):
    return {
        "id": stage_id,
        "title": title,
        "dependencies": tuple(dependencies),
        "tier": tier,
        "fresh_execution": execution,
        "target": target,
        "resource_cap": resource_cap,
        "required_checks": tuple(checks),
        "decision": (
            "The analyzer derives every check from hashed trajectory "
            "ensembles and frozen thresholds."
        ),
    }


STAGES = (
    _stage(
        "R0",
        "Kernel-RG semantic qualification",
        (),
        "kernel_flow",
        "Exact two-dimensional fixtures and a seeded CPU ensemble",
        "Qualify affine innovation blocking, covariance transport, "
        "stationary whitening, embedding rejection, flow beta functions, "
        "hidden-stratum sensitivity, and fluctuation statistics.",
        "1 CPU-hour",
        (
            "exact_affine_innovation_semigroup",
            "stationary_whitening_identity",
            "embedding_accept_and_reject",
            "hidden_stratum_kill",
            "cumulant_gaussianization_kill",
        ),
    ),
    _stage(
        "R1",
        "Two-GPU trajectory micro-pilot",
        ("R0",),
        "kernel_flow",
        "Two width-32 depth-2 seeds, one per GPU; four anchors; "
        "128 continuations with 16 burn-in and 64 measured updates per anchor",
        "Profile complete-state reload, controlled normal perturbations, "
        "hidden-stratum labels, trajectory storage, and block factors "
        "1 through 32 before fixing the larger allocation.",
        "8 GPU-hours total",
        (
            "two_distinct_gpu_runs",
            "complete_anchor_states",
            "retained_slow_basis_16_frozen",
            "normal_complement_gap_positive",
            "normal_paths_and_hidden_labels",
            "all_block_sizes_populated",
            "resource_projection_below_cap",
        ),
    ),
    _stage(
        "R2",
        "Projected kernel and closure",
        ("R1",),
        "kernel_flow",
        "Two width-64 depth-4 seeds; eight anchors; "
        "256 continuations with 32 burn-in and 128 measured updates per anchor",
        "Estimate direct block kernels, compare them with iterated one-step "
        "kernels, and test dependence on tangent, data, optimizer, and "
        "branch strata.",
        "24 GPU-hours total",
        (
            "direct_and_iterated_kernels",
            "empirical_semigroup_defect_bounded",
            "projection_prediction_defect_bounded",
            "innovation_distribution_defect_bounded",
            "innovation_mixing_diagnostics_bounded",
            "stationarity_split_bounded",
            "anchor_kernel_heterogeneity_bounded",
            "heldout_anchor_replay",
        ),
    ),
    _stage(
        "R3",
        "Discrete flow and embeddability",
        ("R2",),
        "kernel_flow",
        "Offline reduction of the R2 ensembles at block factors "
        "1, 2, 4, 8, 16, and 32",
        "Confirm monotone memory loss, a real principal generator, positive "
        "stability and diffusion edges, and agreement of the continuous "
        "flow with direct integer blocks.",
        "8 wall-clock hours, mainly CPU",
        (
            "memory_decreases_under_blocking",
            "principal_generator_reconstructs",
            "generator_stability_edge_positive",
            "stationary_diffusion_nonnegative",
            "continuous_integer_flow_agrees",
        ),
    ),
    _stage(
        "U1",
        "Gaussian fixed-point approach",
        ("R3",),
        "universality",
        "Covariance-whitened nonoverlapping block sums from R2",
        "Test decay of standardized skewness, excess kurtosis, and sliced "
        "distance to the standard Gaussian across registered block scales.",
        "8 wall-clock hours, CPU",
        (
            "stationary_centering_owned",
            "long_run_covariance_positive",
            "higher_cumulants_decrease",
            "gaussian_distance_bounded_at_scale_32",
        ),
    ),
    _stage(
        "U2",
        "Universality transfer",
        ("U1",),
        "universality",
        "Widths 32, 64, and 128 at depths 2, 4, and 6; "
        "two seeds per architecture; 4,096 updates and six burst anchors",
        "After centering, covariance whitening, and correlation-time "
        "registration, compare RG flows and scale-32 fluctuation laws "
        "across architectures and seeds.",
        "78 to 106 GPU-hours total",
        (
            "three_architecture_cells",
            "two_seeds_per_cell",
            "all_systems_kernel_qualified",
            "all_systems_flow_qualified",
            "cross_system_distance_contracts",
            "shared_fixed_point_threshold",
        ),
    ),
    _stage(
        "U3",
        "Independent universality replay",
        ("U2",),
        "universality",
        "Hash and numerical replay of every R0 through U2 artifact",
        "Recompute kernels, closure defects, embeddings, flow levels, "
        "cumulants, cross-system distances, resources, and the final "
        "registered-domain class decision.",
        "16 wall-clock hours, CPU",
        (
            "all_manifests_resolve",
            "all_trajectory_hashes_replay",
            "all_kernel_reductions_replay",
            "all_flow_reductions_replay",
            "all_class_distances_replay",
            "resource_caps_replay",
        ),
    ),
)


ARCHITECTURE_RUNS = (
    {
        "architecture": "w032_d02",
        "width": 32,
        "depth": 2,
        "seeds": (4101, 4102),
        "updates": 4096,
    },
    {
        "architecture": "w064_d04",
        "width": 64,
        "depth": 4,
        "seeds": (4201, 4202),
        "updates": 4096,
    },
    {
        "architecture": "w128_d06",
        "width": 128,
        "depth": 6,
        "seeds": (4301, 4302),
        "updates": 4096,
    },
)


RESOURCE_CAPS = {
    "campaign_is_separate_from_normal_stability_confirmation": True,
    "gpu_count": 2,
    "gpu_model_observed": "NVIDIA GeForce RTX 4090",
    "gpu_physical_bytes_each": 24564 * 1024 ** 2,
    "gpu_allocation_cap_bytes_each": 20 * 1024 ** 3,
    "host_bytes_total": 96 * 1024 ** 3,
    "retained_artifact_bytes": 120 * 1024 ** 3,
    "estimated_gpu_hours_lower": 78,
    "estimated_gpu_hours_upper": 106,
    "hard_gpu_hours": 128,
    "hard_wall_clock_hours": 96,
    "concurrent_gpu_jobs": 2,
        "minimum_free_workspace_bytes": 160 * 1024 ** 3,
    "retained_normal_dimension": 16,
    "retained_basis": (
        "sixteen slow lifted normal modes frozen on construction data; "
        "orthogonal alignment fixed before validation"
    ),
    "complement_requirement": (
        "the common-metric contraction edge of the omitted normal "
        "complement is strictly stronger than the retained slow edge"
    ),
    "stop_rule": (
        "Stop before U2 if R2 projection closure or R3 embeddability fails; "
        "do not spend universality-tier resources on an unclosed kernel."
    ),
}
