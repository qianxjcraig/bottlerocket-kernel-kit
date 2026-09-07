use crate::{MigGpu, MigState, NvidiaMigConfig};
use snafu::{ensure, Snafu};
use std::cmp::min;

const MIG_STRATEGY: &str = "mig";
const FULL_GPU_STRATEGY: &str = "none";

#[derive(Debug, PartialEq, Snafu)]
pub(crate) enum ValidationError {
    #[snafu(display("no NVIDIA GPUs were discovered"))]
    NoGpus,

    #[snafu(display("unsupported NVIDIA GPU model for MIG"))]
    UnsupportedMigGpu,

    #[snafu(display("MIG validation does not support mixed NVIDIA GPU models"))]
    MixedGpuModels,

    #[snafu(display(
        "invalid device partitioning strategy '{}'; expected 'mig' or 'none'",
        strategy
    ))]
    InvalidPartitioningStrategy { strategy: String },

    #[snafu(display("no approved MIG profile is configured for '{}'", model))]
    MissingMigProfile { model: String },

    #[snafu(display("invalid explicit MIG profile '{}'", profile))]
    InvalidMigProfile { profile: String },

    #[snafu(display(
        "GPU {} has MIG state {:?}; expected {}",
        gpu_index,
        actual,
        expected
    ))]
    UnexpectedMigState {
        gpu_index: usize,
        actual: MigState,
        expected: &'static str,
    },

    #[snafu(display(
        "nvidia-smi inventory reported {} GPUs, but NVML state reported {}",
        inventory_count,
        gpu_count
    ))]
    InventoryGpuCount {
        inventory_count: usize,
        gpu_count: usize,
    },

    #[snafu(display("MIG device entry appeared before a GPU entry: '{}'", line))]
    OrphanMigDevice { line: String },

    #[snafu(display("unable to parse nvidia-smi MIG device entry: '{}'", line))]
    InvalidMigInventoryLine { line: String },

    #[snafu(display("nvidia-smi -L did not report any GPUs"))]
    EmptyInventory,

    #[snafu(display(
        "GPU {} MIG inventory mismatch; expected {:?}, found {:?}",
        gpu_index,
        expected,
        actual
    ))]
    MigInventoryMismatch {
        gpu_index: usize,
        expected: Vec<String>,
        actual: Vec<String>,
    },

    #[snafu(display(
        "GPU {} still exposes MIG devices in full-GPU mode: {:?}",
        gpu_index,
        actual
    ))]
    MigDevicesPresent {
        gpu_index: usize,
        actual: Vec<String>,
    },
}

pub(crate) fn validate(
    settings: &NvidiaMigConfig,
    gpu_info: &[MigGpu],
    inventory_output: &str,
) -> Result<(), ValidationError> {
    ensure!(!gpu_info.is_empty(), NoGpusSnafu);
    let inventory = parse_inventory(inventory_output)?;
    ensure!(
        inventory.len() == gpu_info.len(),
        InventoryGpuCountSnafu {
            inventory_count: inventory.len(),
            gpu_count: gpu_info.len(),
        }
    );

    match settings.device_partitioning_strategy.as_str() {
        MIG_STRATEGY => validate_mig(settings, gpu_info, &inventory),
        FULL_GPU_STRATEGY => validate_full_gpu(gpu_info, &inventory),
        strategy => InvalidPartitioningStrategySnafu {
            strategy: strategy.to_string(),
        }
        .fail(),
    }
}

fn validate_mig(
    settings: &NvidiaMigConfig,
    gpu_info: &[MigGpu],
    inventory: &[Vec<String>],
) -> Result<(), ValidationError> {
    let model = gpu_info
        .first()
        .and_then(|gpu| gpu.model.config_key())
        .ok_or(ValidationError::UnsupportedMigGpu)?;

    ensure!(
        gpu_info
            .iter()
            .all(|gpu| gpu.model.config_key() == Some(model)),
        MixedGpuModelsSnafu
    );

    for (gpu_index, gpu) in gpu_info.iter().enumerate() {
        ensure!(
            gpu.state == MigState::Enabled,
            UnexpectedMigStateSnafu {
                gpu_index,
                actual: gpu.state.clone(),
                expected: "Enabled with no pending transition",
            }
        );
    }

    let requested = settings
        .profile
        .get(model)
        .ok_or_else(|| ValidationError::MissingMigProfile {
            model: model.to_string(),
        })?;
    let expected = expected_inventory(model, requested)?;

    for (gpu_index, actual) in inventory.iter().enumerate() {
        let mut actual = actual.clone();
        actual.sort();
        ensure!(
            actual == expected,
            MigInventoryMismatchSnafu {
                gpu_index,
                expected: expected.clone(),
                actual,
            }
        );
    }

    Ok(())
}

fn validate_full_gpu(
    gpu_info: &[MigGpu],
    inventory: &[Vec<String>],
) -> Result<(), ValidationError> {
    for (gpu_index, gpu) in gpu_info.iter().enumerate() {
        ensure!(
            matches!(gpu.state, MigState::Disabled | MigState::Unsupported),
            UnexpectedMigStateSnafu {
                gpu_index,
                actual: gpu.state.clone(),
                expected: "Disabled or unsupported",
            }
        );

        let actual = inventory.get(gpu_index).cloned().unwrap_or_default();
        ensure!(
            actual.is_empty(),
            MigDevicesPresentSnafu { gpu_index, actual }
        );
    }
    Ok(())
}

fn expected_inventory(
    model: &str,
    requested_profile: &str,
) -> Result<Vec<String>, ValidationError> {
    let model_memory = model
        .split_once('.')
        .and_then(|(_, memory)| memory.strip_suffix("gb"))
        .and_then(|memory| memory.parse::<usize>().ok())
        .filter(|memory| *memory > 0)
        .ok_or_else(|| ValidationError::InvalidMigProfile {
            profile: requested_profile.to_string(),
        })?;

    let (compute_slices, profile_memory) = requested_profile
        .split_once("g.")
        .and_then(|(compute, memory)| {
            let memory = memory.strip_suffix("gb")?;
            Some((compute.parse::<usize>().ok()?, memory.parse::<usize>().ok()?))
        })
        .filter(|(compute, memory)| matches!(compute, 1 | 2 | 3 | 4 | 7) && *memory > 0)
        .ok_or_else(|| ValidationError::InvalidMigProfile {
            profile: requested_profile.to_string(),
        })?;

    let count = min(model_memory / profile_memory, 7 / compute_slices);
    ensure!(
        count > 0,
        InvalidMigProfileSnafu {
            profile: requested_profile.to_string(),
        }
    );

    let mut expected = vec![requested_profile.to_string(); count];
    expected.sort();
    Ok(expected)
}

fn parse_inventory(output: &str) -> Result<Vec<Vec<String>>, ValidationError> {
    let mut inventory: Vec<Vec<String>> = Vec::new();

    for line in output.lines() {
        if line.starts_with("GPU ") {
            inventory.push(Vec::new());
            continue;
        }

        let trimmed = line.trim();
        let Some(device) = trimmed.strip_prefix("MIG ") else {
            continue;
        };
        let Some((profile, _)) = device.split_once(" Device ") else {
            return InvalidMigInventoryLineSnafu {
                line: line.to_string(),
            }
            .fail();
        };
        let Some(gpu_inventory) = inventory.last_mut() else {
            return OrphanMigDeviceSnafu {
                line: line.to_string(),
            }
            .fail();
        };
        gpu_inventory.push(profile.to_string());
    }

    ensure!(!inventory.is_empty(), EmptyInventorySnafu);
    Ok(inventory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NvidiaGpu;
    use std::collections::HashMap;

    fn config(strategy: &str, profiles: &[(&str, &str)]) -> NvidiaMigConfig {
        NvidiaMigConfig {
            device_partitioning_strategy: strategy.to_string(),
            profile: profiles
                .iter()
                .map(|(model, profile)| (model.to_string(), profile.to_string()))
                .collect::<HashMap<_, _>>(),
        }
    }

    fn gpu(model: NvidiaGpu, state: MigState) -> MigGpu {
        MigGpu { model, state }
    }

    fn a100_inventory(profile: &str, count: usize) -> String {
        let devices = (0..count)
            .map(|index| {
                format!("  MIG {profile} Device {index}: (UUID: MIG-{index})")
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-0)\n{devices}\n")
    }

    #[test]
    fn validates_exact_shared_inference_geometry() {
        let settings = config(MIG_STRATEGY, &[("a100.40gb", "1g.5gb")]);
        let gpus = [gpu(NvidiaGpu::A100_40GB, MigState::Enabled)];

        assert_eq!(
            validate(&settings, &gpus, &a100_inventory("1g.5gb", 7)),
            Ok(())
        );
    }

    #[test]
    fn rejects_geometry_drift() {
        let settings = config(MIG_STRATEGY, &[("a100.40gb", "1g.5gb")]);
        let gpus = [gpu(NvidiaGpu::A100_40GB, MigState::Enabled)];

        assert!(matches!(
            validate(&settings, &gpus, &a100_inventory("1g.5gb", 6)),
            Err(ValidationError::MigInventoryMismatch { .. })
        ));
    }

    #[test]
    fn rejects_unsupported_compute_shapes() {
        assert_eq!(
            expected_inventory("a100.40gb", "5g.40gb"),
            Err(ValidationError::InvalidMigProfile {
                profile: "5g.40gb".to_string(),
            })
        );
    }

    #[test]
    fn rejects_missing_model_profile() {
        let settings = config(MIG_STRATEGY, &[("h100.80gb", "1g.10gb")]);
        let gpus = [gpu(NvidiaGpu::A100_40GB, MigState::Enabled)];

        assert_eq!(
            validate(&settings, &gpus, &a100_inventory("1g.5gb", 7)),
            Err(ValidationError::MissingMigProfile {
                model: "a100.40gb".to_string(),
            })
        );
    }

    #[test]
    fn rejects_unsupported_gpu_for_shared_inference() {
        let settings = config(MIG_STRATEGY, &[("a100.40gb", "1g.5gb")]);
        let gpus = [gpu(NvidiaGpu::Other, MigState::Unsupported)];
        let inventory = "GPU 0: NVIDIA A10G (UUID: GPU-0)\n";

        assert_eq!(
            validate(&settings, &gpus, inventory),
            Err(ValidationError::UnsupportedMigGpu)
        );
    }

    #[test]
    fn accepts_full_gpu_state_on_mig_unsupported_hardware() {
        let settings = config(FULL_GPU_STRATEGY, &[]);
        let gpus = [gpu(NvidiaGpu::Other, MigState::Unsupported)];
        let inventory = "GPU 0: NVIDIA A10G (UUID: GPU-0)\n";

        assert_eq!(validate(&settings, &gpus, inventory), Ok(()));
    }

    #[test]
    fn rejects_remaining_mig_devices_in_full_gpu_mode() {
        let settings = config(FULL_GPU_STRATEGY, &[]);
        let gpus = [gpu(NvidiaGpu::A100_40GB, MigState::Enabled)];

        assert!(matches!(
            validate(&settings, &gpus, &a100_inventory("1g.5gb", 7)),
            Err(ValidationError::UnexpectedMigState { .. })
        ));
    }

    #[test]
    fn parses_multiple_gpu_inventories() {
        let inventory = parse_inventory(
            "GPU 0: NVIDIA A100 (UUID: GPU-0)\n\
               MIG 3g.20gb Device 0: (UUID: MIG-0)\n\
             GPU 1: NVIDIA A100 (UUID: GPU-1)\n\
               MIG 3g.20gb Device 0: (UUID: MIG-1)\n",
        )
        .unwrap();

        assert_eq!(
            inventory,
            vec![
                vec!["3g.20gb".to_string()],
                vec!["3g.20gb".to_string()]
            ]
        );
    }
}
