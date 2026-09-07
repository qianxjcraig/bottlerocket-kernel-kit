# nvidia-migmanager

Current version: 0.1.0

## NVIDIA MIG Manager
`nvidia-migmanager` ensures that MIG settings are applied to an instance that supports
it. It is called by `nvidia-migmanager.service`.

The binary reads its config file and based on the config, it activates/deactivates MIG
and applies the profile according to the type of GPU present in the instance.

The manager recognizes A100 40/80 GB, H100 80 GB, H200 141 GB, B200 180 GB,
B300 269 GB, and RTX PRO 6000 96 GB GPUs. Unknown future MIG-capable models
may use an explicit model/profile entry, but `validate-mig` rejects them until
their hardware identity and geometry can be verified.

`nvidia-migmanager validate-mig` fails unless the current GPU mode and every
published MIG device exactly match the rendered configuration. It also verifies
that no MIG devices remain when the configuration requests full-GPU mode.
Before changing geometry or disabling MIG, the manager removes existing compute
and GPU instances; active workloads therefore cause a safe failure instead of a
partially applied transition.

### Example:
```toml
[settings.kubelet-device-plugins.nvidia]
device-partitioning-strategy="mig"

[settings.kubelet-device-plugins.nvidia.mig.profile]
"a100.40gb"="2"
"h100.80gb"="4"
"h200.141gb"="3"
```
This would partition the GPUs in an instance with A100 GPU into 2 parts, instance with H100
into 4 parts and instance with H200 into 3 parts.

## Colophon

This text was generated using [cargo-readme](https://crates.io/crates/cargo-readme), and includes the rustdoc from `src/main.rs`.
