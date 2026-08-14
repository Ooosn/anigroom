#!/usr/bin/env bash
set -euo pipefail

export PREFLIGHT_ID=r061_gaussian_only_appearance_fullres_preflight_h100_20260814
export RUN_ID=r061_gaussian_only_appearance_0_30k_h100_20260814
export LABEL=r061_gaussian_only_appearance
export PREFLIGHT_CONFIG=r061_gaussian_only_appearance_fullres_preflight.env
export RUN_CONFIG=r061_gaussian_only_appearance_0_30k.env
export REQUIRE_NO_LOCAL_CHILD_COLOR=1

exec bash "${PROJECT_ROOT:?set PROJECT_ROOT to the clean R061 checkout}/scripts/server/run_r060_relative_shape_amplitudes.sh"
