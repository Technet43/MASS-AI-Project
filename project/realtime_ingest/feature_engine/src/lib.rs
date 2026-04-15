//! MASS-AI Feature Engine — Rust/PyO3
//!
//! Provides vectorised (SIMD-auto) statistical primitives for real-time
//! smart-meter telemetry.  All heavy loops are written to let LLVM emit
//! AVX2 / AVX-512 instructions when compiled with `target-cpu=native`.
//!
//! Exposed Python API
//! ------------------
//! ```python
//! import feature_engine as fe
//!
//! # Rolling statistics over the last `window` samples
//! ma  = fe.moving_average(values, window=60)     # list[float]
//! std = fe.rolling_std(values, window=60)        # list[float]
//!
//! # One-shot feature vector from raw telemetry arrays
//! p_avg, v_std, i_peak = fe.compute_telemetry_features(
//!     voltage, current, active_power, window=60
//! )
//! ```

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ─── helpers ──────────────────────────────────────────────────────────────────

/// Compute the arithmetic mean of `data[start..end]`.
/// The tight inner loop is a prime target for LLVM auto-vectorisation.
#[inline]
fn slice_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    // Manual accumulation lets the compiler pipeline multiple FP units.
    let mut acc0 = 0.0_f64;
    let mut acc1 = 0.0_f64;
    let mut acc2 = 0.0_f64;
    let mut acc3 = 0.0_f64;
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();
    for chunk in chunks {
        acc0 += chunk[0];
        acc1 += chunk[1];
        acc2 += chunk[2];
        acc3 += chunk[3];
    }
    let mut sum = acc0 + acc1 + acc2 + acc3;
    for &v in remainder {
        sum += v;
    }
    sum / data.len() as f64
}

/// Population standard deviation of `data`.
#[inline]
fn slice_std(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let mean = slice_mean(data);
    // Two-pass: variance = mean of squared deviations.
    let mut var_acc0 = 0.0_f64;
    let mut var_acc1 = 0.0_f64;
    let mut var_acc2 = 0.0_f64;
    let mut var_acc3 = 0.0_f64;
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let d0 = chunk[0] - mean;
        let d1 = chunk[1] - mean;
        let d2 = chunk[2] - mean;
        let d3 = chunk[3] - mean;
        var_acc0 += d0 * d0;
        var_acc1 += d1 * d1;
        var_acc2 += d2 * d2;
        var_acc3 += d3 * d3;
    }
    let mut variance = var_acc0 + var_acc1 + var_acc2 + var_acc3;
    for &v in remainder {
        let d = v - mean;
        variance += d * d;
    }
    (variance / data.len() as f64).sqrt()
}

// ─── Python-exported functions ────────────────────────────────────────────────

/// Return the rolling mean of `values` with the given `window`.
///
/// Output length equals `len(values)`.  Positions where fewer than `window`
/// samples are available use however many samples exist (expanding window).
#[pyfunction]
#[pyo3(signature = (values, window))]
fn moving_average(values: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }
    let n = values.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = if i + 1 >= window { i + 1 - window } else { 0 };
        out.push(slice_mean(&values[start..=i]));
    }
    Ok(out)
}

/// Return the rolling population standard deviation of `values`.
///
/// Same windowing semantics as `moving_average`.
#[pyfunction]
#[pyo3(signature = (values, window))]
fn rolling_std(values: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }
    let n = values.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = if i + 1 >= window { i + 1 - window } else { 0 };
        out.push(slice_std(&values[start..=i]));
    }
    Ok(out)
}

/// Compute the three ML features used by the anomaly detector from raw arrays.
///
/// Arguments
/// ---------
/// voltage, current, active_power : list[float]
///     Equal-length time-series of raw meter readings (most recent last).
/// window : int
///     Number of trailing samples to use (e.g. 60 for 1-hour at 1 msg/min).
///
/// Returns
/// -------
/// (p_avg_1h, v_std_1h, i_peak_1h) : tuple[float, float, float]
#[pyfunction]
#[pyo3(signature = (voltage, current, active_power, window))]
fn compute_telemetry_features(
    voltage:      Vec<f64>,
    current:      Vec<f64>,
    active_power: Vec<f64>,
    window:       usize,
) -> PyResult<(f64, f64, f64)> {
    let n = voltage.len();
    if n != current.len() || n != active_power.len() {
        return Err(PyValueError::new_err(
            "voltage, current, and active_power must have the same length",
        ));
    }
    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }

    // Use the last `window` samples (or all if fewer available).
    let start = n.saturating_sub(window);
    let v_slice  = &voltage[start..];
    let c_slice  = &current[start..];
    let p_slice  = &active_power[start..];

    let p_avg  = slice_mean(p_slice);
    let v_std  = slice_std(v_slice);
    // Peak current: single LLVM-vectorisable fold.
    let i_peak = c_slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    Ok((p_avg, v_std, i_peak))
}

/// Batch version: process multiple meters in one call.
///
/// Arguments
/// ---------
/// batch : list[tuple[list[float], list[float], list[float]]]
///     Each element is (voltage, current, active_power) for one meter.
/// window : int
///
/// Returns
/// -------
/// list[tuple[float, float, float]]
///     One (p_avg, v_std, i_peak) per meter.
#[pyfunction]
#[pyo3(signature = (batch, window))]
fn compute_batch_features(
    batch:  Vec<(Vec<f64>, Vec<f64>, Vec<f64>)>,
    window: usize,
) -> PyResult<Vec<(f64, f64, f64)>> {
    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }
    let mut results = Vec::with_capacity(batch.len());
    for (voltage, current, active_power) in batch {
        let (p, v, i) =
            compute_telemetry_features(voltage, current, active_power, window)?;
        results.push((p, v, i));
    }
    Ok(results)
}

// ─── Zero-copy NumPy API ──────────────────────────────────────────────────────
// Accepts numpy arrays directly — no Vec allocation, no memcpy.
// Python side: pass np.asarray(values, dtype=np.float64) for maximum speed.

/// Zero-copy variant of `compute_telemetry_features`.
///
/// Takes numpy `ndarray`s instead of Python lists.  The arrays are borrowed
/// directly from NumPy's memory buffer — no allocation, no copy.
///
/// ```python
/// import numpy as np, feature_engine as fe
/// p, v, i = fe.compute_features_numpy(
///     np.asarray(voltage, dtype=np.float64),
///     np.asarray(current, dtype=np.float64),
///     np.asarray(active_power, dtype=np.float64),
///     window=60,
/// )
/// ```
#[pyfunction]
#[pyo3(signature = (voltage, current, active_power, window))]
fn compute_features_numpy(
    voltage:      PyReadonlyArray1<f64>,
    current:      PyReadonlyArray1<f64>,
    active_power: PyReadonlyArray1<f64>,
    window:       usize,
) -> PyResult<(f64, f64, f64)> {
    let v = voltage.as_slice()
        .map_err(|_| PyValueError::new_err("voltage array must be contiguous (C-order)"))?;
    let c = current.as_slice()
        .map_err(|_| PyValueError::new_err("current array must be contiguous (C-order)"))?;
    let p = active_power.as_slice()
        .map_err(|_| PyValueError::new_err("active_power array must be contiguous (C-order)"))?;

    if window == 0 {
        return Err(PyValueError::new_err("window must be >= 1"));
    }
    let n = v.len();
    if n != c.len() || n != p.len() {
        return Err(PyValueError::new_err(
            "voltage, current, and active_power must have the same length",
        ));
    }

    let start  = n.saturating_sub(window);
    let p_avg  = slice_mean(&p[start..]);
    let v_std  = slice_std(&v[start..]);
    let i_peak = c[start..].iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    Ok((p_avg, v_std, i_peak))
}

// ─── Module registration ──────────────────────────────────────────────────────

#[pymodule]
fn feature_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(moving_average, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std, m)?)?;
    m.add_function(wrap_pyfunction!(compute_telemetry_features, m)?)?;
    m.add_function(wrap_pyfunction!(compute_batch_features, m)?)?;
    m.add_function(wrap_pyfunction!(compute_features_numpy, m)?)?;
    Ok(())
}
