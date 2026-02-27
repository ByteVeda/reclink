//! SIMD utilities: cached feature detection and dispatch macro.

#[cfg(target_arch = "x86_64")]
use std::sync::LazyLock;

/// Cached AVX2 detection (x86_64 only).
#[cfg(target_arch = "x86_64")]
static HAS_AVX2: LazyLock<bool> = LazyLock::new(|| is_x86_feature_detected!("avx2"));

/// Returns `true` if the CPU supports AVX2 (cached after first call).
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn has_avx2() -> bool {
    *HAS_AVX2
}

/// Tiered SIMD dispatch macro.
///
/// Generates: if avx2 → avx2 path, else sse2 (x86_64) / neon (aarch64) / scalar fallback.
///
/// # Usage
/// ```ignore
/// dispatch_simd!(
///     avx2: unsafe { avx2_fn(a, b) },
///     sse2: unsafe { sse2_fn(a, b) },
///     neon: unsafe { neon_fn(a, b) },
///     scalar: scalar_fn(a, b),
/// )
/// ```
macro_rules! dispatch_simd {
    (
        avx2: $avx2:expr,
        sse2: $sse2:expr,
        neon: $neon:expr,
        scalar: $scalar:expr $(,)?
    ) => {{
        #[cfg(target_arch = "x86_64")]
        {
            if $crate::metrics::simd_util::has_avx2() {
                $avx2
            } else {
                $sse2
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            $neon
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            $scalar
        }
    }};
}

pub(crate) use dispatch_simd;
