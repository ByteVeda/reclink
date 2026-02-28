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

/// A heap-allocated vector with 32-byte alignment (AVX2-compatible).
///
/// Ensures that the underlying buffer is aligned to 32 bytes, which is
/// required for AVX2 aligned loads/stores and beneficial for SSE2/NEON.
pub struct AlignedVec<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
}

// SAFETY: AlignedVec owns its data and T: Send implies AlignedVec: Send.
unsafe impl<T: Send> Send for AlignedVec<T> {}
// SAFETY: AlignedVec is &-safe when T: Sync (shared immutable access).
unsafe impl<T: Sync> Sync for AlignedVec<T> {}

const ALIGNMENT: usize = 32;

impl<T> AlignedVec<T> {
    /// Creates a new empty `AlignedVec`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ptr: std::ptr::NonNull::dangling().as_ptr(),
            len: 0,
            cap: 0,
        }
    }

    /// Creates an `AlignedVec` filled with `len` copies of `value`.
    #[must_use]
    pub fn with_len(len: usize, value: T) -> Self
    where
        T: Clone,
    {
        if len == 0 {
            return Self::new();
        }
        let mut v = Self::with_capacity(len);
        for _ in 0..len {
            v.push(value.clone());
        }
        v
    }

    /// Creates an `AlignedVec` with at least `capacity` elements of space.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 || std::mem::size_of::<T>() == 0 {
            return Self::new();
        }
        let layout = Self::layout_for(capacity);
        // SAFETY: layout has non-zero size (capacity > 0, size_of::<T>() > 0).
        let ptr = unsafe { std::alloc::alloc(layout) as *mut T };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self {
            ptr,
            len: 0,
            cap: capacity,
        }
    }

    /// Pushes a value onto the end.
    pub fn push(&mut self, value: T) {
        if self.len == self.cap {
            self.grow();
        }
        // SAFETY: len < cap after grow, so ptr.add(len) is valid.
        unsafe {
            self.ptr.add(self.len).write(value);
        }
        self.len += 1;
    }

    /// Clears the vector, dropping all elements but keeping the allocation.
    pub fn clear(&mut self) {
        // Drop each element
        for i in 0..self.len {
            // SAFETY: all elements 0..len are initialized.
            unsafe {
                std::ptr::drop_in_place(self.ptr.add(i));
            }
        }
        self.len = 0;
    }

    /// Returns the number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a raw pointer to the buffer.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    fn layout_for(cap: usize) -> std::alloc::Layout {
        let size = cap
            .checked_mul(std::mem::size_of::<T>())
            .expect("capacity overflow");
        // SAFETY: ALIGNMENT is a power of two, size > 0.
        unsafe { std::alloc::Layout::from_size_align_unchecked(size, ALIGNMENT) }
    }

    fn grow(&mut self) {
        let new_cap = if self.cap == 0 { 8 } else { self.cap * 2 };
        if std::mem::size_of::<T>() == 0 {
            self.cap = new_cap;
            return;
        }
        let new_layout = Self::layout_for(new_cap);
        let new_ptr = if self.cap == 0 {
            // SAFETY: new_layout has non-zero size.
            unsafe { std::alloc::alloc(new_layout) as *mut T }
        } else {
            let old_layout = Self::layout_for(self.cap);
            // SAFETY: ptr was allocated with old_layout, new_layout has same alignment.
            unsafe {
                std::alloc::realloc(self.ptr as *mut u8, old_layout, new_layout.size()) as *mut T
            }
        };
        if new_ptr.is_null() {
            std::alloc::handle_alloc_error(new_layout);
        }
        self.ptr = new_ptr;
        self.cap = new_cap;
    }
}

impl<T> Default for AlignedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> std::ops::Deref for AlignedVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: ptr is valid for len initialized elements.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> std::ops::DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: ptr is valid for len initialized elements.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.cap == 0 || std::mem::size_of::<T>() == 0 {
            return;
        }
        // Drop all elements
        for i in 0..self.len {
            unsafe {
                std::ptr::drop_in_place(self.ptr.add(i));
            }
        }
        let layout = Self::layout_for(self.cap);
        // SAFETY: ptr was allocated with this layout.
        unsafe {
            std::alloc::dealloc(self.ptr as *mut u8, layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_vec_basic_push_access() {
        let mut v = AlignedVec::<u32>::new();
        v.push(1);
        v.push(2);
        v.push(3);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 3);
    }

    #[test]
    fn aligned_vec_alignment() {
        let v = AlignedVec::<u32>::with_capacity(16);
        if v.as_ptr() != std::ptr::NonNull::dangling().as_ptr() {
            assert_eq!(v.as_ptr() as usize % 32, 0);
        }

        let v2 = AlignedVec::<u32>::with_len(64, 0);
        assert_eq!(v2.as_ptr() as usize % 32, 0);
    }

    #[test]
    fn aligned_vec_grow() {
        let mut v = AlignedVec::<u32>::new();
        for i in 0..100 {
            v.push(i);
        }
        assert_eq!(v.len(), 100);
        for i in 0..100 {
            assert_eq!(v[i], i as u32);
        }
        // Still aligned after growth
        assert_eq!(v.as_ptr() as usize % 32, 0);
    }

    #[test]
    fn aligned_vec_clear() {
        let mut v = AlignedVec::<u32>::new();
        for i in 0..10 {
            v.push(i);
        }
        assert_eq!(v.len(), 10);
        v.clear();
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
        // Can push again after clear
        v.push(42);
        assert_eq!(v[0], 42);
    }

    #[test]
    fn aligned_vec_deref_slice() {
        let v = AlignedVec::<u32>::with_len(5, 7);
        let slice: &[u32] = &v;
        assert_eq!(slice, &[7, 7, 7, 7, 7]);
    }
}
