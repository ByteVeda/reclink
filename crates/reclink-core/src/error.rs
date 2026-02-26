//! Error types for the reclink-core library.

/// All errors that can occur in reclink-core operations.
#[derive(Debug, thiserror::Error)]
pub enum ReclinkError {
    /// Strings must be equal length for this metric (e.g., Hamming distance).
    #[error("strings must have equal length: got {a} and {b}")]
    UnequalLength { a: usize, b: usize },

    /// A required field is missing from a record.
    #[error("missing field: {0}")]
    MissingField(String),

    /// A field has an unexpected type.
    #[error("type mismatch for field `{field}`: expected {expected}, got {got}")]
    TypeMismatch {
        field: String,
        expected: String,
        got: String,
    },

    /// An invalid configuration was provided.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// The pipeline has not been fully configured.
    #[error("pipeline error: {0}")]
    Pipeline(String),

    /// An empty input was provided where non-empty input is required.
    #[error("empty input: {0}")]
    EmptyInput(String),
}

/// Convenience type alias for Results using [`ReclinkError`].
pub type Result<T> = std::result::Result<T, ReclinkError>;
