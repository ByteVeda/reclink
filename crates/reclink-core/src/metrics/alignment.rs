//! Levenshtein edit-distance alignment with visual traceback.
//!
//! Computes the full DP matrix and traces back through it to produce
//! an alignment showing exactly which characters match, are substituted,
//! inserted, deleted, or transposed.

use std::fmt;

/// A single edit operation in an alignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditOp {
    /// Characters match exactly.
    Match(char),
    /// Character in `a` was substituted with character in `b`.
    Substitute(char, char),
    /// Character was inserted (present in `b`, absent in `a`).
    Insert(char),
    /// Character was deleted (present in `a`, absent in `b`).
    Delete(char),
}

/// The result of aligning two strings.
#[derive(Debug, Clone)]
pub struct Alignment {
    /// Ordered sequence of edit operations.
    pub ops: Vec<EditOp>,
    /// Total edit distance.
    pub distance: usize,
}

impl Alignment {
    /// Returns a visual representation of the alignment.
    ///
    /// Format:
    /// ```text
    /// J o h n   S m i t h
    /// | |   | | | |   | |
    /// J o - n   S m y t h
    /// ```
    #[must_use]
    pub fn visual(&self) -> String {
        let mut top = Vec::new();
        let mut mid = Vec::new();
        let mut bot = Vec::new();

        for op in &self.ops {
            match op {
                EditOp::Match(c) => {
                    top.push(c.to_string());
                    mid.push("|".to_string());
                    bot.push(c.to_string());
                }
                EditOp::Substitute(a, b) => {
                    top.push(a.to_string());
                    mid.push(" ".to_string());
                    bot.push(b.to_string());
                }
                EditOp::Delete(a) => {
                    top.push(a.to_string());
                    mid.push(" ".to_string());
                    bot.push("-".to_string());
                }
                EditOp::Insert(b) => {
                    top.push("-".to_string());
                    mid.push(" ".to_string());
                    bot.push(b.to_string());
                }
            }
        }

        format!("{}\n{}\n{}", top.join(" "), mid.join(" "), bot.join(" "))
    }

    /// Returns a list of operation names as strings for Python consumption.
    #[must_use]
    pub fn op_names(&self) -> Vec<String> {
        self.ops
            .iter()
            .map(|op| match op {
                EditOp::Match(c) => format!("match:{c}"),
                EditOp::Substitute(a, b) => format!("sub:{a}->{b}"),
                EditOp::Insert(c) => format!("ins:{c}"),
                EditOp::Delete(c) => format!("del:{c}"),
            })
            .collect()
    }
}

impl fmt::Display for Alignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.visual())
    }
}

/// Compute a Levenshtein alignment between two strings using full-matrix DP
/// with traceback.
#[must_use]
pub fn levenshtein_alignment(a: &str, b: &str) -> Alignment {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let n = a_chars.len();
    let m = b_chars.len();

    // Build DP matrix
    let mut dp = vec![vec![0usize; m + 1]; n + 1];

    for (i, row) in dp.iter_mut().enumerate().take(n + 1) {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate().take(m + 1) {
        *cell = j;
    }

    for i in 1..=n {
        for j in 1..=m {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1) // delete
                .min(dp[i][j - 1] + 1) // insert
                .min(dp[i - 1][j - 1] + cost); // match/substitute
        }
    }

    let distance = dp[n][m];

    // Traceback
    let mut ops = Vec::new();
    let mut i = n;
    let mut j = m;

    while i > 0 || j > 0 {
        if i > 0 && j > 0 && a_chars[i - 1] == b_chars[j - 1] && dp[i][j] == dp[i - 1][j - 1] {
            ops.push(EditOp::Match(a_chars[i - 1]));
            i -= 1;
            j -= 1;
        } else if i > 0
            && j > 0
            && dp[i][j] == dp[i - 1][j - 1] + 1
            && a_chars[i - 1] != b_chars[j - 1]
        {
            ops.push(EditOp::Substitute(a_chars[i - 1], b_chars[j - 1]));
            i -= 1;
            j -= 1;
        } else if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            ops.push(EditOp::Delete(a_chars[i - 1]));
            i -= 1;
        } else {
            // j > 0, insert
            ops.push(EditOp::Insert(b_chars[j - 1]));
            j -= 1;
        }
    }

    ops.reverse();

    Alignment { ops, distance }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_strings() {
        let align = levenshtein_alignment("abc", "abc");
        assert_eq!(align.distance, 0);
        assert_eq!(align.ops.len(), 3);
        assert!(align.ops.iter().all(|op| matches!(op, EditOp::Match(_))));
    }

    #[test]
    fn single_substitution() {
        let align = levenshtein_alignment("cat", "car");
        assert_eq!(align.distance, 1);
        assert!(align
            .ops
            .iter()
            .any(|op| matches!(op, EditOp::Substitute('t', 'r'))));
    }

    #[test]
    fn single_insertion() {
        let align = levenshtein_alignment("cat", "cart");
        assert_eq!(align.distance, 1);
    }

    #[test]
    fn single_deletion() {
        let align = levenshtein_alignment("cart", "cat");
        assert_eq!(align.distance, 1);
    }

    #[test]
    fn empty_strings() {
        let align = levenshtein_alignment("", "");
        assert_eq!(align.distance, 0);
        assert!(align.ops.is_empty());
    }

    #[test]
    fn one_empty() {
        let align = levenshtein_alignment("abc", "");
        assert_eq!(align.distance, 3);
        assert!(align.ops.iter().all(|op| matches!(op, EditOp::Delete(_))));

        let align = levenshtein_alignment("", "xyz");
        assert_eq!(align.distance, 3);
        assert!(align.ops.iter().all(|op| matches!(op, EditOp::Insert(_))));
    }

    #[test]
    fn visual_output() {
        let align = levenshtein_alignment("John Smith", "Jon Smyth");
        let visual = align.visual();
        assert!(visual.contains('\n'));
        // Should have 3 lines
        assert_eq!(visual.lines().count(), 3);
    }

    #[test]
    fn name_alignment() {
        let align = levenshtein_alignment("Smith", "Smyth");
        assert_eq!(align.distance, 1);
        // S, m match; i→y sub; t, h match
        let matches: Vec<_> = align
            .ops
            .iter()
            .filter(|op| matches!(op, EditOp::Match(_)))
            .collect();
        assert_eq!(matches.len(), 4);
    }

    #[test]
    fn op_names_format() {
        let align = levenshtein_alignment("ab", "ac");
        let names = align.op_names();
        assert_eq!(names[0], "match:a");
        assert_eq!(names[1], "sub:b->c");
    }
}
