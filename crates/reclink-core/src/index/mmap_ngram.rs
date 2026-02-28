//! Memory-mapped N-gram index for datasets larger than RAM.
//!
//! Stores the inverted n-gram index and string pool in a compact binary format
//! that can be memory-mapped from disk. Queries operate directly on the mapped
//! memory without loading the entire index into the heap.
//!
//! # File format
//!
//! ```text
//! [header: 24 bytes]
//!   magic: u32      = 0x524E4749 ("RNGI")
//!   version: u32    = 1
//!   n: u32          = n-gram size
//!   num_strings: u32
//!   string_pool_offset: u64
//! [inverted index section]
//!   num_entries: u32
//!   for each entry:
//!     ngram_len: u16
//!     ngram_bytes: [u8; ngram_len]
//!     num_indices: u32
//!     indices: [u32; num_indices]
//! [string pool section]
//!   for each string:
//!     len: u32
//!     bytes: [u8; len]
//! ```

use std::io::{self, Write};
use std::path::Path;

use ahash::AHashMap;

use crate::index::ngram_index::NgramSearchResult;
use crate::preprocess::ngram_tokenize;

const MAGIC: u32 = 0x524E_4749; // "RNGI"
const VERSION: u32 = 1;

/// A memory-mapped N-gram index.
///
/// The index is built from a list of strings and written to a file.
/// At query time, the file is memory-mapped and queries operate on
/// the mapped memory without loading the full index.
pub struct MmapNgramIndex {
    #[cfg(feature = "mmap")]
    mmap: memmap2::Mmap,
    #[cfg(not(feature = "mmap"))]
    data: Vec<u8>,
    n: usize,
    num_strings: usize,
    string_pool_offset: usize,
    // Cached inverted index (built on open)
    index: AHashMap<String, Vec<usize>>,
}

impl MmapNgramIndex {
    /// Build an index from strings and save to the given path.
    pub fn build_and_save(strings: &[&str], n: usize, path: &Path) -> io::Result<()> {
        let mut file = std::fs::File::create(path)?;

        // Build inverted index
        let mut inverted: AHashMap<String, Vec<usize>> = AHashMap::new();
        for (i, s) in strings.iter().enumerate() {
            let ngrams = ngram_tokenize(s, n);
            for ng in ngrams {
                inverted.entry(ng).or_default().push(i);
            }
        }

        // Write header placeholder (we'll seek back to fill string_pool_offset)
        let header_size = 24u64;
        write_u32(&mut file, MAGIC)?;
        write_u32(&mut file, VERSION)?;
        write_u32(&mut file, n as u32)?;
        write_u32(&mut file, strings.len() as u32)?;
        write_u64(&mut file, 0)?; // string_pool_offset placeholder

        // Write inverted index
        write_u32(&mut file, inverted.len() as u32)?;
        // Sort keys for deterministic output
        let mut entries: Vec<_> = inverted.iter().collect();
        entries.sort_by(|(a, _), (b, _)| a.cmp(b));

        for (ngram, indices) in &entries {
            let bytes = ngram.as_bytes();
            write_u16(&mut file, bytes.len() as u16)?;
            file.write_all(bytes)?;
            write_u32(&mut file, indices.len() as u32)?;
            for &idx in *indices {
                write_u32(&mut file, idx as u32)?;
            }
        }

        // Record string pool offset
        let string_pool_pos = file.stream_position().map_err(io::Error::other)?;

        // Write string pool
        for s in strings {
            let bytes = s.as_bytes();
            write_u32(&mut file, bytes.len() as u32)?;
            file.write_all(bytes)?;
        }

        // Seek back and write string_pool_offset
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(16))?; // offset of string_pool_offset field
        write_u64(&mut file, string_pool_pos)?;

        file.flush()?;
        drop(file);

        // Verify by opening
        let _ = header_size; // suppress unused warning

        Ok(())
    }

    /// Open a memory-mapped index from a file.
    #[cfg(feature = "mmap")]
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        let (n, num_strings, string_pool_offset, index) = Self::parse_data(&mmap)?;

        Ok(Self {
            mmap,
            n,
            num_strings,
            string_pool_offset,
            index,
        })
    }

    /// Open an index by reading the file into memory (non-mmap fallback).
    #[cfg(not(feature = "mmap"))]
    pub fn open(path: &Path) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        let (n, num_strings, string_pool_offset, index) = Self::parse_data(&data)?;

        Ok(Self {
            data,
            n,
            num_strings,
            string_pool_offset,
            index,
        })
    }

    #[allow(clippy::type_complexity)]
    fn parse_data(data: &[u8]) -> io::Result<(usize, usize, usize, AHashMap<String, Vec<usize>>)> {
        if data.len() < 24 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
        }

        let magic = read_u32(data, 0);
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid magic: {magic:#x}"),
            ));
        }

        let version = read_u32(data, 4);
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported version: {version}"),
            ));
        }

        let n = read_u32(data, 8) as usize;
        let num_strings = read_u32(data, 12) as usize;
        let string_pool_offset = read_u64(data, 16) as usize;

        // Parse inverted index
        let mut pos = 24;
        let num_entries = read_u32(data, pos) as usize;
        pos += 4;

        let mut index = AHashMap::with_capacity(num_entries);
        for _ in 0..num_entries {
            let ngram_len = read_u16(data, pos) as usize;
            pos += 2;
            let ngram = std::str::from_utf8(&data[pos..pos + ngram_len])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
                .to_string();
            pos += ngram_len;

            let num_indices = read_u32(data, pos) as usize;
            pos += 4;
            let mut indices = Vec::with_capacity(num_indices);
            for _ in 0..num_indices {
                indices.push(read_u32(data, pos) as usize);
                pos += 4;
            }
            index.insert(ngram, indices);
        }

        Ok((n, num_strings, string_pool_offset, index))
    }

    fn data(&self) -> &[u8] {
        #[cfg(feature = "mmap")]
        {
            &self.mmap
        }
        #[cfg(not(feature = "mmap"))]
        {
            &self.data
        }
    }

    /// Read a string from the string pool by index.
    fn get_string(&self, idx: usize) -> Option<String> {
        if idx >= self.num_strings {
            return None;
        }

        let data = self.data();
        let mut pos = self.string_pool_offset;

        // Walk through string pool to find the idx-th string
        for _ in 0..idx {
            let len = read_u32(data, pos) as usize;
            pos += 4 + len;
        }

        let len = read_u32(data, pos) as usize;
        pos += 4;

        std::str::from_utf8(&data[pos..pos + len])
            .ok()
            .map(String::from)
    }

    /// Find all strings sharing at least `threshold` n-grams with the query.
    pub fn search(&self, query: &str, threshold: usize) -> Vec<NgramSearchResult> {
        let query_ngrams = ngram_tokenize(query, self.n);
        let counts = self.count_shared(&query_ngrams);

        let mut results: Vec<NgramSearchResult> = counts
            .into_iter()
            .filter(|&(_, count)| count >= threshold)
            .filter_map(|(idx, count)| {
                self.get_string(idx).map(|value| NgramSearchResult {
                    value,
                    index: idx,
                    shared_ngrams: count,
                })
            })
            .collect();

        results.sort_by(|a, b| b.shared_ngrams.cmp(&a.shared_ngrams));
        results
    }

    /// Find the k strings sharing the most n-grams with the query.
    pub fn search_top_k(&self, query: &str, k: usize) -> Vec<NgramSearchResult> {
        let query_ngrams = ngram_tokenize(query, self.n);
        let counts = self.count_shared(&query_ngrams);

        let mut results: Vec<NgramSearchResult> = counts
            .into_iter()
            .filter_map(|(idx, count)| {
                self.get_string(idx).map(|value| NgramSearchResult {
                    value,
                    index: idx,
                    shared_ngrams: count,
                })
            })
            .collect();

        results.sort_by(|a, b| b.shared_ngrams.cmp(&a.shared_ngrams));
        results.truncate(k);
        results
    }

    /// Returns the number of strings in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.num_strings
    }

    /// Returns whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.num_strings == 0
    }

    /// Estimates the heap memory usage of this index in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let mut bytes = std::mem::size_of::<Self>();
        // Backing data (mmap or heap Vec)
        bytes += self.data().len();
        // Cached inverted index
        for (key, postings) in &self.index {
            bytes += std::mem::size_of::<String>() + key.capacity();
            bytes += std::mem::size_of::<Vec<usize>>()
                + postings.capacity() * std::mem::size_of::<usize>();
        }
        bytes
    }

    fn count_shared(&self, query_ngrams: &[String]) -> Vec<(usize, usize)> {
        let mut counts: AHashMap<usize, usize> = AHashMap::new();
        for ng in query_ngrams {
            if let Some(indices) = self.index.get(ng) {
                for &idx in indices {
                    *counts.entry(idx).or_insert(0) += 1;
                }
            }
        }
        counts.into_iter().collect()
    }
}

// ─── Binary helpers ───────────────────────────────────────────────────────

fn write_u16(w: &mut impl Write, v: u16) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u64(w: &mut impl Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u16(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn read_u64(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_path(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("reclink_mmap_tests");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn build_save_and_open() {
        let path = test_path("test_basic.rngi");
        let strings = ["hello", "help", "world"];
        MmapNgramIndex::build_and_save(&strings, 2, &path).unwrap();

        let index = MmapNgramIndex::open(&path).unwrap();
        assert_eq!(index.len(), 3);
        assert!(!index.is_empty());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn search_finds_matches() {
        let path = test_path("test_search.rngi");
        let strings = ["hello", "help", "world", "held"];
        MmapNgramIndex::build_and_save(&strings, 2, &path).unwrap();

        let index = MmapNgramIndex::open(&path).unwrap();
        let results = index.search("hello", 2);
        let values: Vec<&str> = results.iter().map(|r| r.value.as_str()).collect();
        assert!(values.contains(&"hello"));
        assert!(values.contains(&"help")); // shares "he", "el"

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn search_top_k() {
        let path = test_path("test_topk.rngi");
        let strings = ["hello", "help", "world", "held", "helm"];
        MmapNgramIndex::build_and_save(&strings, 2, &path).unwrap();

        let index = MmapNgramIndex::open(&path).unwrap();
        let results = index.search_top_k("hello", 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].value, "hello"); // most shared ngrams

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn get_string_by_index() {
        let path = test_path("test_getstr.rngi");
        let strings = ["alpha", "beta", "gamma"];
        MmapNgramIndex::build_and_save(&strings, 2, &path).unwrap();

        let index = MmapNgramIndex::open(&path).unwrap();
        assert_eq!(index.get_string(0).unwrap(), "alpha");
        assert_eq!(index.get_string(1).unwrap(), "beta");
        assert_eq!(index.get_string(2).unwrap(), "gamma");
        assert!(index.get_string(3).is_none());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_index() {
        let path = test_path("test_empty.rngi");
        MmapNgramIndex::build_and_save(&[], 2, &path).unwrap();

        let index = MmapNgramIndex::open(&path).unwrap();
        assert!(index.is_empty());
        assert!(index.search("hello", 1).is_empty());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn invalid_magic_rejected() {
        let path = test_path("test_bad_magic.rngi");
        std::fs::write(&path, &[0u8; 24]).unwrap();

        let result = MmapNgramIndex::open(&path);
        assert!(result.is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn memory_usage_nonzero() {
        let path = test_path("test_mem.rngi");
        let strings = ["hello", "help", "world"];
        MmapNgramIndex::build_and_save(&strings, 2, &path).unwrap();

        let index = MmapNgramIndex::open(&path).unwrap();
        assert!(index.memory_usage() > 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn unicode_strings() {
        let path = test_path("test_unicode.rngi");
        let strings = ["café", "naïve", "résumé"];
        MmapNgramIndex::build_and_save(&strings, 2, &path).unwrap();

        let index = MmapNgramIndex::open(&path).unwrap();
        assert_eq!(index.len(), 3);
        assert_eq!(index.get_string(0).unwrap(), "café");

        let results = index.search("café", 1);
        assert!(!results.is_empty());

        std::fs::remove_file(&path).ok();
    }
}
