//! Single-slot prompt prefix cache state.
//!
//! Mirrors the design of llama-server's `server_tokens` struct: a flat sequence
//! of entries that records what is currently materialized in the persistent KV
//! cache for sequence id 0. Text tokens occupy one entry each; an image's
//! `n_tokens` positions all share the same `(hash, group_id)` so the prefix
//! diff matches images atomically.

use llama_cpp_2::token::LlamaToken;

/// One position in the persistent KV cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlotEntry {
    Text(LlamaToken),
    /// Image position. `hash` identifies the image (FNV-1a of the raw bytes);
    /// `group_id` ties together the consecutive entries that belong to the
    /// same image so the matcher can detect partial-overlap mismatches.
    #[cfg_attr(not(feature = "mtmd"), allow(dead_code))]
    Image {
        hash: u64,
        group_id: u32,
    },
}

/// Length of the longest common prefix of `cur` and `new`.
///
/// Image groups are matched atomically: if both sides have an image entry at
/// the same offset, the entire group must agree on `hash` *and* `group_id`'s
/// run length. A partial overlap (same hash but different token count) stops
/// the match at the start of the group.
pub(crate) fn get_common_prefix(cur: &[SlotEntry], new: &[SlotEntry]) -> usize {
    let mut i = 0;
    let max = cur.len().min(new.len());
    while i < max {
        match (cur[i], new[i]) {
            (SlotEntry::Text(a), SlotEntry::Text(b)) if a == b => {
                i += 1;
            }
            (
                SlotEntry::Image {
                    hash: ha,
                    group_id: ga,
                },
                SlotEntry::Image {
                    hash: hb,
                    group_id: gb,
                },
            ) if ha == hb => {
                let len_a = group_run_len(cur, i, ha, ga);
                let len_b = group_run_len(new, i, hb, gb);
                if len_a == len_b && i + len_a <= max {
                    i += len_a;
                } else {
                    return i;
                }
            }
            _ => return i,
        }
    }
    i
}

fn group_run_len(entries: &[SlotEntry], start: usize, hash: u64, group_id: u32) -> usize {
    let mut len = 0;
    while start + len < entries.len() {
        match entries[start + len] {
            SlotEntry::Image {
                hash: h,
                group_id: g,
            } if h == hash && g == group_id => {
                len += 1;
            }
            _ => break,
        }
    }
    len
}

/// FNV-1a 64-bit hash. Matches llama-server's `fnv_hash` choice (they use the
/// 32-bit variant on bitmap bytes; 64-bit gives us more headroom and remains
/// trivially fast).
#[cfg_attr(not(feature = "mtmd"), allow(dead_code))]
pub(crate) fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(id: i32) -> SlotEntry {
        SlotEntry::Text(LlamaToken::new(id))
    }
    fn img(hash: u64, group_id: u32) -> SlotEntry {
        SlotEntry::Image { hash, group_id }
    }

    #[test]
    fn empty_inputs() {
        assert_eq!(get_common_prefix(&[], &[]), 0);
        assert_eq!(get_common_prefix(&[t(1)], &[]), 0);
        assert_eq!(get_common_prefix(&[], &[t(1)]), 0);
    }

    #[test]
    fn identical_text() {
        let a = vec![t(1), t(2), t(3)];
        assert_eq!(get_common_prefix(&a, &a), 3);
    }

    #[test]
    fn text_divergence() {
        let a = vec![t(1), t(2), t(3)];
        let b = vec![t(1), t(2), t(99)];
        assert_eq!(get_common_prefix(&a, &b), 2);
    }

    #[test]
    fn new_extends_cur() {
        let a = vec![t(1), t(2)];
        let b = vec![t(1), t(2), t(3), t(4)];
        assert_eq!(get_common_prefix(&a, &b), 2);
    }

    #[test]
    fn identical_image() {
        let a = vec![t(1), img(0xabcd, 0), img(0xabcd, 0), img(0xabcd, 0), t(9)];
        let b = a.clone();
        assert_eq!(get_common_prefix(&a, &b), 5);
    }

    #[test]
    fn image_hash_mismatch_at_offset() {
        let a = vec![t(1), img(0xabcd, 0), img(0xabcd, 0)];
        let b = vec![t(1), img(0xbeef, 0), img(0xbeef, 0)];
        // hash differs → matcher stops at the start of the group
        assert_eq!(get_common_prefix(&a, &b), 1);
    }

    #[test]
    fn image_size_mismatch_same_hash() {
        // Same hash, but different token count (e.g. different mmproj output) → atomic mismatch
        let a = vec![img(0xabcd, 0), img(0xabcd, 0), img(0xabcd, 0)];
        let b = vec![img(0xabcd, 0), img(0xabcd, 0)];
        assert_eq!(get_common_prefix(&a, &b), 0);
    }

    #[test]
    fn image_vs_text_at_offset() {
        let a = vec![t(1), img(0xabcd, 0)];
        let b = vec![t(1), t(2)];
        assert_eq!(get_common_prefix(&a, &b), 1);
    }

    #[test]
    fn group_boundary_respects_group_id() {
        // Two adjacent images with the same hash (e.g. same image attached twice):
        // distinct group_id values must keep the matcher honest.
        let a = vec![img(0x1, 0), img(0x1, 0), img(0x1, 1), img(0x1, 1)];
        let b = vec![img(0x1, 0), img(0x1, 0), img(0x1, 1), img(0x1, 1)];
        assert_eq!(get_common_prefix(&a, &b), 4);

        // Same hash but second occurrence's group length differs in `b`
        let c = vec![img(0x1, 0), img(0x1, 0), img(0x1, 1), img(0x1, 1)];
        let d = vec![img(0x1, 0), img(0x1, 0), img(0x1, 1)];
        // First group of 2 matches; second group has length 2 vs 1 → stops there
        assert_eq!(get_common_prefix(&c, &d), 2);
    }

    #[test]
    fn fnv1a_64_known_vectors() {
        // Reference values for FNV-1a 64-bit
        // Source: http://www.isthe.com/chongo/tech/comp/fnv/index.html
        assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a_64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn fnv1a_64_image_distinct() {
        // Distinct image bytes must produce distinct hashes (in practice).
        let a = vec![1u8; 1024];
        let b = vec![2u8; 1024];
        assert_ne!(fnv1a_64(&a), fnv1a_64(&b));
    }
}
