#![expect(missing_docs)] // https://github.com/rust-lang/rust/issues/137561
#![cfg(not(miri))] // FIXME: takes too long

use monotable::HashSet;
use rand::{Rng, SeedableRng, distr::Alphanumeric, rngs::SmallRng};
use std::iter;

#[test]
fn test_hashset_insert_cycles() {
    let mut m: HashSet<Vec<char>> = HashSet::new();
    let seed = u64::from_le_bytes(*b"testseed");

    let rng = &mut SmallRng::seed_from_u64(seed);
    let tx: Vec<Vec<char>> = iter::repeat_with(|| {
        rng.sample_iter(&Alphanumeric)
            .take(32)
            .map(char::from)
            .collect()
    })
    .take(4096)
    .collect();

    // more readable with explicit `true` / `false`
    #[expect(clippy::bool_assert_comparison)]
    for _ in 0..32 {
        for x in &tx {
            assert_eq!(m.contains(x), false);
            assert_eq!(m.insert(x.clone()), true);
        }
        assert_eq!(m.len(), tx.len());
        m.clear();
    }
}
