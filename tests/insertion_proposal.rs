#![expect(missing_docs)] // integration coverage for the public proposal API

use monotable::{HashMap, HashSet, HashTable, InsertionProposal};

use std::hash::{BuildHasher, Hasher};

#[derive(Clone, Copy, Default)]
struct ConstState;

#[derive(Default)]
struct ConstHasher;

impl Hasher for ConstHasher {
    fn finish(&self) -> u64 {
        0
    }

    fn write(&mut self, _bytes: &[u8]) {}
}

impl BuildHasher for ConstState {
    type Hasher = ConstHasher;

    fn build_hasher(&self) -> Self::Hasher {
        ConstHasher
    }
}

fn map_proposal<V>(map: &mut HashMap<i32, V, ConstState>, key: &i32) -> InsertionProposal {
    match map.find_or_find_insert_proposal(key) {
        Ok(_) => panic!("expected proposal"),
        Err(proposal) => proposal,
    }
}

fn set_proposal(set: &mut HashSet<i32, ConstState>, value: &i32) -> InsertionProposal {
    match set.find_or_find_insert_proposal(value) {
        Ok(_) => panic!("expected proposal"),
        Err(proposal) => proposal,
    }
}

fn table_proposal(table: &mut HashTable<i32>, value: i32) -> InsertionProposal {
    match table.find_or_find_insert_proposal(0, |entry| *entry == value, |_| 0) {
        Ok(_) => panic!("expected proposal"),
        Err(proposal) => proposal,
    }
}

#[test]
fn hash_map_returns_occupied_entry_for_existing_key() {
    let mut map = HashMap::with_hasher(ConstState);
    map.insert(1, "one");

    match map.find_or_find_insert_proposal(&1) {
        Ok(entry) => {
            assert_eq!(entry.key(), &1);
            assert_eq!(entry.get(), &"one");
        }
        Err(_) => panic!("expected occupied entry"),
    }
}

#[test]
fn hash_map_insert_with_valid_proposal() {
    let mut map = HashMap::with_hasher(ConstState);
    let proposal = map_proposal(&mut map, &1);

    let (key, value) = unsafe { map.insert_with_proposal(proposal, 1, "one") };
    assert_eq!(*key, 1);
    assert_eq!(*value, "one");

    *value = "uno";
    assert_eq!(map.get(&1), Some(&"uno"));
}

#[test]
fn hash_map_insert_with_proposal_after_slot_taken() {
    let mut map = HashMap::with_hasher(ConstState);
    let proposal = map_proposal(&mut map, &1);

    map.insert(2, "two");
    unsafe {
        map.insert_with_proposal(proposal, 1, "one");
    }

    assert_eq!(map.get(&1), Some(&"one"));
    assert_eq!(map.get(&2), Some(&"two"));
}

#[test]
fn hash_map_insert_with_proposal_after_rehash() {
    let mut map = HashMap::with_capacity_and_hasher(1, ConstState);
    let proposal = map_proposal(&mut map, &999);

    for value in 0..128 {
        map.insert(value, value);
    }

    unsafe {
        map.insert_with_proposal(proposal, 999, 999);
    }

    for value in 0..128 {
        assert_eq!(map.get(&value), Some(&value));
    }
    assert_eq!(map.get(&999), Some(&999));
}

#[test]
fn hash_set_returns_occupied_entry_for_existing_value() {
    let mut set = HashSet::with_hasher(ConstState);
    set.insert(1);

    match set.find_or_find_insert_proposal(&1) {
        Ok(entry) => assert_eq!(entry.get(), &1),
        Err(_) => panic!("expected occupied entry"),
    }
}

#[test]
fn hash_set_insert_with_proposal_after_slot_taken() {
    let mut set = HashSet::with_hasher(ConstState);
    let proposal = set_proposal(&mut set, &1);

    set.insert(2);
    let inserted = unsafe { set.insert_with_proposal(proposal, 1) };

    assert_eq!(inserted, &1);
    assert!(set.contains(&1));
    assert!(set.contains(&2));
}

#[test]
fn hash_set_insert_with_proposal_after_rehash() {
    let mut set = HashSet::with_capacity_and_hasher(1, ConstState);
    let proposal = set_proposal(&mut set, &999);

    for value in 0..128 {
        set.insert(value);
    }

    let inserted = unsafe { set.insert_with_proposal(proposal, 999) };
    assert_eq!(inserted, &999);

    for value in 0..128 {
        assert!(set.contains(&value));
    }
    assert!(set.contains(&999));
}

#[test]
fn hash_table_returns_occupied_entry_for_existing_value() {
    let mut table = HashTable::new();
    table.insert_unique(0, 1, |_| 0);

    match table.find_or_find_insert_proposal(0, |entry| *entry == 1, |_| 0) {
        Ok(entry) => assert_eq!(entry.get(), &1),
        Err(_) => panic!("expected occupied entry"),
    }
}

#[test]
fn hash_table_insert_with_proposal_after_slot_taken() {
    let mut table = HashTable::new();
    let proposal = table_proposal(&mut table, 1);

    table.insert_unique(0, 2, |_| 0);
    let entry = unsafe { table.insert_with_proposal(proposal, 0, 1, |_| 0) };

    assert_eq!(entry.get(), &1);
    assert!(table.find_entry(0, |entry| *entry == 1).is_ok());
    assert!(table.find_entry(0, |entry| *entry == 2).is_ok());
}

#[test]
fn hash_table_insert_with_proposal_after_rehash() {
    let mut table = HashTable::with_capacity(1);
    let proposal = table_proposal(&mut table, 999);

    for value in 0..128 {
        table.insert_unique(0, value, |_| 0);
    }

    let entry = unsafe { table.insert_with_proposal(proposal, 0, 999, |_| 0) };
    assert_eq!(entry.get(), &999);

    for value in 0..128 {
        assert!(table.find_entry(0, |entry| *entry == value).is_ok());
    }
    assert!(table.find_entry(0, |entry| *entry == 999).is_ok());
}
