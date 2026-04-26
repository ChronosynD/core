//! Bridge from the storage layer into a scorer's cache, reads every
//! baseline from the supplied store and pushes it into the scorer, a
//! daemon can call this at startup and again when the store changes

use chronosynd_storage::{BaselineStore, StorageError};

use crate::scorer::Scorer;

/// Load every baseline from `store` into `scorer`, returning the count
pub fn warm_from_store(scorer: &mut Scorer, store: &BaselineStore) -> Result<usize, StorageError> {
    let baselines = store.list_baselines()?;
    let count = baselines.len();
    for baseline in baselines {
        scorer.upsert_baseline(baseline);
    }
    Ok(count)
}
