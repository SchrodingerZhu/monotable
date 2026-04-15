use crate::alloc::{Allocator, Global};
use crate::raw::{Bucket, RawIter, RawIterRange, RawTable};
use crate::scopeguard::guard;
use core::mem;
use rayon::iter::{
    ParallelIterator,
    plumbing::{self, Folder, UnindexedConsumer, UnindexedProducer},
};

/// Parallel iterator which returns a raw pointer to every full bucket in the table.
pub(crate) struct RawParIter<T> {
    iter: RawIterRange<T>,
}

impl<T> RawParIter<T> {
    #[cfg_attr(feature = "inline-more", inline)]
    pub(super) unsafe fn iter(&self) -> RawIterRange<T> {
        self.iter.clone()
    }
}

impl<T> Clone for RawParIter<T> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
        }
    }
}

impl<T> From<RawIter<T>> for RawParIter<T> {
    fn from(it: RawIter<T>) -> Self {
        RawParIter { iter: it.iter }
    }
}

impl<T> ParallelIterator for RawParIter<T> {
    type Item = Bucket<T>;

    #[cfg_attr(feature = "inline-more", inline)]
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        let producer = ParIterProducer { iter: self.iter };
        plumbing::bridge_unindexed(producer, consumer)
    }
}

/// Producer which returns a `Bucket<T>` for every element.
struct ParIterProducer<T> {
    iter: RawIterRange<T>,
}

impl<T> UnindexedProducer for ParIterProducer<T> {
    type Item = Bucket<T>;

    #[cfg_attr(feature = "inline-more", inline)]
    fn split(self) -> (Self, Option<Self>) {
        let (left, right) = self.iter.split();
        let left = ParIterProducer { iter: left };
        let right = right.map(|right| ParIterProducer { iter: right });
        (left, right)
    }

    #[cfg_attr(feature = "inline-more", inline)]
    fn fold_with<F>(self, folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        folder.consume_iter(self.iter)
    }
}

/// Parallel iterator which consumes a table and returns elements.
pub(crate) struct RawIntoParIter<T, A: Allocator = Global> {
    table: RawTable<T, A>,
}

impl<T, A: Allocator> RawIntoParIter<T, A> {
    #[cfg_attr(feature = "inline-more", inline)]
    pub(super) unsafe fn par_iter(&self) -> RawParIter<T> {
        unsafe { self.table.par_iter() }
    }
}

impl<T: Send, A: Allocator + Send> ParallelIterator for RawIntoParIter<T, A> {
    type Item = T;

    #[cfg_attr(feature = "inline-more", inline)]
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        let iter = unsafe { self.table.iter().iter };
        let _guard = guard(self.table.into_allocation(), |alloc| {
            if let Some((ptr, layout, ref alloc)) = *alloc {
                unsafe {
                    alloc.deallocate(ptr, layout);
                }
            }
        });
        let producer = ParDrainProducer { iter };
        plumbing::bridge_unindexed(producer, consumer)
    }
}

/// Producer which will consume all elements in the range, even if it is dropped
/// halfway through.
struct ParDrainProducer<T> {
    iter: RawIterRange<T>,
}

impl<T: Send> UnindexedProducer for ParDrainProducer<T> {
    type Item = T;

    #[cfg_attr(feature = "inline-more", inline)]
    fn split(self) -> (Self, Option<Self>) {
        let (left, right) = self.iter.clone().split();
        mem::forget(self);
        let left = ParDrainProducer { iter: left };
        let right = right.map(|right| ParDrainProducer { iter: right });
        (left, right)
    }

    #[cfg_attr(feature = "inline-more", inline)]
    fn fold_with<F>(mut self, mut folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        // Make sure to modify the iterator in-place so that any remaining
        // elements are processed in our Drop impl.
        for item in &mut self.iter {
            folder = folder.consume(unsafe { item.read() });
            if folder.full() {
                return folder;
            }
        }

        // If we processed all elements then we don't need to run the drop.
        mem::forget(self);
        folder
    }
}

impl<T> Drop for ParDrainProducer<T> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn drop(&mut self) {
        // Drop all remaining elements
        if mem::needs_drop::<T>() {
            for item in &mut self.iter {
                unsafe {
                    item.drop();
                }
            }
        }
    }
}

impl<T, A: Allocator> RawTable<T, A> {
    /// Returns a parallel iterator over the elements in a `RawTable`.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) unsafe fn par_iter(&self) -> RawParIter<T> {
        unsafe {
            RawParIter {
                iter: self.iter().iter,
            }
        }
    }

    /// Returns a parallel iterator over the elements in a `RawTable`.
    #[cfg_attr(feature = "inline-more", inline)]
    pub(crate) fn into_par_iter(self) -> RawIntoParIter<T, A> {
        RawIntoParIter { table: self }
    }
}
