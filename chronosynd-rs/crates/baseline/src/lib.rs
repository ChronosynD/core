//! Baseline estimators, the production port of Sediment plus a naive
//! mean-and-stddev reference, both mirror the Python implementation at
//! `chronosynd-py` and CI enforces bit-equivalent outputs across them

#![deny(unsafe_op_in_unsafe_fn)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod independent_gaussian;
mod naive;
mod sediment;

pub use error::BaselineError;
pub use independent_gaussian::Baseline;
pub use naive::NaiveBaseline;
pub use sediment::Sediment;
