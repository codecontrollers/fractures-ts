#![deny(clippy::all)]

#[macro_use]
extern crate napi_derive;
extern crate dyn_clone;
extern crate ndarray;
extern crate ndarray_stats;
extern crate noisy_float;

pub mod cost;
pub mod detection;
pub mod utils;
pub mod wrapper;
