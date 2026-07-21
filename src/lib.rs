//! CT Reconstruction GUI library.
//!
//! The GUI binary (`main.rs`) is a thin shell around these modules; they are
//! exposed here so the non-UI parts (instrument roots, IPTS discovery and
//! validation) can be tested without a display.

pub mod app;
pub mod clean;
pub mod combine;
pub mod config;
pub mod crop;
pub mod instrument;
pub mod ipts;
pub mod logger;
pub mod normalize;
pub mod session;
pub mod tof;
pub mod white_beam;
