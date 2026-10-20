//! Push loads (ADR-063): an engine sends a plan of strided slices, each
//! source store streams its share once in disk order, and each rank
//! receives only its own bytes into a ring it owns.
//!
//! This crate is shared by sparknest and engines: the plan and its checks
//! ([`plan`]), the slot format ([`slot`]), the messages on the engine's
//! socket ([`proto`]), and a small blocking client ([`client`]).

pub mod client;
pub mod plan;
pub mod proto;
pub mod slot;
