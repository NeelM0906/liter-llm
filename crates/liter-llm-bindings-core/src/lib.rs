//! Shared utilities for liter-llm language bindings.
//!
//! This crate provides common functionality that is duplicated across
//! multiple binding crates: case conversion, config parsing, error
//! formatting, JSON helpers, and Tokio runtime management.

pub mod case;
#[cfg(feature = "full")]
pub mod config;
#[cfg(feature = "full")]
pub mod error;
pub mod json;
#[cfg(feature = "full")]
pub mod runtime;
