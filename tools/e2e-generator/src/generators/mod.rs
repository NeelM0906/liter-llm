use anyhow::Result;
use camino::Utf8Path;

use crate::fixture::Fixture;

pub mod c;
pub mod csharp;
pub mod elixir;
pub mod go;
pub mod java;
pub mod php;
pub mod python;
pub mod ruby;
pub mod rust;
pub mod typescript;
pub mod wasm;

/// Common interface implemented by each language generator.
#[allow(dead_code)]
pub trait Generator {
    /// Short identifier for the target language (used in log messages).
    fn name(&self) -> &'static str;

    /// Generate a complete, runnable test project under `output_root/{lang}/`.
    fn generate(&self, fixtures: &[Fixture], output_root: &Utf8Path) -> Result<()>;
}

/// All concrete generator targets (one per supported language).
#[allow(dead_code)]
pub const ALL_TARGETS: &[&str] = &[
    "rust",
    "python",
    "typescript",
    "ruby",
    "go",
    "java",
    "csharp",
    "php",
    "elixir",
    "wasm",
    "c",
];

/// Dispatch to the generator for the given language name.
pub fn run_generator(lang: &str, fixtures: &[Fixture], output_root: &Utf8Path) -> Result<()> {
    match lang {
        "rust" => rust::RustGenerator.generate(fixtures, output_root),
        "python" => python::PythonGenerator.generate(fixtures, output_root),
        "typescript" => typescript::TypeScriptGenerator.generate(fixtures, output_root),
        "go" => go::GoGenerator.generate(fixtures, output_root),
        "ruby" => ruby::RubyGenerator.generate(fixtures, output_root),
        "java" => java::JavaGenerator.generate(fixtures, output_root),
        "csharp" => csharp::CSharpGenerator.generate(fixtures, output_root),
        "php" => php::PhpGenerator.generate(fixtures, output_root),
        "elixir" => elixir::ElixirGenerator.generate(fixtures, output_root),
        "wasm" => wasm::WasmGenerator.generate(fixtures, output_root),
        "c" => c::CGenerator.generate(fixtures, output_root),
        other => {
            println!("TODO: {other} generator (not yet implemented)");
            Ok(())
        }
    }
}
