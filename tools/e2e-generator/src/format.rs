//! Post-generation formatting for each language.
//!
//! Each function runs the language's native formatter on the generated output
//! directory.  Failures are logged but non-fatal — a missing formatter tool
//! should not block generation.

use camino::Utf8Path;
use std::process::Command;

/// Run a formatter command.  Prints a warning on failure instead of
/// propagating the error, since a missing formatter should not block CI.
fn run_formatter(name: &str, cmd: &mut Command) {
    match cmd.output() {
        Ok(output) if output.status.success() => {
            eprintln!("  ✓ {name}");
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("  ⚠ {name} exited with {}: {stderr}", output.status);
        }
        Err(e) => {
            eprintln!("  ⚠ {name} not found or failed to run: {e}");
        }
    }
}

/// Format generated Rust code.
pub fn format_rust(dir: &Utf8Path) {
    let manifest = dir.join("Cargo.toml");
    if manifest.exists() {
        run_formatter(
            "cargo fmt",
            Command::new("cargo")
                .args(["fmt", "--manifest-path"])
                .arg(manifest.as_str()),
        );
    }
}

/// Format generated Python code.
pub fn format_python(dir: &Utf8Path) {
    let tests_dir = dir.join("tests");
    let target = if tests_dir.exists() {
        tests_dir.as_str().to_owned()
    } else {
        dir.as_str().to_owned()
    };
    run_formatter(
        "ruff check --fix",
        Command::new("ruff").args(["check", "--fix", "--quiet", &target]),
    );
    run_formatter("ruff format", Command::new("ruff").args(["format", "--quiet", &target]));
}

/// Format generated TypeScript / WASM code.
pub fn format_typescript(dir: &Utf8Path) {
    run_formatter(
        "biome check --fix",
        Command::new("biome")
            .args(["check", "--fix", "--unsafe", "--write"])
            .arg(dir.as_str()),
    );
    run_formatter(
        "biome format",
        Command::new("biome").args(["format", "--write"]).arg(dir.as_str()),
    );
}

/// Format generated Go code.
pub fn format_go(dir: &Utf8Path) {
    run_formatter("gofmt", Command::new("gofmt").args(["-w"]).arg(dir.as_str()));
    run_formatter("goimports", Command::new("goimports").args(["-w"]).arg(dir.as_str()));
}

/// Format generated Ruby code.
pub fn format_ruby(dir: &Utf8Path) {
    run_formatter(
        "rubocop --autocorrect",
        Command::new("rubocop")
            .args(["--autocorrect-all", "--no-color"])
            .arg(dir.as_str()),
    );
}

/// Format generated Java code.
pub fn format_java(dir: &Utf8Path) {
    let pom = dir.join("pom.xml");
    if pom.exists() {
        run_formatter(
            "mvn spotless:apply",
            Command::new("mvn")
                .args(["spotless:apply", "-q"])
                .current_dir(dir.as_str()),
        );
    }
}

/// Format generated C# code.
pub fn format_csharp(dir: &Utf8Path) {
    run_formatter(
        "dotnet format",
        Command::new("dotnet")
            .args(["format", "--verbosity", "quiet"])
            .arg(dir.as_str()),
    );
}

/// Format generated PHP code.
pub fn format_php(dir: &Utf8Path) {
    run_formatter(
        "php-cs-fixer fix",
        Command::new("php-cs-fixer")
            .args(["fix", "--rules=@PSR12", "--quiet"])
            .arg(dir.as_str()),
    );
}

/// Format generated Elixir code.
pub fn format_elixir(dir: &Utf8Path) {
    if dir.join("mix.exs").exists() {
        run_formatter(
            "mix format",
            Command::new("mix").args(["format"]).current_dir(dir.as_str()),
        );
    }
}

/// Format generated C code.
pub fn format_c(dir: &Utf8Path) {
    run_formatter(
        "clang-format",
        Command::new("clang-format")
            .args(["-i", "--style=file"])
            .arg(dir.join("test_liter_llm.c").as_str()),
    );
}

/// Run the appropriate formatter for a language.
pub fn format_language(lang: &str, output_root: &Utf8Path) {
    let dir = output_root.join(lang);
    if !dir.exists() {
        return;
    }
    eprintln!("Formatting {lang}...");
    match lang {
        "rust" => format_rust(&dir),
        "python" => format_python(&dir),
        "typescript" | "wasm" => format_typescript(&dir),
        "go" => format_go(&dir),
        "ruby" => format_ruby(&dir),
        "java" => format_java(&dir),
        "csharp" => format_csharp(&dir),
        "php" => format_php(&dir),
        "elixir" => format_elixir(&dir),
        "c" => format_c(&dir),
        _ => {}
    }
}
