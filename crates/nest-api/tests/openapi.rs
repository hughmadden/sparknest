//! `openapi.yaml` documents exactly the routes the crate registers: every
//! `.route(...)` in `src/` (the management router, the web listener's page
//! and the RDMA counters listener), method by method. Adding a route
//! without documenting it, or documenting one that is gone, fails here.

use std::collections::BTreeSet;
use std::path::Path;

const METHODS: &[&str] = &["get", "post", "put", "delete", "patch", "head", "options"];

/// (METHOD, path) for every `.route("path", method(h).method(h)...)` in
/// the crate's sources.
fn registered() -> BTreeSet<(String, String)> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out = BTreeSet::new();
    let mut files: Vec<_> = std::fs::read_dir(&src)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .collect();
    files.sort();
    for f in files {
        let text = std::fs::read_to_string(&f).unwrap();
        let mut rest = text.as_str();
        while let Some(at) = rest.find(".route(") {
            rest = &rest[at + ".route(".len()..];
            let args = balanced(rest);
            let path = args
                .split('"')
                .nth(1)
                .unwrap_or_else(|| panic!("{}: a route without a literal path", f.display()));
            let methods = methods_in(args);
            assert!(
                !methods.is_empty(),
                "{}: route {path} has no method",
                f.display()
            );
            for m in methods {
                out.insert((m.to_uppercase(), path.to_string()));
            }
        }
    }
    out
}

/// The text up to the parenthesis closing the one just opened.
fn balanced(s: &str) -> &str {
    let mut depth = 1;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return &s[..i];
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced .route(");
}

/// Method routers named in a route's arguments: `get(`, `.post(`,
/// `axum::routing::delete(`; not handler names like `get_plan`.
fn methods_in(args: &str) -> Vec<&'static str> {
    let b = args.as_bytes();
    let mut found = Vec::new();
    for (i, _) in args.char_indices() {
        if i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_') {
            continue;
        }
        for m in METHODS {
            if args[i..].starts_with(m) && args[i + m.len()..].starts_with('(') {
                found.push(*m);
            }
        }
    }
    found
}

fn spec() -> serde_yaml::Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("openapi.yaml");
    serde_yaml::from_str(&std::fs::read_to_string(p).unwrap()).expect("openapi.yaml parses")
}

fn documented(spec: &serde_yaml::Value) -> BTreeSet<(String, String)> {
    let mut out = BTreeSet::new();
    let paths = spec["paths"].as_mapping().expect("paths");
    for (path, item) in paths {
        let path = path.as_str().unwrap();
        for (k, op) in item.as_mapping().expect("path item") {
            let k = k.as_str().unwrap();
            if METHODS.contains(&k) {
                assert!(
                    op["responses"].as_mapping().is_some_and(|r| !r.is_empty()),
                    "{k} {path} documents no responses"
                );
                out.insert((k.to_uppercase(), path.to_string()));
            }
        }
    }
    out
}

#[test]
fn openapi_documents_every_route() {
    let spec = spec();
    assert_eq!(spec["openapi"].as_str(), Some("3.1.0"));
    let reg = registered();
    let doc = documented(&spec);
    // Sanity: the scan found the router.
    assert!(reg.contains(&("GET".into(), "/v1/status".into())));
    assert!(reg.contains(&("DELETE".into(), "/v1/backups/{id}".into())));
    let undocumented: Vec<_> = reg.difference(&doc).collect();
    let stale: Vec<_> = doc.difference(&reg).collect();
    assert!(
        undocumented.is_empty() && stale.is_empty(),
        "openapi.yaml is out of date\n  registered but not documented: {undocumented:?}\n  documented but not registered: {stale:?}"
    );
}

#[test]
fn openapi_refs_resolve() {
    fn walk(v: &serde_yaml::Value, refs: &mut Vec<String>) {
        match v {
            serde_yaml::Value::Mapping(m) => {
                for (k, v) in m {
                    if k.as_str() == Some("$ref") {
                        refs.push(v.as_str().expect("$ref is a string").to_string());
                    } else {
                        walk(v, refs);
                    }
                }
            }
            serde_yaml::Value::Sequence(s) => s.iter().for_each(|v| walk(v, refs)),
            _ => {}
        }
    }
    let spec = spec();
    let mut refs = Vec::new();
    walk(&spec, &mut refs);
    assert!(!refs.is_empty());
    for r in refs {
        let mut at = &spec;
        for part in r
            .strip_prefix("#/")
            .unwrap_or_else(|| panic!("{r}: only local refs"))
            .split('/')
        {
            at = &at[part];
        }
        assert!(!at.is_null(), "{r} does not resolve");
    }
}
