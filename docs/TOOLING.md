# Required tooling

Everything the build, test loop, or deployment needs, with how to install it.
Add a row the moment a new tool is introduced.

| Tool | Where | Install | Purpose |
|---|---|---|---|
| Rust 1.98.1 | raptor | Homebrew `rust` (already installed) | native amd64 builds |
| Rust 1.98.1 | ostrich (arm64 build host) | rustup: `curl -sSf https://sh.rustup.rs \| sh -s -- -y --profile minimal --default-toolchain 1.98.1 -c clippy -c rustfmt` | native arm64 builds. Homebrew's `rust` bottle depends on a multi-hundred-MB `llvm@22` bottle that downloaded too slowly from ghcr.io, so rustup is used here. |
| libibverbs-dev, librdmacm-dev | all nodes | already installed (rdma-core) | verbs FFI link target |
| fusermount3 | all nodes | already installed (`fuse3`) | mounting as user |
| node ≥ 24, npm | raptor | already installed | web UI: `crates/nest-api/ui` (Svelte 5 + Vite) builds to the committed `crates/nest-api/web/index.html`; `scripts/build.sh` runs `npm ci` and rebuilds when sources change |
| bindgen (cargo) + libclang | raptor only | `cargo install bindgen-cli`; libclang-21 present | regenerate vendored verbs bindings |
| rdmasync, rdmapipe | all nodes | `brew install tpurtell/local-ai/rdmasync rdmapipe` or existing `~/.local/bin` | moving test data, throughput baselines |
| cc, python3 | all nodes | already installed | build the page-cache helper on `sparknest-drop-page-cache --install`; its tests run in `scripts/build.sh --check` |
| sparknest-allow-restart | all nodes | once per host, asks for sudo: `tools/allow-restart/sparknest-allow-restart --hosts` from a checkout (every host in `scripts/cluster.env`), or `sparknest-allow-restart` from the Homebrew package | a sudoers rule for exactly `systemctl restart sparknestd@CLUSTER`, so an agent iterating on the daemon can restart it after a deploy without a password prompt; `--check` tells whether it is in place |
| perftest (`ib_write_bw`) | all nodes | already installed on raptor; `apt`/brew on Sparks if missing | raw fabric ceilings |
| iperf3 | raptor, ostrich, oliver, kakapo | `brew install iperf3` (Sparks may have it from apt) | LAN TCP throughput per stream count (benchmarks/M10-LAN-TCP.md) |

Homebrew (`/home/linuxbrew/.linuxbrew`) is the preferred way to add tools.

## Container test prerequisites

- A minimal Rust toolchain needs `rustup component add clippy --toolchain 1.98.1`
  and `rustfmt` for the required gates.
- Direct-write tests need an io_uring-capable kernel and a container syscall
  policy permitting io_uring, plus sufficient locked memory. Default Docker
  seccomp policies can refuse it even when the host supports it.
- Raise the open-file soft limit to the hard limit, as `scripts/build.sh --check`
  does. Docker can inherit a 1024-descriptor soft limit; long in-process cluster
  suites otherwise exhaust it while opening nodes' SQLite databases.
- The control-socket test requires TCP_RTO_MIN_US. Some 6.14 kernels lack it
  despite the test's version assertion; use a kernel that implements it rather
  than treating a version string as capability evidence.
- HF cache sharing tests require `huggingface_hub==2.2.0`; its HTTP dependencies
  include `httpx2==2.0.0` and `httpcore2==2.0.0`. Use reviewed, digest-verified
  wheels in an isolated test environment, without updating the host's packages.
- On CPU-limited containers, `cargo test --workspace -- --test-threads=1`
  keeps independent Raft timing tests from exhausting the container's CPU turn.

## When ghcr.io downloads stall

Homebrew bottles come from ghcr.io, which sometimes resets long transfers
(seen for the 514 MB `llvm@22` bottle that brew's `rust` needs). Fetch the
blob with resume, then place it in the cache under the name brew expects
(the `.incomplete` file's name without that suffix):

```sh
curl -sSL --http1.1 -C - --retry 5 --retry-all-errors -H "Authorization: Bearer QQ==" \
  -o bottle.tar.gz "https://ghcr.io/v2/homebrew/core/<name>/<version>/blobs/sha256:<digest>"
```
