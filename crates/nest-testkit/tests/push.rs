//! Push loads (ADR-063) end to end: a coordinator opens a group on its
//! host, ranks on other hosts join with memfd rings, and every rank's
//! arena ends up holding exactly the bytes its plan names.

use nest_data::vfs::oflags;
use nest_push::client::{Coordinator, JoinArgs, Rank};
use nest_push::plan::{Dim, Entry, Pattern, Steps};
use nest_push::proto::{Event, RankSpec};
use nest_push::slot;
use nest_testkit::TestCluster;
use nest_types::{FileId, GenState, NodeId};
use std::os::fd::{AsRawFd, OwnedFd};
use std::time::Duration;

async fn ready(n: u64) -> TestCluster {
    let c = TestCluster::start(n).await;
    c.eventually("serving", Duration::from_secs(8), |c| {
        (1..=n).all(|i| c.node(i).data.caught_up() && c.node(i).data.session().is_some())
    })
    .await;
    c.eventually("names registered", Duration::from_secs(8), |c| {
        c.node(1)
            .placer
            .nodes()
            .map(|n| n.iter().all(|h| !h.name.starts_with("node")))
            .unwrap_or(false)
    })
    .await;
    c
}

/// A sealed file of `data` written on node `on`.
async fn sealed(c: &TestCluster, on: u64, name: &str, data: Vec<u8>) -> FileId {
    let v = &c.node(on).vfs;
    let (a, fh, _) = v
        .create(FileId::ROOT, name.as_bytes(), 0o644, oflags::WRONLY)
        .await
        .unwrap();
    let mut off = 0;
    for chunk in data.chunks(1 << 20) {
        v.write(fh, off, chunk.to_vec()).await.unwrap();
        off += chunk.len() as u64;
    }
    v.release(fh, None).await;
    c.eventually("stable", Duration::from_secs(5), |c| {
        c.attr(on, a.id).unwrap().gen_state == GenState::Stable
    })
    .await;
    v.seal(a.id, true).await.unwrap();
    a.id
}

fn memfd(len: usize) -> OwnedFd {
    let fd = nix::sys::memfd::memfd_create(c"push-ring", nix::sys::memfd::MFdFlags::MFD_CLOEXEC)
        .unwrap();
    nix::unistd::ftruncate(&fd, len as i64).unwrap();
    fd
}

/// What an engine rank does: copy each filled slot's fragments into its
/// arena, release the slot, until Done. Returns the arena.
fn run_rank(
    r: Rank,
    ring: OwnedFd,
    slots: u32,
    slot_bytes: u32,
    arena: usize,
) -> Result<Vec<u8>, String> {
    let file = std::fs::File::from(ring);
    let mut out = vec![0u8; arena];
    loop {
        match r.next().map_err(|e| e.to_string())? {
            Some(Event::Filled { slot: i, len }) => {
                assert!(i < slots && len == slot_bytes);
                let mut buf = vec![0u8; slot_bytes as usize];
                use std::os::unix::fs::FileExt;
                file.read_exact_at(&mut buf, i as u64 * slot_bytes as u64)
                    .unwrap();
                let p = slot::parse(&buf, arena as u64).map_err(|e| e.to_string())?;
                let mut at = 0usize;
                for (dst, n) in slot::Fragments::new(&buf, p) {
                    out[dst as usize..dst as usize + n as usize]
                        .copy_from_slice(&buf[at..at + n as usize]);
                    at += n as usize;
                }
                r.release(i).map_err(|e| e.to_string())?;
            }
            Some(Event::Done) => return Ok(out),
            Some(Event::Failed { reason, .. }) => return Err(reason),
            Some(other) => return Err(format!("unexpected {other:?}")),
            None => return Err("socket closed".into()),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn ranks_receive_exactly_their_strided_shares() {
    let c = ready(3).await;
    // Two files: one on node 2, one on node 3 (each written there).
    let (rows, units, unit) = (64u32, 6u32, 1024u32);
    let row = (units * unit) as usize;
    let len = rows as usize * row;
    let a: Vec<u8> = (0..len).map(|i| (i * 7 % 251) as u8).collect();
    let b: Vec<u8> = (0..len).map(|i| (i * 13 % 241) as u8).collect();
    sealed(&c, 2, "a.bin", a.clone()).await;
    sealed(&c, 3, "b.bin", b.clone()).await;

    // Rank 0 (on node 1) takes units {0,2,4} of every row, rank 1 (on
    // node 3) units {1,3,5}, packed; of both files, one after the other
    // in the arena.
    let owned: [Vec<u32>; 2] = [vec![0, 2, 4], vec![1, 3, 5]];
    let share = rows as u64 * 3 * unit as u64;
    let patterns: Vec<Pattern> = owned
        .iter()
        .map(|sel| Pattern {
            run: unit,
            dims: vec![
                Dim {
                    steps: Steps::Count(rows),
                    src_stride: row as u64,
                    dst_stride: sel.len() as u64 * unit as u64,
                },
                Dim {
                    steps: Steps::Indices(sel.clone()),
                    src_stride: unit as u64,
                    dst_stride: unit as u64,
                },
            ],
        })
        .collect();
    let mut entries = Vec::new();
    for f in 0..2u32 {
        for r in 0..2u32 {
            entries.push(Entry {
                file: f,
                pattern: r,
                src_base: 0,
                dst_base: f as u64 * share,
                rank: r,
            });
        }
    }
    let arena = 2 * share;
    let (slots, slot_bytes) = (4u32, 64u32 << 10);
    let hub = c.push_socket(1);
    let rank_sockets = [c.push_socket(1), c.push_socket(3)];

    let res = tokio::task::spawn_blocking(move || {
        let t = Some(Duration::from_secs(20));
        let g = Coordinator::create(
            &hub,
            vec![
                RankSpec {
                    rank: 0,
                    host: "n1".into(),
                    arena_bytes: arena,
                },
                RankSpec {
                    rank: 1,
                    host: "n3".into(),
                    arena_bytes: arena,
                },
            ],
            vec!["/a.bin".into(), "/b.bin".into()],
            t,
        )
        .map_err(|e| format!("create: {e}"))?;
        g.patterns(patterns).unwrap();
        g.entries(entries).unwrap();
        let (group, token) = (g.group, g.token);
        // Ranks join from their own hosts while Start waits for them.
        let ranks: Vec<_> = (0..2u32)
            .map(|r| {
                let sock = rank_sockets[r as usize].clone();
                std::thread::spawn(move || {
                    // Join only once the hub has told this host to expect it.
                    let ring = memfd(slots as usize * slot_bytes as usize);
                    let mut tries = 0;
                    let rank = loop {
                        match Rank::join(
                            &sock,
                            JoinArgs {
                                group,
                                token,
                                rank: r,
                                ring_fd: ring.as_raw_fd(),
                                slots,
                                slot_bytes,
                            },
                            Some(Duration::from_secs(20)),
                        ) {
                            Ok(k) => break k,
                            Err(e) if tries < 100 => {
                                let _ = e;
                                tries += 1;
                                std::thread::sleep(Duration::from_millis(50));
                            }
                            Err(e) => return Err(format!("join: {e}")),
                        }
                    };
                    run_rank(rank, ring, slots, slot_bytes, arena as usize)
                })
            })
            .collect();
        let forecast = g.start();
        let results: Vec<Result<Vec<u8>, String>> =
            ranks.into_iter().map(|h| h.join().unwrap()).collect();
        let forecast = forecast.map_err(|e| {
            format!(
                "start: {e}; ranks: {:?}",
                results.iter().map(|r| r.as_ref().err()).collect::<Vec<_>>()
            )
        })?;
        let arenas: Vec<Vec<u8>> = results
            .into_iter()
            .enumerate()
            .map(|(i, r)| r.map_err(|e| format!("rank {i}: {e}")))
            .collect::<Result<_, _>>()?;
        // The coordinator hears Done after every rank has everything.
        let mut done = false;
        while let Some(e) = g.next().map_err(|e| e.to_string())? {
            match e {
                Event::Done => {
                    done = true;
                    break;
                }
                Event::Progress { .. } => {}
                other => return Err(format!("coordinator got {other:?}")),
            }
        }
        assert!(done);
        Ok::<_, String>((forecast, arenas))
    })
    .await
    .unwrap();
    let (forecast, arenas) = res.unwrap();
    assert_eq!(forecast.rank_bytes, vec![arena, arena]);
    // Each file was streamed by the host holding it.
    let mut stores: Vec<_> = forecast.streams.iter().map(|s| s.store.clone()).collect();
    stores.sort();
    assert_eq!(stores, vec!["n2", "n3"]);
    for (r, got) in arenas.iter().enumerate() {
        let mut want = Vec::new();
        for f in [&a, &b] {
            for rr in 0..rows as usize {
                for u in &owned[r] {
                    let at = rr * row + *u as usize * unit as usize;
                    want.extend_from_slice(&f[at..at + unit as usize]);
                }
            }
        }
        assert_eq!(got.len(), want.len());
        assert!(got == &want, "rank {r}'s arena differs from its share");
    }
    let _ = NodeId(0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_bad_plan_is_refused_before_anything_moves() {
    let c = ready(2).await;
    sealed(&c, 2, "w.bin", vec![1u8; 4096]).await;
    let hub = c.push_socket(1);
    let r = tokio::task::spawn_blocking(move || {
        let g = Coordinator::create(
            &hub,
            vec![RankSpec {
                rank: 0,
                host: "n1".into(),
                arena_bytes: 1024,
            }],
            vec!["/w.bin".into()],
            Some(Duration::from_secs(10)),
        )
        .unwrap();
        g.patterns(vec![Pattern {
            run: 2048,
            dims: vec![],
        }])
        .unwrap();
        // 2 KiB into a 1 KiB arena.
        g.entries(vec![Entry {
            file: 0,
            pattern: 0,
            src_base: 0,
            dst_base: 0,
            rank: 0,
        }])
        .unwrap();
        g.start().map(|_| ()).map_err(|e| e.to_string())
    })
    .await
    .unwrap();
    let e = r.unwrap_err();
    assert!(e.contains("arena"), "{e}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn an_unsealed_file_is_refused() {
    let c = ready(1).await;
    let v = &c.node(1).vfs;
    let (_, fh, _) = v
        .create(FileId::ROOT, b"open.bin", 0o644, oflags::WRONLY)
        .await
        .unwrap();
    v.write(fh, 0, vec![1u8; 4096]).await.unwrap();
    v.release(fh, None).await;
    let hub = c.push_socket(1);
    let r = tokio::task::spawn_blocking(move || {
        let g = Coordinator::create(
            &hub,
            vec![RankSpec {
                rank: 0,
                host: "n1".into(),
                arena_bytes: 4096,
            }],
            vec!["/open.bin".into()],
            Some(Duration::from_secs(10)),
        )
        .unwrap();
        g.patterns(vec![Pattern {
            run: 4096,
            dims: vec![],
        }])
        .unwrap();
        g.entries(vec![Entry {
            file: 0,
            pattern: 0,
            src_base: 0,
            dst_base: 0,
            rank: 0,
        }])
        .unwrap();
        g.start().map(|_| ()).map_err(|e| e.to_string())
    })
    .await
    .unwrap();
    assert!(r.unwrap_err().contains("not sealed"));
}

/// One rank, one file; returns (coordinator, group, token).
fn one_rank_group(hub: &str, arena: u64, path: &str) -> (Coordinator, u64, u64) {
    let g = Coordinator::create(
        hub,
        vec![RankSpec {
            rank: 0,
            host: "n1".into(),
            arena_bytes: arena,
        }],
        vec![path.into()],
        Some(Duration::from_secs(20)),
    )
    .unwrap();
    g.patterns(vec![Pattern {
        run: arena as u32,
        dims: vec![],
    }])
    .unwrap();
    g.entries(vec![Entry {
        file: 0,
        pattern: 0,
        src_base: 0,
        dst_base: 0,
        rank: 0,
    }])
    .unwrap();
    let (group, token) = (g.group, g.token);
    (g, group, token)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_rank_that_leaves_fails_the_group() {
    let c = ready(2).await;
    let len = 1usize << 20;
    sealed(&c, 2, "w.bin", vec![5u8; len]).await;
    let (hub, sock) = (c.push_socket(1), c.push_socket(1));
    let r = tokio::task::spawn_blocking(move || {
        let (g, group, token) = one_rank_group(&hub, len as u64, "/w.bin");
        let (slots, slot_bytes) = (2u32, 64u32 << 10);
        let joiner = std::thread::spawn(move || {
            let ring = memfd(slots as usize * slot_bytes as usize);
            let rank = loop {
                if let Ok(k) = Rank::join(
                    &sock,
                    JoinArgs {
                        group,
                        token,
                        rank: 0,
                        ring_fd: ring.as_raw_fd(),
                        slots,
                        slot_bytes,
                    },
                    Some(Duration::from_secs(10)),
                ) {
                    break k;
                }
                std::thread::sleep(Duration::from_millis(20));
            };
            // Take one slot and leave without releasing it.
            assert!(matches!(rank.next().unwrap(), Some(Event::Filled { .. })));
            drop(rank);
        });
        g.start().unwrap();
        joiner.join().unwrap();
        loop {
            match g.next().unwrap() {
                Some(Event::Progress { .. }) => continue,
                other => return other,
            }
        }
    })
    .await
    .unwrap();
    assert!(
        matches!(&r, Some(Event::Failed { reason, .. }) if reason.contains("closed")),
        "{r:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_coordinator_that_leaves_cancels_the_ranks() {
    let c = ready(2).await;
    let len = 1usize << 20;
    sealed(&c, 2, "w.bin", vec![5u8; len]).await;
    let (hub, sock) = (c.push_socket(1), c.push_socket(1));
    let r = tokio::task::spawn_blocking(move || {
        let (g, group, token) = one_rank_group(&hub, len as u64, "/w.bin");
        let (slots, slot_bytes) = (2u32, 64u32 << 10);
        let ring = memfd(slots as usize * slot_bytes as usize);
        let (fd, s2) = (ring.as_raw_fd(), sock.clone());
        let starter = std::thread::spawn(move || {
            g.start().unwrap();
            // The coordinator goes away once loading has begun.
            std::thread::sleep(Duration::from_millis(100));
            drop(g);
        });
        let rank = loop {
            if let Ok(k) = Rank::join(
                &s2,
                JoinArgs {
                    group,
                    token,
                    rank: 0,
                    ring_fd: fd,
                    slots,
                    slot_bytes,
                },
                Some(Duration::from_secs(10)),
            ) {
                break k;
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        starter.join().unwrap();
        // Never release: the load cannot finish; the rank hears it failed.
        loop {
            match rank.next().unwrap() {
                Some(Event::Filled { .. }) => continue,
                other => return other,
            }
        }
    })
    .await
    .unwrap();
    assert!(
        matches!(&r, Some(Event::Failed { reason, .. }) if reason.contains("coordinator")),
        "{r:?}"
    );
}
