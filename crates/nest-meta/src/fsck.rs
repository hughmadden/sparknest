//! Structural consistency checks over replicated metadata. Used by tests
//! after every step and by `nest fsck`.

use rusqlite::Connection;

/// Returns a list of human-readable problems; empty means consistent.
pub fn check(c: &Connection) -> rusqlite::Result<Vec<String>> {
    let mut problems = Vec::new();
    let mut q = |sql: &str, what: &str| -> rusqlite::Result<()> {
        let mut st = c.prepare(sql)?;
        let rows: Vec<String> = st
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<_>>()?;
        for r in rows {
            problems.push(format!("{what}: {r}"));
        }
        Ok(())
    };
    q(
        "SELECT printf('%d/%s -> %d', d.parent, hex(d.name), d.child) FROM dentries d LEFT JOIN files f ON f.id = d.child WHERE f.id IS NULL",
        "dangling dentry",
    )?;
    q(
        "SELECT printf('%d/%s', d.parent, hex(d.name)) FROM dentries d JOIN files p ON p.id = d.parent WHERE p.kind != 2",
        "dentry under non-directory",
    )?;
    q(
        "SELECT printf('file %d nlink %d names %d', f.id, f.nlink, (SELECT COUNT(*) FROM dentries d WHERE d.child = f.id)) \
         FROM files f WHERE f.kind != 2 AND f.nlink != (SELECT COUNT(*) FROM dentries d WHERE d.child = f.id)",
        "link count",
    )?;
    q(
        "SELECT printf('dir %d nlink %d expected %d', f.id, f.nlink, 2 + (SELECT COUNT(*) FROM files s WHERE s.kind = 2 AND s.parent = f.id AND s.id != f.id)) \
         FROM files f WHERE f.kind = 2 AND f.nlink != 2 + (SELECT COUNT(*) FROM files s WHERE s.kind = 2 AND s.parent = f.id AND s.id != f.id)",
        "directory link count",
    )?;
    q(
        "SELECT printf('dir %d has %d names', f.id, (SELECT COUNT(*) FROM dentries d WHERE d.child = f.id)) \
         FROM files f WHERE f.kind = 2 AND f.id != 1 AND (SELECT COUNT(*) FROM dentries d WHERE d.child = f.id) != 1",
        "directory name count",
    )?;
    q(
        "SELECT printf('dir %d parent %d but named under %d', f.id, f.parent, d.parent) FROM files f JOIN dentries d ON d.child = f.id WHERE f.kind = 2 AND f.parent != d.parent",
        "directory parent mismatch",
    )?;
    q(
        "SELECT printf('file %d', f.id) FROM files f WHERE f.id != 1 AND f.nlink = 0 AND NOT EXISTS (SELECT 1 FROM orphans o WHERE o.file = f.id)",
        "unreferenced file",
    )?;
    q(
        "SELECT printf('file %d', o.file) FROM orphans o JOIN files f ON f.id = o.file WHERE f.nlink != 0",
        "orphan row for linked file",
    )?;
    q(
        "SELECT printf('file %d session %d', o.file, o.session) FROM orphans o LEFT JOIN sessions s ON s.id = o.session WHERE s.id IS NULL",
        "orphan row for dead session",
    )?;
    q(
        "SELECT printf('file %d session %d', l.file, l.session) FROM locks l LEFT JOIN sessions s ON s.id = l.session LEFT JOIN files f ON f.id = l.file WHERE s.id IS NULL OR f.id IS NULL",
        "lock held by dead session or on missing file",
    )?;
    q(
        "SELECT printf('file %d: %d/%d [%d,%d] vs %d/%d [%d,%d]', a.file, a.session, a.owner, a.start, a.end_, b.session, b.owner, b.start, b.end_) \
         FROM locks a JOIN locks b ON a.file = b.file \
         WHERE (a.session < b.session OR (a.session = b.session AND a.owner < b.owner)) \
         AND a.start <= b.end_ AND b.start <= a.end_ AND (a.kind = 1 OR b.kind = 1)",
        "conflicting locks",
    )?;
    q(
        "SELECT printf('file %d owner %d/%d at %d', a.file, a.session, a.owner, b.start) FROM locks a JOIN locks b \
         ON a.file = b.file AND a.session = b.session AND a.owner = b.owner AND a.start < b.start AND a.end_ >= b.start",
        "overlapping ranges of one owner",
    )?;
    q(
        "SELECT printf('file %d gen %d store %d', r.file, r.gen, r.store) FROM replicas r LEFT JOIN files f ON f.id = r.file WHERE f.id IS NULL OR f.kind != 1",
        "replica of missing file",
    )?;
    q(
        "SELECT printf('file %d gen %d current %d', r.file, r.gen, f.gen) FROM replicas r JOIN files f ON f.id = r.file WHERE r.gen != f.gen",
        "replica of non-current generation",
    )?;
    q(
        "SELECT printf('file %d', f.id) FROM files f WHERE f.gen_state = 1 AND EXISTS (SELECT 1 FROM replicas r WHERE r.file = f.id)",
        "owned file with replicas",
    )?;
    q(
        "SELECT printf('file %d', f.id) FROM files f WHERE (f.gen_state = 1) != (f.owner IS NOT NULL)",
        "owner/state mismatch",
    )?;
    q(
        "SELECT printf('file %d', f.id) FROM files f WHERE f.kind != 1 AND (f.gen_state != 0 OR f.owner IS NOT NULL OR f.sealed != 0)",
        "lifecycle state on non-regular file",
    )?;
    q(
        "SELECT printf('%s file %d gen %d', p.key, p.file, p.gen) FROM model_pins p \
         LEFT JOIN files f ON f.id = p.file LEFT JOIN model_artifacts a ON a.key = p.key \
         WHERE f.id IS NULL OR f.kind != 1 OR f.gen != p.gen OR f.gen_state != 0 \
         OR f.sealed != 1 OR f.nlink = 0 OR a.key IS NULL OR a.state NOT IN ('preparing', 'ready', 'unavailable') \
         OR (a.state = 'ready' AND NOT EXISTS (SELECT 1 FROM replicas r WHERE r.file = p.file AND r.gen = p.gen AND r.state = 1))",
        "invalid prepared-artifact pin",
    )?;
    q(
        "SELECT a.key FROM model_artifacts a WHERE \
         (a.state = 'preparing') != (a.owner IS NOT NULL) OR \
         (a.state = 'ready' AND (a.publication IS NULL OR NOT EXISTS (SELECT 1 FROM model_pins p WHERE p.key = a.key)))",
        "invalid prepared-artifact state",
    )?;
    q(
        "SELECT printf('%s lease %s', u.key, u.lease) FROM model_users u LEFT JOIN model_artifacts a \
         ON a.key = u.key WHERE u.released = 0 AND (a.key IS NULL OR a.state NOT IN ('ready', 'unavailable'))",
        "usage lease without a ready artifact",
    )?;
    q(
        "SELECT printf('%s root %d', s.key, s.root) FROM model_snapshots s \
         LEFT JOIN model_artifacts a ON a.key = s.key LEFT JOIN files f ON f.id = s.root \
         WHERE a.key IS NULL OR a.state NOT IN ('preparing', 'ready', 'unavailable') \
         OR f.id IS NULL OR f.kind != 2 OR f.nlink = 0 OR s.root = 1 \
         OR NOT EXISTS(SELECT 1 FROM model_snapshot_files m WHERE m.key = s.key AND m.root = s.root)",
        "invalid prepared snapshot root",
    )?;
    q(
        "SELECT printf('%s root %d file %d', m.key, m.root, m.file) FROM model_snapshot_files m \
         LEFT JOIN model_snapshots s ON s.key = m.key AND s.root = m.root \
         LEFT JOIN dentries d ON d.parent = m.root AND d.name = m.name \
         LEFT JOIN model_pins p ON p.key = m.key AND p.file = m.file AND p.gen = m.gen \
         WHERE s.key IS NULL OR d.child IS NULL OR d.child != m.file OR p.key IS NULL",
        "invalid prepared snapshot manifest",
    )?;
    q(
        "SELECT printf('%s root %d', s.key, s.root) FROM model_snapshots s JOIN dentries d ON d.parent = s.root \
         WHERE NOT EXISTS(SELECT 1 FROM model_snapshot_files m WHERE m.key = s.key AND m.root = s.root AND m.name = d.name AND m.file = d.child)",
        "unbound prepared snapshot entry",
    )?;
    q(
        "SELECT name FROM model_modules WHERE revision <= 0 OR name = '' OR spec = '' OR profile = '' OR key = '' OR qualification_policy = ''",
        "invalid model module declaration",
    )?;
    q(
        "SELECT printf('%s epoch %d', p.key, p.epoch) FROM model_publications p \
         LEFT JOIN model_artifacts a ON a.key = p.key \
         WHERE a.key IS NULL OR p.epoch <= 0 OR p.epoch > a.epoch OR p.root <= 1 \
         OR p.export_file <= 1 OR p.export_gen <= 0 OR p.qualification_file <= 1 OR p.qualification_gen <= 0 \
         OR p.publication = '' OR p.qualification_policy = '' OR p.publication IS NOT a.publication",
        "invalid complete publication evidence",
    )?;
    q(
        "SELECT printf('%s epoch %d', p.key, p.epoch) FROM model_publications p \
         JOIN model_artifacts a ON a.key = p.key AND a.epoch = p.epoch \
         WHERE a.state IN ('ready', 'unavailable') AND ( \
         NOT EXISTS(SELECT 1 FROM model_snapshots s WHERE s.key = p.key AND s.root = p.root AND s.output = 1) \
         OR NOT EXISTS(SELECT 1 FROM model_pins r WHERE r.key = p.key AND r.file = p.export_file AND r.gen = p.export_gen) \
         OR NOT EXISTS(SELECT 1 FROM model_pins r WHERE r.key = p.key AND r.file = p.qualification_file AND r.gen = p.qualification_gen))",
        "complete publication lost its snapshot or receipt pins",
    )?;
    // Historical ownership is retained after cleanup; missing/renamed files
    // are therefore valid. These rows prove an import, not current ownership.
    q(
        "SELECT printf('%s epoch %d file %d', o.key, o.epoch, o.file) FROM model_owned_outputs o \
         LEFT JOIN model_artifacts a ON a.key = o.key \
         WHERE a.key IS NULL OR o.epoch <= 0 OR o.epoch > a.epoch OR o.gen != 1 \
         OR o.parent <= 0 OR o.file <= 1 OR length(o.name) = 0 OR length(o.name) > 255",
        "invalid derived import proof",
    )?;
    // Directory cycles: walk parents from every directory.
    let dirs: Vec<(i64, i64)> = c
        .prepare("SELECT id, parent FROM files WHERE kind = 2")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let parent: std::collections::HashMap<i64, i64> = dirs.iter().copied().collect();
    for (id, _) in &dirs {
        let mut cur = *id;
        let mut steps = 0;
        while cur != 1 {
            cur = match parent.get(&cur) {
                Some(p) => *p,
                None => {
                    problems.push(format!("directory {id} not connected to root"));
                    break;
                }
            };
            steps += 1;
            if steps > dirs.len() {
                problems.push(format!("directory cycle through {id}"));
                break;
            }
        }
    }
    Ok(problems)
}
