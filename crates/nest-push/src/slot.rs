//! A delivery slot (ADR-063 §5): `payload | fragment table | footer`.
//!
//! ```text
//! [payload bytes …][frag 0][frag 1]…[frag n-1][footer]
//! frag   = dst_off u64 | len u32 | 0u32          (16 bytes, little endian)
//! footer = MAGIC u32 | n_frags u32 | payload_len u64   (last 16 bytes)
//! ```
//!
//! Fragments' payloads are in order: fragment i starts where i−1 ended.
//! Sources pack slots with [`Packer`]; a receiver checks a filled slot with
//! [`parse`] before handing it to the engine, and the engine walks it with
//! [`Fragments`].

pub const MAGIC: u32 = 0x5048_5331; // "PHS1"
pub const FRAG: usize = 16;
pub const FOOTER: usize = 16;

/// Fills one slot: payload from the front, fragment table behind it.
/// Runs that continue the previous fragment in the arena extend it.
/// `B` is the slot's memory: a `Vec`, or a sender's registered staging
/// slot, packed in place (bytes past the payload and table are never
/// read, so it need not start zeroed).
pub struct Packer<B = Vec<u8>> {
    buf: B,
    payload: usize,
    frags: Vec<(u64, u32)>,
}

impl<B: AsRef<[u8]> + AsMut<[u8]>> Packer<B> {
    /// A packer filling `buf` (the slot, `slot_bytes` long).
    pub fn new(buf: B) -> Packer<B> {
        assert!(buf.as_ref().len() > FOOTER + FRAG, "slot too small");
        Packer {
            buf,
            payload: 0,
            frags: Vec::new(),
        }
    }

    /// Bytes of payload room left if the next run starts a new fragment.
    pub fn room(&self) -> usize {
        self.free(self.frags.len() + 1)
    }

    /// Payload room with a table of `frags` entries.
    fn free(&self, frags: usize) -> usize {
        self.buf
            .as_ref()
            .len()
            .saturating_sub(self.payload + frags * FRAG + FOOTER)
    }

    pub fn is_empty(&self) -> bool {
        self.frags.is_empty()
    }

    /// Copy as much of `data` (bound for `dst_off`) as fits; how many bytes
    /// were taken. 0 means the slot is full.
    pub fn push(&mut self, dst_off: u64, data: &[u8]) -> usize {
        let extends = self.frags.last().is_some_and(|&(o, n)| {
            o + n as u64 == dst_off && n as usize + data.len() <= u32::MAX as usize
        });
        // Extending needs no new table entry.
        let room = if extends {
            self.free(self.frags.len())
        } else {
            self.room()
        };
        let take = data.len().min(room);
        if take == 0 {
            return 0;
        }
        self.buf.as_mut()[self.payload..self.payload + take].copy_from_slice(&data[..take]);
        self.payload += take;
        match self.frags.last_mut() {
            Some((_, n)) if extends => *n += take as u32,
            _ => self.frags.push((dst_off, take as u32)),
        }
        take
    }

    /// Payload bytes so far.
    pub fn payload(&self) -> usize {
        self.payload
    }

    /// Write the table and footer at the slot's end; the slot (whole: the
    /// footer sits at its end), its payload bytes and fragment count.
    pub fn finish(mut self) -> (B, usize, usize) {
        let buf = self.buf.as_mut();
        let len = buf.len();
        let table = len - FOOTER - self.frags.len() * FRAG;
        for (i, (o, n)) in self.frags.iter().enumerate() {
            let at = table + i * FRAG;
            buf[at..at + 8].copy_from_slice(&o.to_le_bytes());
            buf[at + 8..at + 12].copy_from_slice(&n.to_le_bytes());
            buf[at + 12..at + 16].fill(0);
        }
        let f = len - FOOTER;
        buf[f..f + 4].copy_from_slice(&MAGIC.to_le_bytes());
        buf[f + 4..f + 8].copy_from_slice(&(self.frags.len() as u32).to_le_bytes());
        buf[f + 8..f + 16].copy_from_slice(&(self.payload as u64).to_le_bytes());
        let (p, n) = (self.payload, self.frags.len());
        (self.buf, p, n)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parsed {
    pub payload: usize,
    pub frags: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SlotError {
    #[error("no footer")]
    Footer,
    #[error("fragment table does not fit")]
    Table,
    #[error("fragment {0} lies outside the payload")]
    Payload(usize),
    #[error("fragment {0} writes past the {1}-byte arena")]
    Arena(usize, u64),
}

/// Check a filled slot: the footer, every fragment within the payload and
/// within `arena` bytes of the rank's arena.
pub fn parse(slot: &[u8], arena: u64) -> Result<Parsed, SlotError> {
    let len = slot.len();
    if len < FOOTER {
        return Err(SlotError::Footer);
    }
    let f = &slot[len - FOOTER..];
    if u32::from_le_bytes(f[0..4].try_into().unwrap()) != MAGIC {
        return Err(SlotError::Footer);
    }
    let n = u32::from_le_bytes(f[4..8].try_into().unwrap()) as usize;
    let payload = u64::from_le_bytes(f[8..16].try_into().unwrap());
    let table = n.checked_mul(FRAG).ok_or(SlotError::Table)?;
    if payload > (len - FOOTER) as u64 || table > len - FOOTER - payload as usize {
        return Err(SlotError::Table);
    }
    let mut at = 0u64;
    for (i, (o, l)) in (Fragments { slot, i: 0, n }).enumerate() {
        at += l as u64;
        if at > payload {
            return Err(SlotError::Payload(i));
        }
        if o.checked_add(l as u64).is_none_or(|end| end > arena) {
            return Err(SlotError::Arena(i, arena));
        }
    }
    if at != payload {
        return Err(SlotError::Payload(n));
    }
    Ok(Parsed {
        payload: payload as usize,
        frags: n,
    })
}

/// A checked slot's fragments: (dst_off, len), payloads consecutive from
/// the slot's start.
pub struct Fragments<'a> {
    slot: &'a [u8],
    i: usize,
    n: usize,
}

impl<'a> Fragments<'a> {
    pub fn new(slot: &'a [u8], p: Parsed) -> Fragments<'a> {
        Fragments {
            slot,
            i: 0,
            n: p.frags,
        }
    }
}

impl Iterator for Fragments<'_> {
    type Item = (u64, u32);

    fn next(&mut self) -> Option<(u64, u32)> {
        if self.i == self.n {
            return None;
        }
        let at = self.slot.len() - FOOTER - (self.n - self.i) * FRAG;
        let e = &self.slot[at..at + FRAG];
        self.i += 1;
        Some((
            u64::from_le_bytes(e[0..8].try_into().unwrap()),
            u32::from_le_bytes(e[8..12].try_into().unwrap()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_merges_and_parses() {
        let mut p = Packer::new(vec![0u8; 256]);
        assert_eq!(p.push(1000, &[1; 40]), 40);
        // Continues the first fragment in the arena: merged.
        assert_eq!(p.push(1040, &[2; 30]), 30);
        assert_eq!(p.push(5000, &[3; 10]), 10);
        let (buf, payload, frags) = p.finish();
        assert_eq!((payload, frags), (80, 2));
        let parsed = parse(&buf, 1 << 20).unwrap();
        assert_eq!(
            parsed,
            Parsed {
                payload: 80,
                frags: 2
            }
        );
        let f: Vec<_> = Fragments::new(&buf, parsed).collect();
        assert_eq!(f, vec![(1000, 70), (5000, 10)]);
        assert_eq!(&buf[..40], &[1; 40]);
        assert_eq!(&buf[70..80], &[3; 10]);
        // Past the arena.
        assert_eq!(parse(&buf, 5009), Err(SlotError::Arena(1, 5009)));
    }

    #[test]
    fn a_full_slot_takes_what_fits() {
        let mut p = Packer::new(vec![0u8; 128]);
        let room = p.room();
        assert_eq!(p.push(0, &vec![9; 1000]), room);
        assert_eq!(p.push(1 << 20, &[1]), 0);
        // Extending the last fragment needs no new table entry, but there
        // is no payload room left either.
        assert_eq!(p.push(room as u64, &[1]), 0);
        let (buf, _, _) = p.finish();
        assert!(parse(&buf, 1 << 30).is_ok());
    }

    /// A reused staging slot holds the last slot's bytes: only payload,
    /// table and footer are read, so the image parses the same.
    #[test]
    fn packing_over_old_bytes_reads_the_same() {
        let mut clean = Packer::new(vec![0u8; 256]);
        let mut dirty = Packer::new(vec![0xAAu8; 256]);
        for p in [&mut clean, &mut dirty] {
            p.push(64, &[5; 20]);
            p.push(4096, &[6; 7]);
        }
        let (a, _, _) = clean.finish();
        let (b, _, _) = dirty.finish();
        let (pa, pb) = (parse(&a, 1 << 20).unwrap(), parse(&b, 1 << 20).unwrap());
        assert_eq!(pa, pb);
        assert_eq!(
            Fragments::new(&a, pa).collect::<Vec<_>>(),
            Fragments::new(&b, pb).collect::<Vec<_>>()
        );
        assert_eq!(a[..27], b[..27]);
    }

    #[test]
    fn garbage_is_refused() {
        assert_eq!(parse(&[0u8; 64], 1 << 30), Err(SlotError::Footer));
        let mut p = Packer::new(vec![0u8; 128]);
        p.push(0, &[1; 8]);
        let (mut buf, _, _) = p.finish();
        let n = buf.len();
        buf[n - 12..n - 8].copy_from_slice(&1000u32.to_le_bytes());
        assert_eq!(parse(&buf, 1 << 30), Err(SlotError::Table));
    }
}
