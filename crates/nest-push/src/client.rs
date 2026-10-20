//! A blocking client for the push socket, for engines (ADR-063 §8).
//!
//! ```no_run
//! use nest_push::{client::*, plan::*, proto::*};
//! # let (ranks, files, patterns, entries): (Vec<RankSpec>, Vec<String>, Vec<Pattern>, Vec<Entry>) = Default::default();
//! // Coordinator, on the hub:
//! let g = Coordinator::create(SOCKET, ranks, files, Some(std::time::Duration::from_secs(10)))?;
//! g.patterns(patterns)?;
//! for chunk in entries.chunks(4096) { g.entries(chunk.to_vec())?; }
//! let forecast = g.start()?;
//! while let Some(ev) = g.next()? { /* Progress, then Done or Failed */ }
//! # Ok::<(), std::io::Error>(())
//! ```
//!
//! Each rank opens its own connection to its host's sparknestd with
//! [`Rank::join`], passing the ring's fd, then loops on [`Rank::next`]:
//! copy each `Filled` slot out, then [`Rank::release`] it. Any error, or
//! the socket being absent, means: load through the mount instead.
//!
//! Timeouts: the constructors take one and set it before their first send,
//! so a hung sparknestd fails `create`/`join` with `ErrorKind::TimedOut`.
//! The same timeout is then an idle deadline on every `next()`: no event
//! for that long is `TimedOut` (sparknestd sends `Progress` at least every
//! few seconds while a group runs).

use crate::plan::{Entry, Pattern};
use crate::proto::{self, Event, Forecast, RankSpec, Request};
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};

/// A connected SEQPACKET socket.
pub struct Conn {
    fd: OwnedFd,
}

fn sockaddr(path: &str) -> io::Result<(libc::sockaddr_un, libc::socklen_t)> {
    // SAFETY: sockaddr_un is plain data; zeroed is a valid value.
    let mut a: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    a.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let (bytes, abstract_) = match path.strip_prefix('@') {
        Some(name) => (name.as_bytes(), true),
        None => (path.as_bytes(), false),
    };
    let off = abstract_ as usize;
    if bytes.len() + off >= a.sun_path.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "socket path too long",
        ));
    }
    for (i, b) in bytes.iter().enumerate() {
        a.sun_path[i + off] = *b as libc::c_char;
    }
    let base = std::mem::offset_of!(libc::sockaddr_un, sun_path);
    // An abstract name's length is exact; a path's includes its NUL.
    let len = base + off + bytes.len() + (!abstract_) as usize;
    Ok((a, len as libc::socklen_t))
}

impl Conn {
    /// Connect to `path` (`@name` for an abstract socket). A Unix
    /// socket's connect does not wait: it connects or fails at once
    /// (ECONNREFUSED when nothing listens, ENOENT for a missing path).
    pub fn connect(path: &str) -> io::Result<Conn> {
        let (addr, len) = sockaddr(path)?;
        // SAFETY: plain syscalls; the fd is owned on success.
        unsafe {
            let fd = libc::socket(libc::AF_UNIX, libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC, 0);
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            let fd = OwnedFd::from_raw_fd(fd);
            if libc::connect(
                fd.as_raw_fd(),
                &addr as *const _ as *const libc::sockaddr,
                len,
            ) != 0
            {
                return Err(io::Error::last_os_error());
            }
            Ok(Conn { fd })
        }
    }

    pub fn from_fd(fd: OwnedFd) -> Conn {
        Conn { fd }
    }

    /// Bound every later send and receive: one that waits longer fails
    /// with `ErrorKind::TimedOut`. `None` waits forever (the default).
    pub fn set_timeout(&self, t: Option<std::time::Duration>) -> io::Result<()> {
        let tv = match t {
            Some(d) => libc::timeval {
                tv_sec: d.as_secs() as libc::time_t,
                // A zero timeval means forever; round up to 1 µs.
                tv_usec: (d.subsec_micros().max(u32::from(d.is_zero()))) as libc::suseconds_t,
            },
            None => libc::timeval {
                tv_sec: 0,
                tv_usec: 0,
            },
        };
        for opt in [libc::SO_RCVTIMEO, libc::SO_SNDTIMEO] {
            // SAFETY: plain setsockopt with a local timeval.
            let r = unsafe {
                libc::setsockopt(
                    self.fd.as_raw_fd(),
                    libc::SOL_SOCKET,
                    opt,
                    &tv as *const _ as *const libc::c_void,
                    std::mem::size_of::<libc::timeval>() as libc::socklen_t,
                )
            };
            if r != 0 {
                return Err(io::Error::last_os_error());
            }
        }
        Ok(())
    }

    pub fn fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Send one message, with `pass` as SCM_RIGHTS if given.
    pub fn send(&self, msg: &[u8], pass: Option<RawFd>) -> io::Result<()> {
        let mut iov = libc::iovec {
            iov_base: msg.as_ptr() as *mut _,
            iov_len: msg.len(),
        };
        // Room for one fd.
        let mut cbuf = [0u64; 4];
        // SAFETY: msghdr is plain data; the pointers stay valid for the call.
        unsafe {
            let mut h: libc::msghdr = std::mem::zeroed();
            h.msg_iov = &mut iov;
            h.msg_iovlen = 1;
            if let Some(fd) = pass {
                h.msg_control = cbuf.as_mut_ptr() as *mut _;
                h.msg_controllen = libc::CMSG_SPACE(std::mem::size_of::<RawFd>() as u32) as _;
                let c = libc::CMSG_FIRSTHDR(&h);
                (*c).cmsg_level = libc::SOL_SOCKET;
                (*c).cmsg_type = libc::SCM_RIGHTS;
                (*c).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<RawFd>() as u32) as _;
                std::ptr::write_unaligned(libc::CMSG_DATA(c) as *mut RawFd, fd);
            }
            loop {
                let n = libc::sendmsg(self.fd.as_raw_fd(), &h, libc::MSG_NOSIGNAL);
                if n >= 0 {
                    return Ok(());
                }
                let e = io::Error::last_os_error();
                if e.kind() != io::ErrorKind::Interrupted {
                    return Err(timed_out(e));
                }
            }
        }
    }

    /// Receive one message (and an fd passed with it). `None` when the peer
    /// closed.
    pub fn recv(&self, buf: &mut Vec<u8>) -> io::Result<Option<Option<OwnedFd>>> {
        buf.resize(proto::MAX_MSG, 0);
        let mut iov = libc::iovec {
            iov_base: buf.as_mut_ptr() as *mut _,
            iov_len: buf.len(),
        };
        let mut cbuf = [0u64; 4];
        // SAFETY: as in `send`; a received fd is taken into an OwnedFd once.
        unsafe {
            let mut h: libc::msghdr = std::mem::zeroed();
            h.msg_iov = &mut iov;
            h.msg_iovlen = 1;
            h.msg_control = cbuf.as_mut_ptr() as *mut _;
            h.msg_controllen = std::mem::size_of_val(&cbuf) as _;
            let n = loop {
                let n = libc::recvmsg(self.fd.as_raw_fd(), &mut h, libc::MSG_CMSG_CLOEXEC);
                if n >= 0 {
                    break n as usize;
                }
                let e = io::Error::last_os_error();
                if e.kind() != io::ErrorKind::Interrupted {
                    return Err(timed_out(e));
                }
            };
            if h.msg_flags & libc::MSG_TRUNC != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "message too large",
                ));
            }
            let mut fd = None;
            let mut c = libc::CMSG_FIRSTHDR(&h);
            while !c.is_null() {
                if (*c).cmsg_level == libc::SOL_SOCKET && (*c).cmsg_type == libc::SCM_RIGHTS {
                    let raw = std::ptr::read_unaligned(libc::CMSG_DATA(c) as *const RawFd);
                    fd = Some(OwnedFd::from_raw_fd(raw));
                }
                c = libc::CMSG_NXTHDR(&h, c);
            }
            if n == 0 && fd.is_none() {
                buf.clear();
                return Ok(None);
            }
            buf.truncate(n);
            Ok(Some(fd))
        }
    }

    pub fn request(&self, r: &Request, pass: Option<RawFd>) -> io::Result<()> {
        self.send(&proto::encode(r), pass)
    }

    /// The next event; `None` when sparknestd closed the socket.
    pub fn event(&self) -> io::Result<Option<Event>> {
        let mut buf = Vec::new();
        match self.recv(&mut buf)? {
            None => Ok(None),
            Some(_) => proto::decode(&buf)
                .map(Some)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string())),
        }
    }
}

/// A socket timeout (EAGAIN under SO_RCVTIMEO/SO_SNDTIMEO) as `TimedOut`.
fn timed_out(e: io::Error) -> io::Error {
    if e.kind() == io::ErrorKind::WouldBlock {
        io::Error::new(io::ErrorKind::TimedOut, "push socket timed out")
    } else {
        e
    }
}

fn refused(e: Event) -> io::Error {
    io::Error::other(match e {
        Event::Refused(m) => m,
        Event::Failed { reason, .. } => reason,
        other => format!("unexpected {other:?}"),
    })
}

/// The coordinator's side of a group.
pub struct Coordinator {
    conn: Conn,
    pub group: u64,
    pub token: u64,
}

impl Coordinator {
    /// `timeout` bounds every call on the group's socket from the first
    /// send on (see [`Conn::set_timeout`]); [`Coordinator::set_timeout`]
    /// changes it later.
    pub fn create(
        socket: &str,
        ranks: Vec<RankSpec>,
        files: Vec<String>,
        timeout: Option<std::time::Duration>,
    ) -> io::Result<Coordinator> {
        let conn = Conn::connect(socket)?;
        conn.set_timeout(timeout)?;
        conn.request(
            &Request::Create {
                version: proto::VERSION,
                ranks,
                files,
            },
            None,
        )?;
        match conn.event()? {
            Some(Event::Created { group, token }) => Ok(Coordinator { conn, group, token }),
            Some(e) => Err(refused(e)),
            None => Err(io::Error::other("sparknestd closed the push socket")),
        }
    }

    pub fn set_timeout(&self, t: Option<std::time::Duration>) -> io::Result<()> {
        self.conn.set_timeout(t)
    }

    pub fn patterns(&self, p: Vec<Pattern>) -> io::Result<()> {
        self.conn.request(&Request::Patterns(p), None)
    }

    /// Send entries; keep batches under `proto::MAX_MSG` (a few thousand).
    pub fn entries(&self, e: Vec<Entry>) -> io::Result<()> {
        self.conn.request(&Request::Entries(e), None)
    }

    /// Check the plan; streaming starts once every rank has joined.
    pub fn start(&self) -> io::Result<Forecast> {
        self.conn.request(&Request::Start, None)?;
        match self.conn.event()? {
            Some(Event::Planned(f)) => Ok(f),
            Some(e) => Err(refused(e)),
            None => Err(io::Error::other("sparknestd closed the push socket")),
        }
    }

    /// Progress, then Done or Failed; `None` if the socket closed.
    pub fn next(&self) -> io::Result<Option<Event>> {
        self.conn.event()
    }
}

/// What a rank joins with.
#[derive(Clone, Copy, Debug)]
pub struct JoinArgs {
    pub group: u64,
    pub token: u64,
    pub rank: u32,
    pub ring_fd: RawFd,
    pub slots: u32,
    pub slot_bytes: u32,
}

/// A rank's side of a group.
pub struct Rank {
    conn: Conn,
}

impl Rank {
    /// Join with the ring at `ring_fd` (`slots × slot_bytes`, mappable
    /// MAP_SHARED).
    /// Join `j.group` as `j.rank` with the ring at `j.ring_fd`
    /// (`slots × slot_bytes`, mappable MAP_SHARED). `timeout` bounds every
    /// call from the first send on (see [`Conn::set_timeout`]).
    pub fn join(
        socket: &str,
        j: JoinArgs,
        timeout: Option<std::time::Duration>,
    ) -> io::Result<Rank> {
        let conn = Conn::connect(socket)?;
        conn.set_timeout(timeout)?;
        conn.request(
            &Request::Join {
                version: proto::VERSION,
                group: j.group,
                token: j.token,
                rank: j.rank,
                slots: j.slots,
                slot_bytes: j.slot_bytes,
            },
            Some(j.ring_fd),
        )?;
        match conn.event()? {
            Some(Event::Joined) => Ok(Rank { conn }),
            Some(e) => Err(refused(e)),
            None => Err(io::Error::other("sparknestd closed the push socket")),
        }
    }

    /// Filled slots, then Done or Failed; `None` if the socket closed.
    pub fn next(&self) -> io::Result<Option<Event>> {
        self.conn.event()
    }

    pub fn set_timeout(&self, t: Option<std::time::Duration>) -> io::Result<()> {
        self.conn.set_timeout(t)
    }

    pub fn release(&self, slot: u32) -> io::Result<()> {
        self.conn.request(&Request::Release { slot }, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_and_an_fd_cross_a_socketpair() {
        let mut fds = [0; 2];
        // SAFETY: plain syscall into a local array.
        assert_eq!(
            unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_SEQPACKET, 0, fds.as_mut_ptr()) },
            0
        );
        // SAFETY: fresh fds owned here.
        let (a, b) = unsafe {
            (
                Conn::from_fd(OwnedFd::from_raw_fd(fds[0])),
                Conn::from_fd(OwnedFd::from_raw_fd(fds[1])),
            )
        };
        let f = tempfile_fd();
        a.request(
            &Request::Join {
                version: 1,
                group: 7,
                token: 9,
                rank: 3,
                slots: 2,
                slot_bytes: 4096,
            },
            Some(f.as_raw_fd()),
        )
        .unwrap();
        let mut buf = Vec::new();
        let got = b.recv(&mut buf).unwrap().unwrap();
        assert!(got.is_some(), "fd passed");
        let r: Request = proto::decode(&buf).unwrap();
        assert!(matches!(r, Request::Join { rank: 3, .. }));
        drop(a);
        assert!(b.recv(&mut buf).unwrap().is_none(), "peer closed");
    }

    #[test]
    fn a_silent_peer_times_out() {
        let mut fds = [0; 2];
        // SAFETY: plain syscall into a local array.
        assert_eq!(
            unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_SEQPACKET, 0, fds.as_mut_ptr()) },
            0
        );
        // SAFETY: fresh fds owned here.
        let (a, _b) = unsafe {
            (
                Conn::from_fd(OwnedFd::from_raw_fd(fds[0])),
                Conn::from_fd(OwnedFd::from_raw_fd(fds[1])),
            )
        };
        a.set_timeout(Some(std::time::Duration::from_millis(50)))
            .unwrap();
        let t = std::time::Instant::now();
        let e = a.event().unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::TimedOut);
        assert!(t.elapsed() < std::time::Duration::from_secs(2));
    }

    fn tempfile_fd() -> OwnedFd {
        let name = std::ffi::CString::new("nest-push-test").unwrap();
        // SAFETY: plain syscall; the fd is owned.
        unsafe { OwnedFd::from_raw_fd(libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC)) }
    }

    #[test]
    fn abstract_addresses_have_no_nul() {
        let (_, len) = sockaddr("@sparknest-push").unwrap();
        let base = std::mem::offset_of!(libc::sockaddr_un, sun_path);
        assert_eq!(len as usize, base + 1 + "sparknest-push".len());
    }
}
