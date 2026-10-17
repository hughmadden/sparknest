# Deploying 0.5.0 on the live cluster

The 0.5.0 bottles are installed on every host; the daemons keep 0.4.x until
restarted. Restarting replaces each host's FUSE mount (ADR-059): stop or
restart anything holding files open on `/mnt/sparknest` there, and
containers that bind it, afterwards.

Order: off-grid learners, then the voters, the current leader last. A
stopping 0.5.0 leader hands off to a caught-up 0.5.0 voter, so the last
restart needs no election.

On each host, in turn:

```sh
sparknest-add-lan-standby --dry-run   # fabric hosts: shows the node.toml diff
sparknest-add-lan-standby             # fabric hosts: adds `lan`, checks it
sudo systemctl restart sparknestd@sparknest
nest wait-ready && nest status | head -1
```

`sparknest-add-lan-standby` refuses before the upgrade (0.4.x does not
start with `lan` in node.toml) and does nothing on kakapo or oliver.

Then, once:

```sh
nest net                                  # every peer `fab`, standby up
nest config set preferred-leader raptor   # moves there after 2 min settled
```

Until every host runs 0.5.0, a file owned by a 0.4.x host can lose
concurrent appends from other hosts (benchmarks/M12-REMOTE-WRITES.md):
restart the hosts close together.
