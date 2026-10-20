#!/usr/bin/env bash
# One line per second: epoch host nic_tx_B nic_rx_B disk_read_B sparknestd_cpu_ticks
dev=$(ls /sys/class/infiniband | grep -E '^(rocep1s0f0|mlx5_0)$' | head -1)
c=/sys/class/infiniband/$dev/ports/1/counters
hw=/sys/class/infiniband/$dev/ports/1/hw_counters
disk=$(lsblk -dno NAME,TYPE | awk '$2=="disk"{print $1}' | grep nvme | head -1)
h=$(hostname)
while :; do
  P=$(systemctl show -p MainPID --value sparknestd@sparknest)
  tx=$(( $(cat $c/port_xmit_data) * 4 )); rx=$(( $(cat $c/port_rcv_data) * 4 ))
  rd=$(awk -v d=$disk '$3==d{print $6*512}' /proc/diskstats)
  cpu=$(awk '{print $14+$15}' /proc/$P/stat 2>/dev/null)
  echo "$(date +%s.%N | cut -c1-14) $h $tx $rx $rd ${cpu:-0}"
  sleep 1
done
