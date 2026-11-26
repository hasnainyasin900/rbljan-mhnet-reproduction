#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""RBLJAN ISCX-VPN packet preprocessing (scaled reproduction).

Reimplements the packet path of the official data_preprocessing.py
(PCAP_DPKT.get_idx_saved) with two changes for feasibility:
  1. Fast readers: dpkt.pcap.Reader for .pcap, dpkt.pcapng.Reader for
     .pcapng, dpkt.ip.IP for RAW (DLT_RAW=101, VPN files) — the official
     code falls back to scapy.rdpcap() for all of these, which loads whole
     files into memory and is far too slow for multi-GB files.
  2. Per-file packet caps: only the first N valid packets per file are
     kept (label cap spread evenly across the label's files), so huge
     bulk-transfer files do not dominate preprocessing time.
Packet masking rules are identical to the official code:
  - drop packets with no payload (legitimate task) or total len <= 50
  - drop FIN / SYN-only TCP packets and excluded ports
    (5353, 137, 138, 5355, 17500, 1900)
  - zero IP src/dst, IP id/offset/checksum, TCP/UDP checksum,
    unknown ports (not in service-names-port-numbers.csv), TCP seq/ack
  - header unified to 50 bytes (truncate or zero-pad), payload kept,
    whole packet truncated to 1500 bytes
Output: one {label}.pkl per label (list of byte-int lists) + counts.json
"""
import binascii
import os
import pickle
import sys
import time

import dpkt
import pandas as pd

EXCLUDE_PORTS = [5353, 137, 138, 5355, 17500, 1900]
PKT_MIN_LEN = 50
PKT_MAX_LEN = 1500
HEADER_LEN = 50

# filename (lowercase) -> label. Applied to the basename of each pcap file.
# Order matters: first matching rule wins.
NONVPN_RULES = [
    (('facebook_audio', 'hangouts_audio', 'skype_audio'), 'audio'),
    (('aimchat', 'aim_chat', 'icqchat', 'icq_chat', 'facebook_chat',
      'facebookchat', 'gmailchat', 'hangout_chat', 'hangouts_chat',
      'skype_chat'), 'chat'),
    (('ftps_down', 'ftps_up', 'scp', 'sftp', 'skype_file'), 'file'),
    (('email',), 'mail'),
    (('facebook_video', 'hangouts_video', 'netflix', 'skype_video',
      'spotify', 'vimeo', 'youtube'), 'streaming'),
    (('voipbuster',), 'voip'),
]
VPN_RULES = [
    (('vpn_facebook_audio', 'vpn_hangouts_audio', 'vpn_skype_audio'), 'vpn-audio'),
    (('vpn_aim_chat', 'vpn_facebook_chat', 'vpn_hangouts_chat',
      'vpn_icq_chat', 'vpn_skype_chat'), 'vpn-chat'),
    (('vpn_ftps', 'vpn_sftp', 'vpn_skype_files'), 'vpn-file'),
    (('vpn_email',), 'vpn-mail'),
    (('vpn_vimeo', 'vpn_youtube', 'vpn_netflix', 'vpn_spotify'), 'vpn-streaming'),
    (('vpn_voipbuster',), 'vpn-voip'),
]
# files that must never be used (not one of the paper's 12 ISCX-VPN types)
EXCLUDE_FILES = {'vpn_bittorrent.pcap'}

LABELS = ['audio', 'chat', 'file', 'mail', 'streaming', 'voip',
          'vpn-audio', 'vpn-chat', 'vpn-file', 'vpn-mail',
          'vpn-streaming', 'vpn-voip']


def label_for(basename, is_vpn):
    name = basename.lower()
    if name in EXCLUDE_FILES:
        return None
    rules = VPN_RULES if is_vpn else NONVPN_RULES
    for prefixes, label in rules:
        if any(name.startswith(p) for p in prefixes):
            return label
    return None


def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def get_known_ports(csv_path):
    df = pd.read_csv(csv_path)
    ports = df['Port Number'].unique()
    ports = sorted(int(p) for p in ports if str(p).isdigit())
    return ports


def scapy_icmp_transform(scapy_ip_pkt, ports):
    """Replicate the official data_preprocessing.py scapy branch for ONE
    ICMP packet (used only for pcapng / RAW files, where the official code
    takes the scapy path and KEEPS ICMP error packets, reading sport/dport
    from the embedded original datagram via scapy's layer traversal).
    Returns the byte-int list, or None if the packet must be skipped."""
    import binascii
    payload = scapy_ip_pkt
    try:
        version = payload.version
        payload_len = len(payload.payload.payload)
        header_len = len(payload) - payload_len
        if (payload_len <= 0 or len(payload) <= PKT_MIN_LEN):
            return None
        sport = payload.payload.sport
        dport = payload.payload.dport
        if sport in EXCLUDE_PORTS or dport in EXCLUDE_PORTS:
            return None
        if hasattr(payload.payload, 'flags') and hasattr(payload.payload.flags, 'value'):
            if payload.payload.flags.value % 2 == 1:
                return None
            if payload.payload.flags.value % 4 == 2:
                return None
        if hasattr(payload, 'chksum'):
            payload.chksum = 0
        if hasattr(payload, 'id'):
            payload.id = 0
        if hasattr(payload, 'offset'):
            payload.offset = 0
        if hasattr(payload.payload, 'chksum'):
            payload.payload.chksum = 0
        if binary_search(ports, sport) == -1:
            payload.payload.sport = 0
        if binary_search(ports, dport) == -1:
            payload.payload.dport = 0
        if hasattr(payload.payload, 'ack'):
            payload.payload.ack = 0
        if hasattr(payload.payload, 'seq'):
            payload.payload.seq = 0
        res = str(binascii.hexlify(bytes(payload)).decode())
        res = res[:2 * PKT_MAX_LEN]
        res = [''.join(x).upper() for x in zip(res[::2], res[1::2])]
        word_to_idx = {(hex(i)[2:].zfill(2)).upper(): i for i in range(256)}
        bytes_idx = [word_to_idx[w] for w in res]
        if version == 4:
            bytes_idx[12:20] = [0] * 8
        elif version == 6:
            bytes_idx[8:40] = [0] * 32
        header = bytes_idx[:header_len][:HEADER_LEN]
        if len(header) < HEADER_LEN:
            header = header + [0] * (HEADER_LEN - len(header))
        bytes_idx = header + bytes_idx[header_len:]
        return bytes_idx[:PKT_MAX_LEN]
    except Exception:
        return None


def process_ip_packet(ip, ports, byte_len=PKT_MAX_LEN, zero_checksums=False):
    """Apply the official masking rules to one dpkt IP packet.
    Returns the byte-int list, or None if the packet must be skipped.
    zero_checksums=True reproduces the official scapy path (pcapng / RAW
    files), where the IP and TCP/UDP checksum bytes stay 0; False matches
    the official dpkt path (.pcap Ethernet files), where dpkt recomputes
    the checksums on serialization."""
    try:
        v = ip.v
        payload_len = len(ip.data.data)
        header_len = len(ip) - payload_len
        if (len(ip) <= PKT_MIN_LEN or payload_len <= 0):
            return None
        sport = ip.data.sport
        dport = ip.data.dport
        if sport in EXCLUDE_PORTS or dport in EXCLUDE_PORTS:
            return None
        if hasattr(ip.data, 'flags'):
            if ip.data.flags % 2 == 1:   # FIN
                return None
            if ip.data.flags % 4 == 2:   # SYN
                return None
        if v == 4:
            ip.src = b'\x00' * 4
            ip.dst = b'\x00' * 4
        elif v == 6:
            ip.src = b'\x00' * 16
            ip.dst = b'\x00' * 16
        if hasattr(ip, 'id'):
            ip.id = 0
        if hasattr(ip, 'offset'):
            ip.offset = 0
        if binary_search(ports, sport) == -1:
            ip.data.sport = 0
        if binary_search(ports, dport) == -1:
            ip.data.dport = 0
        if hasattr(ip, 'sum'):
            ip.sum = 0
        if hasattr(ip.data, 'sum'):
            ip.data.sum = 0
        if isinstance(ip.data, dpkt.tcp.TCP):
            ip.data.seq = 0
            ip.data.ack = 0
        res = str(binascii.hexlify(bytes(ip)).decode())
        if byte_len != -1:
            res = res[:2 * byte_len]
        res = [''.join(x).upper() for x in zip(res[::2], res[1::2])]
        word_to_idx = {(hex(i)[2:].zfill(2)).upper(): i for i in range(256)}
        bytes_idx = [word_to_idx[w] for w in res]
        if zero_checksums:
            # IP header checksum (IPv4 only) + TCP/UDP checksum, all of
            # which sit inside the first 50 bytes (the unified header).
            if v == 4 and len(bytes_idx) >= 12:
                bytes_idx[10:12] = [0, 0]
            ihl = (bytes_idx[0] & 0x0F) * 4 if v == 4 else 40
            proto = bytes_idx[9] if v == 4 else bytes_idx[6]
            if proto == 6 and len(bytes_idx) >= ihl + 18:      # TCP
                bytes_idx[ihl + 16:ihl + 18] = [0, 0]
            elif proto == 17 and len(bytes_idx) >= ihl + 8:    # UDP
                bytes_idx[ihl + 6:ihl + 8] = [0, 0]
        header = bytes_idx[:header_len][:HEADER_LEN]
        if len(header) < HEADER_LEN:
            header = header + [0] * (HEADER_LEN - len(header))
        bytes_idx = header + bytes_idx[header_len:]
        return bytes_idx[:byte_len]
    except Exception:
        return None


def iter_packets(path):
    """Yield raw packet buffers using the fastest available reader.
    Tries dpkt pcap, then dpkt pcapng, then scapy streaming reader."""
    # .pcapng -> dpkt.pcapng.Reader
    if path.lower().endswith('.pcapng'):
        try:
            r = dpkt.pcapng.Reader(open(path, 'rb'))
            dl = r.datalink()
            for _ts, buf in r:
                yield dl, buf
            return
        except Exception:
            pass
    else:
        try:
            r = dpkt.pcap.Reader(open(path, 'rb'))
            dl = r.datalink()
            for _ts, buf in r:
                yield dl, buf
            return
        except Exception:
            pass
    # fallback: scapy streaming reader
    from scapy.all import PcapReader
    for pkt in PcapReader(path):
        # scapy packet -> raw bytes; datalink unknown, mark -1
        yield -1, bytes(pkt)


def process_file(path, ports, per_file_cap):
    """Process one pcap/pcapng file, return (valid_packets, invalid_count).
    Stops after per_file_cap valid packets (streaming, never reads the
    whole file). zero_checksums is True for pcapng / RAW files (official
    scapy path) and False for Ethernet .pcap files (official dpkt path)."""
    X = []
    invalid = 0
    zero_checksums = path.lower().endswith('.pcapng')
    for dl, buf in iter_packets(path):
        if len(X) >= per_file_cap:
            break
        if dl in (101, 12):  # DLT_RAW: official code takes the scapy path
            zero_checksums = True
        try:
            if dl == dpkt.pcap.DLT_EN10MB:  # Ethernet
                eth = dpkt.ethernet.Ethernet(buf)
                ip = eth.data
                if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                    invalid += 1
                    continue
            elif dl == 101 or dl == 12:  # DLT_RAW / DLT_RAW2: raw IP
                ip = dpkt.ip.IP(buf)
            elif dl == -1:  # scapy fallback: buf is raw frame bytes
                zero_checksums = True
                # try Ethernet first, then raw IP
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    ip = eth.data
                    if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                        ip = dpkt.ip.IP(buf)
                except Exception:
                    ip = dpkt.ip.IP(buf)
            else:
                # unknown datalink: try Ethernet, then raw IP
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    ip = eth.data
                    if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                        ip = dpkt.ip.IP(buf)
                except Exception:
                    invalid += 1
                    continue
            # ICMP packets: the official dpkt path (.pcap Ethernet) SKIPS
            # them (no sport), but the official scapy path (pcapng / RAW)
            # KEEPS ICMP error packets, reading sport/dport from the
            # embedded original datagram. Replicate exactly.
            if isinstance(ip.data, (dpkt.icmp.ICMP, dpkt.icmp6.ICMP6)):
                if not zero_checksums:
                    invalid += 1
                    continue
                try:
                    from scapy.all import Ether, IP as ScapyIP
                    if dl == dpkt.pcap.DLT_EN10MB or dl == -1:
                        scapy_ip = Ether(buf).payload
                    else:  # RAW datalink: buf is the IP packet itself
                        scapy_ip = ScapyIP(buf)
                    pkt = scapy_icmp_transform(scapy_ip, ports)
                except Exception:
                    pkt = None
                if pkt is None:
                    invalid += 1
                else:
                    X.append(pkt)
                continue
            pkt = process_ip_packet(ip, ports, zero_checksums=zero_checksums)
            if pkt is None:
                invalid += 1
            else:
                X.append(pkt)
        except Exception:
            invalid += 1
    return X, invalid


def build_label_map(input_root):
    """Walk NonVPN-PCAP/ and VPN-PCAP/ under input_root, return
    {label: [file paths]}."""
    label_files = {l: [] for l in LABELS}
    unknown = []
    for sub, is_vpn in (('NonVPN-PCAP', False), ('VPN-PCAP', True)):
        d = os.path.join(input_root, sub)
        if not os.path.isdir(d):
            print(f'[warn] missing dir: {d}')
            continue
        for fn in sorted(os.listdir(d)):
            if not (fn.lower().endswith('.pcap') or fn.lower().endswith('.pcapng')):
                continue
            label = label_for(fn, is_vpn)
            if label is None:
                unknown.append(os.path.join(sub, fn))
            else:
                label_files[label].append(os.path.join(d, fn))
    return label_files, unknown


def main(input_root, out_dir, ports_csv, label_cap=20000):
    os.makedirs(out_dir, exist_ok=True)
    ports = get_known_ports(ports_csv)
    print(f'known ports: {len(ports)}')
    label_files, unknown = build_label_map(input_root)
    print(f'excluded/unknown files: {unknown}')
    for label, files in label_files.items():
        print(f'label {label}: {len(files)} files')
    counts = {}
    t0 = time.time()
    for label in LABELS:
        files = label_files[label]
        if not files:
            counts[label] = {'valid': 0, 'invalid': 0, 'files': 0}
            continue
        per_file_cap = -(-label_cap // len(files))  # ceil
        X, invalid_total = [], 0
        for f in files:
            x, inv = process_file(f, ports, per_file_cap)
            X.extend(x)
            invalid_total += inv
            print(f'  {label}: {os.path.basename(f)} -> {len(x)} valid, {inv} invalid '
                  f'(cum {len(X)})')
        # enforce the label cap (in case per-file ceil overshoots)
        X = X[:label_cap]
        with open(os.path.join(out_dir, label + '.pkl'), 'wb') as fh:
            pickle.dump(X, fh)
        counts[label] = {'valid': len(X), 'invalid': invalid_total,
                         'files': len(files)}
        print(f'>> label {label} DONE: {len(X)} valid packets, '
              f'{invalid_total} invalid, {time.time()-t0:.1f}s elapsed')
    with open(os.path.join(out_dir, 'counts.json'), 'w') as fh:
        import json
        json.dump({'label_cap': label_cap, 'counts': counts,
                   'total_valid': sum(c['valid'] for c in counts.values()),
                   'elapsed_s': time.time() - t0}, fh, indent=2)
    print(f'ALL DONE in {time.time()-t0:.1f}s, '
          f'total valid: {sum(c["valid"] for c in counts.values())}')


if __name__ == '__main__':
    # usage: prep_rbljan.py <input_root> <out_dir> <ports_csv> [label_cap]
    input_root = sys.argv[1]
    out_dir = sys.argv[2]
    ports_csv = sys.argv[3]
    label_cap = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
    main(input_root, out_dir, ports_csv, label_cap)
