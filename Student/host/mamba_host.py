#!/usr/bin/env python3
"""Host driver for the Mamba2 student core on the Arria 10 GX dev kit (pcie_top).

Talks to BAR0 through sysfs (no kernel driver needed, run as root):
    /sys/bus/pci/devices/<bdf>/resource0     mmap, uncached

BAR0 map: see syn/host_if.sv.  Every register / window access is one aligned
8-byte (or 1-byte) store or load done with ctypes, so the order and size of the
PCIe transactions are known (a Python slice copy would be a memcpy of unknown
access width and order, and the CFG window commits on its byte 63).

Usage
    sudo python3 mamba_host.py --list
    sudo python3 mamba_host.py --bdf 0000:01:00.0 --vec ../syn/vec [--tokens 6]
      loads vec/mem/*.hex, runs the tokens of vec/dump/L00_h_in.hex and checks the
      last layer against vec/dump/L11_h_out.hex (the same files as tb_mamba_core).
"""
import argparse
import ctypes
import glob
import json
import mmap
import os
import sys
import time

VENDOR_ALTERA = 0x1172

R_ID, R_STATUS, R_CTRL, R_PTR, R_CAP, R_CYC = 0x00, 0x08, 0x10, 0x18, 0x20, 0x28
WIN, OUT = 0x40, 0x1000
SEL_LINW, SEL_LINS, SEL_NORMW, SEL_CONVP, SEL_SSMP, SEL_H = range(6)
BAR_SIZE = 0x10000


class Bar:
    def __init__(self, bdf):
        path = f"/sys/bus/pci/devices/{bdf}"
        # make sure memory decoding is on (needed when no driver ever enabled the device)
        try:
            with open(f"{path}/enable", "w") as f:
                f.write("1")
        except OSError:
            pass
        self.fd = os.open(f"{path}/resource0", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, BAR_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        self.base = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))

    def w64(self, off, v):
        ctypes.c_uint64.from_address(self.base + off).value = v & 0xFFFFFFFFFFFFFFFF

    def w8(self, off, v):
        ctypes.c_uint8.from_address(self.base + off).value = v & 0xFF

    def r64(self, off):
        return ctypes.c_uint64.from_address(self.base + off).value

    def r32(self, off):
        return ctypes.c_uint32.from_address(self.base + off).value


class Core:
    def __init__(self, bar):
        self.bar = bar
        ident = bar.r64(R_ID)
        if ident >> 32 != 0x4D414D42:
            sys.exit(f"BAR0 ID {ident:#018x}: not the mamba core (expected 'MAMB')")
        self.version = (ident >> 16) & 0xFFFF
        self.d = ident & 0xFFFF

    def set_ptr(self, sel, addr=0):
        self.bar.w64(R_PTR, (sel << 16) | addr)

    def word(self, value, nbytes):
        """One cfg word (int, little endian bytes) of nbytes significant bytes."""
        nq = (nbytes + 7) // 8
        for q in range(nq):
            self.bar.w64(WIN + 8 * q, value >> (64 * q))
        if nq < 8:
            self.bar.w8(WIN + 63, 0)          # commit (byte 63 unused by narrow memories)
        # nq == 8: the last 64-bit store covered byte 63 and committed

    def load(self, sel, words, nbytes):
        self.set_ptr(sel, 0)
        for v in words:
            self.word(v, nbytes)
        st = self.bar.r64(R_PTR)               # read back: also flushes the posted writes
        if st & 0xFFFF != len(words) & 0xFFFF or (st >> 16) & 7 != sel:
            sys.exit(f"load sel {sel}: pointer {st:#x} after {len(words)} words")

    def run_token(self, h, first, cap_layer=None, timeout=1.0):
        if cap_layer is not None:
            self.bar.w64(R_CAP, cap_layer)
        self.set_ptr(SEL_H, 0)
        for v in h:
            self.word(v & 0xFFFFFFFF, 4)
        self.bar.w64(R_CTRL, 1 | (2 if first else 0))
        t0 = time.perf_counter()
        while not (self.bar.r64(R_STATUS) & 2):
            if time.perf_counter() - t0 > timeout:
                sys.exit(f"token timeout, STATUS {self.bar.r64(R_STATUS):#x}")
        out = []
        for i in range(0, self.d, 2):
            q = self.bar.r64(OUT + 4 * i)
            out += [q & 0xFFFFFFFF, q >> 32]
        return out, self.bar.r64(R_CYC) & 0xFFFFFFFF


def read_hex(path):
    with open(path) as f:
        return [int(l, 16) for l in f if l.strip() and not l.startswith("//")]


def list_devices():
    for p in sorted(glob.glob("/sys/bus/pci/devices/*")):
        with open(f"{p}/vendor") as f:
            if int(f.read(), 16) == VENDOR_ALTERA:
                with open(f"{p}/device") as g:
                    print(os.path.basename(p), f"1172:{int(g.read(), 16):04x}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list Altera PCIe functions")
    ap.add_argument("--bdf", help="PCI address, e.g. 0000:01:00.0")
    ap.add_argument("--vec", default="../syn/vec", help="folder with mem/ and dump/")
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--no-load", action="store_true", help="weights already loaded")
    a = ap.parse_args()
    if a.list or not a.bdf:
        list_devices()
        return

    core = Core(Bar(a.bdf))
    print(f"mamba core v{core.version}, D = {core.d}")
    mem, dump = os.path.join(a.vec, "mem"), os.path.join(a.vec, "dump")
    with open(os.path.join(mem, "layout.json")) as f:
        layout = json.load(f)
    nl = (layout["norm_final"]) // layout["norm_stride"]

    if not a.no_load:
        t0 = time.perf_counter()
        for sel, name, nbytes in ((SEL_LINW, "linw", 64), (SEL_LINS, "lins", 2),
                                  (SEL_NORMW, "normw", 3), (SEL_CONVP, "convp", 9),
                                  (SEL_SSMP, "ssmp", 6)):
            core.load(sel, read_hex(os.path.join(mem, name + ".hex")), nbytes)
        print(f"weights loaded in {time.perf_counter() - t0:.2f} s")

    d = core.d
    h_in = read_hex(os.path.join(dump, "L00_h_in.hex"))
    h_out = read_hex(os.path.join(dump, f"L{nl - 1:02d}_h_out.hex"))
    ntok = min(a.tokens, len(h_in) // d)
    bad = 0
    t0 = time.perf_counter()
    for t in range(ntok):
        out, cyc = core.run_token(h_in[t * d:(t + 1) * d], first=(t == 0), cap_layer=nl - 1)
        exp = h_out[t * d:(t + 1) * d]
        nb = sum(o != e for o, e in zip(out, exp))
        bad += nb
        print(f"token {t}: {cyc} cycles ({cyc / 250:.1f} us at 250 MHz), {nb} mismatches")
    dt = time.perf_counter() - t0
    print(f"{ntok} tokens in {dt * 1e3:.1f} ms on the host side ({ntok / dt:.0f} tok/s incl. I/O)")
    print("PASS" if bad == 0 else f"FAIL: {bad} mismatches")


if __name__ == "__main__":
    main()
