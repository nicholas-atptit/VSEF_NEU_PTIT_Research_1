# VN30 Stock + Index Joint Panel Universe Correction

The latest joint panel protocol used `DGC` and `VPL`.

The active January 2025 VN30 universe should use `BCM` and `BVH` instead.

`VPL` should not block the January 2025 joint panel if it is not in the active universe.

The correct stock universe must contain exactly 30 tickers:

- `ACB`, `BID`, `BCM`, `BVH`, `CTG`, `FPT`, `GAS`, `GVR`, `HDB`, `HPG`, `LPB`, `MBB`, `MSN`, `MWG`, `PLX`, `SAB`, `SHB`, `SSB`, `SSI`, `STB`, `TCB`, `TPB`, `VCB`, `VHM`, `VIB`, `VIC`, `VJC`, `VNM`, `VPB`, `VRE`

Removed from the previous mistaken list:

- `DGC`
- `VPL`

Supported indices remain:

- `VNINDEX`
- `VN30`
- `HNXINDEX`
- `HNX30`
- `UPCOMINDEX`
- `VN100`

The total joint panel remains 36 instruments: 30 stocks and 6 indices.
