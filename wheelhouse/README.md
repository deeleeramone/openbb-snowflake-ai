# Wheelhouse

This directory contains pre-built wheel packages for all supported platforms.

## Installation

To install a wheel for your platform, use:

```bash
pip install wheelhouse/<wheel-file-for-your-platform>.whl
```

## Platform-specific Requirements

- **Linux AMD64**: See `requirements-linux-amd64.txt`
- **Linux ARM64**: See `requirements-linux-aarch64.txt`
- **macOS x86_64**: See `requirements-macos-x86_64.txt`
- **macOS ARM64**: See `requirements-macos-arm64.txt`
- **Windows AMD64**: See `requirements-windows-amd64.txt`
- **Windows ARM64**: See `requirements-windows-arm64.txt`

## Verification

All files include SHA256 checksums. To verify:

```bash
sha256sum -c wheelhouse/<file>.sha256
```

Or check against the master list in `CHECKSUMS.sha256`.

## Build Information

- Build Date: $(date -u)
- Workflow Run: 
- Commit: 
