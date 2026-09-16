# Frozen checkpoint bundle

This directory contains the submitted checkpoint split into browser-uploadable
parts. Reconstruct it from the repository's `code/` directory:

```bash
python checkpoint_bundle.py restore \
  --bundle-dir ../checkpoint \
  --output ../checkpoint/P0-step3000-cache-on.pt
```

The command verifies every part and the reconstructed file before atomically
making it visible. Expected checkpoint SHA-256:

```text
25e2c6adcdc312726317e6abe06df6f7035bde5675ae70d5eb7f51978db71152
```

The checkpoint is 42,438,297 bytes. Together with `code/cache_model.py` and
`code/model.py`, the uncompressed inference assets total 42,444,568 bytes
(40.478294 MiB), below the 64 MiB limit. See `manifest.json` for part hashes.

A clean reconstruction was validation-scored before the test split was
restored. It obtained 1.5987543293376403 BPB, within 1.02e-7 BPB of the original
recorded 1.598754227700291; this is ordinary CPU floating-point reduction
variation, not a checkpoint-content difference.
