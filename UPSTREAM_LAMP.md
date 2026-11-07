# Official LAMP source

- URL: https://github.com/FabriDeCastelli/LAMP
- Pinned commit: `69f6eb53316ef19296fce4fd25e0a57cb128c231`
- Official source snapshot is in `external/LAMP/`; its MIT license and original copyright are preserved.
- The nested Git metadata was removed only from this new isolated clone; the exact commit and per-file SHA-256 are recorded here.
- `upstream_adapter.py` ports `lamp_conv.py`, `simple_conv.py`, the two parametrizations, dissipativity and Euler conventions. It replaces torch_scatter with torch index_add and explicitly sizes all nodes. Lifecycle conditioning is a separate extension in `lifecycle_lamp.py`.
- Reference parity is checked against an independent dense reconstruction. Direct official execution requires torch_scatter, absent in the unchanged SAGIN conda environment.

| File | SHA-256 |
|---|---|
| `src/models/lamp.py` | `46109b1caed79d5ae1b8d8f0ff84e1347f6bbd19d1e73d85945a09dcc01e90a7` |
| `src/layers/convolutions/lamp_conv.py` | `14f9ababde46cb6e59d60bbeec8b916a1fa1805a190deefc7d54bee1292c9830` |
| `src/layers/convolutions/simple_conv.py` | `b46e13423c8d8adc8e825ba79b8839aa543950bf9db8a71d42203c64d3f1cc91` |
| `src/layers/parameterizations/antisymmetric.py` | `f6176b9ddd365474ac2facf6fe4f707a2df5c9a6ce0b09f68da76291006815de` |
| `src/layers/parameterizations/symmetric.py` | `4929bb3656ea2493d30d3fccb612099c711c05ae874b1ddf7152b8a6656f76ad` |
| `src/layers/parameterizations/dissipative_antisymmetric.py` | `0360e47faed803b2b6071ecf7fb34ff1bafd8e7e86e4c8af479fad7cb51bdcd3` |
| `src/layers/attention/dissipative.py` | `34cdcd79977dfa440261d6f9c679f4b0d596b3ad7c3becd1ca279d81f5475e90` |
| `src/layers/discretizations/euler.py` | `e1bf795fa2957b2d05a0c2c6994a992c0797f6c2d29b6bc6c3b14db5649dba1e` |
| `LICENSE` | `d1554b8c728b8636a925a83e01a023ba93b96423558cb7061c7072491c9fe916` |
| `pyproject.toml` | `5b3190521611e873d0c2094cc865e2f7b90719084a4165acad715575626baae5` |
