# Test fixtures (annotations only, no images)

| Path | Origin | Licence |
|------|--------|---------|
| `sroie/*.json` | Verbatim SROIE key files from github.com/zzzDavid/ICDAR-2019-SROIE @ 27be427 (`data/key/`) | SROIE: ICDAR 2019 research use; mirror repo MIT |
| `wildreceipt/data.txt`, `class_list.txt` | Verbatim WildReceipt annotations shipped in MMOCR's test suite (`tests/data/kie_toy_dataset/wildreceipt/`, open-mmlab/mmocr @ 966296f) | MMOCR Apache-2.0 |
| `cord/gt_parse_*.json` | **Not verbatim.** Written to follow the CORD v2 `gt_parse` structure documented in the CORD README and visible in Donut's `token2json` output; huggingface.co was unreachable when they were written. `tests/test_converters.py::test_cord_real_data` exercises the converter on the real dataset when it is present in `data/raw/cord`. | CC BY 4.0 (structure) |
