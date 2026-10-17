<!-- Written by `accuscan evaluate` / `accuscan baseline`. Do not edit. -->

| Test set | Samples | CER (greedy) | WER (greedy) | CER (dictionary) | WER (dictionary) |
|---|---:|---:|---:|---:|---:|
| GNHK test, words | 8808 | 20.1% | 45.4% | 18.0% | 37.9% |
| GNHK test, lines | 1563 | 21.5% | 55.8% | n/a | n/a |
| Canvas style, held-out fonts | 2000 | 1.4% | 5.8% | 2.5% | 9.4% |

Canvas gate: CER 1.4% against a 15.0% bar, so the demo opens in **free** mode.

Model file 4.81 MB (int8). CPU latency per line, one thread: 3.06 ms at width 128, 5.97 ms at width 256, 11.83 ms at width 512.

| Model | Test set | Samples | CER | WER | Size | ms per crop (CPU) |
|---|---|---:|---:|---:|---:|---:|
| AccuScan CRNN | GNHK test, words | 8808 | 20.1% | 45.4% | 4.81 MB | 5.97 |
| microsoft/trocr-small-handwritten | GNHK test, words | 8822 | 51.4% | 89.0% | 246.4 MB | 80.1 |
| AccuScan CRNN | GNHK test, lines | 1563 | 21.5% | 55.8% | 4.81 MB | 5.97 |
| microsoft/trocr-small-handwritten | GNHK test, lines | 1568 | 38.8% | 73.5% | 246.4 MB | 105.2 |
