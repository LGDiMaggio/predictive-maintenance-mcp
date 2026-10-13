# Engineer's Guide: After the First Diagnosis

> **Not there yet?** The [README Quick Start](../README.md#quick-start) takes you from a fresh copy of the repository to a first diagnosis in Claude Desktop. This guide covers what comes after: your own measurements, machine manuals, complete workflows, anomaly detection, and the questions that come up along the way. Everything happens in conversation; the server computes, you judge.

---

## What This Guide Covers

- Loading your own vibration files so that every tool, including the ISO severity assessment, can work on them
- Reading equipment manuals to get bearing designations and operating speeds
- Complete workflows: manual, bearing, fault frequencies, envelope analysis, report
- Training an anomaly model on your healthy baselines
- Troubleshooting the answers that look wrong

---

## Use Your Own Data

The server reads a signal file and the companion metadata file next to it. Three things decide whether a diagnosis works on your data: the file layout, the companion keys, and the shaft speed.

### The signal file

- One channel per file: a CSV with a single column of acceleration or velocity samples and **no header row**. A header row is loaded as text and the analysis fails later; a time column in first position is analysed as if it were the signal.
- Formats: CSV, TXT, MAT, WAV, NPY, Parquet, and raw binary (`.bin`, `.raw`, `.dat`) with the decode parameters declared in the companion, see the [Adapter Guide](ADAPTER_GUIDE.md#supported-formats).
- Where: a folder of yours under `data/signals/` (for example `data/signals/pump_p101/`), or an absolute path given in the prompt. Keep your files out of `real_train/` and `real_test/`: those folders hold the labelled samples that the anomaly workflows treat as references.

### The companion file

Name it `<signal name>_metadata.json`, next to the signal (`drive_end.csv` and `drive_end_metadata.json`). Two keys are read as parameters:

```json
{
  "sampling_rate": 25600,
  "signal_unit": "g"
}
```

- `sampling_rate` in Hz. Without it the signal cannot be analysed.
- `signal_unit`: `g` or `m/s2` for acceleration, `mm/s` or `m/s` for velocity. Without a declared unit the ISO 20816-3 severity assessment is refused with a structured reason; the unit is never guessed from the amplitude.
- Any other top-level key is kept for your reference (`BPFO`, `BPFI`, `BSF`, `FTF`, a description) but is not read as a parameter: a key named `sampling_rate_hz` or `unit` is ignored without a warning. The envelope report does read `BPFO`, `BPFI`, `BSF` and `FTF` from the companion to mark the spectrum.
- Both values can also be given in the prompt ("load drive_end.csv at 25600 Hz, unit g"); an explicit value wins over the companion.
- To record which asset and measurement point a file belongs to, and to build a history over time, add the `measurement` object described in the [Adapter Guide](ADAPTER_GUIDE.md#declaring-a-measurement).

### The shaft speed and the bearing

- Every diagnosis needs the shaft speed. State it in the prompt ("the shaft runs at 1480 rpm") or declare it in the `measurement` object; without a source the diagnosis is refused rather than defaulted.
- Fault-frequency matching needs the bearing: a designation present in the catalog (for example `6205`), the four characteristic frequencies in Hz, or the geometry (number of rolling elements, ball and pitch diameters). Give it in the prompt, or ask the server to read it from the machine manual (next section).

### Example

With `data/signals/pump_p101/drive_end.csv` and its companion in place:

```
Load pump_p101/drive_end.csv. The shaft runs at 1480 rpm and the bearing is a 6205:
check the fault frequencies, assess the ISO severity and generate the envelope report.
```

---

## Machine Manuals

Copy PDF or TXT manuals to `resources/machine_manuals/`. A sample is included, `test_pump_manual.pdf`. Then ask:

```
What bearings are specified in test_pump_manual.pdf?
```

The server extracts bearing designations and operating speeds from the manual and can compute the characteristic frequencies from the catalog geometry. Targeted questions ("what is the maximum allowed vibration for this pump?") search the manual text and return the passages they rest on.

---

## Complete Workflows

### Bearing diagnosis from the manual to the report

```
1. Read the specifications from test_pump_manual.pdf
2. Look up the bearing geometry for the bearing it specifies
3. Calculate the characteristic frequencies at the operating speed in the manual
4. Run the envelope analysis on real_train/OuterRaceFault_1.csv looking for those frequencies
5. Generate a report with the findings
```

The assistant chains the tools; every number in the answer comes from the server.

### Severity for a batch of signals

```
Generate ISO 20816-3 severity reports for all the signals in real_train
```

### Compare two spectra

```
Compare the FFT spectrum of real_train/baseline_1.csv with real_train/OuterRaceFault_1.csv.
What differences do you see?
```

---

## Anomaly Detection

Train on healthy references, then score the rest:

```
Train an anomaly detection model using real_train/baseline_1.csv and real_train/baseline_2.csv
as healthy references. Then evaluate all the fault signals in real_train against it.
```

The model learns the normal pattern from the baselines; each scored signal returns the share of segments outside that pattern. With your own data, train on acquisitions you know to be healthy and keep them in a folder of their own.

---

## Troubleshooting

| Symptom | What to do |
|---------|------------|
| The severity assessment is refused | Declare `signal_unit` in the companion file or in the prompt; the unit is never guessed |
| The diagnosis asks for the shaft speed | State the rpm in the prompt, or declare it in the `measurement` object of the companion |
| "Signal file not found" | Ask for the list of available signals; a relative name resolves under `data/signals/`, otherwise give the absolute path |
| The companion values are ignored | Check the file name (`<signal name>_metadata.json`) and the keys (`sampling_rate`, `signal_unit`) |
| The report does not open | Open the HTML file in the `reports/` folder with a browser |
| No signals are listed, or the server is missing | See [INSTALL.md, Troubleshooting](../INSTALL.md#troubleshooting) |

---

## What's Next

- **More workflows**: [EXAMPLES.md](../EXAMPLES.md), step-by-step tutorials with the included data
- **The included dataset**: [data/README.md](../data/README.md)
- **Give feedback**: your real-world experience shapes what gets built next, [tell us what worked and what did not](https://github.com/LGDiMaggio/predictive-maintenance-mcp/discussions)
- **Help validate**: [test the diagnostics with your data](https://github.com/LGDiMaggio/predictive-maintenance-mcp/issues)

---

**You are the expert on your machines. This tool gives you an assistant that speaks your language.**
