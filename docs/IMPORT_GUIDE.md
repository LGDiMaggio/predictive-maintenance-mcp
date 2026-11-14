# Signal import guide

The signal import reads delimited-text exports (`.csv` and `.txt`) that
`load_signal` cannot read as they are: files with a header, a units row, a
preamble or several channels. Nothing is assumed: every value the import
writes into a signal or the asset ledger needs your confirmation.

## The four steps

1. **Inspect** with `inspect_signal_file(filepath=...)`. It reads a bounded head
   and tail of the file, writes nothing, and reports the regions (preamble,
   header, units row, data, trailer) with their line numbers, the delimiter
   and decimal mark, each column by its physical 0-based index, and the
   values the file states. It also returns a draft configuration and the
   confirmations still missing.
2. **Configure.** Complete the draft: the sampling rate and where it comes
   from, each selected channel's quantity, unit and scaling, and, to record
   the channels in the asset ledger, the asset, measurement point, direction
   and acquisition time. Each value carries a basis that names your act:
   `user_stated`, or `user_accepted_file_value` with the line of the file it
   cites.
3. **Validate** with `validate_signal_import(filepath=..., config=...)`. It reads
   the whole file, writes nothing, and returns one of three outcomes:
   - `ready`: every value has a basis its evidence supports; an
     `import_digest` is returned;
   - `needs_confirmation`: `required_confirmations` lists what you still have
     to state or choose;
   - `invalid`: a contradiction or a problem in the file that no confirmation
     resolves.

   Each value comes with a statement that says what the import will do with
   it. Read the statements before importing.
4. **Import** with `import_signal_file(filepath=..., config=..., import_digest=...)`.
   The file is read and validated again. If anything changed since the
   validation (a byte of the file, a value, a basis), the import is refused
   with "validate again". Otherwise the import record is written, every
   channel is stored as `<file name>_c<index>` (all of them or none), and each
   bound channel is recorded in the asset ledger with its own
   interpretation. The signals are then analyzed by `signal_id` like any
   loaded signal.

## What a confirmation means

A basis records who confirmed a value and on what evidence. Validation
checks that a value accepted from the file equals what the cited line says,
and refuses a stated value that contradicts the file unless the
configuration names an override with a reason.

Bases sent through the MCP tools are recorded as **caller-asserted**: they
record what the calling assistant claimed, never who the caller is. The
assistant is told never to supply a basis you did not give.

## Values the file can supply

A value the file prints can be accepted with `user_accepted_file_value` and the
line that prints it, for these fields only:

- the sampling rate (`time_base`): a rate or a sample interval in the preamble;
- a channel's unit (`channels[<column>].unit`): its token in the units row, for
  a channel without a sensor sensitivity;
- a sensor sensitivity in the preamble (`channels[<column>].scaling`), with its
  unit;
- the shaft speed (`defaults.rpm`) and the acquisition instant
  (`defaults.acquired_at`) in the preamble.

Validation checks that the cited line prints that value. Every other value is
stated by you, or accepted from an earlier import of the same bytes or from the
measurement point's declaration: the dialect, the data lines, the asset,
measurement point, direction and sensor of each channel, the load and the
operating state. A column label, a file name or a folder name never supplies a
value.

## The time base

The sampling rate has one of three sources (`time_base.source`):

- `time_column`: derived from a time column of the file; accept it with
  `user_accepted_derived_value` and the column it cites;
- `stated`: a value you state (`user_stated`);
- `file_declared`: a rate or sample interval the preamble prints; accept it
  with `user_accepted_file_value` and the line that prints it.

Every time column listed in the configuration is checked, also when the rate
comes from another source:

- a step of 1.5 sample intervals or more (missing samples), or a step below
  half a sample interval (a repeated, reordered or restarted stamp), makes the
  validation `invalid` and names the line;
- stamps that leave a uniform grid by more than half a sample interval also
  make it `invalid`;
- stamps that stray from the grid by less, and a rate whose sources disagree
  (the difference is given in ppm), need your decision
  (`needs_confirmation`);
- a column printed with too few decimals to resolve one sample interval proves
  only the duration: the rate then needs a stated or file-declared value.

A time column is left unassessed only by removing it from `time_columns` and
naming an override with a reason; the reason is recorded. Nothing is
resampled, interpolated or filled in: a gap is reported, never repaired.

## What reaches the assistant

The import never returns sample arrays. At most three values per column, the
labels, unit tokens, preamble lines and file names reach the assistant, and
only inside `file_text` fields: escaped, capped, and marked as file text, so
they are read as data, never as instructions.

## Vocabulary

<!-- import-vocabulary:start -->
| Field | Values | Meaning |
|-------|--------|---------|
| `basis` | `user_stated`, `user_accepted_file_value`, `user_accepted_derived_value`, `user_accepted_profile_value`, `user_accepted_prior_value`, `user_accepted_point_value` | Your act behind a value. `user_stated` cites nothing; `user_accepted_file_value` cites a `line`, `user_accepted_derived_value` a time `column`, `user_accepted_prior_value` an `import_id`, `user_accepted_point_value` a `declaration_version`. `user_accepted_profile_value` is reserved: this version has no import profiles and refuses it. |
| `time_base_source` | `time_column`, `stated`, `file_declared` | Where the sampling rate comes from (`time_base.source`). |
| `role` | `vibration`, `time`, `tachometer`, `displacement`, `temperature`, `current`, `voltage`, `index`, `other`, `ignore` | What a column holds. Only `vibration` columns become signals; the others are shown and never imported. |
| `quantity` | `acceleration`, `velocity` | The quantity of a `vibration` channel. |
| `unit` | `g`, `m/s2`, `mm/s`, `m/s` | The stored unit: `g` or `m/s2` for acceleration, `mm/s` or `m/s` for velocity. |
| `source_unit` | `mV`, `V` | The electrical unit of a column that a sensor sensitivity converts. |
| `sensitivity_unit` | `mV/g`, `V/g`, `mV/(m/s2)`, `V/(m/s2)`, `mV/(mm/s)`, `V/(m/s)` | The unit of a sensor sensitivity. |
| `sensitivity_source` | `datasheet_nominal`, `calibration_certificate`, `teds`, `user_estimate` | Where a sensitivity value comes from; it is recorded with the value. |
| `dc_policy` | `keep` | The mean of an electrical channel is kept as recorded. |
| `time_unit` | `s`, `ms`, `us`, `ns` | The unit of a time column (`us` for microseconds). |
| `delimiter` | `,`, `;`, `tab`, `whitespace` | The field delimiter. |
| `decimal` | `.`, `,` | The decimal mark, never the same character as the delimiter. |
| `encoding` | `utf-8`, `utf-8-sig`, `cp1252` | The text encoding; `cp1252` (Windows-1252) only when you choose it. |
| `value_status` | `detected`, `derived`, `proposed`, `ambiguous`, `absent` | How inspection obtained a value; inspection never marks a value as confirmed. |
| `outcome` | `ready`, `needs_confirmation`, `invalid` | The result of a validation. Only `ready` returns an `import_digest`. |
<!-- import-vocabulary:end -->

## Repeating, restoring and correcting

- Importing the same file with the same configuration again returns
  `already_imported` with the same signal ids, and stores again any channel
  the memory cache dropped.
- `get_signal_imports` lists the imports, newest first. After a restart,
  `import_signal_file(import_id=...)` stores the signals of an import again
  without new confirmations, provided the original file still holds the
  recorded bytes at a recorded location.
- A corrected configuration (a sensitivity, a data range) of the same bytes
  supersedes the channel's earlier declaration in the asset ledger; the
  earlier versions stay in its history. Pass `overwrite=True` to replace the
  signals the earlier import stored.

## Where imports are read from and recorded

- Files are read only from the data directory and the folders listed in
  `PMM_IMPORT_ROOTS` (absolute paths separated by `;` on Windows and `:`
  elsewhere; a network share through a mapped drive letter, never a UNC
  path). The original file is never modified, and nothing is written next
  to it.
- Import records are written under `PMM_IMPORT_DIR` (default
  `data/imports`), one append-only file per import. Deleting a record does
  not affect the asset ledger, which replays an imported channel from its
  own declaration and the original file.

## Deleting an import record

The server never deletes an import record. To delete one, stop the server and
remove `records/<import_id>.jsonl` under `PMM_IMPORT_DIR`. After that,
`import_signal_file(import_id=...)` refuses to restore the import and names
its id; the asset ledger keeps the imported measurements and still replays
them from its own declarations and the original file. Measurements are removed
from the asset ledger only by its own rule: deletion is per asset, by removing
the asset's ledger file while the server is stopped.

## Network transports

The signal import tools refuse file access over non-loopback network
transports. Other existing MCP tools have separate access rules and are not
covered by this restriction. Every import tool refuses a call that
arrives over SSE or streamable HTTP unless the server was started from its
command line on a loopback address (such as `127.0.0.1`), the call reaches it
directly (no proxy forwarding header) and names a loopback host. Setting
`PMM_IMPORT_ALLOW_REMOTE` with a network transport bound to any other address
stops the server at startup. Run imports over stdio, or with the server
started on `127.0.0.1` on the machine that holds the file.

## Limits of this version

- Delimited text only (`.csv`, `.txt`), UTF-8 or Windows-1252.
- Acceleration and velocity channels only; other columns are shown but never
  become signals. A channel in mV or V needs its sensor sensitivity.
- A column of raw converter counts has no scaling route in this version: the
  only conversion is a sensor sensitivity from `mV` or `V`. A unit stated for
  such a column is recorded as your statement, and nothing in the file can
  check it.
- Validation reads earlier imports and the asset ledgers within fixed
  bounds (250 import records, 250 ledgers, 8 MiB per file, 32 MiB and
  30,000 ledger events in total). Beyond them a validation is `invalid`, not
  ready, until the history can be read in full.
- A server version that predates the import refuses to replay an imported
  measurement instead of reading the file another way.
