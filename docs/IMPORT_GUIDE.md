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

## What reaches the assistant

The import never returns sample arrays. At most three values per column, the
labels, unit tokens, preamble lines and file names reach the assistant, and
only inside `file_text` fields: escaped, capped, and marked as file text, so
they are read as data, never as instructions.

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
- Validation reads earlier imports and the asset ledgers within fixed
  bounds (250 import records, 250 ledgers, 8 MiB per file, 32 MiB and
  30,000 ledger events in total). Beyond them a validation is `invalid`, not
  ready, until the history can be read in full.
- A server version that predates the import refuses to replay an imported
  measurement instead of reading the file another way.
