"""
Validation of a signal import configuration against the whole file.

ISO 13374 Block 1 (data acquisition). :func:`validate_import` reads every
byte of the file through the same strict decoder an import uses (digest
mode: no sample is kept), assesses every time column, checks each value's
basis against the evidence it cites, applies the unit, scaling and binding
rules, runs a read-only ledger pre-check and compares with earlier
interpretations of the same bytes. It returns ``ready``,
``needs_confirmation`` or ``invalid`` with one statement per engineering
value, and the ``import_digest`` an import must present when, and only
when, the outcome is ``ready``.

Validation writes nothing: not next to the file, not in the signal store,
the asset ledger or the import record store.

Rules that keep a missing or contradicted fact from passing as a confirmed
one:

- a value the import writes needs a basis naming the user's act; a basis
  that accepts a file, derived, earlier-import or point-declaration value
  must cite that evidence, and the configured value must equal what it
  holds;
- a unit token on a selected column, or on a time column, that does not
  map to the configured unit is a contradiction, refused unless the
  override of that field names a reason;
- every time column, configured or inspected, is assessed over the whole
  data region, whatever the time base; a gap, reversal or timing defect
  refuses the file, and a time column alone never sets the rate;
- the structure (encoding, dialect, header and units rows, data range)
  needs a basis unless the whole file parses as configured, no other
  reading parses the scanned data, the encoding is unambiguous and the
  header and units rows are the ones the inspection rule finds; a header
  or units row the configuration leaves out or names elsewhere is still
  evidence, and every other line above the data is read for printed
  values;
- every value the file prints (a rate or interval, a sensitivity with its
  unit, a unit, the shaft speed, the acquisition instant) is compared with
  the configuration, not only the line a basis cites: a contradiction is
  refused unless the override of that field names a reason; a printed
  value that cannot be read, cannot be compared without assuming a time
  zone, or that the file does not tie to one channel needs the user, and
  only that field's override records the reason;
- history this validation needs and cannot read (a damaged import record,
  an unreadable ledger record, a recorded measurement of these bytes that
  cannot be read in full or disagrees with its own id, a move that reaches
  no holder) or does not read in full (a count, byte or event bound
  reached) makes the outcome ``invalid``: no user confirmation stands in
  for it; a record store or ledger directory that cannot be read at all
  raises.

The package is internal apart from :func:`validate_import` and, for the
import service, :func:`validate_for_import`: ``_entry`` runs
a validation; ``_evidence``, ``_time``, ``_checks``, ``_printed`` and
``_history`` hold the checks; ``_results`` builds the statements and
per-channel values; ``_common`` holds the run they share.
"""

from ._entry import Validated, validate_for_import, validate_import

__all__ = ["Validated", "validate_for_import", "validate_import"]
