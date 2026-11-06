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
- a unit token on a selected column that does not map to the configured
  unit is a contradiction, refused unless an override names a reason;
- every time column, configured or inspected, is assessed over the whole
  data region, whatever the time base; a gap, reversal or timing defect
  refuses the file, and a time column alone never sets the rate;
- the structure (encoding, dialect, header and units rows, data range)
  needs a basis unless the whole file parses as configured, no other
  reading parses the scanned data, the encoding is unambiguous and the
  header and units rows are the ones the inspection rule finds; a units
  row the configuration leaves out is still evidence for the unit check;
- a value the file prints (a rate or interval, a sensitivity with its
  unit, the shaft speed, the acquisition instant) that the configuration
  states otherwise is a contradiction or a disagreement, never silently
  ignored;
- history this validation needs and cannot read (a damaged import record,
  an unreadable ledger line or index entry) or does not read in full (a
  record store over its scan limit) makes the outcome ``invalid``: no user
  confirmation stands in for it; a record store or ledger that cannot be
  read at all raises.

The package is internal apart from :func:`validate_import`: ``_entry`` runs
a validation; ``_evidence``, ``_time``, ``_checks`` and ``_history`` hold
the checks; ``_results`` builds the statements and per-channel values;
``_common`` holds the run they share.
"""

from ._entry import validate_import

__all__ = ["validate_import"]
