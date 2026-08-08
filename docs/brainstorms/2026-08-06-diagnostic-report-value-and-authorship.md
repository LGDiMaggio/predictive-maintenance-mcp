---
date: 2026-08-06
topic: diagnostic-report-value-and-authorship
---

# Diagnostic Report: Value and Authorship

## Summary

A single integrated diagnostic report for the bearing-fault path, generated deterministically by the server, that states the finding, the reasoning behind it, and the recommended action — with an annotated envelope spectrum as the explanatory evidence. A conversational reading layer sits on top and cites that document rather than re-authoring it.

---

## Problem Frame

The server currently emits one HTML report per analysis tool (`envelope_analysis_*`, `fft_spectrum_*`, `feature_comparison_*` in `reports/`). The only output that synthesises across tools is DOCX. Report tools return a file path plus a short summary to the caller, so a client LLM never holds enough structured evidence to reason about the case itself.

No reader outside the project author has ever opened one of these reports. They function today as demonstration artefacts, not work artefacts. The intended readers are a maintenance technician deciding whether to act on a machine, and a company or evaluating stakeholder judging whether the diagnosis is credible. Both share two properties: they receive the output **detached from the conversation that produced it**, and they **cannot verify any claim it makes**.

A hand-composed alternative (an LLM rendering a rich report inline in chat) was structurally far better than the current HTML: it placed calculated versus measured bearing frequencies side by side with a match verdict, explicitly reconciled an acceptable ISO zone against a 100% anomaly ratio, and ranked maintenance actions by urgency. The numeric values it displayed were faithful to the server. Its **labels were not**:

- It titled the severity block with the bare, edition-less standard number followed by "Machine Class II". `src/diagnostics/iso20816.py` is the single source of zone truth and deliberately emits `"ISO 20816-3 (thresholds from ISO 10816-3:2009)"` together with a mandatory caveat about the 2022 edition merging zones A and B ([iso20816.py:141](src/diagnostics/iso20816.py:141)). Both the edition and the caveat were dropped. (The repository's own guard test forbids writing that imprecise citation anywhere, including here — which is why this sentence describes it rather than quoting it.)
- "Machine Class II" does not exist in this codebase. The taxonomy is `(machine_group, support_type)` — Group 1/2 × rigid/flexible ([iso20816.py:41](src/diagnostics/iso20816.py:41)). The class vocabulary was imported from a standard the project does not use.
- It displayed a "Confidence: high" chip. The codebase refuses to produce one, in three separate places and on purpose: [recommendations.py:40](src/decision_support/recommendations.py:40) takes no confidence input because it would be repeated verbatim in advisory output; [diagnostics_tools.py:958](src/mcp_tools/diagnostics_tools.py:958) derives no confidence label; [models.py:516](src/models.py:516) warns that the available fit metric is a heuristic and must not be presented as a probability.

The cost shape differs by reader. For an evaluating stakeholder, one label the tool does not actually stand behind discredits the whole artefact — and the author is not in the room to correct it. For a technician, it is a machine stopped, or left running, for a reason nobody can reconstruct afterwards.

---

## Actors

- A1. Maintenance technician: receives the report outside the session, decides whether and when to intervene.
- A2. Evaluating stakeholder (customer, reviewer): judges whether the diagnosis and the tool behind it are trustworthy.
- A3. Analyst: runs the MCP session, produces the report, and answers follow-up questions about it.
- A4. Client LLM: orchestrates the tools, renders and comments on results in the conversation.

---

## Key Flows

- F1. Produce a bearing diagnosis of record
  - **Trigger:** A3 asks the client to diagnose a vibration signal with bearing metadata available.
  - **Actors:** A3, A4, server
  - **Steps:** signal loaded → analyses run across envelope, ISO severity, anomaly detection, characteristic-frequency matching → server composes one integrated document containing every evaluative statement in its own words → document written to disk and its authored content returned to the client.
  - **Outcome:** a self-contained document exists that states finding, reasoning, evidence and recommended action, and can be opened by A1 or A2 with no access to the conversation.
  - **Covered by:** R1, R3, R5, R6, R7, R8, R10, R12, R13, R14

- F2. Read and interrogate the diagnosis
  - **Trigger:** A3 (or A4 on A3's behalf) discusses the result in the session — summarising it, or testing an alternative hypothesis such as misalignment.
  - **Actors:** A3, A4
  - **Steps:** A4 works from the authored statements returned in F1 → summarises, reorders or visualises them → when asked something the document does not answer, says so rather than filling the gap.
  - **Outcome:** the conversation adds interpretation and exploration without introducing any claim absent from the document.
  - **Covered by:** R4, R15

---

## Requirements

**Authorship boundary**

- R1. Every evaluative statement in the report is authored by the server as a complete string: the verdict, the standard's name with its caveat, the zone and its description, the reason a frequency match counts as a match, and each recommended action.
- R2. The server emits no confidence score or confidence label. Where the report needs to convey strength of evidence, it does so by listing the facts that support the finding, not by grading them.
- R3. What the server returns to the client includes the authored statements and the data behind each figure — not only a file path and a short summary as today.
- R4. When the client LLM presents or comments on a result, it reuses the server's statements. It does not coin standard names, machine classes, zone letters, severity words, or confidence levels of its own.

The boundary in one view:

| Element | Authored by server | Rendered / discussed by LLM |
|---|---|---|
| Numeric values, peaks, zone boundaries | Yes | Displays |
| Standard name and its caveat | Yes | Quotes verbatim |
| Fault verdict and its justification | Yes | Quotes, may summarise |
| Contradiction between indicators | Yes | Quotes, may expand |
| Recommended actions and urgency | Yes | Quotes, may reorder for reading |
| Layout, typography, emphasis, chart styling | No | Free |
| Exploration of alternative hypotheses | No | Free, marked as conversation, not as report content |
| Confidence level | Never produced | Never introduced |

**Integrated report content**

- R5. One document covers the whole bearing case: signal overview, ISO severity, anomaly state, characteristic-frequency matching, spectral energy distribution, and recommended actions. It replaces the need to open several per-tool reports to reach a conclusion.
- R6. When indicators disagree — for example an acceptable ISO zone alongside a high anomaly ratio — the document states the disagreement explicitly and explains which indicator governs the recommended action and why.
- R7. Every recommendation carries its motivation and names the evidence it rests on.
- R8. When a healthy baseline signal is available for the same machine, the document compares against it and expresses the key indicators as deltas, not only as absolute values.
- R9. When an input is missing — no baseline, no bearing metadata, signal outside the ISO evaluation scope — the document says what is missing and what conclusion is therefore unavailable, rather than silently omitting the section.

**Explanatory graphics**

- R10. The envelope spectrum is annotated: BPFO, BPFI, BSF and FTF lines overlaid with a visible tolerance band, so that a reader can see the dominant peak falling inside one band and outside the others. The annotation carries the argument without interaction, so the figure makes the same point in a static rendering as in an interactive one.
- R11. A figure earns its place only if it closes a piece of reasoning. If removing it would not weaken any conclusion in the document, it does not belong in the document.

**Document of record**

- R12. The document exists in two renderings produced from the same authored content: a self-contained HTML working view and a PDF deliverable. Both open on their own with no access to the session that produced them, and both carry the same statements — a reader comparing the two finds no claim in one that is absent from the other.
- R13. The same signal analysed with the same parameters produces the same document, apart from generation timestamp. This holds per rendering.
- R14. The document carries its own provenance: source signal, the analysis parameters used, and the server version that produced it.

**Conversational reading layer**

- R15. The reading layer works from the same authored statements as the document and introduces no assertion the document does not contain. When asked something outside them, it says the analysis does not answer it.

---

## Acceptance Examples

- AE1. **Covers R6.** Given a signal whose RMS velocity places it in an acceptable ISO zone while the anomaly detector flags every segment, when the document is generated, it states that the two indicators describe different stages of the same condition and names which one governs the recommended action.
- AE2. **Covers R8, R9.** Given no baseline signal is available for the machine, when the document is generated, the comparison section is present and states that no baseline was available and which conclusion is therefore unavailable — it does not disappear.
- AE3. **Covers R9.** Given bearing metadata is absent, when the document is generated, no characteristic-frequency matching is presented and the document states that the match could not be attempted and why.
- AE4. **Covers R1, R4.** Given the client renders a summary of the result in conversation, when it names the standard, the zone or the recommended action, those appear exactly as the server authored them, caveat included.
- AE5. **Covers R13.** Given the same signal is analysed twice with identical parameters, when both documents are compared, they differ only in generation timestamp.
- AE6. **Covers R2.** Given a textbook-clear outer-race signature, when the document is generated, it presents the supporting facts (match within tolerance, harmonic present, energy concentrated in the high band) and assigns no confidence grade to the finding.

---

## Success Criteria

- A maintenance technician handed only the document, with no access to the conversation, can say what is wrong, why the tool concluded that, and what to do next.
- An external technical reviewer reading the document finds no label, standard name, class, or confidence claim that the codebase does not actually produce.
- The reference case that motivated this work, regenerated under the new model, contains neither a confidence grade nor a standard name the project does not use.
- `ce-plan` can proceed without having to decide who is allowed to assert what — that boundary is settled here.

---

## Scope Boundaries

- The existing per-tool HTML reports and the DOCX generator stay as they are. One case is taken to full depth before anything is generalised.
- Diagnostic paths other than rolling-element bearing faults (gears, imbalance, misalignment, looseness).
- Multi-acquisition trending, historical evolution, and RUL presentation.
- Fleet or multi-machine views.
- Aesthetic redesign of the current templates as an end in itself — the visual weakness is a symptom of missing synthesis, and restyling without the authored reasoning would leave the output equally mute.
- Implementing the conversational reading layer. Its constraints are defined here (R4, R15) so that the document design does not foreclose it, but the document of record is built first.

---

## Key Decisions

- **Deterministic document first, conversational layer second.** Both readers who matter receive the output detached from the session, so the artefact that survives the conversation is the one that must exist. Once the server authors its reasoning in structured form, the conversational layer costs little more.
- **The server authors claims; the LLM renders them.** This is a direct response to an observed failure, not a precaution — the rendered alternative carried faithful numbers under invented labels, and the readers who receive it cannot detect the difference.
- **No confidence label, preserved deliberately.** The codebase already refuses to emit one in three places. Making the payload richer must not become a back door for reintroducing it.
- **Baseline comparison is in scope.** For a technician, a delta against a healthy machine carries more decision value than an absolute figure, and healthy baseline signals with metadata already exist in the dataset.
- **One end-to-end case before generalising.** The open question is whether an integrated, reasoned document changes what a reader does. That is answerable with one path and not more cheaply with several.
- **Two renderings, one authored source.** HTML serves the analyst's working view and PDF serves the technician and the evaluating stakeholder, who attach and archive rather than browse. Splitting the rendering is acceptable precisely because the authored content is shared — the divergence risk that motivated the authorship boundary would return if each rendering composed its own text.

---

## Dependencies / Assumptions

- `src/diagnostics/iso20816.py` is the single source of zone boundaries and already carries the mandatory standard caveat with every result. Verified.
- Healthy baseline signals with accompanying metadata exist in `data/signals/real_train/` (`baseline_1`, `baseline_2`). Verified.
- No reader outside the project author has yet used the generated reports. Stated by the project author; the reports are therefore treated as having no installed-base compatibility constraint.
- Assumption, unverified: the clients in use can render the returned authored content richly. If a target client cannot, the document of record still satisfies both external readers on its own.

---

## Outstanding Questions

### Deferred to Planning

- [Affects R3][Technical] How the authored content reaches the client — an enriched tool return versus an MCP resource — and what that implies for context cost on large cases.
- [Affects R10][Technical] How the annotated spectrum is rendered server-side while keeping the figure readable at report scale in both renderings.
- [Affects R12][Technical] How the PDF rendering is produced, and what dependency that adds. The project pins dependencies deliberately and keeps core processing free of anything beyond NumPy/SciPy, so the addition needs to sit outside that boundary.
- [Affects R12][Technical] Whether both renderings are always emitted or the caller selects one, and how R13's determinism is demonstrated across the pair.
- [Affects R13][Technical] How determinism is achieved and demonstrated given that the document carries a generation timestamp.
- [Affects R8][Needs research] How a signal is associated with the correct baseline — explicit caller argument, metadata field, or convention — and what happens when several candidates exist.
- [Affects R6][Needs research] Which indicator disagreements occur in practice beyond the ISO-versus-anomaly case, so that the authored explanations cover them rather than falling back to generic wording.
