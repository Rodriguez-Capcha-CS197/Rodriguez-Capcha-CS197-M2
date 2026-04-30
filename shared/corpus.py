"""Synthetic corpus generation. Supports query paraphrasing for Week 6 scaling."""

import numpy as np
from ska_agent.evaluation.officeqa import OfficeQAQuestion
from .scoring import dollars

AGENCIES = [
    "Treasury Operations Office", "Public Debt Service",
    "Revenue Analysis Bureau", "Federal Grants Division",
    "Cash Management Office", "Budget Review Unit",
    "Infrastructure Finance Office", "Economic Stabilization Fund",
    "Intergovernmental Transfers Office", "Audit and Compliance Division",
    "Fiscal Policy Oversight Bureau", "National Revenue Collections Office",
    "Capital Markets Regulation Division", "Federal Disbursement Authority",
    "Strategic Reserve Management Office", "Debt Issuance and Settlement Bureau",
    "Interagency Coordination Office", "Financial Stability Monitoring Unit",
    "Public Investment Accounting Division", "Congressional Appropriations Liaison",
    "Municipal Finance Advisory Board", "Federal Asset Liquidation Office",
    "Risk and Compliance Analytics Bureau", "Sovereign Debt Operations Center",
    "Macroeconomic Policy Analysis Unit",
]

PROGRAMS = [
    "Debt Servicing", "Emergency Grants", "Infrastructure Loans",
    "Tax Refund Processing", "Agency Payroll", "Treasury Securities",
    "Municipal Support", "Disaster Relief", "Technology Modernization",
    "Public Health Transfers", "Capital Investment Funding",
    "Federal Reserve Contributions", "Strategic Infrastructure Bonds",
    "Social Safety Net Disbursements", "Environmental Compliance Grants",
    "Defense Procurement Transfers", "Rural Development Loans",
    "Export Credit Assistance", "Pension Obligation Funding",
    "Education Finance Transfers", "Cybersecurity Infrastructure",
    "Border Security Allocations", "Judicial Operations Funding",
    "Space Program Appropriations", "Veterans Benefits Processing",
]

YEARS = list(range(1999, 2025))

METRICS = [
    "outlays", "obligations", "receipts", "expenditures", "appropriations",
    "disbursements", "allocations", "transfers", "revenues", "liabilities",
    "assets", "deficits", "surpluses", "balances", "collections",
    "payments", "subsidies", "loan guarantees", "interest costs",
    "administrative costs", "operating expenses", "capital expenditures",
    "program costs", "budget authority", "net borrowing",
]

LOOKUP_TEMPLATES = [
    "What was the {metric} for {program} in {year} under {agency}, according to the record?",
    "What were the reported {metric} for {program} under {agency} in {year}?",
    "Find the {metric} for {program} in {agency} during {year}.",
    "Under {agency}, give me the {metric} for {program} during {year}.",
    "According to the record, how much did {agency} report in {metric} for {program} in {year}?",
    "For fiscal year {year}, what {metric} amount was listed for {program} under {agency}?",
    "What value was recorded for {program} {metric} by {agency} in {year}?",
]

MULTI_DOC_TEMPLATES = [
    "Comparing {agency_a} and {agency_b}, which agency had larger {metric} for {program} in {year}?",
    "Between {agency_a} and {agency_b}, who reported higher {metric} for {program} in {year}?",
    "In {year}, comparing {agency_a} and {agency_b}, which agency reported a greater {metric} for {program}?",
    "Using both records, identify which agency had the greater {metric} for {program} in {year}.",
    "Which source shows the larger {metric} amount for {program}: {agency_a} or {agency_b}?",
    "Using both sources, between {agency_a} and {agency_b}, who had a greater {metric} amount for {program} in {year}?",
    "Did {agency_a} or {agency_b} have a greater {metric} in {year} for {program}?",
]

COMPUTE_TEMPLATES = [
    "Give me the amount that {agency}'s {metric} for {program} changed from {year_a} to {year_b}.",
    "What was the increase in {metric} for {program} under {agency} from {year_a} to {year_b}?",
    "Compute the dollar change in {metric} for {program} between {year_a} and {year_b}.",
    "How much did {agency}'s {metric} for {program} change from {year_a} to {year_b}?",
    "Find the difference between the {year_b} and {year_a} {metric} values for {program}.",
    "How much more {metric} did {agency} have in {year_b} compared to {year_a}?",
    "Between {year_a} and {year_b}, how much did {agency}'s {metric} increase?",
]

# MULTI_STEP templates don't use {metric} — priority outlays are always "outlays".
MULTI_STEP_TEMPLATES = [
    "For {agency} in {year}, what were the combined priority outlays for {program_a} and {program_b}?",
    "Using the priority program records, calculate the total outlays for {program_a} and {program_b} under {agency}.",
    "Identify the two priority programs for {agency} in {year} and report their combined outlays.",
    "What total amount did {agency} allocate to its listed priority programs in {year}?",
    "After identifying {agency}'s priority programs in {year}, what was the combined outlay amount for {program_a} and {program_b}?",
    "Using the records for {program_a} and {program_b}, what total priority outlay did {agency} report in {year}?",
    "For {year}, combine the priority outlays for {program_a} and {program_b} under {agency}; what is the total?",
]


def make_officeqa_synthetic_corpus(n_per_mode=25, n_paraphrases=1, seed=0):
    """Generate synthetic OfficeQA corpus.

    Args:
        n_per_mode: number of distinct (agency, program, year, metric) triples per mode.
        n_paraphrases: number of question phrasings per triple. 1 = original behavior.
        seed: random seed for paraphrase template selection.

    Returns:
        segment_texts: list of raw segment strings.
        questions: list of OfficeQAQuestion objects.
        relevant_by_qid: dict qid -> list of gold segment indices.
        triples_by_qid: dict qid -> hashable tuple identifying the underlying data triple.
            Used in Week 6 for held-out splitting at the triple level so paraphrases
            of the same question don't bleed between train and test.
    """
    rng = np.random.default_rng(seed)
    segment_texts = []
    questions = []
    relevant_by_qid = {}
    triples_by_qid = {}

    def add_segment(text):
        idx = len(segment_texts)
        segment_texts.append(text)
        return idx

    def make_paraphrases(templates, **kwargs):
        """Always include the first (canonical) template, then add n-1 random others."""
        chosen = [templates[0]]
        if n_paraphrases > 1:
            extras = rng.choice(
                templates[1:],
                size=min(n_paraphrases - 1, len(templates) - 1),
                replace=False,
            )
            chosen.extend(extras)
        return [t.format(**kwargs) for t in chosen]

    # ------------------------------------------------------------------
    # LOOKUP
    # ------------------------------------------------------------------
    for i in range(n_per_mode):
        agency = AGENCIES[i % len(AGENCIES)]
        program = PROGRAMS[(i * 2) % len(PROGRAMS)]
        year = YEARS[i % len(YEARS)]
        metric = METRICS[i % len(METRICS)]
        outlay = 5000 + 137 * i
        answer = dollars(outlay)

        record_idx = add_segment(
            f"[NODE: lookup_{i}_record] type=row | Year: {year} | Agency: {agency} | "
            f"Program: {program} | {metric.capitalize()}: {answer} | "
            f"The reported {metric} for {program} under {agency} in {year} were {answer}."
        )
        add_segment(
            f"[NODE: lookup_{i}_distractor] type=note | Year: {year - 1} | Agency: {agency} | "
            f"Category: Administrative Overhead | Internal reference only. "
            f"Do not use for budget reporting."
        )

        for p_idx, q_text in enumerate(make_paraphrases(
            LOOKUP_TEMPLATES, agency=agency, program=program, year=year, metric=metric
        )):
            qid = f"lookup_{i}_p{p_idx}"
            questions.append(OfficeQAQuestion(
                question_id=qid, question=q_text, answer=answer,
                source_documents=[f"{agency}_{year}.pdf"],
                question_type="LOOKUP", difficulty="easy",
            ))
            relevant_by_qid[qid] = [record_idx]
            triples_by_qid[qid] = (agency, program, year, metric, "LOOKUP")

    # ------------------------------------------------------------------
    # MULTI_DOC
    # ------------------------------------------------------------------
    for i in range(n_per_mode):
        agency_a = AGENCIES[i % len(AGENCIES)]
        agency_b = AGENCIES[(i + 3) % len(AGENCIES)]
        program = PROGRAMS[(i + 1) % len(PROGRAMS)]
        year = YEARS[i % len(YEARS)]
        metric = METRICS[(i + 1) % len(METRICS)]
        value_a = 7000 + 91 * i
        value_b = 6500 + 83 * i
        answer = agency_a if value_a > value_b else agency_b

        doc_a_idx = add_segment(
            f"[NODE: multidoc_{i}_doc_a] type=row | Source: Bulletin A | Year: {year} | "
            f"Agency: {agency_a} | Program: {program} | "
            f"{metric.capitalize()}: {dollars(value_a)}."
        )
        doc_b_idx = add_segment(
            f"[NODE: multidoc_{i}_doc_b] type=row | Source: Bulletin B | Year: {year} | "
            f"Agency: {agency_b} | Program: {program} | "
            f"{metric.capitalize()}: {dollars(value_b)}."
        )

        for p_idx, q_text in enumerate(make_paraphrases(
            MULTI_DOC_TEMPLATES,
            agency_a=agency_a, agency_b=agency_b,
            program=program, year=year, metric=metric,
        )):
            qid = f"multidoc_{i}_p{p_idx}"
            questions.append(OfficeQAQuestion(
                question_id=qid, question=q_text, answer=answer,
                source_documents=[f"{agency_a}_{year}.pdf", f"{agency_b}_{year}.pdf"],
                question_type="MULTI_DOC", difficulty="medium",
            ))
            relevant_by_qid[qid] = [doc_a_idx, doc_b_idx]
            triples_by_qid[qid] = ((agency_a, agency_b), program, year, metric, "MULTI_DOC")

    # ------------------------------------------------------------------
    # COMPUTE
    # ------------------------------------------------------------------
    for i in range(n_per_mode):
        agency = AGENCIES[(i + 2) % len(AGENCIES)]
        program = PROGRAMS[(i + 4) % len(PROGRAMS)]
        year_a = YEARS[i % len(YEARS)]
        year_b = year_a + 1 if year_a < 2024 else 2024
        metric = METRICS[(i + 2) % len(METRICS)]
        base_receipts = 10000 + 111 * i
        new_receipts = base_receipts + 250 + 13 * (i % 9)
        difference = new_receipts - base_receipts
        answer = dollars(difference)

        year_a_idx = add_segment(
            f"[NODE: compute_{i}_year_a] type=row | Year: {year_a} | Agency: {agency} | "
            f"Program: {program} | {metric.capitalize()}: {dollars(base_receipts)}."
        )
        year_b_idx = add_segment(
            f"[NODE: compute_{i}_year_b] type=row | Year: {year_b} | Agency: {agency} | "
            f"Program: {program} | {metric.capitalize()}: {dollars(new_receipts)}."
        )

        for p_idx, q_text in enumerate(make_paraphrases(
            COMPUTE_TEMPLATES,
            agency=agency, program=program,
            year_a=year_a, year_b=year_b, metric=metric,
        )):
            qid = f"compute_{i}_p{p_idx}"
            questions.append(OfficeQAQuestion(
                question_id=qid, question=q_text, answer=answer,
                source_documents=[f"{agency}_{year_a}.pdf", f"{agency}_{year_b}.pdf"],
                question_type="COMPUTE", difficulty="medium",
            ))
            relevant_by_qid[qid] = [year_a_idx, year_b_idx]
            triples_by_qid[qid] = (agency, program, (year_a, year_b), metric, "COMPUTE")

    # ------------------------------------------------------------------
    # MULTI_STEP — metric is fixed to "outlays" since templates don't vary it
    # ------------------------------------------------------------------
    for i in range(n_per_mode):
        agency = AGENCIES[(i + 5) % len(AGENCIES)]
        program_a = PROGRAMS[i % len(PROGRAMS)]
        program_b = PROGRAMS[(i + 5) % len(PROGRAMS)]
        year = YEARS[i % len(YEARS)]
        amount_a = 8000 + 101 * i
        amount_b = 8500 + 77 * i
        total = amount_a + amount_b
        answer = dollars(total)

        classification_idx = add_segment(
            f"[NODE: multistep_{i}_classification] type=metadata | Year: {year} | "
            f"Agency: {agency} | Priority programs: {program_a}; {program_b}."
        )
        program_a_idx = add_segment(
            f"[NODE: multistep_{i}_program_a] type=row | Year: {year} | Agency: {agency} | "
            f"Program: {program_a} | Priority Outlays: {dollars(amount_a)}."
        )
        program_b_idx = add_segment(
            f"[NODE: multistep_{i}_program_b] type=row | Year: {year} | Agency: {agency} | "
            f"Program: {program_b} | Priority Outlays: {dollars(amount_b)}."
        )

        for p_idx, q_text in enumerate(make_paraphrases(
            MULTI_STEP_TEMPLATES,
            agency=agency, program_a=program_a, program_b=program_b, year=year,
        )):
            qid = f"multistep_{i}_p{p_idx}"
            questions.append(OfficeQAQuestion(
                question_id=qid, question=q_text, answer=answer,
                source_documents=[f"{agency}_{year}_priority.pdf"],
                question_type="MULTI_STEP", difficulty="hard",
            ))
            relevant_by_qid[qid] = [classification_idx, program_a_idx, program_b_idx]
            triples_by_qid[qid] = (agency, (program_a, program_b), year, "outlays", "MULTI_STEP")

    # ------------------------------------------------------------------
    # Distractor pool — unchanged
    # ------------------------------------------------------------------
    for i in range(50):
        agency = AGENCIES[i % len(AGENCIES)]
        program = PROGRAMS[(i + 6) % len(PROGRAMS)]
        year = YEARS[i % len(YEARS)]
        value = 3000 + 211 * i
        add_segment(
            f"[NODE: distractor_{i}] type=note | Year: {year} | Agency: {agency} | "
            f"Program: {program} | Administrative note value: {dollars(value)} | "
            f"This note is background context and is not the final answer for the training questions."
        )

    return segment_texts, questions, relevant_by_qid, triples_by_qid


def build_segments_from_texts(segment_texts, embedder):
    """Embed and wrap raw segment texts into Segment objects."""
    import numpy as np
    from ska_agent.core.structures import Segment

    segment_vectors = embedder.embed(segment_texts)
    segments = []
    for i, (text, vec) in enumerate(zip(segment_texts, segment_vectors)):
        vec = np.asarray(vec, dtype=np.float64)
        seg = Segment(
            text=text, vector=vec, start_idx=i, end_idx=i + 1,
            sentences=[text], internal_cost=0.0,
        )
        segments.append(seg)
    return segments