"""The companies whose 10-Ks the filings pull reads, and how each lays out Item 1A (#297).

A filer is one company, under every CIK it has filed as: Alphabet took over
Google Inc.'s reports in 2015, and Exxon Mobil's reports moved to a new holding
company in 2025. Of two copies of one report, the copy under the last CIK
listed is preferred (see ``mci_gru.filings.edgar.build_filing_index``).

``layout`` is the ``read_item_1a`` layout that reads the company's risk
headings (see ``mci_gru.filings.risk_factors``). It was chosen by reading every
fiscal 2013 to 2025 filing and checked by ``mci_gru.filings.checks``; a company
added here is read with the default until its filings show otherwise.

The ten companies after Alphabet are those experiment E5 asked for: one or two
per sector, each a 10-K filer through the whole window. The ten after those are
experiment E9's (#304), chosen the same way. American Airlines Group filed as
AMR Corporation until the 2013 merger under the same CIK, so that one CIK covers
every year.

``first_fiscal_year`` leaves out a company's earlier reports when they hold no
risk factors to read, and ``note`` says why.
"""

from __future__ import annotations

from dataclasses import dataclass

from mci_gru.filings.edgar import ALPHABET_CIK, GOOGLE_INC_CIK


@dataclass(frozen=True)
class Filer:
    """One company: the key that names its output files, its name, and its CIKs."""

    key: str
    name: str
    ciks: tuple[str, ...]
    layout: str = "blocks"
    first_fiscal_year: int | None = None
    note: str = ""

    @property
    def preferred_cik(self) -> str:
        return self.ciks[-1]


FILERS: dict[str, Filer] = {
    filer.key: filer
    for filer in (
        Filer(
            "alphabet", "Alphabet Inc. (Google Inc. before 2015)", (GOOGLE_INC_CIK, ALPHABET_CIK)
        ),
        Filer("jpm", "JPMorgan Chase & Co.", ("0000019617",)),
        Filer("dal", "Delta Air Lines, Inc.", ("0000027904",)),
        Filer(
            "xom",
            "Exxon Mobil Corporation (ExxonMobil Holdings Corp. from 2025)",
            ("0000034088", "0002115436"),
            layout="run_in",
        ),
        Filer("pfe", "Pfizer Inc.", ("0000078003",)),
        Filer("wmt", "Walmart Inc.", ("0000104169",)),
        Filer("ba", "The Boeing Company", ("0000012927",)),
        Filer("msft", "Microsoft Corporation", ("0000789019",), layout="sentences"),
        Filer("ko", "The Coca-Cola Company", ("0000021344",)),
        Filer("unh", "UnitedHealth Group Incorporated", ("0000731766",)),
        Filer("cat", "Caterpillar Inc.", ("0000018230",)),
        Filer("aapl", "Apple Inc.", ("0000320193",)),
        Filer("hd", "The Home Depot, Inc.", ("0000354950",)),
        Filer(
            "jnj",
            "Johnson & Johnson",
            ("0000200406",),
            first_fiscal_year=2016,
            note=(
                "Fiscal 2013-2015 reports' Item 1A only points to Exhibit 99, which holds "
                "the risk factors, so those years are left out"
            ),
        ),
        Filer("bac", "Bank of America Corporation", ("0000070858",)),
        Filer("aal", "American Airlines Group Inc. (AMR Corporation before 2013)", ("0000006201",)),
        Filer("cvx", "Chevron Corporation", ("0000093410",), layout="run_in"),
        Filer("pg", "The Procter & Gamble Company", ("0000080424",)),
        Filer("hon", "Honeywell International Inc.", ("0000773840",)),
        Filer("vz", "Verizon Communications Inc.", ("0000732712",)),
        Filer("nee", "NextEra Energy, Inc.", ("0000753308",)),
    )
}
