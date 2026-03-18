"""
data_ingestion.py
-----------------
Downloads Zillow Research datasets and converts them into
LangChain Documents ready for embedding into ChromaDB.

Zillow Public Data URLs (all free, no login required):
  - Median Sale Price (all homes, monthly)
  - Rental Index (ZORI)
  - Days to Pending (market velocity)
  - Inventory (total listings)
"""

import os
import pandas as pd
import requests
from langchain.schema import Document

# ---------------------------------------------------------------------------
# Zillow Research CSV endpoints (public, no API key needed)
# ---------------------------------------------------------------------------
ZILLOW_DATASETS = {
    "median_sale_price": {
        "url": "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        "description": "Zillow Home Value Index – median home value by metro",
    },
    "rental_index": {
        "url": "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfrcondomfr_sm_month.csv",
        "description": "Zillow Observed Rent Index – median asking rent by metro",
    },
    "days_to_pending": {
        "url": "https://files.zillowstatic.com/research/public_csvs/market_temp_index/Metro_market_temp_index_uc_sfrcondo_month.csv",
        "description": "Zillow Market Temperature Index – market heat by metro",
    },
    "inventory": {
        "url": "https://files.zillowstatic.com/research/public_csvs/invt_fs/Metro_invt_fs_uc_sfrcondo_sm_month.csv",
        "description": "Zillow For-Sale Inventory – active listings by metro",
    },
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download_datasets() -> None:
    """Download all Zillow CSV files into the /data directory."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, meta in ZILLOW_DATASETS.items():
        dest = os.path.join(DATA_DIR, f"{name}.csv")
        if os.path.exists(dest):
            print(f"  [skip] {name}.csv already downloaded")
            continue
        print(f"  [download] {name} ...")
        try:
            r = requests.get(meta["url"], timeout=30)
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
            print(f"  [ok] saved to {dest}")
        except Exception as e:
            print(f"  [error] {name}: {e}")


def _melt_zillow_csv(path: str) -> pd.DataFrame:
    """
    Zillow CSVs are wide-format (one column per month).
    Melt them into: RegionName | StateName | date | value
    """
    df = pd.read_csv(path)

    # Identify metadata columns vs date columns
    id_cols = [c for c in df.columns if not c.startswith("19") and not c.startswith("20")]
    date_cols = [c for c in df.columns if c not in id_cols]

    melted = df.melt(id_vars=id_cols, value_vars=date_cols,
                     var_name="date", value_name="value")
    melted.dropna(subset=["value"], inplace=True)
    melted["date"] = pd.to_datetime(melted["date"])
    return melted


def build_documents(top_n_metros: int = 50) -> list[Document]:
    """
    Convert Zillow CSVs into LangChain Documents.

    Each document = one metro + one dataset + recent 12-month summary.
    This keeps chunk size manageable while preserving rich context.

    Args:
        top_n_metros: Number of metros to include (ranked by data completeness).

    Returns:
        List of LangChain Document objects.
    """
    documents = []

    for name, meta in ZILLOW_DATASETS.items():
        path = os.path.join(DATA_DIR, f"{name}.csv")
        if not os.path.exists(path):
            print(f"  [warn] {name}.csv not found – run download_datasets() first")
            continue

        try:
            df = _melt_zillow_csv(path)
        except Exception as e:
            print(f"  [error] parsing {name}: {e}")
            continue

        # Keep only the most recent 12 months
        latest_date = df["date"].max()
        cutoff = latest_date - pd.DateOffset(months=12)
        recent = df[df["date"] >= cutoff].copy()

        # Determine the region column (varies by dataset)
        region_col = "RegionName" if "RegionName" in recent.columns else recent.columns[0]
        state_col  = "StateName"  if "StateName"  in recent.columns else None

        # Select top N metros by row count (proxy for completeness)
        top_metros = (
            recent.groupby(region_col)["value"]
            .count()
            .nlargest(top_n_metros)
            .index.tolist()
        )

        for metro in top_metros:
            metro_df = recent[recent[region_col] == metro].sort_values("date")
            if metro_df.empty:
                continue

            values     = metro_df["value"].dropna()
            state      = metro_df[state_col].iloc[0] if state_col else "N/A"
            avg_val    = values.mean()
            min_val    = values.min()
            max_val    = values.max()
            latest_val = values.iloc[-1]
            pct_change = ((values.iloc[-1] - values.iloc[0]) / values.iloc[0] * 100
                          if len(values) > 1 else 0)

            # Build human-readable text chunk
            unit = "$" if "price" in name or "rent" in name or "index" in name else ""
            text = (
                f"Dataset: {meta['description']}\n"
                f"Metro: {metro}, State: {state}\n"
                f"Period: {metro_df['date'].min().strftime('%b %Y')} "
                f"to {metro_df['date'].max().strftime('%b %Y')}\n"
                f"Latest value: {unit}{latest_val:,.0f}\n"
                f"12-month average: {unit}{avg_val:,.0f}\n"
                f"12-month range: {unit}{min_val:,.0f} – {unit}{max_val:,.0f}\n"
                f"12-month change: {pct_change:+.1f}%\n"
                f"Monthly trend: "
                + ", ".join(
                    f"{row['date'].strftime('%b %Y')}={unit}{row['value']:,.0f}"
                    for _, row in metro_df.iterrows()
                )
            )

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "dataset": name,
                        "metro": metro,
                        "state": state,
                        "latest_date": latest_date.strftime("%Y-%m"),
                        "latest_value": round(latest_val, 2),
                        "pct_change_12m": round(pct_change, 2),
                    },
                )
            )

    print(f"  [ok] built {len(documents)} documents from {len(ZILLOW_DATASETS)} datasets")
    return documents


if __name__ == "__main__":
    print("Downloading Zillow datasets...")
    download_datasets()
    print("\nBuilding LangChain documents...")
    docs = build_documents()
    print(f"\nSample document:\n{docs[0].page_content[:500]}")
