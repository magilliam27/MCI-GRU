"""
LSEG/Refinitiv data loader for MCI-GRU.

This module provides functionality to fetch stock data from Refinitiv/LSEG
via the refinitiv-data library. Requires Refinitiv Workspace to be running.
"""

import time
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime
from tqdm import tqdm

from mci_gru.data.universes import UNIVERSES, get_universe_info, get_chain_ric


class LSEGLoader:
    """
    LSEG/Refinitiv data loader.
    
    Connects to Refinitiv Workspace app to fetch historical price data,
    index constituents, and market indicators.
    
    Note: Requires Refinitiv Workspace desktop app to be running.
    """
    
    def __init__(self, session_type: str = "desktop"):
        """
        Initialize LSEG loader.
        
        Args:
            session_type: Type of Refinitiv session ('desktop' or 'platform')
        """
        self.session_type = session_type
        self.rd = None
        self.connected = False
        
    def connect(self) -> bool:
        """
        Connect to Refinitiv Workspace.
        
        Returns:
            True if connection successful
            
        Raises:
            ImportError: If refinitiv-data not installed
            RuntimeError: If connection fails
        """
        try:
            import refinitiv.data as rd
            self.rd = rd
        except ImportError:
            raise ImportError(
                "refinitiv-data package not installed. "
                "Install with: pip install refinitiv-data"
            )
        
        try:
            self.rd.open_session()
            self.connected = True
            print("Connected to Refinitiv Workspace")
            return True
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Refinitiv Workspace. "
                f"Make sure the app is running. Error: {e}"
            )
    
    def disconnect(self):
        """Disconnect from Refinitiv."""
        if self.connected and self.rd is not None:
            try:
                self.rd.close_session()
            except:
                pass
            self.connected = False
            print("Disconnected from Refinitiv")
    
    def _ensure_connected(self):
        """Ensure we have an active connection."""
        if not self.connected:
            raise RuntimeError("Not connected to Refinitiv. Call connect() first.")
    
    def get_universe_constituents(self, universe: str) -> List[str]:
        """
        Get constituent RICs for a stock universe.
        
        Args:
            universe: Universe name (e.g., 'sp500', 'russell1000')
            
        Returns:
            List of constituent RIC codes
        """
        self._ensure_connected()
        
        chain_ric = get_chain_ric(universe)
        info = get_universe_info(universe)
        
        print(f"Fetching constituents for {info['name']} ({chain_ric})...")
        
        try:
            # Fetch constituent list
            constituents = self.rd.get_data(
                universe=[chain_ric],
                fields=['TR.RIC', 'TR.CommonName']
            )
            
            if constituents is None or len(constituents) == 0:
                raise ValueError(f"No constituents found for {universe}")
            
            rics = constituents['Instrument'].tolist()
            print(f"  Found {len(rics)} constituents")
            
            return rics
            
        except Exception as e:
            print(f"  Error fetching constituents: {e}")
            raise
    
    def get_historical_prices(
        self,
        rics: List[str],
        start: str,
        end: str,
        fields: List[str] = None,
        batch_size: int = 100,
        delay_between_batches: float = 1.0,
        convert_ric_to_ticker: bool = False
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for multiple stocks.
        
        Args:
            rics: List of RIC codes
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            fields: Fields to fetch (default: None, uses API default OHLCV)
            batch_size: Number of RICs per request (for rate limiting)
            delay_between_batches: Seconds to wait between batches
            convert_ric_to_ticker: If True, convert RICs like 'AAPL.O' to 
                                   simple tickers like 'AAPL'
            
        Returns:
            DataFrame with columns [kdcode, dt, open, high, low, close, volume]
        """
        self._ensure_connected()
        
        # Note: For daily historical pricing, do NOT specify fields.
        # The API returns OHLCV by default. Specifying field names like
        # 'OPEN', 'HIGH', etc. causes errors because these aren't valid
        # field codes for the historical pricing endpoint.
        
        print(f"Fetching historical prices for {len(rics)} stocks...")
        print(f"  Date range: {start} to {end}")
        print(f"  Batch size: {batch_size}")
        
        all_data = []
        
        for i in tqdm(range(0, len(rics), batch_size), desc="Fetching data"):
            batch_rics = rics[i:i + batch_size]
            
            try:
                # Don't pass 'fields' for interday data - API returns default OHLCV
                df = self.rd.get_history(
                    universe=batch_rics,
                    start=start,
                    end=end,
                    interval='1D'
                )
                
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    
            except Exception as e:
                print(f"  Warning: Error fetching batch {i//batch_size + 1}: {e}")
            
            # Rate limiting
            if i + batch_size < len(rics):
                time.sleep(delay_between_batches)
        
        if not all_data:
            raise ValueError("No data fetched")
        
        # Combine all batches
        combined = pd.concat(all_data)
        
        # Reshape to standard format
        df = self._reshape_to_standard_format(combined)
        
        # Optionally convert RICs to simple ticker symbols
        if convert_ric_to_ticker and 'kdcode' in df.columns:
            df['kdcode'] = df['kdcode'].apply(self._ric_to_ticker)
        
        print(f"  Fetched {len(df)} rows for {df['kdcode'].nunique()} stocks")
        
        return df
    
    def _reshape_to_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reshape Refinitiv data to standard format.
        
        Refinitiv get_history() returns data in MultiIndex format:
        - Index: Date (datetime)
        - Columns: MultiIndex with (Instrument, Field)
        
        This normalizes to: [kdcode, dt, open, high, low, close, volume]
        
        Based on official LSEG example:
        https://github.com/LSEG-API-Samples/Example.DataLibrary.Python/blob/lseg-data-examples/Examples/1-Access/EX-1.01.02-GetHistory.ipynb
        """
        # Column name mapping from LSEG field names to standard OHLCV
        # Based on official LSEG API documentation and examples
        column_mapping = {
            # Official LSEG daily summary field names (from EX-1.01.02-GetHistory.ipynb)
            'MKT_OPEN': 'open',
            'MKT_HIGH': 'high',
            'MKT_LOW': 'low',
            'TRDPRC_1': 'close',
            'ACVOL_UNS': 'volume',
            # Alternative field names that may appear
            'OPEN_PRC': 'open',
            'HIGH_1': 'high',
            'LOW_1': 'low',
            'HST_CLOSE': 'close',
            # Simple variants
            'OPEN': 'open',
            'HIGH': 'high',
            'LOW': 'low',
            'CLOSE': 'close',
            'VOLUME': 'volume',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        }
        
        records = []
        
        # Check if we have MultiIndex columns (typical for multi-instrument requests)
        if isinstance(df.columns, pd.MultiIndex):
            # Get unique instruments from the first level of columns
            instruments = df.columns.get_level_values(0).unique().tolist()
            
            for instrument in instruments:
                # Skip if this is not actually an instrument column
                if instrument in ['Date', 'Instrument', 'index']:
                    continue
                
                try:
                    # Extract data for this instrument
                    instrument_data = df[instrument].copy()
                    
                    # Rename columns using mapping
                    instrument_data = instrument_data.rename(columns=column_mapping)
                    
                    # Add instrument identifier
                    instrument_data['kdcode'] = instrument
                    
                    # Add date from index
                    instrument_data['dt'] = df.index
                    
                    records.append(instrument_data)
                except KeyError:
                    continue
            
            if records:
                df = pd.concat(records, ignore_index=True)
            else:
                raise ValueError("No valid instrument data found in DataFrame")
        else:
            # Single-level columns (single instrument or already flat)
            df = df.reset_index()
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Handle instrument/date columns
            if 'Instrument' in df.columns:
                df = df.rename(columns={'Instrument': 'kdcode'})
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'dt'})
            elif 'index' in df.columns:
                df = df.rename(columns={'index': 'dt'})
        
        # Convert date to string format
        if 'dt' in df.columns:
            df['dt'] = pd.to_datetime(df['dt']).dt.strftime('%Y-%m-%d')
        
        # Select only the columns we need
        required_cols = ['kdcode', 'dt', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [c for c in required_cols if c in df.columns]
        df = df[available_cols]
        
        # Drop rows where ALL OHLCV values are NaN (empty rows)
        ohlcv_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
        if ohlcv_cols:
            df = df.dropna(subset=ohlcv_cols, how='all')
        
        # Drop rows where close price is NaN (essential for trading data)
        if 'close' in df.columns:
            df = df.dropna(subset=['close'])
        
        # Remove duplicates - keep first valid row per (stock, date)
        if 'kdcode' in df.columns and 'dt' in df.columns:
            df = df.drop_duplicates(subset=['kdcode', 'dt'], keep='first')
        
        # Sort by stock and date
        if 'kdcode' in df.columns and 'dt' in df.columns:
            df = df.sort_values(['kdcode', 'dt']).reset_index(drop=True)
        
        return df
    
    def _ric_to_ticker(self, ric: str) -> str:
        """
        Convert RIC (Reuters Instrument Code) to simple ticker symbol.
        
        Examples:
            'AAPL.O' -> 'AAPL'  (NASDAQ)
            'IBM.N' -> 'IBM'    (NYSE)
            'MSFT.O' -> 'MSFT'  (NASDAQ)
        
        Args:
            ric: RIC code like 'AAPL.O'
            
        Returns:
            Ticker symbol like 'AAPL'
        """
        if '.' in ric:
            return ric.rsplit('.', 1)[0]
        return ric
    
    def _is_permission_error(self, error: Exception) -> bool:
        """Check if an exception is a permission/entitlement error."""
        error_str = str(error).lower()
        permission_indicators = [
            'usernotpermission',
            'no permission',
            'not permitted',
            'not entitled',
            'entitlement',
            'access denied',
            'unauthorized'
        ]
        return any(indicator in error_str for indicator in permission_indicators)
    
    def get_vix(self, start: str, end: str) -> Optional[pd.DataFrame]:
        """
        Fetch VIX index data.
        
        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns [dt, vix], or None if data unavailable
        """
        self._ensure_connected()
        
        print(f"Fetching VIX data from {start} to {end}...")
        
        # Try multiple VIX RICs - different ones may be available
        # depending on user's entitlements
        vix_rics = ['.VIX', 'VIX.N', '^VIX', 'CBOE/VIX']
        
        for ric in vix_rics:
            try:
                df = self.rd.get_history(
                    universe=[ric],
                    start=start,
                    end=end,
                    interval='1D'
                )
                
                if df is None or len(df) == 0:
                    continue  # Try next RIC
                
                # Reshape
                df = df.reset_index()
                df = df.rename(columns={'Date': 'dt', 'CLOSE': 'vix', 'Close': 'vix'})
                df['dt'] = pd.to_datetime(df['dt']).dt.strftime('%Y-%m-%d')
                
                # Keep only needed columns
                if 'vix' in df.columns:
                    df = df[['dt', 'vix']]
                else:
                    # Handle single column case - take first non-date column as vix
                    cols = [c for c in df.columns if c != 'dt']
                    if cols:
                        df = df.rename(columns={cols[0]: 'vix'})
                        df = df[['dt', 'vix']]
                    else:
                        continue  # Try next RIC
                
                print(f"  Fetched {len(df)} VIX observations (using {ric})")
                return df
                
            except Exception as e:
                if self._is_permission_error(e):
                    print(f"  No permission for {ric}, trying alternatives...")
                    continue
                # For other errors, log and try next RIC
                print(f"  Error with {ric}: {e}")
                continue
        
        # All RICs failed
        print("  Warning: VIX data unavailable (no permission or data). Skipping.")
        return None
    
    def get_treasury_yields(self, start: str, end: str) -> Optional[pd.DataFrame]:
        """
        Fetch 10Y and 2Y Treasury yield data.
        
        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns [dt, yield_10y, yield_2y, yield_curve],
            or None if data unavailable
        """
        self._ensure_connected()
        
        print(f"Fetching Treasury yields from {start} to {end}...")
        
        # Try multiple RIC combinations for Treasury yields
        treasury_rics_options = [
            ['US10YT=RR', 'US2YT=RR'],
            ['US10YT=X', 'US2YT=X'],
            ['US10Y=RR', 'US2Y=RR'],
        ]
        
        for rics in treasury_rics_options:
            try:
                df = self.rd.get_history(
                    universe=rics,
                    start=start,
                    end=end,
                    interval='1D'
                )
                
                if df is None or len(df) == 0:
                    continue  # Try next RIC set
                
                # Reshape - handle multi-column format
                df = df.reset_index()
                
                # Rename based on structure
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                  for col in df.columns]
                
                # Standardize
                df = df.rename(columns={'Date': 'dt'})
                df['dt'] = pd.to_datetime(df['dt']).dt.strftime('%Y-%m-%d')
                
                # Compute yield curve (10Y - 2Y)
                if 'yield_10y' in df.columns and 'yield_2y' in df.columns:
                    df['yield_curve'] = df['yield_10y'] - df['yield_2y']
                
                print(f"  Fetched {len(df)} Treasury observations (using {rics})")
                return df
                
            except Exception as e:
                if self._is_permission_error(e):
                    print(f"  No permission for {rics}, trying alternatives...")
                    continue
                # For other errors, log and try next RIC set
                print(f"  Error with {rics}: {e}")
                continue
        
        # All RIC options failed
        print("  Warning: Treasury yield data unavailable (no permission or data). Skipping.")
        return None

    def get_series(self, ric: str, start: str, end: str, value_name: str) -> Optional[pd.DataFrame]:
        """
        Fetch a single LSEG time series and normalize to [dt, value_name].

        Uses historical close as the proxy value. Returns None if unavailable.
        """
        self._ensure_connected()
        try:
            raw = self.rd.get_history(
                universe=[ric],
                start=start,
                end=end,
                interval='1D'
            )
            if raw is None or len(raw) == 0:
                return None
            shaped = self._reshape_to_standard_format(raw)
            if "close" not in shaped.columns:
                return None
            series = shaped[["dt", "close"]].copy()
            series = series.rename(columns={"close": value_name})
            series[value_name] = pd.to_numeric(series[value_name], errors="coerce")
            series = series.dropna(subset=[value_name])
            if len(series) == 0:
                return None
            return series.sort_values("dt").drop_duplicates(subset=["dt"], keep="last")
        except Exception as e:
            if self._is_permission_error(e):
                print(f"  No permission for {ric}, skipping.")
                return None
            print(f"  Error fetching series {ric}: {e}")
            return None
    
    def fetch_universe_data(
        self,
        universe: str,
        start: str,
        end: str,
        include_vix: bool = False,
        cache_path: Optional[str] = None,
        convert_ric_to_ticker: bool = False
    ) -> pd.DataFrame:
        """
        Fetch complete dataset for a stock universe.
        
        Convenience method that fetches constituents and their historical data.
        
        Args:
            universe: Universe name
            start: Start date
            end: End date
            include_vix: Whether to merge VIX data
            cache_path: Optional path to save CSV (for caching)
            convert_ric_to_ticker: If True, convert RICs like 'AAPL.O' to 
                                   simple tickers like 'AAPL'
            
        Returns:
            DataFrame with all stock data
        """
        self._ensure_connected()
        
        print(f"Fetching complete data for {universe}...")
        
        # Get constituents
        rics = self.get_universe_constituents(universe)
        
        # Fetch historical data
        df = self.get_historical_prices(
            rics, start, end, 
            convert_ric_to_ticker=convert_ric_to_ticker
        )
        
        # Merge VIX if requested
        if include_vix:
            vix_df = self.get_vix(start, end)
            if vix_df is not None and len(vix_df) > 0:
                df = df.merge(vix_df, on='dt', how='left')
                # Use ffill() instead of deprecated fillna(method='ffill')
                df['vix'] = df['vix'].ffill().fillna(20)
            else:
                # VIX data unavailable - use default value
                print("  Using default VIX value (20) since data unavailable")
                df['vix'] = 20.0
        
        # Cache if path provided
        if cache_path:
            df.to_csv(cache_path, index=False)
            print(f"  Cached data to {cache_path}")
        
        return df


# Convenience function
def load_from_lseg(
    universe: str,
    start: str,
    end: str,
    include_vix: bool = False,
    convert_ric_to_ticker: bool = False
) -> pd.DataFrame:
    """
    Convenience function to load data from LSEG.
    
    Args:
        universe: Universe name
        start: Start date
        end: End date
        include_vix: Whether to include VIX
        convert_ric_to_ticker: If True, convert RICs like 'AAPL.O' to 
                               simple tickers like 'AAPL'
        
    Returns:
        DataFrame with stock data
    """
    loader = LSEGLoader()
    try:
        loader.connect()
        df = loader.fetch_universe_data(
            universe, start, end, include_vix,
            convert_ric_to_ticker=convert_ric_to_ticker
        )
        return df
    finally:
        loader.disconnect()
