"""
Stock universe definitions for MCI-GRU.

This module provides definitions for different stock universes
including S&P 500, Russell 1000, and MSCI World.
"""

from typing import Dict, Any, List, Optional


# Universe definitions with Refinitiv RICs
UNIVERSES: Dict[str, Dict[str, Any]] = {
    "sp500": {
        "chain_ric": "0#.SPX",
        "name": "S&P 500",
        "description": "500 largest US companies by market cap",
        "calendar": "NYSE",
        "expected_stocks": 500,
        "currency": "USD",
        "country": "US",
    },
    "russell1000": {
        "chain_ric": "0#.RUI",
        "name": "Russell 1000",
        "description": "1000 largest US companies by market cap",
        "calendar": "NYSE",
        "expected_stocks": 1000,
        "currency": "USD",
        "country": "US",
    },
    "russell2000": {
        "chain_ric": "0#.RUT",
        "name": "Russell 2000",
        "description": "2000 small-cap US companies",
        "calendar": "NYSE",
        "expected_stocks": 2000,
        "currency": "USD",
        "country": "US",
    },
    "msci_world": {
        "chain_ric": "0#.MIWO00000PUS",
        "name": "MSCI World",
        "description": "Large and mid-cap stocks across 23 developed markets",
        "calendar": "MULTI",
        "expected_stocks": 1500,
        "currency": "USD",  # USD denominated index
        "country": "MULTI",
    },
    "msci_acwi": {
        "chain_ric": "0#.MIACWI00PUS",
        "name": "MSCI ACWI",
        "description": "All Country World Index - developed and emerging markets",
        "calendar": "MULTI",
        "expected_stocks": 2900,
        "currency": "USD",
        "country": "MULTI",
    },
    "nasdaq100": {
        "chain_ric": "0#.NDX",
        "name": "NASDAQ 100",
        "description": "100 largest non-financial NASDAQ companies",
        "calendar": "NASDAQ",
        "expected_stocks": 100,
        "currency": "USD",
        "country": "US",
    },
}


def get_universe_info(universe: str) -> Dict[str, Any]:
    """
    Get information about a stock universe.
    
    Args:
        universe: Universe name (e.g., 'sp500', 'russell1000')
        
    Returns:
        Dictionary with universe information
        
    Raises:
        ValueError: If universe not found
    """
    if universe not in UNIVERSES:
        available = list(UNIVERSES.keys())
        raise ValueError(f"Unknown universe '{universe}'. Available: {available}")
    return UNIVERSES[universe]


def get_chain_ric(universe: str) -> str:
    """
    Get the chain RIC for a universe.
    
    Args:
        universe: Universe name
        
    Returns:
        Chain RIC string for Refinitiv API
    """
    return get_universe_info(universe)["chain_ric"]


def list_universes() -> List[str]:
    """
    List all available universe names.
    
    Returns:
        List of universe names
    """
    return list(UNIVERSES.keys())


def is_multi_country(universe: str) -> bool:
    """
    Check if a universe spans multiple countries.
    
    Args:
        universe: Universe name
        
    Returns:
        True if multi-country universe
    """
    return get_universe_info(universe).get("country") == "MULTI"
