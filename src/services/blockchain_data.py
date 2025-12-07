"""Blockchain Data Service.

Provides global blockchain data (transactions, blocks) for supported chains.

Currently supported:
- Solana (SOL) via Solana RPC API (free, no API key required)

Future support (requires finding free APIs):
- Bitcoin (BTC)
- Ethereum (ETH)
- Ripple (XRP)
- Zcash (ZEC)

Note: Blockchair was removed due to aggressive rate limiting on free tier.
"""

import os
import aiohttp
from typing import Any


class BlockchainDataService:
    """Service for fetching blockchain data from multiple chains.
    
    Currently uses Solana public RPC. Other chains require finding
    alternative free APIs (Blockchair was removed due to rate limits).
    """
    
    # Solana RPC endpoints
    SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"
    
    # Chain name mappings (aliases to canonical names)
    CHAIN_ALIASES = {
        "btc": "bitcoin",
        "bitcoin": "bitcoin",
        "eth": "ethereum",
        "ethereum": "ethereum",
        "xrp": "ripple",
        "ripple": "ripple",
        "zec": "zcash",
        "zcash": "zcash",
        "sol": "solana",
        "solana": "solana",
    }
    
    # Chains with working free APIs
    SUPPORTED_CHAINS = {"solana"}
    
    # Chains that need alternative APIs (placeholder for future)
    UNSUPPORTED_CHAINS = {"bitcoin", "ethereum", "ripple", "zcash"}
    
    def __init__(self):
        """Initialize the blockchain data service."""
        pass
    
    def get_supported_chains(self) -> list[str]:
        """Get list of supported blockchain chains.
        
        Returns:
            List of supported chain names
        """
        return list(self.SUPPORTED_CHAINS)
    
    def get_all_known_chains(self) -> list[str]:
        """Get all known chains (including unsupported ones).
        
        Returns:
            List of all chain names
        """
        return list(self.SUPPORTED_CHAINS | self.UNSUPPORTED_CHAINS)
    
    def _normalize_chain(self, chain: str) -> str:
        """Normalize chain name to canonical form.
        
        Args:
            chain: Chain name or alias (e.g., "btc", "bitcoin", "BTC")
            
        Returns:
            Canonical chain name (e.g., "bitcoin")
        """
        return self.CHAIN_ALIASES.get(chain.lower(), chain.lower())
    
    async def _make_solana_request(self, method: str, params: list[Any]) -> dict[str, Any]:
        """Make JSON-RPC request to Solana RPC.
        
        Args:
            method: RPC method name
            params: RPC parameters
            
        Returns:
            JSON response as dict
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.SOLANA_RPC_URL,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                return await response.json()
    
    async def get_recent_transactions(
        self, 
        chain: str, 
        limit: int = 10
    ) -> dict[str, Any]:
        """Get recent transactions for a blockchain.
        
        Args:
            chain: Blockchain name or alias (currently only sol/solana supported)
            limit: Number of transactions to return
            
        Returns:
            Dict with status, transactions list, and chain info
        """
        normalized_chain = self._normalize_chain(chain)
        
        if normalized_chain in self.UNSUPPORTED_CHAINS:
            return {
                "status": "not_supported",
                "chain": chain,
                "message": (
                    f"Blockchain data for {chain.upper()} is not currently available. "
                    f"Only Solana (SOL) is supported via free API at this time. "
                    f"Alternative free APIs for {chain.upper()} are being researched."
                )
            }
        
        if normalized_chain not in self.SUPPORTED_CHAINS:
            return {
                "status": "error",
                "message": f"Unknown chain: {chain}. Supported: {', '.join(self.get_supported_chains())}"
            }
        
        try:
            if normalized_chain == "solana":
                return await self._get_solana_recent_transactions(limit)
            else:
                return {
                    "status": "error",
                    "message": f"No API configured for {chain}"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error fetching transactions: {str(e)}"
            }
    
    async def _get_solana_recent_transactions(self, limit: int) -> dict[str, Any]:
        """Fetch recent Solana transactions via RPC.
        
        For Solana, we get the latest block and its transactions.
        
        Args:
            limit: Number of transactions
            
        Returns:
            Formatted transaction response
        """
        # Get latest slot
        slot_response = await self._make_solana_request("getSlot", [])
        if "error" in slot_response:
            return {
                "status": "error",
                "message": f"Solana RPC error: {slot_response.get('error')}"
            }
        
        latest_slot = slot_response.get("result")
        
        # Get block with transactions
        block_response = await self._make_solana_request("getBlock", [
            latest_slot,
            {
                "encoding": "json",
                "transactionDetails": "signatures",
                "maxSupportedTransactionVersion": 0
            }
        ])
        
        if "error" in block_response:
            return {
                "status": "error",
                "message": f"Solana RPC error: {block_response.get('error')}"
            }
        
        block_data = block_response.get("result", {})
        signatures = block_data.get("signatures", [])[:limit]
        
        # Format transactions (signatures in Solana are transaction identifiers)
        formatted_txs = []
        for sig in signatures:
            formatted_txs.append({
                "signature": sig,
                "slot": latest_slot,
                "block_time": block_data.get("blockTime"),
            })
        
        return {
            "status": "success",
            "chain": "solana",
            "slot": latest_slot,
            "count": len(formatted_txs),
            "transactions": formatted_txs
        }
    
    async def get_latest_block(self, chain: str) -> dict[str, Any]:
        """Get latest block info for a blockchain.
        
        Args:
            chain: Blockchain name or alias
            
        Returns:
            Dict with block info
        """
        normalized_chain = self._normalize_chain(chain)
        
        if normalized_chain in self.UNSUPPORTED_CHAINS:
            return {
                "status": "not_supported",
                "chain": chain,
                "message": (
                    f"Block data for {chain.upper()} is not currently available. "
                    f"Only Solana (SOL) is supported via free API at this time."
                )
            }
        
        if normalized_chain not in self.SUPPORTED_CHAINS:
            return {
                "status": "error",
                "message": f"Unknown chain: {chain}"
            }
        
        try:
            if normalized_chain == "solana":
                return await self._get_solana_latest_block()
            else:
                return {
                    "status": "error",
                    "message": f"No API configured for {chain}"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error fetching block: {str(e)}"
            }
    
    async def _get_solana_latest_block(self) -> dict[str, Any]:
        """Get latest Solana slot/block info."""
        slot_response = await self._make_solana_request("getSlot", [])
        
        if "error" in slot_response:
            return {
                "status": "error",
                "message": f"Solana RPC error: {slot_response.get('error')}"
            }
        
        latest_slot = slot_response.get("result")
        
        # Get epoch info
        epoch_response = await self._make_solana_request("getEpochInfo", [])
        epoch_info = epoch_response.get("result", {})
        
        return {
            "status": "success",
            "chain": "solana",
            "block": {
                "slot": latest_slot,
                "epoch": epoch_info.get("epoch"),
                "slot_index": epoch_info.get("slotIndex"),
                "slots_in_epoch": epoch_info.get("slotsInEpoch"),
            }
        }
    
    async def get_blockchain_stats(self, chain: str) -> dict[str, Any]:
        """Get general statistics for a blockchain.
        
        Args:
            chain: Blockchain name or alias
            
        Returns:
            Dict with blockchain statistics
        """
        normalized_chain = self._normalize_chain(chain)
        
        if normalized_chain in self.UNSUPPORTED_CHAINS:
            return {
                "status": "not_supported",
                "chain": chain,
                "message": (
                    f"Stats for {chain.upper()} are not currently available. "
                    f"Only Solana (SOL) is supported via free API at this time."
                )
            }
        
        if normalized_chain not in self.SUPPORTED_CHAINS:
            return {
                "status": "error",
                "message": f"Unknown chain: {chain}"
            }
        
        try:
            if normalized_chain == "solana":
                return await self._get_solana_stats()
            else:
                return {
                    "status": "error",
                    "message": f"No API configured for {chain}"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error fetching stats: {str(e)}"
            }
    
    async def _get_solana_stats(self) -> dict[str, Any]:
        """Get Solana network stats."""
        # Get epoch info
        epoch_response = await self._make_solana_request("getEpochInfo", [])
        epoch_info = epoch_response.get("result", {})
        
        # Get supply info
        supply_response = await self._make_solana_request("getSupply", [])
        supply_info = supply_response.get("result", {}).get("value", {})
        
        return {
            "status": "success",
            "chain": "solana",
            "stats": {
                "epoch": epoch_info.get("epoch"),
                "slot_height": epoch_info.get("absoluteSlot"),
                "block_height": epoch_info.get("blockHeight"),
                "total_supply_sol": supply_info.get("total", 0) / 1e9,
                "circulating_supply_sol": supply_info.get("circulating", 0) / 1e9,
            }
        }
