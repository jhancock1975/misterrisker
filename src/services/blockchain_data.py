"""Blockchain Data Service.

Provides global blockchain data (transactions, blocks) for multiple chains:
- Bitcoin (BTC) via Blockchair API
- Ethereum (ETH) via Blockchair API
- Ripple (XRP) via Blockchair API
- Zcash (ZEC) via Blockchair API
- Solana (SOL) via Solana RPC API
"""

import os
import aiohttp
from typing import Any


class BlockchainDataService:
    """Service for fetching blockchain data from multiple chains.
    
    Uses Blockchair API for BTC, ETH, XRP, ZEC and Solana RPC for SOL.
    """
    
    # Blockchair base URL
    BLOCKCHAIR_BASE_URL = "https://api.blockchair.com"
    
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
    
    # Chains supported by Blockchair
    BLOCKCHAIR_CHAINS = {"bitcoin", "ethereum", "ripple", "zcash"}
    
    # Chains with native RPC
    SOLANA_CHAIN = "solana"
    
    def __init__(self, blockchair_api_key: str | None = None):
        """Initialize the blockchain data service.
        
        Args:
            blockchair_api_key: Optional Blockchair API key for higher rate limits
        """
        self.blockchair_api_key = blockchair_api_key or os.getenv("BLOCKCHAIR_API_KEY")
    
    def get_supported_chains(self) -> list[str]:
        """Get list of supported blockchain chains.
        
        Returns:
            List of supported chain names
        """
        return ["bitcoin", "ethereum", "ripple", "zcash", "solana"]
    
    def _normalize_chain(self, chain: str) -> str:
        """Normalize chain name to canonical form.
        
        Args:
            chain: Chain name or alias (e.g., "btc", "bitcoin", "BTC")
            
        Returns:
            Canonical chain name (e.g., "bitcoin")
        """
        return self.CHAIN_ALIASES.get(chain.lower(), chain.lower())
    
    async def _make_request(self, url: str) -> dict[str, Any]:
        """Make HTTP request to Blockchair API.
        
        Args:
            url: Full URL to request
            
        Returns:
            JSON response as dict
        """
        # Add API key if available
        if self.blockchair_api_key:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}key={self.blockchair_api_key}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()
    
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
            chain: Blockchain name or alias (btc, eth, xrp, sol, zec)
            limit: Number of transactions to return (max 100)
            
        Returns:
            Dict with status, transactions list, and chain info
        """
        normalized_chain = self._normalize_chain(chain)
        
        if normalized_chain not in self.get_supported_chains():
            return {
                "status": "error",
                "message": f"Unsupported chain: {chain}. Supported chains: {', '.join(self.get_supported_chains())}"
            }
        
        try:
            if normalized_chain == self.SOLANA_CHAIN:
                return await self._get_solana_recent_transactions(limit)
            else:
                return await self._get_blockchair_recent_transactions(normalized_chain, limit)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error fetching transactions: {str(e)}"
            }
    
    async def _get_blockchair_recent_transactions(
        self, 
        chain: str, 
        limit: int
    ) -> dict[str, Any]:
        """Fetch recent transactions from Blockchair API.
        
        Args:
            chain: Normalized chain name
            limit: Number of transactions
            
        Returns:
            Formatted transaction response
        """
        # Limit to max 100 per Blockchair API
        limit = min(limit, 100)
        
        url = f"{self.BLOCKCHAIR_BASE_URL}/{chain}/transactions?limit={limit}&s=time(desc)"
        response = await self._make_request(url)
        
        if response.get("context", {}).get("code") != 200:
            return {
                "status": "error",
                "message": f"Blockchair API error: {response.get('context', {}).get('error', 'Unknown error')}"
            }
        
        transactions = response.get("data", [])
        
        # Format transactions for display
        formatted_txs = []
        for tx in transactions:
            formatted_tx = {
                "hash": tx.get("hash"),
                "block": tx.get("block_id"),
                "time": tx.get("time"),
            }
            
            # Add chain-specific fields
            if chain == "bitcoin":
                formatted_tx["input_total_btc"] = tx.get("input_total", 0) / 100000000
                formatted_tx["output_total_btc"] = tx.get("output_total", 0) / 100000000
                formatted_tx["fee_btc"] = tx.get("fee", 0) / 100000000
            elif chain == "ethereum":
                formatted_tx["value_eth"] = tx.get("value", 0) / 1e18
                formatted_tx["gas_used"] = tx.get("gas_used")
            elif chain == "ripple":
                formatted_tx["ledger_index"] = tx.get("ledger_index")
            elif chain == "zcash":
                formatted_tx["input_total_zec"] = tx.get("input_total", 0) / 100000000
                formatted_tx["output_total_zec"] = tx.get("output_total", 0) / 100000000
            
            formatted_txs.append(formatted_tx)
        
        return {
            "status": "success",
            "chain": chain,
            "count": len(formatted_txs),
            "transactions": formatted_txs
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
        
        if normalized_chain not in self.get_supported_chains():
            return {
                "status": "error",
                "message": f"Unsupported chain: {chain}"
            }
        
        try:
            if normalized_chain == self.SOLANA_CHAIN:
                return await self._get_solana_latest_block()
            else:
                return await self._get_blockchair_latest_block(normalized_chain)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error fetching block: {str(e)}"
            }
    
    async def _get_blockchair_latest_block(self, chain: str) -> dict[str, Any]:
        """Get latest block info from Blockchair stats endpoint."""
        url = f"{self.BLOCKCHAIR_BASE_URL}/{chain}/stats"
        response = await self._make_request(url)
        
        if response.get("context", {}).get("code") != 200:
            return {
                "status": "error",
                "message": f"Blockchair API error"
            }
        
        data = response.get("data", {})
        
        return {
            "status": "success",
            "chain": chain,
            "block": {
                "height": data.get("best_block_height") or data.get("blocks"),
                "hash": data.get("best_block_hash"),
                "time": data.get("best_block_time"),
                "total_transactions": data.get("transactions"),
            }
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
        
        if normalized_chain not in self.get_supported_chains():
            return {
                "status": "error",
                "message": f"Unsupported chain: {chain}"
            }
        
        try:
            if normalized_chain == self.SOLANA_CHAIN:
                return await self._get_solana_stats()
            else:
                return await self._get_blockchair_stats(normalized_chain)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error fetching stats: {str(e)}"
            }
    
    async def _get_blockchair_stats(self, chain: str) -> dict[str, Any]:
        """Get blockchain stats from Blockchair."""
        url = f"{self.BLOCKCHAIR_BASE_URL}/{chain}/stats"
        response = await self._make_request(url)
        
        if response.get("context", {}).get("code") != 200:
            return {
                "status": "error",
                "message": "Blockchair API error"
            }
        
        data = response.get("data", {})
        
        return {
            "status": "success",
            "chain": chain,
            "stats": {
                "blocks": data.get("blocks"),
                "transactions": data.get("transactions"),
                "difficulty": data.get("difficulty"),
                "hashrate": data.get("hashrate_24h"),
                "mempool_transactions": data.get("mempool_transactions"),
                "mempool_size": data.get("mempool_size"),
                "market_price_usd": data.get("market_price_usd"),
                "market_cap_usd": data.get("market_cap_usd"),
            }
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
