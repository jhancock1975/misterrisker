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
        """Fetch recent Solana transactions via RPC with full details.
        
        For Solana, we get the latest block and then fetch details for each transaction.
        
        Args:
            limit: Number of transactions
            
        Returns:
            Formatted transaction response with parsed details
        """
        # Get latest slot
        slot_response = await self._make_solana_request("getSlot", [])
        if "error" in slot_response:
            return {
                "status": "error",
                "message": f"Solana RPC error: {slot_response.get('error')}"
            }
        
        latest_slot = slot_response.get("result")
        
        # Get block with full transaction details
        block_response = await self._make_solana_request("getBlock", [
            latest_slot,
            {
                "encoding": "json",
                "transactionDetails": "full",
                "maxSupportedTransactionVersion": 0
            }
        ])
        
        if "error" in block_response:
            return {
                "status": "error",
                "message": f"Solana RPC error: {block_response.get('error')}"
            }
        
        block_data = block_response.get("result", {})
        block_time = block_data.get("blockTime")
        transactions = block_data.get("transactions", [])[:limit]
        
        # Parse each transaction for meaningful details
        formatted_txs = []
        for tx_data in transactions:
            parsed_tx = self._parse_solana_transaction(tx_data, latest_slot, block_time)
            formatted_txs.append(parsed_tx)
        
        return {
            "status": "success",
            "chain": "solana",
            "slot": latest_slot,
            "block_time": block_time,
            "count": len(formatted_txs),
            "transactions": formatted_txs
        }
    
    def _parse_solana_transaction(
        self, 
        tx_data: dict[str, Any], 
        slot: int,
        block_time: int | None
    ) -> dict[str, Any]:
        """Parse a Solana transaction to extract meaningful details.
        
        Args:
            tx_data: Raw transaction data from RPC
            slot: The slot number
            block_time: Unix timestamp of the block
            
        Returns:
            Parsed transaction with human-readable details
        """
        tx = tx_data.get("transaction", {})
        meta = tx_data.get("meta", {})
        
        # Get signature
        message = tx.get("message", {})
        signatures = tx.get("signatures", [])
        signature = signatures[0] if signatures else "unknown"
        
        # Get account keys (addresses involved)
        account_keys = message.get("accountKeys", [])
        
        # Pre/post balances (in lamports, 1 SOL = 1e9 lamports)
        pre_balances = meta.get("preBalances", [])
        post_balances = meta.get("postBalances", [])
        
        # Calculate balance changes
        balance_changes = []
        for i, (pre, post) in enumerate(zip(pre_balances, post_balances)):
            if pre != post:
                change_lamports = post - pre
                change_sol = change_lamports / 1e9
                if i < len(account_keys):
                    balance_changes.append({
                        "account": account_keys[i],
                        "change_lamports": change_lamports,
                        "change_sol": change_sol
                    })
        
        # Fee paid
        fee_lamports = meta.get("fee", 0)
        fee_sol = fee_lamports / 1e9
        
        # Transaction status
        err = meta.get("err")
        status = "failed" if err else "success"
        
        # Identify transaction type from instructions
        instructions = message.get("instructions", [])
        tx_type = self._identify_solana_tx_type(instructions, account_keys, balance_changes)
        
        # Find the main accounts (sender and receiver for transfers)
        sender = None
        receiver = None
        transfer_amount_sol = None
        
        # For transfers, the fee payer is usually the sender
        if account_keys:
            sender = account_keys[0]  # First account is usually fee payer/sender
        
        # Look for significant balance changes to identify transfers
        for change in balance_changes:
            if change["change_sol"] > 0 and change["account"] != sender:
                receiver = change["account"]
                transfer_amount_sol = abs(change["change_sol"])
                break
            elif change["change_sol"] < 0 and abs(change["change_sol"]) > fee_sol:
                # Large negative balance = likely the sender
                transfer_amount_sol = abs(change["change_sol"]) - fee_sol
        
        # Total value moved (absolute sum of all SOL changes)
        total_value_sol = sum(abs(c["change_sol"]) for c in balance_changes) / 2
        
        # Log data (for NFT/token info)
        log_messages = meta.get("logMessages", [])
        
        # Token balance changes (for SPL tokens)
        pre_token_balances = meta.get("preTokenBalances", [])
        post_token_balances = meta.get("postTokenBalances", [])
        
        token_transfers = []
        if pre_token_balances or post_token_balances:
            token_transfers = self._parse_token_transfers(
                pre_token_balances, 
                post_token_balances,
                account_keys
            )
        
        return {
            "signature": signature,
            "slot": slot,
            "block_time": block_time,
            "status": status,
            "fee_sol": fee_sol,
            "tx_type": tx_type,
            "sender": sender,
            "receiver": receiver,
            "transfer_amount_sol": transfer_amount_sol,
            "total_value_sol": total_value_sol,
            "num_accounts": len(account_keys),
            "num_instructions": len(instructions),
            "balance_changes": balance_changes[:5],  # Limit for display
            "token_transfers": token_transfers[:3],  # Limit for display
            "has_error": err is not None,
            "error": str(err) if err else None
        }
    
    def _identify_solana_tx_type(
        self, 
        instructions: list[dict], 
        account_keys: list[str],
        balance_changes: list[dict]
    ) -> str:
        """Identify the type of Solana transaction.
        
        Args:
            instructions: Transaction instructions
            account_keys: Accounts involved
            balance_changes: Balance changes
            
        Returns:
            Transaction type string
        """
        if not instructions:
            return "unknown"
        
        # Common Solana program IDs
        SYSTEM_PROGRAM = "11111111111111111111111111111111"
        TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
        ASSOCIATED_TOKEN = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
        COMPUTE_BUDGET = "ComputeBudget111111111111111111111111111111"
        SERUM_DEX = "9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin"
        RAYDIUM_AMM = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
        
        program_ids = set()
        for instr in instructions:
            prog_idx = instr.get("programIdIndex", 0)
            if prog_idx < len(account_keys):
                program_ids.add(account_keys[prog_idx])
        
        # Determine type based on programs used
        if RAYDIUM_AMM in program_ids or SERUM_DEX in program_ids:
            return "🔄 DEX Swap"
        elif TOKEN_PROGRAM in program_ids and ASSOCIATED_TOKEN in program_ids:
            return "🪙 Token Transfer"
        elif TOKEN_PROGRAM in program_ids:
            return "🪙 Token Operation"
        elif SYSTEM_PROGRAM in program_ids and len(balance_changes) >= 2:
            return "💸 SOL Transfer"
        elif SYSTEM_PROGRAM in program_ids:
            return "⚙️ System Operation"
        elif len(instructions) > 3:
            return "📜 Smart Contract"
        else:
            return "📝 Transaction"
    
    def _parse_token_transfers(
        self,
        pre_balances: list[dict],
        post_balances: list[dict],
        account_keys: list[str]
    ) -> list[dict]:
        """Parse SPL token transfers from balance changes.
        
        Args:
            pre_balances: Pre-transaction token balances
            post_balances: Post-transaction token balances
            account_keys: Account addresses
            
        Returns:
            List of token transfers
        """
        transfers = []
        
        # Build a map of token balances by account index
        pre_map = {b.get("accountIndex"): b for b in pre_balances}
        post_map = {b.get("accountIndex"): b for b in post_balances}
        
        all_indices = set(pre_map.keys()) | set(post_map.keys())
        
        for idx in all_indices:
            pre = pre_map.get(idx, {})
            post = post_map.get(idx, {})
            
            pre_amount = float(pre.get("uiTokenAmount", {}).get("uiAmount") or 0)
            post_amount = float(post.get("uiTokenAmount", {}).get("uiAmount") or 0)
            
            if pre_amount != post_amount:
                mint = post.get("mint") or pre.get("mint", "unknown")
                owner = post.get("owner") or pre.get("owner", "unknown")
                change = post_amount - pre_amount
                
                transfers.append({
                    "mint": mint,
                    "owner": owner[:8] + "..." if len(owner) > 8 else owner,
                    "change": change,
                    "direction": "received" if change > 0 else "sent"
                })
        
        return transfers
    
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
