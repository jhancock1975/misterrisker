"""Tests for the Blockchain Data Service.

This service provides global blockchain data (transactions, blocks).
Bitcoin and Solana are supported:
- Bitcoin via mempool.space API (free, no API key)
- Solana via public RPC API (free, no API key)
Other chains (ETH, XRP, ZEC) return not_supported status.
"""

import pytest
from unittest.mock import AsyncMock, patch


class TestBlockchainDataService:
    """Tests for BlockchainDataService."""

    @pytest.mark.asyncio
    async def test_get_recent_transactions_solana(self):
        """Should fetch recent Solana transactions via Solana RPC."""
        from src.services.blockchain_data import BlockchainDataService
        
        # Mock the slot response
        mock_slot_response = {"result": 300000000}
        
        # Mock the block response with full transaction data
        mock_block_response = {
            "result": {
                "blockHeight": 250000000,
                "blockTime": 1701864000,
                "transactions": [
                    {
                        "transaction": {
                            "signatures": ["sig1abc123"],
                            "message": {
                                "accountKeys": ["addr1", "addr2"],
                                "instructions": [{"programIdIndex": 0}]
                            }
                        },
                        "meta": {
                            "err": None,
                            "fee": 5000,
                            "preBalances": [1000000000, 500000000],
                            "postBalances": [999995000, 500000000]
                        }
                    },
                    {
                        "transaction": {
                            "signatures": ["sig2def456"],
                            "message": {
                                "accountKeys": ["addr3", "addr4"],
                                "instructions": [{"programIdIndex": 0}]
                            }
                        },
                        "meta": {
                            "err": None,
                            "fee": 5000,
                            "preBalances": [2000000000, 100000000],
                            "postBalances": [1999995000, 100000000]
                        }
                    }
                ]
            }
        }
        
        with patch.object(BlockchainDataService, '_make_solana_request', new_callable=AsyncMock) as mock_request:
            # First call returns slot, second returns block
            mock_request.side_effect = [mock_slot_response, mock_block_response]
            
            service = BlockchainDataService()
            result = await service.get_recent_transactions("solana", limit=10)
            
            assert result["status"] == "success"
            assert result["chain"] == "solana"
            assert "transactions" in result
            assert len(result["transactions"]) == 2
            assert result["transactions"][0]["signature"] == "sig1abc123"

    @pytest.mark.asyncio
    async def test_get_recent_transactions_bitcoin_supported(self):
        """Should fetch recent Bitcoin transactions via mempool.space."""
        from src.services.blockchain_data import BlockchainDataService
        
        # Mock blocks response
        mock_blocks_response = [
            {
                "id": "0000000000000000000abc123",
                "height": 926000,
                "timestamp": 1701864000,
                "tx_count": 3000
            }
        ]
        
        # Mock transactions response
        mock_txs_response = [
            {
                "txid": "txid123abc",
                "vin": [{"prevout": {"scriptpubkey_address": "bc1sender", "value": 100000}}],
                "vout": [{"scriptpubkey_address": "bc1receiver", "value": 90000}],
                "fee": 1000,
                "size": 250,
                "weight": 600,
                "status": {"confirmed": True, "block_height": 926000}
            }
        ]
        
        with patch.object(BlockchainDataService, '_make_mempool_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = [mock_blocks_response, mock_txs_response]
            
            service = BlockchainDataService()
            result = await service.get_recent_transactions("bitcoin", limit=5)
            
            assert result["status"] == "success"
            assert result["chain"] == "bitcoin"
            assert "transactions" in result
            assert len(result["transactions"]) == 1
            assert result["transactions"][0]["txid"] == "txid123abc"

    @pytest.mark.asyncio
    async def test_get_recent_transactions_ethereum_not_supported(self):
        """Should return not_supported for Ethereum (no free API)."""
        from src.services.blockchain_data import BlockchainDataService
        
        service = BlockchainDataService()
        result = await service.get_recent_transactions("ethereum", limit=10)
        
        assert result["status"] == "not_supported"

    @pytest.mark.asyncio
    async def test_get_latest_block_solana(self):
        """Should fetch the latest Solana slot info."""
        from src.services.blockchain_data import BlockchainDataService
        
        mock_slot_response = {"result": 300000000}
        mock_epoch_response = {
            "result": {
                "epoch": 650,
                "slotIndex": 150000,
                "slotsInEpoch": 432000,
            }
        }
        
        with patch.object(BlockchainDataService, '_make_solana_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = [mock_slot_response, mock_epoch_response]
            
            service = BlockchainDataService()
            result = await service.get_latest_block("solana")
            
            assert result["status"] == "success"
            assert result["chain"] == "solana"
            assert "block" in result
            assert result["block"]["slot"] == 300000000

    @pytest.mark.asyncio
    async def test_get_latest_block_bitcoin_supported(self):
        """Should fetch Bitcoin block info via mempool.space."""
        from src.services.blockchain_data import BlockchainDataService
        
        mock_blocks_response = [
            {
                "id": "0000000000000000000abc123",
                "height": 926000,
                "timestamp": 1701864000,
                "tx_count": 3000,
                "size": 1500000,
                "weight": 4000000,
                "difficulty": 149000000000000,
                "nonce": 12345678,
                "merkle_root": "abc123merkle"
            }
        ]
        
        with patch.object(BlockchainDataService, '_make_mempool_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_blocks_response
            
            service = BlockchainDataService()
            result = await service.get_latest_block("bitcoin")
            
            assert result["status"] == "success"
            assert result["chain"] == "bitcoin"
            assert "block" in result
            assert result["block"]["height"] == 926000

    @pytest.mark.asyncio
    async def test_get_chain_stats_solana(self):
        """Should fetch Solana chain statistics."""
        from src.services.blockchain_data import BlockchainDataService
        
        # getEpochInfo response
        mock_epoch_response = {
            "result": {
                "epoch": 650,
                "slotIndex": 150000,
                "slotsInEpoch": 432000,
                "absoluteSlot": 300000000,
                "blockHeight": 250000000,
            }
        }
        # getSupply response
        mock_supply_response = {
            "result": {
                "value": {
                    "total": 580000000000000000,
                    "circulating": 450000000000000000,
                    "nonCirculating": 130000000000000000,
                }
            }
        }
        
        with patch.object(BlockchainDataService, '_make_solana_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = [
                mock_epoch_response,
                mock_supply_response,
            ]
            
            service = BlockchainDataService()
            result = await service.get_blockchain_stats("solana")
            
            assert result["status"] == "success"
            assert result["chain"] == "solana"
            assert "stats" in result
            assert result["stats"]["epoch"] == 650

    @pytest.mark.asyncio
    async def test_get_chain_stats_ethereum_not_supported(self):
        """Should return not_supported for Ethereum stats."""
        from src.services.blockchain_data import BlockchainDataService
        
        service = BlockchainDataService()
        result = await service.get_blockchain_stats("ethereum")
        
        assert result["status"] == "not_supported"

    @pytest.mark.asyncio
    async def test_supported_chains(self):
        """Should list supported and known chains correctly."""
        from src.services.blockchain_data import BlockchainDataService
        
        service = BlockchainDataService()
        
        # get_supported_chains returns only fully supported chains
        supported = service.get_supported_chains()
        assert "solana" in supported
        assert "bitcoin" in supported  # Bitcoin is now supported via mempool.space
        
        # Ethereum etc are not fully supported
        assert "ethereum" not in supported
        
        # get_all_known_chains includes both supported and unsupported
        all_known = service.get_all_known_chains()
        assert "bitcoin" in all_known
        assert "ethereum" in all_known
        assert "ripple" in all_known
        assert "zcash" in all_known
        assert "solana" in all_known

    @pytest.mark.asyncio
    async def test_unsupported_chain_returns_error(self):
        """Should return error for completely unknown chains."""
        from src.services.blockchain_data import BlockchainDataService
        
        service = BlockchainDataService()
        result = await service.get_recent_transactions("unsupported_chain", limit=10)
        
        # Unknown chains return not_supported
        assert result["status"] in ("error", "not_supported")

    @pytest.mark.asyncio
    async def test_chain_aliases(self):
        """Should support common aliases for chains (BTC, ETH, XRP, SOL, ZEC)."""
        from src.services.blockchain_data import BlockchainDataService
        
        service = BlockchainDataService()
        
        # Test that aliases map to correct chains
        assert service._normalize_chain("btc") == "bitcoin"
        assert service._normalize_chain("eth") == "ethereum"
        assert service._normalize_chain("xrp") == "ripple"
        assert service._normalize_chain("sol") == "solana"
        assert service._normalize_chain("zec") == "zcash"
        assert service._normalize_chain("bitcoin") == "bitcoin"
        assert service._normalize_chain("SOLANA") == "solana"

    @pytest.mark.asyncio
    async def test_solana_rpc_error_handling(self):
        """Should handle Solana RPC errors gracefully."""
        from src.services.blockchain_data import BlockchainDataService
        
        with patch.object(BlockchainDataService, '_make_solana_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("RPC connection failed")
            
            service = BlockchainDataService()
            result = await service.get_recent_transactions("solana", limit=10)
            
            assert result["status"] == "error"
            assert "error" in result["message"].lower() or "failed" in result["message"].lower()
