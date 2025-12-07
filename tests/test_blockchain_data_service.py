"""Tests for the Blockchain Data Service.

This service provides global blockchain data (transactions, blocks).
Currently only Solana is supported via free public RPC API.
Other chains (BTC, ETH, XRP, ZEC) return not_supported status.
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
        
        # Mock the block response - Solana returns signatures list directly
        mock_block_response = {
            "result": {
                "blockHeight": 250000000,
                "blockTime": 1701864000,
                "signatures": ["sig1abc123", "sig2def456"]
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
    async def test_get_recent_transactions_bitcoin_not_supported(self):
        """Should return not_supported for Bitcoin (no free API)."""
        from src.services.blockchain_data import BlockchainDataService
        
        service = BlockchainDataService()
        result = await service.get_recent_transactions("bitcoin", limit=10)
        
        assert result["status"] == "not_supported"
        assert "chain" in result
        assert "message" in result
        assert "free API" in result["message"]

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
    async def test_get_latest_block_bitcoin_not_supported(self):
        """Should return not_supported for Bitcoin block info."""
        from src.services.blockchain_data import BlockchainDataService
        
        service = BlockchainDataService()
        result = await service.get_latest_block("bitcoin")
        
        assert result["status"] == "not_supported"
        assert "free API" in result["message"]

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
        
        # Bitcoin/Ethereum/etc are not fully supported
        assert "bitcoin" not in supported
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
