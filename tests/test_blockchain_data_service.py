"""Tests for the Blockchain Data Service.

This service provides global blockchain data (transactions, blocks) for multiple chains:
- Bitcoin (BTC) via Blockchair API
- Ethereum (ETH) via Blockchair API
- Ripple (XRP) via Blockchair API
- Zcash (ZEC) via Blockchair API
- Solana (SOL) via Solana RPC API
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp


class TestBlockchainDataService:
    """Tests for BlockchainDataService."""

    @pytest.mark.asyncio
    async def test_get_recent_transactions_bitcoin(self):
        """Should fetch recent Bitcoin transactions from Blockchair."""
        from src.services.blockchain_data import BlockchainDataService
        
        mock_response = {
            "data": [
                {
                    "block_id": 873500,
                    "hash": "abc123",
                    "time": "2025-12-06 10:00:00",
                    "input_total": 100000000,
                    "output_total": 99990000,
                    "fee": 10000,
                },
                {
                    "block_id": 873500,
                    "hash": "def456",
                    "time": "2025-12-06 09:59:00",
                    "input_total": 50000000,
                    "output_total": 49995000,
                    "fee": 5000,
                }
            ],
            "context": {"code": 200}
        }
        
        with patch.object(BlockchainDataService, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            service = BlockchainDataService()
            result = await service.get_recent_transactions("bitcoin", limit=10)
            
            assert result["status"] == "success"
            assert "transactions" in result
            assert len(result["transactions"]) == 2
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_recent_transactions_ethereum(self):
        """Should fetch recent Ethereum transactions from Blockchair."""
        from src.services.blockchain_data import BlockchainDataService
        
        mock_response = {
            "data": [
                {
                    "block_id": 19000000,
                    "hash": "0xabc123",
                    "time": "2025-12-06 10:00:00",
                    "value": 1000000000000000000,
                    "gas_used": 21000,
                }
            ],
            "context": {"code": 200}
        }
        
        with patch.object(BlockchainDataService, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            service = BlockchainDataService()
            result = await service.get_recent_transactions("ethereum", limit=10)
            
            assert result["status"] == "success"
            assert "transactions" in result
            assert len(result["transactions"]) == 1

    @pytest.mark.asyncio
    async def test_get_recent_transactions_ripple(self):
        """Should fetch recent Ripple (XRP) transactions from Blockchair."""
        from src.services.blockchain_data import BlockchainDataService
        
        mock_response = {
            "data": [
                {
                    "ledger_index": 85000000,
                    "hash": "xyz789",
                    "close_time": "2025-12-06 10:00:00",
                }
            ],
            "context": {"code": 200}
        }
        
        with patch.object(BlockchainDataService, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            service = BlockchainDataService()
            result = await service.get_recent_transactions("ripple", limit=10)
            
            assert result["status"] == "success"
            assert "transactions" in result

    @pytest.mark.asyncio
    async def test_get_recent_transactions_zcash(self):
        """Should fetch recent Zcash transactions from Blockchair."""
        from src.services.blockchain_data import BlockchainDataService
        
        mock_response = {
            "data": [
                {
                    "block_id": 2500000,
                    "hash": "zec123",
                    "time": "2025-12-06 10:00:00",
                }
            ],
            "context": {"code": 200}
        }
        
        with patch.object(BlockchainDataService, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            service = BlockchainDataService()
            result = await service.get_recent_transactions("zcash", limit=10)
            
            assert result["status"] == "success"
            assert "transactions" in result

    @pytest.mark.asyncio
    async def test_get_recent_transactions_solana(self):
        """Should fetch recent Solana transactions via Solana RPC."""
        from src.services.blockchain_data import BlockchainDataService
        
        # Solana uses a different API structure
        mock_block_response = {
            "result": {
                "transactions": [
                    {
                        "transaction": {
                            "signatures": ["sig1"],
                        },
                        "meta": {
                            "fee": 5000,
                        }
                    },
                    {
                        "transaction": {
                            "signatures": ["sig2"],
                        },
                        "meta": {
                            "fee": 5000,
                        }
                    }
                ]
            }
        }
        
        with patch.object(BlockchainDataService, '_make_solana_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_block_response
            
            service = BlockchainDataService()
            result = await service.get_recent_transactions("solana", limit=10)
            
            assert result["status"] == "success"
            assert "transactions" in result

    @pytest.mark.asyncio
    async def test_get_latest_block_bitcoin(self):
        """Should fetch the latest Bitcoin block info."""
        from src.services.blockchain_data import BlockchainDataService
        
        mock_response = {
            "data": {
                "blocks": 873500,
                "transactions": 1050000000,
                "best_block_hash": "abc123hash",
                "best_block_height": 873500,
                "best_block_time": "2025-12-06 10:00:00",
            },
            "context": {"code": 200}
        }
        
        with patch.object(BlockchainDataService, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            service = BlockchainDataService()
            result = await service.get_latest_block("bitcoin")
            
            assert result["status"] == "success"
            assert "block" in result

    @pytest.mark.asyncio
    async def test_supported_chains(self):
        """Should list all supported blockchain chains."""
        from src.services.blockchain_data import BlockchainDataService
        
        service = BlockchainDataService()
        chains = service.get_supported_chains()
        
        assert "bitcoin" in chains
        assert "ethereum" in chains
        assert "ripple" in chains
        assert "zcash" in chains
        assert "solana" in chains

    @pytest.mark.asyncio
    async def test_unsupported_chain_returns_error(self):
        """Should return error for unsupported chains."""
        from src.services.blockchain_data import BlockchainDataService
        
        service = BlockchainDataService()
        result = await service.get_recent_transactions("unsupported_chain", limit=10)
        
        assert result["status"] == "error"
        assert "unsupported" in result["message"].lower()

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
