import logging
from openai import AzureOpenAI
from config.env import ServiceProvidersSettings
from django.conf import settings
import json
from moralis import evm_api
import pandas as pd
from datetime import datetime, timedelta, timezone
import numpy as np


class RiskCalculatorAgentController:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.open_ai_model_name = "gpt-4o"
        self.configs: ServiceProvidersSettings = settings.ENV.service_providers
        self.azure_client = AzureOpenAI(
            api_key=self.configs.azure_openai_api_token,
            api_version=self.configs.azure_openai_api_version,
            azure_endpoint=self.configs.azure_openai_endpoint,
        )

    @staticmethod
    def get_wallet_information(wallet_address: str) -> dict:
        moralis_api_token = settings.ENV.service_providers.moralis_api_token

        wallet_information = {}
        params = {
            "order": "ASC",
            "limit": 1,
            "address": wallet_address,
        }

        wallet_history = evm_api.wallets.get_wallet_history(
            api_key=moralis_api_token,
            params=params,
        )

        wallet_information["first_transaction_datetime"] = pd.to_datetime(
            wallet_history["result"][0]["block_timestamp"]
        )
        wallet_information["days_passed_since_first_transaction"] = (
            datetime.now(timezone.utc)
            - wallet_information["first_transaction_datetime"]
        ).days
        wallet_information["first_transaction_datetime"] = str(
            wallet_information["first_transaction_datetime"]
        )

        params = {"address": wallet_address}

        profitability_summary = evm_api.wallets.get_wallet_profitability_summary(
            api_key=moralis_api_token,
            params=params,
        )
        wallet_information["total_count_of_trades"] = profitability_summary[
            "total_count_of_trades"
        ]
        wallet_information["avg_daily_trades"] = (
            wallet_information["total_count_of_trades"]
            / wallet_information["days_passed_since_first_transaction"]
        )
        wallet_information["total_realized_profit_percentage"] = profitability_summary[
            "total_realized_profit_percentage"
        ]
        wallet_information["avg_daily_realized_profit_percentage"] = (
            wallet_information["total_realized_profit_percentage"]
            / wallet_information["days_passed_since_first_transaction"]
        )
        wallet_information["total_realized_profit_usd"] = float(
            profitability_summary["total_realized_profit_usd"]
        )
        wallet_information["avg_daily_realized_profit_usd"] = (
            wallet_information["total_realized_profit_usd"]
            / wallet_information["days_passed_since_first_transaction"]
        )

        wallet_stats = evm_api.wallets.get_wallet_stats(
            api_key=moralis_api_token,
            params=params,
        )
        wallet_information["total_nfts"] = float(wallet_stats["nfts"])
        wallet_information["total_nft_collections"] = float(wallet_stats["collections"])
        wallet_information["avg_nfts_per_collection"] = (
            wallet_information["total_nfts"]
            / wallet_information["total_nft_collections"]
            if wallet_information["total_nft_collections"] > 0
            else 0
        )
        wallet_information["total_transactions"] = float(
            wallet_stats["transactions"]["total"]
        )
        wallet_information["avg_daily_transactions"] = (
            wallet_information["total_transactions"]
            / wallet_information["days_passed_since_first_transaction"]
        )
        wallet_information["total_nft_transfers"] = float(
            wallet_stats["nft_transfers"]["total"]
        )
        wallet_information["avg_daily_nft_transfers"] = (
            wallet_information["total_nft_transfers"]
            / wallet_information["days_passed_since_first_transaction"]
        )
        wallet_information["total_token_transfers"] = float(
            wallet_stats["token_transfers"]["total"]
        )
        wallet_information["avg_daily_token_transfers"] = (
            wallet_information["total_token_transfers"]
            / wallet_information["days_passed_since_first_transaction"]
        )

        params = {
            "address": wallet_address,
            "min_pair_side_liquidity_usd": 100,
        }

        token_balances = evm_api.wallets.get_wallet_token_balances_price(
            api_key=moralis_api_token,
            params=params,
        )

        wallet_information["stable-coins-portfolio-percentage"] = np.sum(
            [
                token_balance["portfolio_percentage"]
                for token_balance in token_balances["result"]
                if abs(token_balance["usd_price"] - 1) < 0.001
            ]
        )
        wallet_information["stable-coins-usd-value"] = np.sum(
            [
                token_balance["usd_value"]
                for token_balance in token_balances["result"]
                if abs(token_balance["usd_price"] - 1) < 0.001
            ]
        )

        wallet_information["non-stable-coins-portfolio-percentage"] = np.sum(
            [
                token_balance["portfolio_percentage"]
                for token_balance in token_balances["result"]
                if abs(token_balance["usd_price"] - 1) >= 0.001
            ]
        )
        wallet_information["non-stable-coins-usd-value"] = np.sum(
            [
                token_balance["usd_value"]
                for token_balance in token_balances["result"]
                if abs(token_balance["usd_price"] - 1) >= 0.001
            ]
        )
        wallet_information["total_worth"] = (
            wallet_information["stable-coins-usd-value"]
            + wallet_information["non-stable-coins-usd-value"]
        )

        params = {"address": wallet_address}

        protocols = evm_api.wallets.get_defi_summary(
            api_key=moralis_api_token,
            params=params,
        )

        wallet_information["total_defi_positions_usd_value"] = protocols[
            "total_usd_value"
        ]
        wallet_information["total_defi_positions_portfolio_percentage"] = (
            (
                wallet_information["total_defi_positions_usd_value"]
                / wallet_information["total_worth"]
                * 100
            )
            if wallet_information["total_worth"] > 0
            else 0
        )
        wallet_information["total_defi_active_protocols"] = protocols[
            "active_protocols"
        ]
        wallet_information["total_defi_positions"] = protocols["total_positions"]
        wallet_information["avg_defi_positions_per_protocol"] = (
            (
                wallet_information["total_defi_positions"]
                / wallet_information["total_defi_active_protocols"]
            )
            if wallet_information["total_defi_active_protocols"] > 0
            else 0
        )
        wallet_information["defi_protocols"] = [
            {
                "protocol_name": protocol["protocol_name"],
                "total_usd_value": protocol["total_usd_value"],
                "total_positions": protocol["positions"],
            }
            for protocol in protocols["protocols"]
        ]

        return wallet_information

    @staticmethod
    def get_wallet_risk(wallet_address: str) -> dict:
        """
        Calculate a crypto wallet's risk score and return the internal metrics used for the decision.
        This allows the assistant to explain the reasoning to the user.
        """
        wallet_information = RiskCalculatorAgentController.get_wallet_information(
            wallet_address
        )

        age_risk = 1 - min(
            1, wallet_information["days_passed_since_first_transaction"] / (3 * 365)
        )
        total_worth_risk = 1 - (
            min(1, wallet_information["total_worth"] / 100000)
            if (wallet_information["total_worth"] > 20000)
            else 0
        )
        avg_daily_trades_risk = min(1, wallet_information["avg_daily_trades"] / 10)
        avg_daily_realized_profit_risk = min(
            1, abs(wallet_information["avg_daily_realized_profit_usd"]) / 5000
        )
        total_nfts_risk = min(1, wallet_information["total_nfts"] / 100)
        avg_nfts_per_collection_risk = min(
            1, wallet_information["avg_nfts_per_collection"] / 10
        )
        avg_daily_transactions_risk = min(
            1, wallet_information["avg_daily_transactions"] / 50
        )
        avg_daily_nft_transfers_risk = min(
            1, wallet_information["avg_daily_nft_transfers"] / 10
        )
        avg_daily_token_transfers_risk = min(
            1, wallet_information["avg_daily_token_transfers"] / 20
        )
        portfolio_risk = (
            min(1, wallet_information["non-stable-coins-portfolio-percentage"] / 60)
            if (wallet_information["total_worth"] > 20000)
            else 0
        )
        total_defi_positions_portfolio_percentage_risk = (
            min(1, wallet_information["total_defi_positions_portfolio_percentage"] / 30)
            if (wallet_information["total_worth"] > 20000)
            else 0
        )
        avg_defi_positions_per_protocol_risk = min(
            1, wallet_information["avg_defi_positions_per_protocol"] / 10
        )

        risks = {
            "age_risk": (age_risk, 10),
            "total_worth_risk": (total_worth_risk, 15),
            "avg_daily_trades_risk": (avg_daily_trades_risk, 10),
            "avg_daily_realized_profit_risk": (avg_daily_realized_profit_risk, 10),
            "total_nfts_risk": (total_nfts_risk, 5),
            "avg_nfts_per_collection_risk": (avg_nfts_per_collection_risk, 5),
            "avg_daily_transactions_risk": (avg_daily_transactions_risk, 10),
            "avg_daily_nft_transfers_risk": (avg_daily_nft_transfers_risk, 5),
            "avg_daily_token_transfers_risk": (avg_daily_token_transfers_risk, 5),
            "portfolio_risk": (portfolio_risk, 15),
            "total_defi_positions_portfolio_percentage_risk": (
                total_defi_positions_portfolio_percentage_risk,
                10,
            ),
            "avg_defi_positions_per_protocol_risk": (
                avg_defi_positions_per_protocol_risk,
                5,
            ),
        }

        total_weighted_risk = sum(value * weight for value, weight in risks.values())
        total_possible_weight = sum(weight for _, weight in risks.values())

        wallet_information["wallet_address"] = wallet_address
        wallet_information["wallet_risk"] = round(
            (total_weighted_risk / total_possible_weight) * 100, 2
        )

        return wallet_information

    def __get_wallet_risk_function_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "get_wallet_risk",
                "description": "Calculate a crypto wallet's risk score between 0 to 100 and return the internal metrics used for the decision. This allows the assistant to explain the reasoning to the user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "wallet_address": {
                            "type": "string",
                            "description": "The public wallet address (e.g., 0x...)",
                        }
                    },
                    "required": ["wallet_address"],
                },
            },
        }

    def __run_agent(self):
        intro_message = "Hi I'm assistant from Nuvolari.ai that evaluates the risk of crypto wallets."
        messages = [
            {
                "role": "system",
                "content": "You are assistant from Nuvolari.ai that evaluates the risk of crypto wallets.",
            },
            {"role": "assistant", "content": intro_message},
        ]
        print(f"##########################\n\n\nAssistant: {intro_message}")

        while True:
            user_input = input(f"User: ")
            messages.append({"role": "user", "content": user_input})
            print("Assistant is thinking ...")
            response = self.azure_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=[self.__get_wallet_risk_function_schema()],
                tool_choice="auto",
            )

            message = response.choices[0].message
            assistant_response = None
            if message.tool_calls:
                messages.append(message.model_dump())
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "get_wallet_risk":
                        args = json.loads(tool_call.function.arguments)
                        result = self.get_wallet_risk(**args)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": json.dumps(result),
                            }
                        )

                response = self.azure_client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                )
                assistant_response = response.choices[0].message.content
            else:
                assistant_response = message.content

            messages.append({"role": "assistant", "content": assistant_response})
            print(f"##########################\n\n\nAssistant: {assistant_response}")

    def run(self):
        self.__run_agent()
