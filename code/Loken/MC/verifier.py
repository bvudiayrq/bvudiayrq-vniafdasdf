# file: verifier.py
import subprocess
import re
import os
import tempfile
from typing import Dict, Any

# --- Scenario Definition ---
# This dictionary defines a complete verification scenario.
# You can change 'dapp_to_test' to any of the implemented DApp models
# to verify its specific logic against the full LTL specification suite.
scenario = {
    "dapp_to_test": "UniswapV2Pair",  # Options: 'UniswapV2Pair', 'UniswapV3CLPool', 'CurveStableswapPool', 
                                     # 'BalancerWeightedPool', 'SushiSwapPair', 'PancakeSwapV3Pool',
                                     # 'OneInchAggregator', 'CoWProtocol', 'UniswapV4HookPool'
    "constants": {
        "NUM_USERS": 3,
        "MAX_BALANCE": 100000,
        "MAX_SUPPLY": 100000,
        "PAIR_ADDR": 2  # A designated address for the pair contract
    },
    "initial_state": {
        "users": {
            # User 0: The primary liquidity provider and swapper
            0: {"token0": 50000, "token1": 50000, "lp_token": 0},
            # User 1: Another swapper and NFT recipient
            1: {"token0": 10000, "token1": 10000, "lp_token": 0}
        },
        "allowances": {
            # owner -> spender -> amount
            0: {1: 0}
        },
        "pool": {
            "reserve0": 0,
            "reserve1": 0
        },
        "nft": {
            "owner": 0,  # User 0 owns the NFT
            "approved": 0  # No one is approved initially
        },
        "rebase": {
            "total_supply": 10000,
            "total_staking": 50000  # Hypothetical total staked amount
        }
    },
    # A sequence of actions to test. The model checker will execute these in order.
    "actions": [
        {"step": 0, "user": 0, "type": "approve", "params": {"spender": 1, "amount": 2000}},
        {"step": 1, "user": 1, "type": "transfer_from", "params": {"from_user": 0, "to_user": 1, "value": 1500}},
        {"step": 2, "user": 0, "type": "add_liquidity", "params": {"amount0_in": 10000, "amount1_in": 10000}},
        {"step": 3, "user": 1, "type": "swap", "params": {"amount0_in": 1000, "amount1_in": 0}},
        {"step": 4, "user": 0, "type": "nft_approve", "params": {"approved_user": 1}},
        {"step": 5, "user": 1, "type": "nft_transfer_from", "params": {"from_user": 0, "to_user": 1}},
        {"step": 6, "user": 0, "type": "remove_liquidity", "params": {"amount_lp_in": 5000}},
        {"step": 7, "user": 0, "type": "rebase", "params": {"rebase_amount": 500}},
        {"step": 8, "user": 0, "type": "airdrop", "params": {"user1_gets": 100, "user2_gets": 150}},
        {"step": 9, "user": 1, "type": "nft_burn", "params": {}},
        {"step": 10, "user": 0, "type": "nft_mint", "params": {"to_user": 0}},
    ]
}

def generate_scenario_smv(template_path: str, scenario_data: Dict[str, Any]) -> str:
    """
    Generates scenario-specific SMV file content from a base template and scenario data.
    """
    with open(template_path, 'r') as f:
        template_content = f.read()

    # Replace constants and select the DApp to test
    template_content = template_content.replace("{{DAPP_TO_TEST}}", scenario_data["dapp_to_test"])
    for key, value in scenario_data["constants"].items():
        template_content = template_content.replace(f"{{{key}}}", str(value))

    # Build initial state assignments
    init_lines =
    # User balances
    for user_id, state in scenario_data["initial_state"]["users"].items():
        init_lines.append(f"    init(dapp.token0.balance[{user_id}]) := {state['token0']};")
        init_lines.append(f"    init(dapp.token1.balance[{user_id}]) := {state['token1']};")
        init_lines.append(f"    init(dapp.lp_token.balance[{user_id}]) := {state['lp_token']};")
    # Allowances
    for owner, spenders in scenario_data["initial_state"]["allowances"].items():
        for spender, amount in spenders.items():
            init_lines.append(f"    init(dapp.token0.allowance[{owner}][{spender}]) := {amount};")
    # Pool reserves
    init_lines.append(f"    init(dapp.reserve0) := {scenario_data['initial_state']['pool']['reserve0']};")
    init_lines.append(f"    init(dapp.reserve1) := {scenario_data['initial_state']['pool']['reserve1']};")
    # NFT state
    init_lines.append(f"    init(dapp.nft_module.owner) := {scenario_data['initial_state']['nft']['owner']};")
    init_lines.append(f"    init(dapp.nft_module.approved) := {scenario_data['initial_state']['nft']['approved']};")
    # Rebase state
    init_lines.append(f"    init(dapp.rebase_module.totalSupply) := {scenario_data['initial_state']['rebase']['total_supply']};")
    init_lines.append(f"    init(dapp.rebase_module.totalStaking) := {scenario_data['initial_state']['rebase']['total_staking']};")

    template_content = template_content.replace("-- {{INITIAL_STATE_ASSIGNMENTS}}", "\n".join(init_lines))

    # Build action sequence constraints
    trans_lines =
    num_actions = len(scenario_data["actions"])
    trans_lines.append(f"    next(state_step) := (state_step < {num_actions - 1})? state_step + 1 : {num_actions - 1};")
    trans_lines.append("    TRANS")
    for action in scenario_data["actions"]:
        step = action["step"]
        user = action["user"]
        action_type = action["type"]
        params = action["params"]
        param_str = " & ".join([f"{k} = {v}" for k, v in params.items()])
        trans_lines.append(f"        (state_step = {step}) -> (user_action = {action_type} & actor = {user} & {param_str});")
    
    template_content = template_content.replace("-- {{ACTION_SEQUENCE_TRANS}}", "\n".join(trans_lines))

    return template_content

def run_verifier(model_content: str, verbose: bool = False):
    """
    Executes NuSMV on a given model content string and parses the output.
    """
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.smv') as temp_f:
        temp_f.write(model_content)
        model_path = temp_f.name

    print(f"[*] Verifying generated model for DApp '{scenario['dapp_to_test']}': {model_path}")
    
    try:
        result = subprocess.run(
           ,
            capture_output=True, text=True, check=True, timeout=600  # 10-minute timeout
        )
        output = result.stdout
        if verbose:
            print("\n--- NuSMV Output ---")
            print(output)
            print("--------------------")

        spec_results = re.findall(r"-- specification (.+?) is (true|false)", output)
        if not spec_results:
            print("[!] Could not parse verification results from output.")
            return False

        print("\n--- Verification Results ---")
        all_true = True
        for name, status in spec_results:
            print(f"    - Specification '{name.strip()}': {status.upper()}")
            if status == 'false':
                all_true = False
        print("--------------------------")
        
        if not all_true:
            print("\n[!] One or more properties were violated. Check NuSMV output for counterexamples.")
        else:
            print("\n[+] All properties verified successfully.")
        return all_true

    except FileNotFoundError:
        print("[!] Error: 'NuSMV' executable not found. Please ensure it is in your system's PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[!] Error during NuSMV execution. Return code: {e.returncode}")
        print("Stderr:", e.stderr)
        return False
    except subprocess.TimeoutExpired:
        print("[!] Error: NuSMV execution timed out. The model may be too complex.")
        return False
    finally:
        os.remove(model_path)

if __name__ == "__main__":
    base_model_path = "models.smv"
    if not os.path.exists(base_model_path):
        print(f"[!] Error: Base model file '{base_model_path}' not found.")
    else:
        print("[*] Generating scenario-specific SMV model...")
        scenario_model_content = generate_scenario_smv(base_model_path, scenario)
        
        # For debugging, you can save the generated model
        # with open("generated_scenario.smv", "w") as f:
        #     f.write(scenario_model_content)
            
        run_verifier(scenario_model_content, verbose=False)