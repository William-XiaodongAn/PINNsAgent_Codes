# src/run_experiments.py

import argparse
import os
import sys
import time
import json
import csv
import math

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.retriever import PGKR
from database.knowledge_base import KnowledgeBase
from database.memory_tree import MemoryTree
from agents.programmer import Programmer
from agents.planner import Planner
from utils.llm_client import LLMClient
from utils.config_loader import ConfigLoader
from utils.logger import ExperimentLogger
from utils.formatter import format_mse, format_time
from database.pde_encoder import PDE_LABELS


def _load_parent_llms():
    """LLM allow-list from the parent cardiac-agent/experiment_config.json ('llms') --
    the single source of truth shared with experiment.py / baselines. Each key is a
    selectable model; its value carries the provider used for native-SDK routing in
    utils/llm_client.py. API keys are read from the parent .env."""
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "experiment_config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f).get("llms", {})
    except (OSError, ValueError) as e:
        print(f"Warning: could not read parent LLM allow-list at {cfg_path}: {e}. "
              f"--llm will be unconstrained.")
        return {}


PARENT_LLMS = _load_parent_llms()


def parse_instances(spec):
    """Parse an instance spec like '0-49' or '0,1,2' or '0-3,7' into a deduped,
    order-preserving list of ints."""
    ids = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            lo, hi = tok.split("-")
            ids += list(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(tok))
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def resolve_instances(args):
    """Return (instance_key, [ids]) for the batch 'search-once-reuse-rest' flow,
    or (None, None) when no instance list was given. instance_key is the yaml/
    benchmark argument name for the chosen PDE family."""
    for family, key in (("heat", "heat_instance"), ("fk", "fk_instance"),
                        ("adv", "adv_instance"), ("bg", "bg_instance"),
                        ("tnnp", "tnnp_instance")):
        spec = getattr(args, f"{family}_instances")
        if spec:
            return key, parse_instances(spec)
    return None, None


def _finite(xs):
    """Keep only real, finite numbers (drop None/nan/inf)."""
    return [x for x in xs if x is not None and not (math.isnan(x) or math.isinf(x))]


def pick_best_iteration(iteration_history):
    """Lowest-MSE iteration, ignoring nan/inf; falls back to the raw min if all bad."""
    valid = [it for it in iteration_history
             if not (math.isnan(it["mse"]) or math.isinf(it["mse"]))]
    pool = valid if valid else iteration_history
    return min(pool, key=lambda x: x["mse"])


def parse_args():
    parser = argparse.ArgumentParser(description="PINNsAgent experiment script")
    # Basic experiment settings
    parser.add_argument('--mode', type=str, choices=['random', 'llm'], default='random',
                       help='Optimization mode: random or llm')
    # Prompt strategy arguments
    parser.add_argument('--prompt_strategy', type=str, default='zero_shot',
                       choices=['zero_shot', 'full_history', 'memory_tree', 'pgkr', 'pinns_agent'],
                       help='Prompt strategy for LLM mode (default: zero_shot)')
    # PGKR arguments for pgkr and pinns_agent strategies
    parser.add_argument('--use_pgkr', action='store_true',
                       help='Enable PGKR (PDE-Guided Knowledge Retrieval) for pinns_agent strategy')
    parser.add_argument('--use_memory_tree', action='store_true',
                       help='Enable MemoryTree exploration scores for pinns_agent strategy')
    parser.add_argument('--pgkr_top_k', type=int, default=1,
                       help='Number of similar PDEs to retrieve from knowledge base (default: 1)')
    parser.add_argument('--use_composite_score', action='store_true',
                       help='Use composite score (MSE + runtime) for PGKR best config selection')
    
    # NEW: UCT-related arguments
    parser.add_argument('--use_uct', action='store_true',
                       help='Use UCT (Upper Confidence Bound) scores instead of static exploration scores')
    parser.add_argument('--uct_lambda', type=float, default=1.4,
                       help='UCT exploration weight lambda (default: 1.4)')
    
    parser.add_argument('--pde_type', type=str, choices=['1d', '2d', '3d', 'nd'], default=None,
                       help='PDE dimension type, choose one between --pde_type and --pde_name, cannot specify both')
    parser.add_argument('--pde_name', type=str, default=None,
                       help='Specify single PDE name, choose one between --pde_type and --pde_name, cannot specify both')
    parser.add_argument('--num_iters', type=int, default=5,
                       help='Number of optimization iterations per PDE')
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./outputs/experiments',
                       help='Experiment relative directory output path for both agent logs and PINNacle results')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Experiment run name, used to distinguish different experiments')
    # Training settings
    parser.add_argument('--device', type=str, default='0', help='GPU device ID')
    parser.add_argument('--llm', type=str, default=None,
                        choices=(list(PARENT_LLMS.keys()) or None),
                        help="LLM model that drives the search (llm mode only). Restricted to the "
                             "parent cardiac-agent/experiment_config.json 'llms' allow-list; provider "
                             "routing and the API key (from .env) are derived from that entry. "
                             "Default: the model in configs/default_config.yaml llm_config.")
    parser.add_argument('--iter', type=int, default=20000, help='Training iteration count')
    parser.add_argument('--fk_instance', type=int, default=None,
                       help='Fenton-Karma test-set instance id: train on the batch-generated '
                            'ref/fenton_karma_<i>.dat + its IC. Default: the un-indexed default pair (instance 0).')
    parser.add_argument('--heat_instance', type=int, default=None,
                       help='Heat2D-cardiac test-set instance id: train on the batch-generated '
                            'ref/heat2d_cardiac_<i>.dat + its IC. Default: the un-indexed default pair (instance 0).')
    parser.add_argument('--adv_instance', type=int, default=None,
                       help='Advection-cardiac test-set instance id: train on the batch-generated '
                            'ref/advection_beta*_<i>.dat + its IC. Default: the un-indexed default pair (instance 0).')
    parser.add_argument('--bg_instance', type=int, default=None,
                       help='Burgers-cardiac test-set instance id: train on the batch-generated '
                            'ref/burgers_nu*_<i>.dat + its IC. Default: the un-indexed default pair (instance 0).')
    parser.add_argument('--tnnp_instance', type=int, default=None,
                       help='TNNP test-set instance id: train on the batch-generated '
                            'ref/tnnp_<i>.dat + its 20 IC fields. Default: the un-indexed default pair (instance 0).')
    # Batch "search-once, reuse-rest" instance lists. When given, the FIRST instance
    # runs the full LLM hyperparameter search (--num_iters); every remaining instance
    # reuses that best config for a single training run (no LLM calls).
    parser.add_argument('--heat_instances', type=str, default=None,
                       help='Heat2D-cardiac instance list/range, e.g. "0-49" or "0,1,2". '
                            'First instance is searched; the rest reuse its best config.')
    parser.add_argument('--fk_instances', type=str, default=None,
                       help='Fenton-Karma instance list/range, e.g. "10-59" or "0,1,2". '
                            'First instance is searched; the rest reuse its best config.')
    parser.add_argument('--adv_instances', type=str, default=None,
                       help='Advection-cardiac instance list/range, e.g. "10-59" or "0,1,2". '
                            'First instance is searched; the rest reuse its best config.')
    parser.add_argument('--bg_instances', type=str, default=None,
                       help='Burgers-cardiac instance list/range, e.g. "10-59" or "0,1,2". '
                            'First instance is searched; the rest reuse its best config.')
    parser.add_argument('--tnnp_instances', type=str, default=None,
                       help='TNNP instance list/range, e.g. "2-9" or "0,1,2". '
                            'First instance is searched; the rest reuse its best config.')
    parser.add_argument('--seed', type=int, default=44, help='Random seed')
    # Path settings
    parser.add_argument('--config_path', type=str, default=None,
                       help='Configuration file path')
    parser.add_argument('--csv_path', type=str,
                       default='./data/dataset_for_retrieval.csv',
                       help='Knowledge base CSV file path')
    parser.add_argument('--train_code_dir', type=str, default="./pinnacle",
                       help='Training code directory (where benchmark.py is located)')
    parser.add_argument('--conda_python', type=str, default="python",
                       help='Conda environment Python executable path')
    # Repeated experiment settings
    parser.add_argument('--num_runs', type=int, default=1,
                       help='Number of repeated runs per PDE')
    # New PDE scenario simulation
    parser.add_argument('--simulate_new_pde', action='store_true',
                       help='Simulate new PDE scenario: do not use this PDE\'s historical records for retrieval')
    # Knowledge base save control
    parser.add_argument('--save_kb', action='store_true',
                       help='Save updated knowledge base to CSV file after experiments')
    parser.add_argument('--kb_save_path', type=str, default=None,
                       help='Custom path to save knowledge base (default: overwrite original CSV)')
    # Verbose control
    parser.add_argument('--verbose_llm', action='store_true',
                       help='Whether to print LLM interaction process (including retry information)')
    parser.add_argument('--verbose_training', action='store_true',
                       help='Whether to print detailed output of PINNacle training process')
    
    args = parser.parse_args()
    
    # Validate mutual exclusivity of pde_type and pde_name
    if args.pde_type is None and args.pde_name is None:
        parser.error("Must specify either --pde_type or --pde_name")
    if args.pde_type is not None and args.pde_name is not None:
        parser.error("Cannot specify both --pde_type and --pde_name, please choose only one")

    # Validate PGKR/MemoryTree arguments - only valid for pinns_agent strategy
    if args.prompt_strategy != "pinns_agent" and args.use_pgkr:
        parser.error("--use_pgkr can only be used with --prompt_strategy pinns_agent")
    if args.prompt_strategy != "pinns_agent" and args.use_memory_tree:
        parser.error("--use_memory_tree can only be used with --prompt_strategy pinns_agent")
    
    # Validate UCT arguments
    if args.use_uct and not args.use_memory_tree:
        parser.error("--use_uct requires --use_memory_tree to be enabled")

    # Validate batch instance arguments (search-once, reuse-rest flow).
    # Families: heat / fk / adv (advection) / bg (burgers). One family per run.
    families = ("heat", "fk", "adv", "bg", "tnnp")
    plural = {f: getattr(args, f"{f}_instances") for f in families}
    active = [f for f, v in plural.items() if v]
    if len(active) > 1:
        parser.error("Specify only one of --heat_instances/--fk_instances/--adv_instances/"
                     "--bg_instances/--tnnp_instances; one instance family per run")
    for f in families:
        if plural[f] and getattr(args, f"{f}_instance") is not None:
            parser.error(f"Use either --{f}_instance (single) or --{f}_instances (batch), not both")
    if active and args.pde_name is None:
        parser.error("--*_instances require --pde_name (instances are specific to a single PDE)")

    return args

def setup_experiment(args):
    """Set up experiment environment"""
    # Load configuration
    config_loader = ConfigLoader(args.config_path)
    
    # Update fixed parameters using args
    config_loader.update_fixed_params(
        device=args.device,
        iter=args.iter,
        seed=args.seed
    )
    
    # Create output directory (absolute path)
    base_output = os.path.abspath(args.output_dir)
    output_dir = os.path.join(base_output, args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    return config_loader, output_dir

def initialize_agents(args, config_loader, output_dir):
    """Initialize all agents"""
    # Knowledge base
    kb = KnowledgeBase(args.csv_path)
    
    # Retriever (PGKR)
    pgkr = PGKR()
    
    # Memory Tree (for memory_tree and pinns_agent strategies)
    memory_tree = None
    if args.mode == "llm" and args.prompt_strategy in ["memory_tree", "pinns_agent"]:
        memory_tree = MemoryTree(knowledge_base=kb)
    
    # Planner
    if args.mode == "llm":
        llm_config = config_loader.get_llm_config()
        if args.llm:
            # Model picked from the parent allow-list: route by its provider and let
            # LLMClient pull the matching key from .env (api_key omitted on purpose).
            provider = PARENT_LLMS.get(args.llm, {}).get("provider")
            llm_client = LLMClient(
                model=args.llm,
                provider=provider,
                base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
            )
            print(f"Using LLM '{args.llm}' (provider={provider or 'inferred'}) from the parent allow-list")
        else:
            llm_client = LLMClient(
                api_key=llm_config.get("api_key"),
                base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
                model=llm_config.get("model", "gpt-4o"),
            )
        # Prepare kwargs for different prompt strategies
        prompt_kwargs = {}
        if args.prompt_strategy == "pinns_agent":
            prompt_kwargs['use_pgkr'] = args.use_pgkr
            prompt_kwargs['use_memory_tree'] = args.use_memory_tree
        
        planner = Planner(
            mode=args.mode, 
            llm_client=llm_client, 
            log_dir=output_dir,
            search_space=config_loader.get_search_space(),
            verbose=args.verbose_llm,
            prompt_strategy=args.prompt_strategy,
            max_iterations=args.num_iters,
            **prompt_kwargs
        )
    else:
        planner = Planner(
            mode=args.mode, 
            log_dir=output_dir,
            search_space=config_loader.get_search_space(),
            verbose=args.verbose_llm,
            max_iterations=args.num_iters,
        )
    
    # Programmer - initialize with basic parameters only, specific output_dir set in each run
    fixed_params = config_loader.get_fixed_params()
    fixed_params.update({
        "name": "iter",  # Initial name, will be updated in each run
        "output_dir": "./outputs/temp",  # Temporary directory, will be updated in each run
        "general_method": "none",
        # [pde, ic, periodic] — up-weight the IC term 100x so the net can't ignore
        # the initial condition. ONLY valid for 3-loss PDEs (burgers/advection);
        # FK/TNNP/heat have a different number of loss terms, set back to "none".
        "loss_weight": "1,100,1",
        "num_test_points": "Default"
    })
    # Pass the Fenton-Karma instance selector through to benchmark.py (via the yaml),
    # only when set so other PDEs' configs stay untouched.
    if args.fk_instance is not None:
        fixed_params["fk_instance"] = args.fk_instance
    if args.heat_instance is not None:
        fixed_params["heat_instance"] = args.heat_instance
    if args.adv_instance is not None:
        fixed_params["adv_instance"] = args.adv_instance
    if args.bg_instance is not None:
        fixed_params["bg_instance"] = args.bg_instance
    if args.tnnp_instance is not None:
        fixed_params["tnnp_instance"] = args.tnnp_instance

    programmer = Programmer(
        template_fixed=fixed_params, 
        train_dir=args.train_code_dir,
        conda_python=args.conda_python,
        verbose=args.verbose_training
    )
    
    # Logger
    logger = ExperimentLogger(output_dir)
    
    return kb, pgkr, memory_tree, planner, programmer, logger

def run_single_pde_experiment(pde_name, args, config_loader, kb, pgkr, memory_tree, planner,
                              programmer, logger, output_dir, run_id=1, run_output_dir=None):
    """Run experiment for a single PDE"""
    print(f"\n{'='*120}")
    # Base info string
    info_str = (f"Experiment: {pde_name} | Run {run_id}/{args.num_runs} | Mode: {args.mode.upper()} | "
                f"Num Iters: {args.num_iters} | Prompt Strategy: {args.prompt_strategy}")
    print(info_str)
    
    # Add strategy-specific info
    if args.mode == "llm":
        if args.prompt_strategy == "pinns_agent":
            # PINNsAgent: show both PGKR and MemoryTree status
            info = f"PGKR: {'ON' if args.use_pgkr else 'OFF'}"
            if args.use_pgkr:
                info += f" (Top-K={args.pgkr_top_k} | Composite Score={args.use_composite_score})"
            info += f" | MemoryTree: {'ON' if args.use_memory_tree else 'OFF'}"
            # NEW: Show UCT status
            if args.use_memory_tree:
                info += f" | UCT: {'ON' if args.use_uct else 'OFF'}"
                if args.use_uct:
                    info += f" (λ={args.uct_lambda})"
            info += f" | Simulate New PDE: {args.simulate_new_pde}"
            print(info)
        elif args.prompt_strategy == "pgkr":
            # pgkr strategy always uses PGKR
            pgkr_info = f"PGKR Top-K: {args.pgkr_top_k} | Simulate New PDE: {args.simulate_new_pde}"
            pgkr_info += f" | Use Composite Score: {args.use_composite_score}"
            print(pgkr_info)
        elif args.prompt_strategy == "memory_tree":
            # memory_tree strategy always uses MemoryTree
            memory_info = f"Use MemoryTree: True | Simulate New PDE: {args.simulate_new_pde}"
            # NEW: Show UCT status for memory_tree strategy
            memory_info += f" | UCT: {'ON' if args.use_uct else 'OFF'}"
            if args.use_uct:
                memory_info += f" (λ={args.uct_lambda})"
            print(memory_info)
    
    print(f"{'='*120}")
    
    # NEW: Reset online visit counts for this PDE before starting new run
    if memory_tree and args.use_uct:
        memory_tree.reset_online_visit_counts(pde_name)
        print(f"\n🔄 Reset online visit counts for {pde_name}")
    
    # Retrieve similar PDEs and their best configurations if using PGKR
    similar_pdes_configs = None
    if args.mode == "llm" and args.prompt_strategy in ["pgkr", "pinns_agent"]:
        # For 'pgkr' strategy: always use PGKR
        # For 'pinns_agent' strategy: use PGKR only if args.use_pgkr is True
        should_use_pgkr = (args.prompt_strategy == "pgkr") or (args.use_pgkr)
        
        if should_use_pgkr:
            print(f"\nRetrieving best configurations from {args.pgkr_top_k} similar PDEs...")
            similar_pdes_configs = pgkr.retrieve_similar_pdes_configs(
                target_pde=pde_name,
                kb=kb,
                pgkr_top_k=args.pgkr_top_k,
                simulate_new_pde=args.simulate_new_pde,
                use_composite_score=args.use_composite_score
            )
            
            if similar_pdes_configs:
                print(f"Retrieved configurations from {len(similar_pdes_configs)} similar PDEs:")
                for pde, info in similar_pdes_configs.items():
                    print(f"  • {pde}: Similarity={info['similarity']:.4f}, Best MSE={info['best_mse']:.2e}")
            else:
                print("Warning: No similar PDEs configurations retrieved")
    
    # Experiment loop
    iteration_history = []
    if run_output_dir is None:
        run_output_dir = os.path.join(output_dir, f"{pde_name}_run_{run_id}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    for iter_id in range(1, args.num_iters + 1):
        print(f"\n{'─'*80}")
        print(f"Iteration {iter_id}/{args.num_iters}")
        print(f"{'─'*80}")
        
        # NEW: Get exploration scores (static or UCT-based)
        exploration_scores = None
        if args.mode == "llm" and args.prompt_strategy in ["memory_tree", "pinns_agent"] and memory_tree:
            # For 'memory_tree' strategy: always use MemoryTree
            # For 'pinns_agent' strategy: use MemoryTree only if args.use_memory_tree is True
            should_use_memory_tree = (args.prompt_strategy == "memory_tree") or (args.use_memory_tree)
            
            if should_use_memory_tree:
                if args.use_uct:
                    # Use UCT scores (dynamic)
                    print(f"\n🎯 Retrieving UCT scores from MemoryTree (λ={args.uct_lambda})...")
                    exploration_scores = memory_tree.get_uct_scores(
                        pde_name=pde_name,
                        simulate_new_pde=args.simulate_new_pde,
                        lambda_val=args.uct_lambda
                    )
                    score_type = "UCT"
                else:
                    # Use static exploration scores
                    print(f"\n📊 Retrieving static exploration scores from MemoryTree...")
                    exploration_scores = memory_tree.get_scores_for_pde(
                        pde_name=pde_name,
                        simulate_new_pde=args.simulate_new_pde
                    )
                    score_type = "Static"
                
                if exploration_scores:
                    print(f"Top-10 parameters by {score_type} score:")
                    top_5 = list(exploration_scores.items())
                    for param, score in top_5:
                        # Show visit count if using UCT
                        if args.use_uct and pde_name in memory_tree.online_visit_counts:
                            visits = memory_tree.online_visit_counts[pde_name].get(param, 0)
                            print(f"  • [{visits}x] {param}: {score:.3f}")
                        else:
                            print(f"  • {param}: {score:.3f}")
                else:
                    print("Warning: No exploration scores retrieved")
        
        # Snapshot cumulative LLM token usage before config generation so we can
        # attribute the tokens spent (incl. retries) to this iteration.
        llm_client = getattr(planner, "llm_client", None)
        tok_before = llm_client.get_usage() if llm_client is not None else None

        # Generate configuration - pass exploration_scores (UCT or static)
        config = planner.generate_config(
            history=iteration_history,
            pde_name=pde_name,
            run_id=run_id,
            iter_id=iter_id,
            similar_pdes_configs=similar_pdes_configs,
            exploration_scores=exploration_scores
        )
        config["task"] = pde_name
        config["pde_list"] = [pde_name]

        # Token cost spent on this iteration's LLM calls (zeros in random mode).
        if llm_client is not None and tok_before is not None:
            tok_after = llm_client.get_usage()
            token_cost = {
                "input": tok_after["input_tokens"] - tok_before["input_tokens"],
                "output": tok_after["output_tokens"] - tok_before["output_tokens"],
            }
        else:
            token_cost = {"input": 0, "output": 0}
        
        # Simplified configuration display
        key_params = {k: v for k, v in config.items() 
                     if k not in ['task', 'pde_list']}
        print(f"Config: {key_params}")
        
        # Set iteration directory
        iter_dir = os.path.join(run_output_dir, f"iter_{iter_id}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # Set PINNacle output directory
        programmer.fixed["output_dir"] = run_output_dir
        programmer.fixed["name"] = f"iter_{iter_id}"
        
        # Generate configuration file
        yaml_path = os.path.join(iter_dir, "config.yaml")
        programmer.write_yaml(config, yaml_path)
        
        print(f"YAML Config File: {yaml_path}")
        
        # Run experiment
        print("Training...")
        mse, run_time, nrmse = programmer.run_exp(yaml_path)

        # Display results
        print(f"✓ Completed | MSE: {format_mse(mse)} | nRMSE: {format_mse(nrmse)} | "
              f"Time: {format_time(run_time)}s | Tokens(in/out): {token_cost['input']}/{token_cost['output']}")
        
        # NEW: Update visit counts if using UCT
        if memory_tree and args.use_uct:
            memory_tree.update_visit_count(pde_name, config)
            print(f"📈 Updated visit counts")
        
        # Record experiment history
        iteration_record = {
            "iter_id": iter_id,
            "config": config,
            "mse": mse,
            "nrmse": nrmse,
            "run_time": run_time,
            "token_cost": token_cost,
            "pde_name": pde_name,
            "run_id": run_id
        }
        iteration_history.append(iteration_record)
        
        # Add to knowledge base
        record = dict(config)
        record["task"] = pde_name
        record["mse"] = mse
        record["run_time"] = run_time
        kb.add_record(record)
        
        # Display current best result
        best_iteration = min(iteration_history, key=lambda x: x["mse"])
        print(f"Best So Far | MSE: {format_mse(best_iteration['mse'])} | run_time: {format_time(best_iteration['run_time'])}s")
    
    # Save single run experiment summary
    logger.save_run_summary(iteration_history, run_output_dir, pde_name, run_id)
    
    # Summarize this round of experiment
    best_iteration = min(iteration_history, key=lambda x: x["mse"])
    total_in = sum((it.get("token_cost") or {}).get("input", 0) for it in iteration_history)
    total_out = sum((it.get("token_cost") or {}).get("output", 0) for it in iteration_history)
    print(f"\n{'='*80}")
    print(f"Run {run_id} Summary for {pde_name}:")
    print(f"  Best MSE: {format_mse(best_iteration['mse'])}")
    print(f"  Best nRMSE: {format_mse(best_iteration.get('nrmse', float('nan')))}")
    print(f"  Best Time: {format_time(best_iteration['run_time'])}s")
    print(f"  Token Cost (this run): input={total_in}, output={total_out}")
    print(f"  Best Config: {best_iteration['config']}")
    print(f"{'='*80}")
    
    return iteration_history


def run_instance_batch(pde_name, inst_key, inst_ids, args, config_loader, kb, pgkr,
                       memory_tree, planner, programmer, logger, output_dir, run_id=1):
    """Search-once, reuse-rest over a list of PDE instances.

    The FIRST instance runs the full LLM hyperparameter search (--num_iters); its
    best config is then reused for a single training run on every remaining
    instance (no LLM calls). All instances share the hyperparameters; only the IC
    and reference solution change per instance.
    """
    first, rest = inst_ids[0], inst_ids[1:]
    parent_dir = os.path.join(output_dir, f"{pde_name}_instances_run_{run_id}")
    os.makedirs(parent_dir, exist_ok=True)

    # ---- Phase 1: full hyperparameter search on the first instance -------------
    print(f"\n{'#'*120}")
    print(f"# Phase 1/2 — full search on {inst_key}={first}  "
          f"({pde_name}, run {run_id}/{args.num_runs}, {args.num_iters} iters)")
    print(f"{'#'*120}")
    programmer.fixed[inst_key] = first
    search_dir = os.path.join(parent_dir, f"search_instance_{first}")
    search_history = run_single_pde_experiment(
        pde_name, args, config_loader, kb, pgkr, memory_tree, planner,
        programmer, logger, output_dir, run_id=run_id, run_output_dir=search_dir
    )

    best = pick_best_iteration(search_history)
    # The hyperparameters to reuse. Strip instance selectors (they come from
    # programmer.fixed, set per instance below).
    best_config = dict(best["config"])
    best_config.pop("fk_instance", None)
    best_config.pop("heat_instance", None)
    hyper = {k: v for k, v in best_config.items() if k not in ("task", "pde_list")}

    search_tokens = {
        "input": sum((it.get("token_cost") or {}).get("input", 0) for it in search_history),
        "output": sum((it.get("token_cost") or {}).get("output", 0) for it in search_history),
    }

    print(f"\n✅ Phase 1 done. Best config from {inst_key}={first} "
          f"(iter {best['iter_id']} | MSE {format_mse(best['mse'])} | "
          f"nRMSE {format_mse(best.get('nrmse', float('nan')))}):")
    print(f"   {hyper}")

    instance_results = [{
        "instance": first, "phase": "search", "mse": best["mse"],
        "nrmse": best.get("nrmse", float("nan")), "run_time": best["run_time"],
    }]

    # ---- Phase 2: reuse the best config on every remaining instance ------------
    if rest:
        print(f"\n{'#'*120}")
        print(f"# Phase 2/2 — reuse best config on {len(rest)} instance(s), no LLM: {rest}")
        print(f"{'#'*120}")
    for inst in rest:
        print(f"\n{'─'*80}")
        print(f"Reuse on {inst_key}={inst}")
        print(f"{'─'*80}")
        inst_dir = os.path.join(parent_dir, f"reuse_instance_{inst}")
        os.makedirs(inst_dir, exist_ok=True)

        programmer.fixed["output_dir"] = parent_dir
        programmer.fixed["name"] = f"reuse_instance_{inst}"
        programmer.fixed[inst_key] = inst

        yaml_path = os.path.join(inst_dir, "config.yaml")
        programmer.write_yaml(best_config, yaml_path)

        print("Training (reused config)...")
        mse, run_time, nrmse = programmer.run_exp(yaml_path)
        print(f"✓ Completed | MSE: {format_mse(mse)} | nRMSE: {format_mse(nrmse)} | "
              f"Time: {format_time(run_time)}s")

        instance_results.append({
            "instance": inst, "phase": "reuse", "mse": mse,
            "nrmse": nrmse, "run_time": run_time,
        })

        # Record the reused config + this instance's result in the knowledge base.
        record = dict(best_config)
        record["task"] = pde_name
        record["mse"] = mse
        record["run_time"] = run_time
        kb.add_record(record)

    save_instance_batch_summary(parent_dir, pde_name, inst_key, inst_ids, first,
                                best_config, best, search_tokens, instance_results, args)
    return instance_results


def save_instance_batch_summary(parent_dir, pde_name, inst_key, inst_ids, first_inst,
                                best_config, best_search_iter, search_tokens,
                                instance_results, args):
    """Write per-instance CSV + JSON summary and print the aggregate table.

    The headline metric is nRMSE = mean over instances of each instance's relative
    L2 error (4_l2rel), matching scripts/eval_test_set.py and the paper convention.
    """
    hyper = {k: v for k, v in best_config.items() if k not in ("task", "pde_list")}
    nrmses = _finite([r["nrmse"] for r in instance_results])
    mses = _finite([r["mse"] for r in instance_results])
    nrmse_mean = sum(nrmses) / len(nrmses) if nrmses else float("nan")
    mse_mean = sum(mses) / len(mses) if mses else float("nan")

    # ---- per-instance CSV ------------------------------------------------------
    csv_path = os.path.join(parent_dir, "instance_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "phase", "mse", "nrmse", "run_time"])
        for r in instance_results:
            w.writerow([r["instance"], r["phase"], r["mse"], r["nrmse"], r["run_time"]])
        w.writerow([])
        w.writerow(["nRMSE (mean over instances)", nrmse_mean])
        w.writerow(["mean_mse", mse_mean])
        w.writerow(["num_instances", len(instance_results)])

    # ---- JSON summary ----------------------------------------------------------
    summary = {
        "pde_name": pde_name,
        "instance_kind": inst_key,
        "instances": inst_ids,
        "search_instance": first_inst,
        "num_instances": len(instance_results),
        "best_config": best_config,
        "search_best": {
            "iter_id": best_search_iter["iter_id"],
            "mse": best_search_iter["mse"],
            "nrmse": best_search_iter.get("nrmse", float("nan")),
            "run_time": best_search_iter["run_time"],
        },
        "search_token_cost": search_tokens,
        "aggregate": {"nrmse_mean": nrmse_mean, "mse_mean": mse_mean},
        "per_instance": instance_results,
        "run_config": {
            "num_iters": args.num_iters,
            "iter": args.iter,
            "simulate_new_pde": args.simulate_new_pde,
            "seed": args.seed,
        },
    }
    json_path = os.path.join(parent_dir, "batch_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- console table ---------------------------------------------------------
    print(f"\n{'='*120}")
    print(f"📦 INSTANCE BATCH SUMMARY — {pde_name}  ({inst_key})")
    print(f"{'='*120}")
    print(f"Searched on instance {first_inst}; reused best config on {len(instance_results) - 1} more.")
    print(f"Best config: {hyper}")
    print(f"\n{'instance':>10} {'phase':>8} {'MSE':>16} {'nRMSE':>16} {'time(s)':>10}")
    print("-" * 64)
    for r in instance_results:
        print(f"{r['instance']:>10} {r['phase']:>8} {format_mse(r['mse']):>16} "
              f"{format_mse(r['nrmse']):>16} {format_time(r['run_time']):>10}")
    print("-" * 64)
    print(f"nRMSE (mean over {len(nrmses)} instances): {format_mse(nrmse_mean)}")
    print(f"mean MSE: {format_mse(mse_mean)}")
    print(f"Search LLM tokens (in/out): {search_tokens['input']}/{search_tokens['output']}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"{'='*120}")


def save_knowledge_base(args, kb):
    """Persist the knowledge base to CSV when --save_kb is set."""
    if args.save_kb:
        save_path = args.kb_save_path if args.kb_save_path else None
        kb.save(save_path)
        saved_to = save_path if save_path else args.csv_path
        print(f"\n Knowledge base saved to: {saved_to}")
    else:
        print(f"\n Knowledge base not saved (use --save_kb to enable saving)")


def main():
    args = parse_args()
    
    # Set up experiment
    config_loader, output_dir = setup_experiment(args)
    
    # Initialize agents
    kb, pgkr, memory_tree, planner, programmer, logger = initialize_agents(args, config_loader, output_dir)
    
    # Get PDE list
    if args.pde_name:
        pde_list = [args.pde_name]
    else:
        pde_list = config_loader.get_pde_list(args.pde_type)
    
    print(f"\n{'='*120}")
    print(f"Created Output Directory: {output_dir}")
    print(f"PDEs to run: {pde_list}")
    print(f"{'='*120}")

    # --- Batch "search-once, reuse-rest" over instances --------------------------
    inst_key, inst_ids = resolve_instances(args)
    if inst_ids:
        pde_name = pde_list[0]  # validated: instance lists require a single --pde_name
        print(f"Instance batch ({inst_key}): {inst_ids}")
        print(f"  → search on {inst_ids[0]}, reuse best config on {len(inst_ids) - 1} more")
        for run_id in range(1, args.num_runs + 1):
            run_instance_batch(
                pde_name, inst_key, inst_ids, args, config_loader,
                kb, pgkr, memory_tree, planner, programmer, logger,
                output_dir, run_id
            )
        save_knowledge_base(args, kb)
        return

    # Run experiments
    all_results = {}
    
    for pde_name in pde_list:
        pde_results = []
        
        for run_id in range(1, args.num_runs + 1):
            iteration_history = run_single_pde_experiment(
                pde_name, args, config_loader, 
                kb, pgkr, memory_tree, planner, programmer, logger,
                output_dir, run_id
            )
            pde_results.extend(iteration_history)
            all_results[pde_name] = pde_results
            
            # Update overall results in real-time after each run completion
            logger.save_experiment_summary(all_results, args, completed_runs=run_id)
        
        # After all runs for this PDE are completed, save PDE-level summary
        logger.save_pde_summary(pde_name, all_results)
        print(f"\n✅ Completed all {args.num_runs} runs for {pde_name}\n")
    
    # Save knowledge base (controlled by argument)
    save_knowledge_base(args, kb)

    # Final summary
    logger.save_experiment_summary(all_results, args, completed_runs=args.num_runs)
    
    # Print concise summary table for all PDEs
    logger.print_all_pdes_summary(all_results)

if __name__ == "__main__":
    main()