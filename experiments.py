"""
PPCL experiments — all phases with disconnect protection.

Depends on core.py. In Colab: run core.py first, then this file.
If core.py wasn't run, this module will load it automatically.
"""
import os
import json
import time
import importlib.util

import numpy as np


def _ensure_ppcl_core():
    """Pull core symbols into this namespace if %%run core.py was skipped."""
    g = globals()
    if g.get("preflight_check") and g.get("run_method") and g.get("SAVE_DIR") is not None:
        return
    _here = os.path.dirname(os.path.abspath(__file__))
    _core_path = os.path.join(_here, "core.py")
    if not os.path.isfile(_core_path):
        raise FileNotFoundError(
            f"core.py not found beside experiments.py ({_core_path}). "
            "Upload both to the same folder, then run: %run core.py"
        )
    spec = importlib.util.spec_from_file_location("ppcl_core", _core_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    for _name in (
        "SAVE_DIR",
        "CKPT_DIR",
        "preflight_check",
        "run_method",
        "save_result",
        "load_all_results",
        "clear_results_cache",
        "_write_crash_log",
        "_atomic_json_write",
    ):
        if hasattr(mod, _name):
            g[_name] = getattr(mod, _name)


_ensure_ppcl_core()


# --- Helpers ---
def _cached(method, seed, tag):
    """Check Drive first, then local backup. Returns loaded dict or None."""
    fname = f"{method}_seed{seed}_{tag}.json"
    for base in [SAVE_DIR, "./results"]:
        fpath = os.path.join(base, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath) as f:
                    r = json.load(f)
                loc = "Drive" if base == SAVE_DIR else "local"
                print(f"  cached ({loc}) {method} seed={seed} [{tag}] ({r.get('avg_acc', 0):.1f}%)")
                return r
            except Exception as e:
                print(f"  could not load {fpath}: {e}")
    return None


def _run_and_save(method, seed, tag, **kwargs):
    """Run method, save result, return (result, model)."""
    try:
        kwargs.setdefault("ckpt_tag", tag)
        kwargs.setdefault("load_ckpt", True)
        kwargs.setdefault("save_ckpt", True)
        kwargs.setdefault("save_ckpt_each_task", True)
        result, model = run_method(method, seed=seed, **kwargs)
        result['method'] = method
        result['seed'] = seed
        save_result(result, tag)
        return result, model
    except Exception as e:
        try:
            _write_crash_log({"method": method, "seed": seed, "tag": tag, "kwargs": kwargs}, e)
        except Exception:
            pass
        print(f"Run failed for {method} seed={seed} tag={tag}: {e}")
        raise


def _timer_start():
    return time.time()


def _timer_end(t0, label=""):
    mins = (time.time() - t0) / 60
    print(f"  {label} {mins:.1f} min")


def _key(method, seed, tag):
    return f"{method}_s{seed}_{tag}"


def fresh_start_setup(clear_cache=False, clear_checkpoints=False, clear_figures=False):
    """One-call helper for a clean slate before running phases."""
    print("=== Fresh Start Setup ===")
    ok = preflight_check()
    if not ok:
        print("Preflight failed. Fix environment before running phases.")
        return False
    if clear_cache:
        clear_results_cache(
            delete_checkpoints=clear_checkpoints,
            delete_figures=clear_figures,
        )
    print("Ready. Next: phase1()")
    return True


def _mean_std_ci(vals):
    vals = np.array(vals, dtype=float)
    if len(vals) == 0:
        return 0.0, 0.0, 0.0
    m = float(np.mean(vals))
    s = float(np.std(vals))
    ci95 = float(1.96 * s / np.sqrt(max(len(vals), 1)))
    return m, s, ci95


def summarize_methods(results_by_method):
    """Print mean/std/95% CI for a dict of method -> list[result]."""
    print(f"\n{'Method':<12} {'Mean':>8} {'Std':>8} {'95%CI':>8} {'n':>4}")
    print("-" * 46)
    for mn in sorted(results_by_method.keys()):
        vals = [r.get('avg_acc', 0.0) for r in results_by_method[mn]]
        m, s, ci = _mean_std_ci(vals)
        print(f"{mn:<12} {m:>7.1f}% {s:>7.1f}% {ci:>7.1f} {len(vals):>4d}")


def validate_phase_results(results, expected_methods=None, expected_seeds=None, strict=False):
    """Check run integrity. Optionally raise on issues."""
    issues = []
    if expected_methods:
        for m in expected_methods:
            if isinstance(results, dict) and m not in results and not any(
                k.startswith(m) for k in results.keys()
            ):
                issues.append(f"Missing method: {m}")
    if expected_seeds is not None:
        if isinstance(results, dict):
            for m, v in results.items():
                if isinstance(v, list):
                    seeds = sorted([x.get("seed", None) for x in v])
                    for s in expected_seeds:
                        if s not in seeds:
                            issues.append(f"Missing seed {s} for {m}")
    if issues:
        print("\nValidation issues:")
        for it in issues:
            print(f"  - {it}")
        if strict:
            raise RuntimeError("Validation failed.")
    else:
        print("Validation passed.")
    return issues


def export_results_csv(results, out_name="paper_results.csv"):
    """Export flat CSV for the paper appendix."""
    rows = []
    if isinstance(results, dict):
        for k, v in results.items():
            if isinstance(v, list):
                for r in v:
                    rows.append(r)
            elif isinstance(v, dict):
                rows.append(v)
    out_path = os.path.join(SAVE_DIR, out_name)
    fields = sorted(set().union(*[set(r.keys()) for r in rows])) if rows else []
    if not rows:
        print("No rows to export.")
        return None
    import csv
    with open(out_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"Exported CSV: {out_path} ({len(rows)} rows)")
    return out_path


# --- Phase 1: PPCL vs SimCLR ---
def phase1():
    """~1 hr. SimCLR baseline + PPCL sweep over r in {0.05, 0.10, 0.15, 0.20}.
    Returns (r_sim, r_ppcl_best, best_r).
    """
    print("=" * 60)
    print("PHASE 1 -- PPCL vs SimCLR (blocker gate)")
    print("=" * 60)
    tag = "phase1"
    r_sim = _cached("simclr", 0, tag)
    if r_sim is None:
        print("\nRunning SimCLR baseline...")
        t0 = _timer_start()
        r_sim, _ = _run_and_save("simclr", 0, tag)
        _timer_end(t0, "SimCLR")
    reserves = [0.05, 0.10, 0.15, 0.20]
    best_r, best_acc, best_res = None, -1, None
    for r in reserves:
        rtag = f"phase1_r{r:.2f}"
        cached = _cached("ppcl", 0, rtag)
        if cached is not None:
            if cached['avg_acc'] > best_acc:
                best_acc = cached['avg_acc']
                best_r = r
                best_res = cached
            continue
        print(f"\nRunning PPCL reserve={r:.2f}...")
        t0 = _timer_start()
        res, _ = _run_and_save("ppcl", 0, rtag, ppcl_reserve=r)
        _timer_end(t0, f"PPCL r={r}")
        if res['avg_acc'] > best_acc:
            best_acc = res['avg_acc']
            best_r = r
            best_res = res
    print("\n" + "=" * 60)
    print(f"SimCLR:     {r_sim['avg_acc']:.1f}%")
    print(f"Best PPCL:  {best_acc:.1f}% (r={best_r:.2f})")
    gap = best_acc - r_sim['avg_acc']
    if gap > 0:
        print(f"PPCL beats SimCLR by {gap:.1f}pp -- PROCEED")
    else:
        print(f"PPCL lost to SimCLR by {-gap:.1f}pp -- STOP AND DIAGNOSE")
    print("=" * 60)
    return r_sim, best_res, best_r


# --- Phase 2: Full benchmark ---
def phase2(best_r=0.10, seeds=(0, 1, 2, 3, 4), linear_probe_for_all=False):
    """~6 hrs. Full CIFAR-100 benchmark, 18 methods x 3+ seeds."""
    print("=" * 60)
    print("PHASE 2 -- Full CIFAR-100 benchmark")
    print("=" * 60)
    tag = "phase2_lp" if linear_probe_for_all else "phase2"
    methods = [
        ("simclr", {}),
        ("simclr_ema_l2", {}),
        ("ewc_ssl", {}),
        ("lwf_ssl", {}),
        ("replay_ssl", {}),
        ("ppcl", {"ppcl_reserve": best_r}),
        ("ppcl_stable", {"ppcl_reserve": best_r}),
        ("ppcl_adaptive", {}),
        ("ppcl_mom", {"ppcl_reserve": best_r}),
        ("cassle", {}),
        ("pnr", {}),
        ("byol", {}),
        ("barlow", {}),
        ("simsiam", {}),
        ("vicreg", {}),
        ("lump", {}),
        ("der_ssl", {}),
        ("freeze_enc", {}),
        ("supervised", {}),
        ("oracle", {}),
    ]
    all_results = {}
    for method, kwargs in methods:
        for seed in seeds:
            cached = _cached(method, seed, tag)
            if cached is not None:
                all_results[f"{method}_seed{seed}"] = cached
                continue
            print(f"\nRunning {method} seed={seed}...")
            t0 = _timer_start()
            sckpt = (seed == 0)
            res, _ = _run_and_save(
                method,
                seed,
                tag,
                save_ckpt=sckpt,
                ckpt_tag=tag,
                eval_linear_probe=linear_probe_for_all,
                **kwargs,
            )
            all_results[f"{method}_seed{seed}"] = res
            _timer_end(t0, method)
    print_summary(all_results)
    validate_phase_results(all_results, expected_methods=[m for m, _ in methods], strict=False)
    return all_results


# --- Phase 12: Random dims ablation ---
def phase12_ppcl_random_dims(best_r=0.10, seeds=(0, 1, 2, 3, 4), n_tasks=20):
    """Compare variance-selected vs random-dims PPCL on a long task horizon."""
    print("=" * 60)
    print(f"PHASE 12 -- PPCL random-dims ablation ({n_tasks} tasks)")
    print("=" * 60)
    tag = f"phase12_ppcl_rand_{n_tasks}t_r{best_r:.2f}"
    methods = [
        ("ppcl", {"ppcl_reserve": best_r}),
        ("ppcl_rand", {"ppcl_reserve": best_r}),
    ]
    results = {}
    for method, kwargs in methods:
        for seed in seeds:
            cached = _cached(method, seed, tag)
            if cached is not None:
                results.setdefault(method, []).append(cached)
                continue
            print(f"\nRunning {method} seed={seed}...")
            t0 = _timer_start()
            res, _ = _run_and_save(method, seed, tag, n_tasks=n_tasks, **kwargs)
            results.setdefault(method, []).append(res)
            _timer_end(t0, method)
    if "ppcl" in results and "ppcl_rand" in results:
        pa = [r["avg_acc"] for r in results["ppcl"]]
        ra = [r["avg_acc"] for r in results["ppcl_rand"]]
        print("\nPPCL vs PPCL-rand (kNN avg_acc):")
        print(f"  ppcl     : {np.mean(pa):.1f}% +/- {np.std(pa):.1f}%")
        print(f"  ppcl_rand: {np.mean(ra):.1f}% +/- {np.std(ra):.1f}%")
        try:
            print(f"  paired_stats: {paired_stats(pa, ra)}")
        except Exception:
            pass
        print(f"  sig_test: {sig_test(pa, ra)}")
    return results


# --- Phase 13: BYOL tau sweep (20 tasks) ---
def phase13_byol_tau_20t(taus=(0.99, 0.996, 0.999), seeds=(0, 1, 2, 3, 4)):
    """BYOL tau sweep on long-horizon sequences (20 tasks)."""
    print("=" * 60)
    print("PHASE 13 -- BYOL tau sweep on 20 tasks")
    print("=" * 60)
    n_tasks = 20
    results = {}
    for tau in taus:
        for seed in seeds:
            tag = f"phase13_byol20_tau{tau}"
            cached = _cached("byol", seed, tag)
            if cached is not None:
                results.setdefault(tau, []).append(cached)
                continue
            print(f"\nRunning byol tau={tau} seed={seed}...")
            t0 = _timer_start()
            res, _ = _run_and_save("byol", seed, tag, n_tasks=n_tasks, byol_mom=tau)
            results.setdefault(tau, []).append(res)
            _timer_end(t0, f"byol tau={tau}")
    print("\nBYOL tau results (kNN avg_acc):")
    for tau in taus:
        accs = [r["avg_acc"] for r in results.get(tau, [])]
        if accs:
            print(f"  tau={tau:<6} -> {np.mean(accs):.1f}% +/- {np.std(accs):.1f}%")
    return results


# --- Phase NeurIPS: BYOL bridge ---
def phase_neurips_byol_bridge(seeds=(0, 1, 2), explicit_lambdas=(1e-4, 5e-4, 1e-3)):
    """SimCLR vs explicit L2-to-EMA-shadow vs BYOL stability ablation."""
    print("=" * 60)
    print("PHASE NeurIPS -- BYOL vs explicit EMA L2 (theory ablation)")
    print("=" * 60)
    results = {}
    for method, kwargs in [("simclr", {}), ("byol", {})]:
        for seed in seeds:
            tag = "phase_neurips_byol_bridge"
            cached = _cached(method, seed, tag)
            if cached is not None:
                results[f"{method}_s{seed}"] = cached
                continue
            print(f"\n{method} seed={seed}...")
            t0 = _timer_start()
            res, _ = _run_and_save(method, seed, tag, **kwargs)
            results[f"{method}_s{seed}"] = res
            _timer_end(t0, method)
    for lam in explicit_lambdas:
        lam_slug = ("%g" % float(lam)).replace(".", "p").replace("+", "").replace("e", "e")
        for seed in seeds:
            tag = f"phase_neurips_byol_bridge_lam{lam_slug}"
            cached = _cached("simclr_ema_l2", seed, tag)
            if cached is not None:
                results[f"simclr_ema_l2_lam{lam}_s{seed}"] = cached
                continue
            print(f"\nsimclr_ema_l2 lam={lam} seed={seed}...")
            t0 = _timer_start()
            res, _ = _run_and_save("simclr_ema_l2", seed, tag, explicit_ema_l2_lambda=float(lam))
            results[f"simclr_ema_l2_lam{lam}_s{seed}"] = res
            _timer_end(t0, "simclr_ema_l2")
    print("\n--- Summary (mean avg_acc %) ---")
    by_m = {}
    for k, r in results.items():
        by_m.setdefault(r.get("method", "?"), []).append(r.get("avg_acc", 0.0))
    for m in sorted(by_m.keys()):
        v = by_m[m]
        print(f"  {m:<16} {np.mean(v):.1f} +/- {np.std(v):.1f}  (n={len(v)})")
    return results


def phase16_ppcl_vs_stable(best_r=0.10, seeds=(0, 1, 2, 3, 4)):
    """Head-to-head: vanilla PPCL vs PPCL-Stable (EMA subspace + decorrelation)."""
    print("=" * 60)
    print("PHASE 16 -- PPCL vs PPCL-Stable")
    print("=" * 60)
    tag = f"phase16_ppcl_vs_stable_r{best_r:.2f}"
    methods = [
        ("ppcl", {"ppcl_reserve": best_r}),
        ("ppcl_stable", {"ppcl_reserve": best_r}),
    ]
    results = {}
    for method, kwargs in methods:
        for seed in seeds:
            cached = _cached(method, seed, tag)
            if cached is not None:
                results.setdefault(method, []).append(cached)
                continue
            print(f"\nRunning {method} seed={seed}...")
            t0 = _timer_start()
            res, _ = _run_and_save(method, seed, tag, **kwargs)
            results.setdefault(method, []).append(res)
            _timer_end(t0, method)
    if "ppcl" in results and "ppcl_stable" in results:
        pa = [r["avg_acc"] for r in results["ppcl"]]
        sa = [r["avg_acc"] for r in results["ppcl_stable"]]
        fa = [r.get("fwd_transfer", 0.0) for r in results["ppcl"]]
        fb = [r.get("fwd_transfer", 0.0) for r in results["ppcl_stable"]]
        print("\nPPCL vs PPCL-Stable (kNN avg_acc):")
        print(f"  ppcl        : {np.mean(pa):.1f}% +/- {np.std(pa):.1f}%")
        print(f"  ppcl_stable : {np.mean(sa):.1f}% +/- {np.std(sa):.1f}%")
        print(f"  FWT ppcl={np.mean(fa):.1f}% ppcl_stable={np.mean(fb):.1f}%")
        print(f"  sig_test(acc): {sig_test(pa, sa)}")
    return results


def run_supplementary(best_r=0.10, seeds=(0, 1, 2, 3, 4), byol_taus=(0.99, 0.996, 0.999)):
    """Run all non-blocker supplementary experiments."""
    print("=" * 60)
    print("RUN SUPPLEMENTARY -- non-blockers")
    print("=" * 60)
    p12 = phase12_ppcl_random_dims(best_r=best_r, seeds=seeds, n_tasks=20)
    p13 = phase13_byol_tau_20t(taus=byol_taus, seeds=seeds)
    p16 = phase16_ppcl_vs_stable(best_r=best_r, seeds=seeds)
    return {"phase12": p12, "phase13": p13, "phase16": p16}


# --- Phase 3: BYOL momentum ablation ---
def phase3(seeds=(0,)):
    """~30 min. BYOL with tau in {0.0, 0.9, 0.99, 0.996, 0.999, 1.0}."""
    print("=" * 60)
    print("PHASE 3 -- BYOL momentum ablation")
    print("=" * 60)
    taus = [0.0, 0.9, 0.99, 0.996, 0.999, 1.0]
    results = {}
    for tau in taus:
        for seed in seeds:
            tag = f"phase3_tau{tau}"
            cached = _cached("byol", seed, tag)
            if cached is not None:
                results[f"tau{tau}_s{seed}"] = cached
                continue
            print(f"\nRunning BYOL tau={tau} seed={seed}...")
            t0 = _timer_start()
            res, _ = _run_and_save("byol", seed, tag, byol_mom=tau)
            results[f"tau{tau}_s{seed}"] = res
            _timer_end(t0, f"tau={tau}")
    print("\n" + "=" * 60)
    for tau in taus:
        accs = [results[f"tau{tau}_s{s}"]["avg_acc"] for s in seeds
                if f"tau{tau}_s{s}" in results]
        if accs:
            print(f"  tau={tau:<6} -> {np.mean(accs):.1f}%")
    return results


# --- Phase 4: Reserve fraction sweep ---
def phase4(seeds=(0,)):
    """~2 hrs. PPCL with r in {0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50}."""
    print("=" * 60)
    print("PHASE 4 -- Reserve fraction sweep")
    print("=" * 60)
    reserves = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    results = {}
    for r in reserves:
        method = "simclr" if r == 0.0 else "ppcl"
        for seed in seeds:
            tag = f"phase4_r{r:.2f}"
            cached = _cached(method, seed, tag)
            if cached is not None:
                results[f"r{r:.2f}_s{seed}"] = cached
                continue
            print(f"\nRunning r={r:.2f} seed={seed}...")
            t0 = _timer_start()
            kw = {"ppcl_reserve": r} if method == "ppcl" else {}
            res, _ = _run_and_save(method, seed, tag, **kw)
            results[f"r{r:.2f}_s{seed}"] = res
            _timer_end(t0, f"r={r}")
    print("\n" + "=" * 60)
    for r in reserves:
        accs = [results[f"r{r:.2f}_s{s}"]["avg_acc"] for s in seeds
                if f"r{r:.2f}_s{s}" in results]
        if accs:
            print(f"  r={r:<5} -> {np.mean(accs):.1f}%")
    return results


# --- Phase 5: TinyImageNet ---
def phase5(best_r=0.10, seeds=(0, 1, 2, 3, 4)):
    """~8 hrs. Key methods on TinyImageNet (200 classes, 20x10 tasks)."""
    print("=" * 60)
    print("PHASE 5 -- TinyImageNet benchmark")
    print("=" * 60)
    _download_tinyimagenet()
    tag = "phase5_tiny"
    methods = [
        ("simclr", {}),
        ("ppcl", {"ppcl_reserve": best_r}),
        ("cassle", {}),
        ("byol", {}),
        ("oracle", {}),
    ]
    results = {}
    for method, kwargs in methods:
        for seed in seeds:
            cached = _cached(method, seed, tag)
            if cached is not None:
                results[f"{method}_s{seed}"] = cached
                continue
            print(f"\nRunning {method} seed={seed} (TinyImageNet)...")
            t0 = _timer_start()
            res, _ = _run_and_save(method, seed, tag,
                                   dataset_name="tinyimagenet", n_tasks=20, **kwargs)
            results[f"{method}_s{seed}"] = res
            _timer_end(t0, method)
    return results


# --- Phase 6: 20-task CIFAR-100 ---
def phase6(best_r=0.10, seeds=(0, 1, 2, 3, 4)):
    """~2 hrs. 20 tasks x 5 classes. SimCLR, BYOL, PPCL with sig tests."""
    print("=" * 60)
    print("PHASE 6 -- 20-task CIFAR-100 (high-risk)")
    print("=" * 60)
    tag = "phase6_20t"
    methods = [
        ("simclr", {}),
        ("byol", {}),
        ("ppcl", {"ppcl_reserve": best_r}),
    ]
    results = {}
    for method, kwargs in methods:
        for seed in seeds:
            cached = _cached(method, seed, tag)
            if cached is not None:
                results.setdefault(method, []).append(cached)
                continue
            print(f"\nRunning {method} seed={seed} (20 tasks)...")
            t0 = _timer_start()
            res, _ = _run_and_save(method, seed, tag, n_tasks=20, **kwargs)
            results.setdefault(method, []).append(res)
            _timer_end(t0, method)
    print("\n" + "=" * 60)
    print("PHASE 6 RESULTS -- 20-task CIFAR-100")
    print("=" * 60)
    for mn in ["simclr", "byol", "ppcl"]:
        if mn in results:
            accs = [r['avg_acc'] for r in results[mn]]
            print(f"  {mn:<12} -> {np.mean(accs):.1f}% +/- {np.std(accs):.1f}%")
    if all(mn in results for mn in ["ppcl", "simclr", "byol"]):
        pa = [r['avg_acc'] for r in results['ppcl']]
        sa = [r['avg_acc'] for r in results['simclr']]
        ba = [r['avg_acc'] for r in results['byol']]
        print(f"\nPPCL vs SimCLR: {sig_test(pa, sa)}")
        print(f"PPCL vs BYOL:   {sig_test(pa, ba)}")
        print(f"Delta(PPCL-SimCLR): {np.mean(pa)-np.mean(sa):+.2f}pp")
        print(f"Delta(PPCL-BYOL):   {np.mean(pa)-np.mean(ba):+.2f}pp")
        print(f"Stats(PPCL vs SimCLR): {paired_stats(pa, sa)}")
        print(f"Stats(PPCL vs BYOL):   {paired_stats(pa, ba)}")
    summarize_methods(results)
    validate_phase_results(results, expected_methods=["simclr", "byol", "ppcl"],
                           expected_seeds=seeds, strict=False)
    return results


# --- Phase 7: Transfer learning ---
def phase7(seeds=(0,)):
    """~2 hrs. Linear probe on CIFAR-10 and STL-10 from Phase 2 checkpoints."""
    print("=" * 60)
    print("PHASE 7 -- Transfer learning")
    print("=" * 60)
    tag = "phase7_transfer"
    methods = ["simclr", "simclr_ema_l2", "ppcl", "ppcl_stable", "ppcl_mom", "byol", "cassle"]
    results = {}
    for method in methods:
        for seed in seeds:
            cached = _cached(method, seed, tag)
            if cached is not None:
                results[f"{method}_s{seed}"] = cached
                continue
            is_ema = method in ("byol", "ppcl_mom")
            is_pred = method in ("byol", "ppcl_mom", "simsiam")
            model = SSLModel(use_pred=is_pred, use_ema=is_ema).to(DEVICE)
            loaded = load_checkpoint(model, method, seed, "phase2")
            if not loaded:
                print(f"  No Phase 2 checkpoint for {method} seed={seed}. Retraining...")
                _, model = run_method(method, seed=seed, save_ckpt=True, ckpt_tag="phase2")
            print(f"\nTransfer eval for {method} seed={seed}...")
            c10_acc = transfer_eval(model, "cifar10")
            stl_acc = transfer_eval(model, "stl10")
            p2 = _cached(method, seed, "phase2")
            cl_acc = p2['avg_acc'] if p2 else 0.0
            result = {
                'method': method, 'seed': seed, 'avg_acc': cl_acc,
                'cifar10_transfer': c10_acc, 'stl10_transfer': stl_acc,
                'forgetting': 0.0, 'fwd_transfer': 0.0, 'acc_matrix': [],
                'sranks': [], 'eranks': [], 'uniforms': [], 'lsis': [],
                'ppcl_eranks': [], 'cka_drifts': [], 'grad_aligns': [],
                'plast_ratios': [], 'time_min': 0, 'n_params': 0, 'extra_params': 0,
            }
            save_result(result, tag)
            results[f"{method}_s{seed}"] = result
    print(f"\n{'Method':<16} {'CL Acc':>8} {'CIFAR-10':>10} {'STL-10':>10}")
    print("-" * 46)
    for k, r in sorted(results.items()):
        print(f"{r['method']:<16} {r.get('avg_acc', 0):>7.1f}% "
              f"{r.get('cifar10_transfer', 0):>9.1f}% "
              f"{r.get('stl10_transfer', 0):>9.1f}%")
    return results


# --- Phase 8: Sensitivity ---
def phase8(best_r=0.10, seeds=(0,)):
    """~3 hrs. Grid search over batch_size x lr."""
    print("=" * 60)
    print("PHASE 8 -- Sensitivity analysis")
    print("=" * 60)
    batch_sizes = [128, 256, 512]
    lrs = [1e-4, 3e-4, 1e-3]
    results = {}
    for bs in batch_sizes:
        for lr in lrs:
            for method in ["simclr", "ppcl"]:
                for seed in seeds:
                    tag = f"phase8_bs{bs}_lr{lr}"
                    k = _key(method, seed, tag)
                    cached = _cached(method, seed, tag)
                    if cached is not None:
                        results[k] = cached
                        continue
                    print(f"\n{method} bs={bs} lr={lr}...")
                    t0 = _timer_start()
                    kw = {"ppcl_reserve": best_r} if method == "ppcl" else {}
                    res, _ = _run_and_save(method, seed, tag,
                                          batch_size=bs, lr=lr, epochs=30, **kw)
                    results[k] = res
                    _timer_end(t0, f"{method} bs={bs}")
    print(f"\n{'Config':<24} {'SimCLR':>10} {'PPCL':>10} {'Delta':>8}")
    print("-" * 54)
    for bs in batch_sizes:
        for lr in lrs:
            sk = _key("simclr", 0, f"phase8_bs{bs}_lr{lr}")
            pk = _key("ppcl", 0, f"phase8_bs{bs}_lr{lr}")
            sa = results.get(sk, {}).get('avg_acc', 0)
            pa = results.get(pk, {}).get('avg_acc', 0)
            print(f"  bs={bs} lr={lr:<8} {sa:>9.1f}% {pa:>9.1f}% {pa-sa:>+7.1f}")
    return results


# --- Phase 9: Task order permutation ---
def phase9(best_r=0.10, order_seeds=(0, 100, 200)):
    """~1.5 hrs. 3 different random task orderings."""
    print("=" * 60)
    print("PHASE 9 -- Task order permutation")
    print("=" * 60)
    results = {}
    for os_seed in order_seeds:
        for method in ["simclr", "ppcl", "byol"]:
            tag = f"phase9_order{os_seed}"
            cached = _cached(method, 0, tag)
            if cached is not None:
                results.setdefault(method, []).append(cached)
                continue
            print(f"\n{method} order_seed={os_seed}...")
            t0 = _timer_start()
            kw = {"ppcl_reserve": best_r} if method == "ppcl" else {}
            # Vary task order only; keep optimization seed fixed
            res, _ = run_method(method, seed=0, task_seed=os_seed, **kw)
            res['method'] = method
            res['seed'] = 0
            res['task_seed'] = os_seed
            save_result(res, tag)
            results.setdefault(method, []).append(res)
            _timer_end(t0, method)
    print(f"\n{'Method':<12} {'Mean':>8} {'Std':>8}")
    print("-" * 30)
    for mn in ["simclr", "ppcl", "byol"]:
        if mn in results:
            accs = [r['avg_acc'] for r in results[mn]]
            print(f"  {mn:<10} {np.mean(accs):>7.1f}% {np.std(accs):>7.1f}%")
    return results


# --- Phase 10: Task granularity ---
def phase10(best_r=0.10, seeds=(0,)):
    """~1 hr. Splits: 20x5, 10x10, 5x20, 2x50."""
    print("=" * 60)
    print("PHASE 10 -- Task granularity")
    print("=" * 60)
    configs = [(20, 5), (10, 10), (5, 20), (2, 50)]
    results = {}
    for nt, cpt in configs:
        for method in ["simclr", "ppcl", "byol"]:
            for seed in seeds:
                tag = f"phase10_{nt}t{cpt}c"
                k = _key(method, seed, tag)
                cached = _cached(method, seed, tag)
                if cached is not None:
                    results[k] = cached
                    continue
                print(f"\n{method} {nt}Tx{cpt}C seed={seed}...")
                t0 = _timer_start()
                kw = {"ppcl_reserve": best_r} if method == "ppcl" else {}
                res, _ = _run_and_save(method, seed, tag, n_tasks=nt, **kw)
                results[k] = res
                _timer_end(t0, f"{method} {nt}T")
    return results


# --- Phase 11: Online vs offline ---
def phase11(best_r=0.10, seeds=(0,)):
    """~30 min. 1 epoch (online) vs 50 epochs (offline)."""
    print("=" * 60)
    print("PHASE 11 -- Online vs offline")
    print("=" * 60)
    epoch_configs = [1, 50]
    results = {}
    for ep in epoch_configs:
        for method in ["simclr", "ppcl", "byol"]:
            for seed in seeds:
                tag = f"phase11_ep{ep}"
                k = _key(method, seed, tag)
                cached = _cached(method, seed, tag)
                if cached is not None:
                    results[k] = cached
                    continue
                print(f"\n{method} epochs={ep} seed={seed}...")
                t0 = _timer_start()
                kw = {"ppcl_reserve": best_r} if method == "ppcl" else {}
                res, _ = _run_and_save(method, seed, tag, epochs=ep, **kw)
                results[k] = res
                _timer_end(t0, f"{method} ep={ep}")
    print(f"\n{'Method':<12} {'1 epoch':>10} {'50 epochs':>12}")
    print("-" * 36)
    for mn in ["simclr", "ppcl", "byol"]:
        a1 = results.get(_key(mn, 0, "phase11_ep1"), {}).get('avg_acc', 0)
        a50 = results.get(_key(mn, 0, "phase11_ep50"), {}).get('avg_acc', 0)
        print(f"  {mn:<10} {a1:>9.1f}% {a50:>11.1f}%")
    return results


# --- Aggregate runners ---
def run_all_phases(best_r=None):
    """Phases 1-6 in sequence. ~18 hours total."""
    r_sim, r_ppcl, best_r_found = phase1()
    if best_r is None:
        best_r = best_r_found
    if r_ppcl['avg_acc'] <= r_sim['avg_acc']:
        print("\nPPCL did not beat SimCLR. Stopping.")
        return {"phase1": (r_sim, r_ppcl, best_r)}
    p6 = phase6(best_r)
    p2 = phase2(best_r)
    p3 = phase3()
    p4 = phase4()
    p5 = phase5(best_r)
    out = {
        "phase1": (r_sim, r_ppcl, best_r),
        "phase2": p2, "phase3": p3,
        "phase4": p4, "phase5": p5, "phase6": p6,
    }
    export_results_csv(p2, out_name="phase2_main_table.csv")
    return out


def run_main_track(best_r=0.1):
    """Phases 7-11. ~8 hours total."""
    p7 = phase7()
    p8 = phase8(best_r)
    p9 = phase9(best_r)
    p10 = phase10(best_r)
    p11 = phase11(best_r)
    return {"phase7": p7, "phase8": p8, "phase9": p9, "phase10": p10, "phase11": p11}


# --- NeurIPS v2 phases ---
def phase_ppcl_fix_grid(seeds=(0,)):
    """P0-A: 6x6 grid of lambda x r on CIFAR-100, 10 tasks.
    Returns (results, best_lam, best_r).
    """
    print("=" * 60)
    print("PHASE P0A -- PPCL lambda x r grid")
    print("=" * 60)
    lambdas = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    reserves = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    results = {}
    best_acc, best_lam, best_r = -1, None, None
    for lam in lambdas:
        for r in reserves:
            for seed in seeds:
                tag = f"p0a_lam{lam}_r{r:.2f}"
                cached = _cached("ppcl", seed, tag)
                if cached is not None:
                    results[f"lam{lam}_r{r}_s{seed}"] = cached
                    if cached['avg_acc'] > best_acc:
                        best_acc = cached['avg_acc']
                        best_lam = lam
                        best_r = r
                    continue
                print(f"\n  PPCL lam={lam} r={r} seed={seed}...")
                t0 = _timer_start()
                res, _ = _run_and_save("ppcl", seed, tag, ppcl_reserve=r, ppcl_lam=lam)
                results[f"lam{lam}_r{r}_s{seed}"] = res
                if res['avg_acc'] > best_acc:
                    best_acc = res['avg_acc']
                    best_lam = lam
                    best_r = r
                _timer_end(t0, f"lam={lam} r={r}")
    sim = _cached("simclr", 0, "p0a_baseline")
    if sim is None:
        sim, _ = _run_and_save("simclr", 0, "p0a_baseline")
    print(f"\nBest PPCL: lam={best_lam}, r={best_r} -> {best_acc:.1f}%")
    print(f"SimCLR baseline: {sim['avg_acc']:.1f}%")
    gap = best_acc - sim['avg_acc']
    if gap > 0:
        print(f"  PPCL beats SimCLR by {gap:.1f}pp")
    else:
        print(f"  PPCL loses to SimCLR by {-gap:.1f}pp -- proceed to soft routing")
    return results, best_lam, best_r


def phase_ppcl_soft_routing(best_lam=0.05, seeds=(0, 1, 2, 3, 4)):
    """P0-A continued: Soft-PPCL + warmup variants vs SimCLR."""
    print("=" * 60)
    print("PHASE P0A-SOFT -- Soft PPCL + warmup evaluation")
    print("=" * 60)
    tag = "p0a_soft"
    methods_configs = [
        ("simclr", {}, "simclr_baseline"),
        ("ppcl_soft", {"ppcl_lam": best_lam}, "soft_ppcl"),
        ("ppcl_soft", {"ppcl_lam": best_lam, "ppcl_warmup_tasks": 1}, "soft_warmup"),
        ("ppcl_soft", {"ppcl_lam": best_lam, "ppcl_warmup_tasks": 1,
                       "ppcl_lam_ramp_epochs": 10}, "soft_warmup_ramp"),
        ("ppcl_enc", {"ppcl_lam": best_lam}, "encoder_ppcl"),
    ]
    results = {}
    for method, kwargs, slug in methods_configs:
        for seed in seeds:
            mtag = f"{tag}_{slug}"
            cached = _cached(method, seed, mtag)
            if cached is not None:
                results.setdefault(slug, []).append(cached)
                continue
            print(f"\n  {slug} seed={seed}...")
            t0 = _timer_start()
            res, _ = _run_and_save(method, seed, mtag, **kwargs)
            results.setdefault(slug, []).append(res)
            _timer_end(t0, slug)
    for slug, runs in results.items():
        accs = [r['avg_acc'] for r in runs]
        m, s, _ = _mean_std_ci(accs)
        print(f"  {slug:<25} -> {m:.1f}% +/- {s:.1f}%")
    return results


def phase_byol_disentangle(seeds=(0, 1, 2, 3, 4)):
    """P0-B: 6-method BYOL component attribution table.
    Isolates EMA, predictor, and no-negatives independently.
    """
    print("=" * 60)
    print("PHASE P0B -- BYOL disentangling ablation")
    print("=" * 60)
    tag = "p0b_byol_disentangle"
    methods = [
        ("simclr", {}),
        ("simclr_ema_l2", {}),
        ("simclr_pred", {}),
        ("simclr_ema_l2_pred", {}),
        ("byol_no_pred", {}),
        ("byol", {}),
    ]
    results = {}
    for method, kwargs in methods:
        for seed in seeds:
            cached = _cached(method, seed, tag)
            if cached is not None:
                results.setdefault(method, []).append(cached)
                continue
            print(f"\n  {method} seed={seed}...")
            t0 = _timer_start()
            res, _ = _run_and_save(method, seed, tag, **kwargs)
            results.setdefault(method, []).append(res)
            _timer_end(t0, method)
    print("\n" + "=" * 60)
    print("BYOL DISENTANGLING RESULTS")
    print("=" * 60)
    component_map = {
        "simclr":             ("No",  "No",  "Yes"),
        "simclr_ema_l2":      ("Yes", "No",  "Yes"),
        "simclr_pred":        ("No",  "Yes", "Yes"),
        "simclr_ema_l2_pred": ("Yes", "Yes", "Yes"),
        "byol_no_pred":       ("Yes", "No",  "No"),
        "byol":               ("Yes", "Yes", "No"),
    }
    print(f"{'Method':<22} {'EMA':>5} {'Pred':>5} {'Neg':>5} {'Acc':>12} {'Forget':>12}")
    print("-" * 65)
    for method, _ in methods:
        if method in results:
            accs = [r['avg_acc'] for r in results[method]]
            fgts = [r.get('forgetting', 0) for r in results[method]]
            ema, pred, neg = component_map[method]
            m_a, s_a, _ = _mean_std_ci(accs)
            m_f, s_f, _ = _mean_std_ci(fgts)
            print(f"  {method:<20} {ema:>5} {pred:>5} {neg:>5} "
                  f"{m_a:>6.1f}+/-{s_a:.1f} {m_f:>6.1f}+/-{s_f:.1f}")
    if "simclr" in results and "simclr_ema_l2" in results:
        sa = [r['avg_acc'] for r in results['simclr']]
        ea = [r['avg_acc'] for r in results['simclr_ema_l2']]
        print(f"\nEMA contribution: SimCLR->SimCLR+EMA = {np.mean(ea)-np.mean(sa):+.1f}pp "
              f"({sig_test(sa, ea)})")
    if "simclr" in results and "simclr_pred" in results:
        sa = [r['avg_acc'] for r in results['simclr']]
        pa = [r['avg_acc'] for r in results['simclr_pred']]
        print(f"Pred contribution: SimCLR->SimCLR+Pred = {np.mean(pa)-np.mean(sa):+.1f}pp "
              f"({sig_test(sa, pa)})")
    if "simclr" in results and "byol" in results:
        sa = [r['avg_acc'] for r in results['simclr']]
        ba = [r['avg_acc'] for r in results['byol']]
        print(f"Full BYOL gap: SimCLR->BYOL = {np.mean(ba)-np.mean(sa):+.1f}pp "
              f"({sig_test(sa, ba)})")
    return results


def phase_dose_response(seeds=(0, 1, 2, 3, 4)):
    """P0-D: BYOL tau sweep + SimCLR-EMA-L2 lambda sweep for dose-response curves."""
    print("=" * 60)
    print("PHASE P0D -- Dose-response curves")
    print("=" * 60)
    taus = [0.0, 0.5, 0.9, 0.95, 0.99, 0.996, 0.999, 1.0]
    tau_results = {}
    for tau in taus:
        for seed in seeds:
            tag = f"p0d_tau{tau}"
            cached = _cached("byol", seed, tag)
            if cached is not None:
                tau_results.setdefault(tau, []).append(cached)
                continue
            print(f"\n  BYOL tau={tau} seed={seed}...")
            t0 = _timer_start()
            res, _ = _run_and_save("byol", seed, tag, byol_mom=tau)
            tau_results.setdefault(tau, []).append(res)
            _timer_end(t0, f"tau={tau}")
    lambdas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    lam_results = {}
    for lam in lambdas:
        for seed in seeds:
            lam_slug = f"{lam:.0e}".replace("+", "").replace("-", "m")
            tag = f"p0d_ema_lam{lam_slug}"
            cached = _cached("simclr_ema_l2", seed, tag)
            if cached is not None:
                lam_results.setdefault(lam, []).append(cached)
                continue
            print(f"\n  SimCLR-EMA-L2 lam={lam} seed={seed}...")
            t0 = _timer_start()
            res, _ = _run_and_save("simclr_ema_l2", seed, tag, explicit_ema_l2_lambda=lam)
            lam_results.setdefault(lam, []).append(res)
            _timer_end(t0, f"lam={lam}")
    print("\n--- BYOL tau dose-response ---")
    for tau in taus:
        if tau in tau_results:
            accs = [r['avg_acc'] for r in tau_results[tau]]
            m, s, _ = _mean_std_ci(accs)
            print(f"  tau={tau:<6} -> {m:.1f}% +/- {s:.1f}%")
    print("\n--- SimCLR-EMA-L2 lambda dose-response ---")
    for lam in lambdas:
        if lam in lam_results:
            accs = [r['avg_acc'] for r in lam_results[lam]]
            m, s, _ = _mean_std_ci(accs)
            print(f"  lam={lam:<8.0e} -> {m:.1f}% +/- {s:.1f}%")
    return {"tau_results": tau_results, "lam_results": lam_results}


def phase_imagenet100(best_r=0.10, best_lam=0.05, seeds=(0, 1, 2)):
    """P1-C: Key methods on ImageNet-100 (224x224, 100 classes, 10x10 tasks)."""
    print("=" * 60)
    print("PHASE P1C -- ImageNet-100 benchmark")
    print("=" * 60)
    tag = "p1c_imagenet100"
    methods = [
        ("simclr", {}),
        ("byol", {}),
        ("simclr_ema_l2", {}),
        ("ppcl", {"ppcl_reserve": best_r, "ppcl_lam": best_lam}),
        ("ppcl_mom", {"ppcl_reserve": best_r, "ppcl_lam": best_lam}),
        ("replay_ssl", {}),
        ("cassle", {}),
    ]
    results = {}
    for method, kwargs in methods:
        for seed in seeds:
            cached = _cached(method, seed, tag)
            if cached is not None:
                results.setdefault(method, []).append(cached)
                continue
            print(f"\n  {method} seed={seed} (ImageNet-100)...")
            t0 = _timer_start()
            res, _ = _run_and_save(
                method, seed, tag,
                dataset_name="imagenet100", n_tasks=10,
                epochs=100, batch_size=256,
                **kwargs,
            )
            results.setdefault(method, []).append(res)
            _timer_end(t0, method)
    summarize_methods(results)
    return results


def compute_all_statistics(results=None, output_path=None):
    """Wilcoxon signed-rank for all pairwise method comparisons.
    Saves to statistics.json for traceability.
    """
    from itertools import combinations

    if results is None:
        results = load_all_results("*.json")
    by_method = {}
    for k, r in results.items():
        m = r.get('method', '?')
        by_method.setdefault(m, []).append(r.get('avg_acc', 0))
    stats = {}
    for (m1, m2) in combinations(sorted(by_method.keys()), 2):
        a1, a2 = by_method[m1], by_method[m2]
        if len(a1) >= 3 and len(a2) >= 3 and len(a1) == len(a2):
            ps = paired_stats(a1, a2)
            stats[f"{m1}_vs_{m2}"] = ps
    if output_path is None:
        output_path = os.path.join(SAVE_DIR, "statistics.json")
    _atomic_json_write(stats, output_path)
    print(f"  Saved {len(stats)} pairwise tests to {output_path}")
    return stats


def run_neurips_pipeline(best_r=None, best_lam=None):
    """Full NeurIPS experiment pipeline in dependency order with decision gates."""
    print("\n" + "=" * 70)
    print("STAGE 1: PPCL FIX PROTOCOL (P0-A)")
    print("=" * 70)
    grid_results, grid_lam, grid_r = phase_ppcl_fix_grid()
    soft_results = phase_ppcl_soft_routing(best_lam=grid_lam or 0.05)

    if best_r is None:
        best_r = grid_r or 0.10
    if best_lam is None:
        best_lam = grid_lam or 0.05

    print("\n" + "=" * 70)
    print("STAGE 2: BYOL DISENTANGLING (P0-B)")
    print("=" * 70)
    byol_results = phase_byol_disentangle()

    print("\n" + "=" * 70)
    print("STAGE 3: FULL BENCHMARK (P0-C)")
    print("=" * 70)
    p2 = phase2(best_r=best_r)

    print("\n" + "=" * 70)
    print("STAGE 4: DOSE-RESPONSE (P0-D)")
    print("=" * 70)
    dose = phase_dose_response()

    print("\n" + "=" * 70)
    print("STAGE 5: PPCL-RAND ABLATION (P1-A)")
    print("=" * 70)
    p12 = phase12_ppcl_random_dims(best_r=best_r)

    print("\n" + "=" * 70)
    print("STAGE 6: TINYIMAGENET (P1-B)")
    print("=" * 70)
    p5 = phase5(best_r=best_r, seeds=(0, 1, 2))

    print("\n" + "=" * 70)
    print("STAGE 7: IMAGENET-100 (P1-C)")
    print("=" * 70)
    in100 = phase_imagenet100(best_r=best_r, best_lam=best_lam)

    print("\n" + "=" * 70)
    print("STAGE 8: SECONDARY ABLATIONS (P2)")
    print("=" * 70)
    p8 = phase8(best_r=best_r)
    p9 = phase9(best_r=best_r)
    p10 = phase10(best_r=best_r)
    p6 = phase6(best_r=best_r)
    p11 = phase11(best_r=best_r)
    p7 = phase7()

    print("\n" + "=" * 70)
    print("STAGE 9: STATISTICS + FIGURES (P3)")
    print("=" * 70)
    all_results = load_all_results("*.json")
    compute_all_statistics(all_results)
    export_results_csv(p2, out_name="neurips_main_table.csv")

    print("\n" + "=" * 70)
    print("NEURIPS PIPELINE COMPLETE")
    print(f"Best PPCL: lam={best_lam}, r={best_r}")
    print("=" * 70)

    return {
        "p0a_grid": grid_results,
        "p0a_soft": soft_results,
        "p0b_byol": byol_results,
        "p0c_benchmark": p2,
        "p0d_dose": dose,
        "p1a_rand": p12,
        "p1b_tiny": p5,
        "p1c_in100": in100,
        "p2_sensitivity": p8,
        "p2_order": p9,
        "p2_granularity": p10,
        "p2_stress": p6,
        "p2_online": p11,
        "p2_transfer": p7,
    }


print("[experiments] All phase functions loaded.")
