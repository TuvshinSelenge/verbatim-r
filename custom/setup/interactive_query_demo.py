#!/usr/bin/env python3
"""
Pick a bank → scope retrieval to that PDF + use that bank's span JSON for suites.

Repo root:
    PYTHONPATH=. python -m custom.setup.interactive_query_demo
    PYTHONPATH=. python -m custom.setup.interactive_query_demo --db-path custom/milvus_verbatim_new.db --bank bawag

List banks / indexed PDF mapping:
    PYTHONPATH=. python -m custom.setup.interactive_query_demo --list-banks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

_SETUP_DIR = Path(__file__).resolve().parent
_CUSTOM_ROOT = _SETUP_DIR.parent
_REPO_ROOT = _CUSTOM_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from custom.setup.bank_context import (
    get_profile_by_id,
    load_bank_profiles,
    pick_scope_for_profile,
    span_path_for_profile,
)
from custom.setup.connect_index import connect_to_index
from custom.setup.paths import resolve_custom_milvus_db_path
from custom.setup.report_scope import (
    ReportScope,
    list_indexed_source_records,
    list_report_scopes,
)

DEFAULT_DB = resolve_custom_milvus_db_path()


def _chunk_index_from_meta(meta: Dict[str, Any]) -> Any:
    return meta.get("chunk_index", meta.get("chunk_number"))


def _normalize_expected_chunk_idxs(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    return [raw]


def run_query(
    db_path: str,
    question: str,
    scope: Optional[ReportScope],
    k: int,
    device: str,
) -> None:
    filt = scope.filter_expr() if scope and scope.meta_value else None

    print()
    print(f"Question: {question}")
    if scope and scope.meta_value:
        print(f"Scope:    {scope.label}")
    else:
        print("Scope:    all reports")
    print("-" * 60)

    rag_index, _ = connect_to_index(db_path=db_path, device=device, verbose=False)
    results = rag_index.query(text=question, k=k, filter=filt)

    if not results:
        print("(no results)")
        return

    for i, r in enumerate(results, 1):
        meta = getattr(r, "metadata", {}) or {}
        src = meta.get("source_file") or meta.get("source") or "?"
        page = meta.get("page", meta.get("page_number", "?"))
        cidx = _chunk_index_from_meta(meta)
        score = getattr(r, "score", None)
        text = getattr(r, "text", None) or ""
        preview = text[:700] + ("…" if len(text) > 700 else "")

        print(f"\n[{i}] score={score} | chunk={cidx} | {src} | page={page}\n{preview}")


def run_span_suite(
    db_path: str,
    span_path: Path,
    scope: Optional[ReportScope],
    k: int,
    device: str,
    skip_sentinel_1300: bool,
    check_chunk_hits: bool,
) -> None:
    items = json.loads(span_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("Expected JSON array")

    filt = scope.filter_expr() if scope and scope.meta_value else None
    print(f"\nSpan suite: {span_path.name} ({len(items)} items)")
    if scope and scope.meta_value:
        print(f"PDF scope: {scope.label}")
    print("-" * 60)

    rag_index, _ = connect_to_index(db_path=db_path, device=device, verbose=False)
    hits = 0
    total_eval = 0

    for i, item in enumerate(items, 1):
        if not isinstance(item, dict) or "query" not in item:
            continue
        q = item["query"]
        expected = _normalize_expected_chunk_idxs(item.get("expected_chunk_index"))
        if skip_sentinel_1300 and expected and all(x == 1300 for x in expected):
            print(f"[{i}] (skip sentinel 1300) {q[:70]}…")
            continue

        results = rag_index.query(text=q, k=k, filter=filt)

        gold = None
        if check_chunk_hits and expected:
            gold = set(expected)
            total_eval += 1
            retrieved = {_chunk_index_from_meta(getattr(r, "metadata", {}) or {}) for r in results}
            retrieved.discard(None)
            ok = bool(gold & retrieved)
            hits += int(ok)
            flag = "HIT" if ok else "miss"
        else:
            flag = "—"

        print(f"\n[{i}] {flag}  {q[:100]}{'…' if len(q) > 100 else ''}")
        for j, r in enumerate(results[: min(5, len(results))], 1):
            meta = getattr(r, "metadata", {}) or {}
            cidx = _chunk_index_from_meta(meta)
            score = getattr(r, "score", None)
            print(f"    {j}) chunk={cidx} score={score}")

    if check_chunk_hits and total_eval:
        print(f"\n--- chunk-id hit rate (top-{k}): {hits}/{total_eval} = {hits/total_eval:.1%} ---")


def print_bank_menu(db_path: str) -> None:
    profiles = load_bank_profiles()
    try:
        scopes = list_report_scopes(db_path)
    except Exception as e:
        print(f"Could not list indexed PDFs: {e}")
        scopes = []

    print(f"Database: {db_path}\n")
    print("Banks (edit custom/data/bank_profiles.json to tune names / source_substrings):\n")
    for idx, p in enumerate(profiles, 1):
        bid = p.get("id", "?")
        label = p.get("label", bid)
        span = p.get("span_file", "")
        match = pick_scope_for_profile(p, scopes, db_path)
        if match and match.meta_value:
            mline = f"matched index → {match.label}"
        else:
            mline = "NO indexed PDF matched (add substrings that appear in the file path)"
        print(f"  {idx}) [{bid}] {label}")
        print(f"      spans: {span}")
        print(f"      {mline}\n")


def resolve_bank_choice(
    db_path: str,
    bank_arg: Optional[str],
    scope_index: Optional[int],
    interactive: bool,
) -> tuple[Any, Optional[ReportScope]]:
    """Return (profile dict, ReportScope or None)."""
    profiles = load_bank_profiles()
    if not profiles:
        raise SystemExit("No banks in custom/data/bank_profiles.json")

    scopes = list_report_scopes(db_path)
    known_scopes = [s for s in scopes if s.meta_value]
    env_default = os.getenv("CUSTOM_BANK_ID", "rbi")

    if bank_arg:
        prof = get_profile_by_id(bank_arg)
        if not prof:
            raise SystemExit(f"Unknown bank id {bank_arg!r}. Use --list-banks.")
        scope = pick_scope_for_profile(prof, scopes, db_path)
        return prof, scope

    if scope_index is not None:
        if scope_index == 0:
            prof = get_profile_by_id(env_default) or profiles[0]
            return prof, None
        if scope_index < 1 or scope_index > len(known_scopes):
            raise SystemExit(f"--scope-index must be 0..{len(known_scopes)}")
        scope = known_scopes[scope_index - 1]
        prof = None
        for p in profiles:
            m = pick_scope_for_profile(p, scopes, db_path)
            if (
                m
                and scope
                and m.meta_key == scope.meta_key
                and m.meta_value == scope.meta_value
            ):
                prof = p
                break
        if prof is None:
            prof = get_profile_by_id(env_default) or profiles[0]
            print(
                f"Note: PDF #{scope_index} not matched to a bank profile; "
                f"prompts use [{prof.get('id')}]. Tune source_substrings in bank_profiles.json."
            )
        return prof, scope

    if interactive:
        print_bank_menu(db_path)
        while True:
            raw = input(f"Select bank [1-{len(profiles)} / q]: ").strip().lower()
            if raw in ("q", "quit"):
                raise SystemExit(0)
            try:
                n = int(raw)
            except ValueError:
                print("Invalid.\n")
                continue
            if n < 1 or n > len(profiles):
                print("Invalid.\n")
                continue
            prof = profiles[n - 1]
            scope = pick_scope_for_profile(prof, scopes, db_path)
            return prof, scope

    prof = get_profile_by_id(env_default) or profiles[0]
    scope = pick_scope_for_profile(prof, scopes, db_path)
    return prof, scope


def main() -> int:
    parser = argparse.ArgumentParser(description="Bank-scoped retrieval + span JSON suites.")
    parser.add_argument("--db-path", default=DEFAULT_DB)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--bank", type=str, default=None, help="bank_profiles id, e.g. rbi, bawag, erste, uni")
    parser.add_argument(
        "--scope-index",
        type=int,
        default=None,
        metavar="N",
        help="Pick PDF by 1-based index from raw Milvus list (advanced); 0 = no filter",
    )
    parser.add_argument("--list-banks", action="store_true", help="Show banks and index match, exit")
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Print distinct source_file/source/dataset_id values from Milvus (for bank_profiles), exit",
    )
    parser.add_argument("--run-spans", action="store_true", help="Run that bank's span_*.json queries")
    parser.add_argument(
        "--no-chunk-check",
        action="store_true",
        help="Span suite: do not compare expected_chunk_index",
    )
    parser.add_argument(
        "--no-skip-sentinel",
        action="store_true",
        help="Include items whose only expected_chunk_index is 1300",
    )
    args = parser.parse_args()
    db_path = os.path.abspath(os.path.expanduser(str(args.db_path)))

    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    if args.list_banks:
        print_bank_menu(db_path)
        return 0

    if args.list_sources:
        try:
            rows = list_indexed_source_records(db_path)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(f"Database: {db_path}\n")
        print(f"{'field':<14} {'chunks':>8}  basename / value")
        print("-" * 80)
        for r in rows:
            fld = r["metadata_field"]
            cnt = r["chunk_count"]
            val = r["value"] or "(empty)"
            base = r.get("basename") or ""
            line = f"{fld:<14} {cnt:>8}  "
            if base:
                line += f"{base}\n{'':>14} {'':>8}  {val}"
            else:
                line += val
            print(line)
            print()
        return 0

    try:
        profile, scope = resolve_bank_choice(
            db_path,
            args.bank,
            args.scope_index,
            interactive=(args.bank is None and args.scope_index is None),
        )
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1

    bid = profile.get("id", "")
    legal = profile.get("legal_name", "")
    short = profile.get("short_name", "")
    print(f"\nActive bank: [{bid}] {legal} ({short})")
    if scope and scope.meta_value:
        print(f"Milvus filter: {scope.label}")
    else:
        print("Milvus filter: none (all chunks)")

    if args.run_spans:
        spath = span_path_for_profile(profile)
        run_span_suite(
            db_path,
            spath,
            scope,
            k=args.k,
            device=args.device,
            skip_sentinel_1300=not args.no_skip_sentinel,
            check_chunk_hits=not args.no_chunk_check,
        )
        return 0

    print("\nInteractive questions (empty line to change bank, q to quit)\n")
    while True:
        q = input("Your question: ").strip()
        if q.lower() in ("q", "quit"):
            break
        if not q:
            profile, scope = resolve_bank_choice(db_path, None, None, interactive=True)
            print(f"\nSwitched to [{profile.get('id')}] scope={scope.label if scope else 'all'}\n")
            continue
        try:
            run_query(db_path, q, scope, k=args.k, device=args.device)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
        print("\n" + "=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
