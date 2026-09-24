/**
 * Council-as-jury for settlement.
 *
 * During settlement the oracle BUYS each eligible persona's verdict via a USDC
 * payment (into the persona's own wallet through the paid /api/council/vote
 * endpoint), then tallies the votes into the on-chain verdict.
 * This is what binds the 10 personas to settlement: every juror is paid, and
 * the consensus — not a single oracle call — decides the payout.
 *
 * Best-effort: any persona that errors or times out simply abstains. If fewer
 * than `quorum` decisive votes come back, returns null so the caller falls back
 * to its own (solo) verdict — council voting never blocks a settlement.
 *
 * Self-resolving mode (opt-in via `selfResolving` config) implements the
 * mechanism from "Self-Resolving Prediction Markets for Unverifiable Outcomes"
 * (Srinivasan, Karger, Chen — arXiv:2306.04305): jurors report sequentially in
 * random order, each seeing the prior reports; the market stops with
 * probability alpha after each vote; and jurors are later scored with a
 * cross-entropy market scoring rule against the terminal (reference) report —
 * the oracle's own history-informed assessment. Informative updates toward the
 * reference earn a bonus, parroting the prior earns exactly zero.
 */

import { listCouncilPersonas, type PersonaSpec } from "../council/personas";
import { fetchWithBudget, type PayingWallet } from "../../lib/x402/buyer";
import { transferUsdc, type AgentWallet } from "../../lib/agent-wallets";
import { isAccountAddress } from "../../lib/stellar";
import { usdcToUnits } from "../../lib/usdc";
import { isVerdict, type Verdict } from "../../lib/verdict";

export type { Verdict };

export interface CouncilVote {
  slug: string;
  displayName: string;
  verdict: Verdict;
  confidence: number;
  pricePaidUnits: string | null; // USDC atomic units (7dp), null if free/unsettled
  /** Persona wallet that received the vote fee (bonus transfer target). */
  walletAddress?: string;
  /** q_t = P(CHALLENGERS_WIN) implied by this report (self-resolving mode). */
  probability?: number;
  /** Cross-entropy score vs the terminal reference report (self-resolving mode). */
  score?: number;
  /** Bonus paid out for a positive score (self-resolving mode). */
  bonusUsdc?: number;
}

export interface CouncilVerdict {
  verdict: Verdict;
  confidence: number;
  explanation: string;
  tally: { creator: number; challengers: number; draw: number; unresolvable: number; decisive: number };
  votes: CouncilVote[];
  totalPaidUnits: bigint;
  /** Sequential q_t reports, prior first implicit at Q_PRIOR (self-resolving mode). */
  qHistory?: number[];
  /** Human-readable juror reports, in voting order — fed to the terminal
   *  (reference) assessment (self-resolving mode). */
  reports?: string[];
}

export interface SelfResolvingConfig {
  /** Stop probability after each vote once `minVotes` decisive reports exist. */
  alpha: number;
  /** Decisive reports required before random termination may trigger. */
  minVotes: number;
}

/**
 * Clamp self-resolving knobs to operational bounds.
 * Malformed env / config must not pause settlement or force an infinite jury.
 */
export function normalizeSelfResolvingConfig(cfg: SelfResolvingConfig): SelfResolvingConfig {
  const rawAlpha = Number(cfg.alpha);
  const rawMin = Number(cfg.minVotes);
  const alpha = Number.isFinite(rawAlpha) ? Math.min(1, Math.max(0, rawAlpha)) : 0.25;
  const minVotes = Number.isFinite(rawMin) ? Math.max(1, Math.trunc(rawMin)) : 1;
  return { alpha, minVotes };
}

// ── Self-resolving mechanism math (pure, unit-tested) ─────────────────────────

/** Common prior. The on-chain pool ratio would be the natural prior but is
 *  manipulable by the creator's own stake, so we start neutral. */
export const Q_PRIOR = 0.5;
// Keep q away from {0,1} so the log scores stay finite.
const Q_MIN = 0.02;
const Q_MAX = 0.98;
/** Bonus shares below this are dust — skipped rather than transferred. */
export const BONUS_DUST_USDC = 0.0005;

function clampQ(q: number): number {
  // Round to 4dp so float artifacts (0.5 − 80/200 = 0.09999…8) don't leak
  // into logs, commits, and score math.
  const rounded = Math.round(q * 1e4) / 1e4;
  return Math.min(Q_MAX, Math.max(Q_MIN, rounded));
}

/**
 * Maps a verdict+confidence report to q = P(CHALLENGERS_WIN).
 * DRAW/UNRESOLVABLE carry no directional information → q stays at qPrev,
 * which makes the report's cross-entropy score exactly zero.
 */
export function verdictToProbability(verdict: Verdict, confidence: number, qPrev: number): number {
  const c = Math.max(0, Math.min(100, confidence));
  if (verdict === "CHALLENGERS_WIN") return clampQ(0.5 + c / 200);
  if (verdict === "CREATOR_WINS") return clampQ(0.5 - c / 200);
  return qPrev;
}

/**
 * Cross-entropy market scoring rule: the juror's marginal information
 * contribution, judged by the terminal reference belief qT.
 *   S = qT·ln(qt/qPrev) + (1−qT)·ln((1−qt)/(1−qPrev))
 * Zero when qt === qPrev (no update); positive for updates toward qT.
 *
 * Inputs are clamped into (Q_MIN, Q_MAX). Malformed / non-finite values
 * collapse to the prior clamp rather than minting Inf/NaN bonuses that
 * would corrupt USDC allocation after a funded settlement.
 */
export function crossEntropyScore(qT: number, qt: number, qPrev: number): number {
  const ref = clampQ(Number.isFinite(qT) ? qT : Q_PRIOR);
  const q = clampQ(Number.isFinite(qt) ? qt : Q_PRIOR);
  const prev = clampQ(Number.isFinite(qPrev) ? qPrev : Q_PRIOR);
  if (q === prev) return 0;
  return ref * Math.log(q / prev) + (1 - ref) * Math.log((1 - q) / (1 - prev));
}

/**
 * Fills each vote's `score` from its q_t against the reference report.
 * The q chain starts at Q_PRIOR; abstaining reports (probability undefined)
 * score zero and do not advance the chain.
 */
export function scoreCouncilVotes(votes: CouncilVote[], referenceQ: number): CouncilVote[] {
  const qT = clampQ(referenceQ);
  let qPrev = Q_PRIOR;
  return votes.map((v) => {
    if (v.probability === undefined) return { ...v, score: 0 };
    const score = crossEntropyScore(qT, v.probability, qPrev);
    qPrev = v.probability;
    return { ...v, score };
  });
}

/**
 * Splits `poolUsdc` proportionally across positive scores; non-positive
 * scores and dust-sized shares get nothing. May under-distribute (dust is
 * kept, never redistributed) and never exceeds the pool.
 */
export function allocateBonus(scores: number[], poolUsdc: number): number[] {
  // Non-finite pool or scores are treated as zero — a NaN bonus must never
  // reach transferUsdc after on-chain settlement.
  if (!Number.isFinite(poolUsdc) || poolUsdc <= 0) return scores.map(() => 0);
  const positives = scores.map((s) => (Number.isFinite(s) && s > 0 ? s : 0));
  const total = positives.reduce((a, b) => a + b, 0);
  if (total <= 0) return scores.map(() => 0);
  // Integer micro-USDC with a float-noise epsilon: floors guarantee the sum
  // never exceeds the pool.
  const poolMicro = Math.round(poolUsdc * 1e6);
  return positives.map((s) => {
    const shareMicro = Math.floor((poolMicro * s) / total + 1e-6);
    const share = shareMicro / 1e6;
    return share >= BONUS_DUST_USDC ? share : 0;
  });
}

/** Personas that can judge a claim: evidence-reasoning (have a promptBias) and,
 *  for specialists, only within their category. Rule-based traders abstain. */
function eligiblePersonas(category: string): PersonaSpec[] {
  const cat = category.toLowerCase();
  return listCouncilPersonas().filter(
    (p) =>
      !!p.promptBias &&
      (!p.categoryFilter || p.categoryFilter.some((c) => c.toLowerCase() === cat)),
  );
}

interface VoteResponse {
  verdict?: Verdict;
  confidence?: number;
  explanation?: string;
  paidTo?: string;
}

function shuffled<T>(items: T[]): T[] {
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

export async function gatherCouncilVerdict(args: {
  claimId: number;
  category: string;
  baseUrl: string;
  payer: PayingWallet;
  votePriceBot?: number;
  capUsdc?: number;
  quorum?: number;
  /** Enables the sequential self-resolving mechanism (see module header). */
  selfResolving?: SelfResolvingConfig;
}): Promise<CouncilVerdict | null> {
  const capUnits = usdcToUnits(args.capUsdc ?? 0.005);
  const quorum = args.quorum ?? 3;
  const sr = args.selfResolving ? normalizeSelfResolvingConfig(args.selfResolving) : undefined;
  // Random order prevents the same persona from always reporting first
  // (uninformed) or last (most informed) — part of the mechanism's
  // resistance to juror position gaming.
  const personas = sr ? shuffled(eligiblePersonas(args.category)) : eligiblePersonas(args.category);
  if (personas.length === 0) return null;

  const votes: CouncilVote[] = [];
  const qHistory: number[] = [];
  const history: string[] = [];
  let qPrev = Q_PRIOR;
  let decisiveSoFar = 0;
  let totalPaidUnits = 0n;

  // Sequential: keeps within LLM free-tier RPM and RPC rate limits —
  // and in self-resolving mode, sequencing is the mechanism itself.
  for (const p of personas) {
    let url = `${args.baseUrl.replace(/\/$/, "")}/api/council/vote?claimId=${args.claimId}&persona=${encodeURIComponent(p.slug)}`;
    if (sr && history.length > 0) {
      url += `&history=${encodeURIComponent(JSON.stringify(history.slice(-8)))}`;
    }
    try {
      const r = await fetchWithBudget(url, args.payer, capUnits);
      if (!r.response.ok) continue;
      const body = (await r.response.json()) as VoteResponse;
      const verdict = body.verdict;
      if (!isVerdict(verdict)) {
        continue;
      }
      const priceUnits = r.payment?.priceUnits ?? null;
      if (priceUnits != null) totalPaidUnits += priceUnits;
      const confidence = Math.max(0, Math.min(100, Math.round(body.confidence ?? 0)));
      const vote: CouncilVote = {
        slug: p.slug,
        displayName: p.displayName,
        verdict,
        confidence,
        pricePaidUnits: priceUnits != null ? priceUnits.toString() : null,
        walletAddress: typeof body.paidTo === "string" ? body.paidTo : undefined,
      };
      if (sr) {
        const q = verdictToProbability(verdict, confidence, qPrev);
        vote.probability = q;
        qHistory.push(q);
        const reasoning = (body.explanation ?? "").slice(0, 220);
        history.push(
          `${p.displayName}: ${Math.round(q * 100)}% challengers — ${reasoning}`,
        );
        qPrev = q;
      }
      votes.push(vote);
      if (verdict === "CREATOR_WINS" || verdict === "CHALLENGERS_WIN") {
        decisiveSoFar++;
      }
      // Random termination: once enough decisive reports exist, each further
      // vote only happens with probability 1−alpha. Keeps the reference
      // report's position unpredictable and bounds LLM spend per settlement.
      if (sr && decisiveSoFar >= sr.minVotes && Math.random() < sr.alpha) {
        break;
      }
    } catch {
      // persona abstains on any error
    }
  }

  const tally = { creator: 0, challengers: 0, draw: 0, unresolvable: 0, decisive: 0 };
  for (const v of votes) {
    if (v.verdict === "CREATOR_WINS") tally.creator++;
    else if (v.verdict === "CHALLENGERS_WIN") tally.challengers++;
    else if (v.verdict === "DRAW") tally.draw++;
    else tally.unresolvable++;
  }
  tally.decisive = tally.creator + tally.challengers;

  // Not enough jurors voted decisively — let the caller settle solo.
  if (tally.decisive < quorum) return null;

  let verdict: Verdict;
  let winningVotes: CouncilVote[];
  if (tally.creator > tally.challengers) {
    verdict = "CREATOR_WINS";
    winningVotes = votes.filter((v) => v.verdict === "CREATOR_WINS");
  } else if (tally.challengers > tally.creator) {
    verdict = "CHALLENGERS_WIN";
    winningVotes = votes.filter((v) => v.verdict === "CHALLENGERS_WIN");
  } else {
    // Split jury — refund rather than guess.
    verdict = "UNRESOLVABLE";
    winningVotes = [];
  }

  // Confidence = avg of the majority's confidence, scaled by how lopsided the
  // decisive vote was (a 7–1 majority is firmer than 4–3).
  const avgConf =
    winningVotes.length > 0
      ? winningVotes.reduce((s, v) => s + v.confidence, 0) / winningVotes.length
      : 0;
  const agreement = tally.decisive > 0 ? Math.max(tally.creator, tally.challengers) / tally.decisive : 0;
  const confidence = Math.round(avgConf * agreement);

  const side = verdict === "CREATOR_WINS" ? "CREATOR" : verdict === "CHALLENGERS_WIN" ? "CHALLENGERS" : "SPLIT";
  const abstain = tally.draw + tally.unresolvable;
  const explanation = `[council ${tally.creator}–${tally.challengers} → ${side}${abstain ? `, ${abstain} abstain` : ""}] ${winningVotes[0]?.displayName ?? "Jury"} et al.`.slice(0, 500);

  return {
    verdict,
    confidence,
    explanation,
    tally,
    votes,
    totalPaidUnits,
    ...(sr ? { qHistory, reports: history } : {}),
  };
}

export interface BonusReceipt {
  slug: string;
  bonusUsdc: number;
  txHash: string | null; // null when the transfer failed (logged, non-fatal)
}

/**
 * Pays the cross-entropy bonuses to positive-scoring jurors as classic Stellar
 * USDC payments from the oracle wallet. Every transfer is individually best-effort:
 * a failed payout is logged and skipped, never thrown — settlement must not
 * depend on payout success. Call AFTER the claim is settled on-chain.
 */
export async function payCouncilBonuses(
  votes: CouncilVote[],
  poolUsdc: number,
  payerWallet: AgentWallet,
): Promise<BonusReceipt[]> {
  const bonuses = allocateBonus(votes.map((v) => v.score ?? 0), poolUsdc);
  const receipts: BonusReceipt[] = [];
  for (let i = 0; i < votes.length; i++) {
    const vote = votes[i];
    const bonus = bonuses[i];
    if (bonus <= 0) continue;
    vote.bonusUsdc = bonus;
    if (!isAccountAddress(vote.walletAddress ?? "")) {
      console.warn(`[council] no Stellar account for ${vote.slug} — bonus ${bonus} USDC skipped`);
      receipts.push({ slug: vote.slug, bonusUsdc: bonus, txHash: null });
      continue;
    }
    try {
      // 7 decimals, not 6: Stellar USDC's SAC exposes 7, and Stellar rejects an
      // amount with an eighth decimal place outright rather than truncating it.
      const txHash = await transferUsdc({
        wallet: payerWallet,
        to: vote.walletAddress!,
        amountUsdc: bonus.toFixed(7),
      });
      receipts.push({ slug: vote.slug, bonusUsdc: bonus, txHash });
    } catch (err) {
      console.warn(`[council] bonus transfer to ${vote.slug} failed:`, err);
      receipts.push({ slug: vote.slug, bonusUsdc: bonus, txHash: null });
    }
  }
  return receipts;
}
