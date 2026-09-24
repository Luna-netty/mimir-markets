import assert from "node:assert/strict";
import test from "node:test";

import {
  Q_PRIOR,
  BONUS_DUST_USDC,
  verdictToProbability,
  crossEntropyScore,
  scoreCouncilVotes,
  allocateBonus,
  normalizeSelfResolvingConfig,
  type CouncilVote,
} from "../../agents/oracle/council-vote";

// ── verdictToProbability ──────────────────────────────────────────────────────

test("confidence maps symmetrically around the prior", () => {
  assert.equal(verdictToProbability("CHALLENGERS_WIN", 80, Q_PRIOR), 0.9);
  assert.equal(verdictToProbability("CREATOR_WINS", 80, Q_PRIOR), 0.1);
  assert.equal(verdictToProbability("CHALLENGERS_WIN", 0, Q_PRIOR), 0.5);
});

test("q is clamped away from 0 and 1 so log scores stay finite", () => {
  assert.equal(verdictToProbability("CHALLENGERS_WIN", 100, Q_PRIOR), 0.98);
  assert.equal(verdictToProbability("CREATOR_WINS", 100, Q_PRIOR), 0.02);
  assert.equal(verdictToProbability("CHALLENGERS_WIN", 250, Q_PRIOR), 0.98);
});

test("DRAW and UNRESOLVABLE carry no information — q stays at qPrev", () => {
  assert.equal(verdictToProbability("DRAW", 90, 0.7), 0.7);
  assert.equal(verdictToProbability("UNRESOLVABLE", 90, 0.3), 0.3);
});

test("negative and oversize confidence clamp rather than inventing probability", () => {
  assert.equal(verdictToProbability("CHALLENGERS_WIN", -40, Q_PRIOR), 0.5);
  assert.equal(verdictToProbability("CREATOR_WINS", 1000, Q_PRIOR), 0.02);
});

// ── crossEntropyScore ─────────────────────────────────────────────────────────

test("no update scores exactly zero — parroting the prior pays nothing", () => {
  assert.equal(crossEntropyScore(0.9, 0.5, 0.5), 0);
});

test("updates toward the reference score positive, away score negative", () => {
  assert.ok(crossEntropyScore(0.9, 0.8, 0.5) > 0);
  assert.ok(crossEntropyScore(0.9, 0.2, 0.5) < 0);
  // Mirror case: reference favors the creator.
  assert.ok(crossEntropyScore(0.1, 0.2, 0.5) > 0);
  assert.ok(crossEntropyScore(0.1, 0.8, 0.5) < 0);
});

test("reporting the reference belief itself maximizes the score", () => {
  const qT = 0.85;
  const atReference = crossEntropyScore(qT, qT, 0.5);
  for (const q of [0.55, 0.65, 0.75, 0.95]) {
    assert.ok(atReference > crossEntropyScore(qT, q, 0.5));
  }
});

test("scores are additive along the chain (market scoring rule telescopes)", () => {
  // Two sequential jurors moving 0.5→0.7→0.9 together earn what one juror
  // moving 0.5→0.9 would — payment splits by marginal contribution.
  const qT = 0.9;
  const combined = crossEntropyScore(qT, 0.7, 0.5) + crossEntropyScore(qT, 0.9, 0.7);
  const direct = crossEntropyScore(qT, 0.9, 0.5);
  assert.ok(Math.abs(combined - direct) < 1e-12);
});

test("malformed non-finite probabilities collapse safely — no Inf/NaN bonuses", () => {
  for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
    const score = crossEntropyScore(bad, bad, bad);
    assert.equal(score, 0);
    assert.ok(Number.isFinite(score));
  }
  // Extreme raw inputs clamp into (0.02, 0.98) and remain finite.
  const extreme = crossEntropyScore(2, -1, 0);
  assert.ok(Number.isFinite(extreme));
});

test("boundary clamp makes 0/1 reports score as Q_MIN/Q_MAX, never -Infinity", () => {
  const atFloor = crossEntropyScore(0.9, 0, 0.5);
  const atCeil = crossEntropyScore(0.9, 1, 0.5);
  assert.ok(Number.isFinite(atFloor));
  assert.ok(Number.isFinite(atCeil));
  assert.equal(crossEntropyScore(0.9, 0, 0.5), crossEntropyScore(0.9, 0.02, 0.5));
  assert.equal(crossEntropyScore(0.9, 1, 0.5), crossEntropyScore(0.9, 0.98, 0.5));
});

// ── scoreCouncilVotes ─────────────────────────────────────────────────────────

function makeVote(overrides: Partial<CouncilVote>): CouncilVote {
  return {
    slug: "optimist",
    displayName: "The Optimist",
    verdict: "CHALLENGERS_WIN",
    confidence: 80,
    pricePaidUnits: null,
    ...overrides,
  };
}

test("scoreCouncilVotes chains q from the prior and skips abstainers", () => {
  const votes = [
    makeVote({ slug: "a", probability: 0.8 }),
    makeVote({ slug: "b", probability: undefined }), // abstained — no q
    makeVote({ slug: "c", probability: 0.9 }),
  ];
  const scored = scoreCouncilVotes(votes, 0.9);
  assert.ok(scored[0].score! > 0);              // 0.5 → 0.8 toward reference
  assert.equal(scored[1].score, 0);             // abstainer scores zero
  assert.ok(scored[2].score! > 0);              // 0.8 → 0.9, chain skipped b
  const direct = crossEntropyScore(0.9, 0.9, 0.8);
  assert.ok(Math.abs(scored[2].score! - direct) < 1e-12);
});

test("empty jury and all-abstainer rounds score nothing (dependency failure)", () => {
  assert.deepEqual(scoreCouncilVotes([], 0.9), []);
  const abstained = scoreCouncilVotes(
    [makeVote({ slug: "a" }), makeVote({ slug: "b" })], // no probability
    0.9,
  );
  assert.equal(abstained[0].score, 0);
  assert.equal(abstained[1].score, 0);
});

test("duplicate slug reports still advance the chain independently", () => {
  // A stale retry that reuses a slug must not collapse scores — each report
  // is a sequential information update, not a set keyed by persona.
  const scored = scoreCouncilVotes(
    [
      makeVote({ slug: "a", probability: 0.7 }),
      makeVote({ slug: "a", probability: 0.9 }),
    ],
    0.9,
  );
  assert.ok(scored[0].score! > 0);
  assert.ok(scored[1].score! > 0);
  const telescoped =
    crossEntropyScore(0.9, 0.7, 0.5) + crossEntropyScore(0.9, 0.9, 0.7);
  assert.ok(Math.abs((scored[0].score! + scored[1].score!) - telescoped) < 1e-12);
});

test("cancelled-style DRAW/UNRESOLVABLE mapped q leaves score at zero when q unchanged", () => {
  const qPrev = 0.7;
  const drawQ = verdictToProbability("DRAW", 99, qPrev);
  const unresolvedQ = verdictToProbability("UNRESOLVABLE", 99, qPrev);
  assert.equal(drawQ, qPrev);
  assert.equal(unresolvedQ, qPrev);
  assert.equal(crossEntropyScore(0.9, drawQ, qPrev), 0);
  assert.equal(crossEntropyScore(0.9, unresolvedQ, qPrev), 0);
});

test("stale reference outside (0,1) is clamped before scoring", () => {
  const votes = [makeVote({ slug: "a", probability: 0.8 })];
  const high = scoreCouncilVotes(votes, 5);
  const low = scoreCouncilVotes(votes, -3);
  assert.ok(Number.isFinite(high[0].score!));
  assert.ok(Number.isFinite(low[0].score!));
  // Same as scoring against the clamp bounds.
  assert.equal(high[0].score, scoreCouncilVotes(votes, 0.98)[0].score);
  assert.equal(low[0].score, scoreCouncilVotes(votes, 0.02)[0].score);
});

// ── allocateBonus ─────────────────────────────────────────────────────────────

test("bonus splits proportionally across positive scores only", () => {
  const bonuses = allocateBonus([0.3, 0.1, -0.5, 0], 0.008);
  assert.equal(bonuses[0], 0.006);
  assert.equal(bonuses[1], 0.002);
  assert.equal(bonuses[2], 0);
  assert.equal(bonuses[3], 0);
});

test("total payout never exceeds the pool", () => {
  const bonuses = allocateBonus([1.7, 0.9, 0.4], 0.01);
  const total = bonuses.reduce((a, b) => a + b, 0);
  assert.ok(total <= 0.01 + 1e-9);
});

test("dust shares are skipped, all-negative rounds pay nothing", () => {
  // 1% of the pool is below the dust floor.
  const bonuses = allocateBonus([99, 1], 0.01);
  assert.ok(bonuses[0] > 0);
  assert.equal(bonuses[1], 0);
  assert.ok((0.01 * 1) / 100 < BONUS_DUST_USDC);
  assert.deepEqual(allocateBonus([-1, -2, 0], 0.01), [0, 0, 0]);
});

test("malformed pool or NaN scores pay nothing — funded bonus stays safe", () => {
  assert.deepEqual(allocateBonus([1, 2], Number.NaN), [0, 0]);
  assert.deepEqual(allocateBonus([1, 2], Number.POSITIVE_INFINITY), [0, 0]);
  assert.deepEqual(allocateBonus([1, 2], -0.01), [0, 0]);
  assert.deepEqual(allocateBonus([Number.NaN, Number.POSITIVE_INFINITY, 1], 0.01), [0, 0, 0.01]);
  assert.deepEqual(allocateBonus([], 0.01), []);
});

// ── normalizeSelfResolvingConfig ──────────────────────────────────────────────

test("normalizeSelfResolvingConfig clamps alpha into [0,1] and minVotes >= 1", () => {
  assert.deepEqual(normalizeSelfResolvingConfig({ alpha: 0.25, minVotes: 3 }), {
    alpha: 0.25,
    minVotes: 3,
  });
  assert.deepEqual(normalizeSelfResolvingConfig({ alpha: -1, minVotes: 0 }), {
    alpha: 0,
    minVotes: 1,
  });
  assert.deepEqual(normalizeSelfResolvingConfig({ alpha: 2, minVotes: 4.9 }), {
    alpha: 1,
    minVotes: 4,
  });
});

test("normalizeSelfResolvingConfig replaces non-finite knobs with safe defaults", () => {
  assert.deepEqual(
    normalizeSelfResolvingConfig({ alpha: Number.NaN, minVotes: Number.POSITIVE_INFINITY }),
    { alpha: 0.25, minVotes: 1 },
  );
});

// ── regression: end-to-end score → bonus invariant ────────────────────────────

test("regression: positive CE scores alone can claim the bonus pool", () => {
  const votes = scoreCouncilVotes(
    [
      makeVote({ slug: "a", probability: 0.8 }),
      makeVote({ slug: "b", probability: 0.2 }), // away from qT → negative
      makeVote({ slug: "c", probability: 0.9 }),
    ],
    0.9,
  );
  const bonuses = allocateBonus(
    votes.map((v) => v.score ?? 0),
    0.01,
  );
  assert.ok(bonuses[0] > 0);
  assert.equal(bonuses[1], 0);
  assert.ok(bonuses[2] > 0);
  assert.ok(bonuses.reduce((a, b) => a + b, 0) <= 0.01 + 1e-9);
});
