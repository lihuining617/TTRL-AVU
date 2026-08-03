from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ConsensusDebugInfo:
    mean_entropies: list[float]
    weights: list[float]
    consensus_scores: dict[str, float]
    p0: dict[str, float]
    high_answers: set[str]
    low_answers: set[str]


def _validate_answers(answers: Sequence[str]) -> None:
    if not isinstance(answers, Sequence) or isinstance(answers, (str, bytes)):
        raise TypeError("answers must be a sequence of strings.")
    if len(answers) == 0:
        raise ValueError("answers must contain at least one answer.")
    if any(not isinstance(answer, str) for answer in answers):
        raise TypeError("every answer must be a string.")


def _validate_common(lambda_: float, tau: float, delta: float, eps: float) -> None:
    if not all(np.isfinite(value) for value in (lambda_, tau, delta, eps)):
        raise ValueError("lambda_, tau, delta, and eps must be finite.")
    if lambda_ < 0:
        raise ValueError("lambda_ must be nonnegative.")
    if not 0 <= tau <= 1:
        raise ValueError("tau must be in [0, 1].")
    if not 0 <= delta <= 1:
        raise ValueError("delta must be in [0, 1].")
    if eps <= 0:
        raise ValueError("eps must be positive.")


def mean_token_entropies_from_probs(token_probs: Sequence[np.ndarray], eps: float = 1e-12) -> list[float]:
    if len(token_probs) == 0:
        raise ValueError("token_probs must contain at least one probability matrix.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    mean_entropies = []
    vocab_size = None
    for i, probs in enumerate(token_probs):
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 2:
            raise ValueError(f"token_probs[{i}] must be a 2D array.")
        if probs.shape[0] == 0 or probs.shape[1] == 0:
            raise ValueError(f"token_probs[{i}] must have nonzero length and vocabulary size.")
        if vocab_size is None:
            vocab_size = probs.shape[1]
        elif probs.shape[1] != vocab_size:
            raise ValueError("all token probability arrays must have the same vocabulary size.")
        if np.any(probs < 0):
            raise ValueError(f"token_probs[{i}] contains negative probabilities.")
        if not np.allclose(probs.sum(axis=1), 1.0, atol=eps, rtol=0.0):
            raise ValueError(f"each row of token_probs[{i}] must sum to 1.")

        nonzero = probs > 0
        entropy_sum = -np.sum(probs[nonzero] * np.log(probs[nonzero]))
        mean_entropies.append(float(entropy_sum / probs.shape[0]))

    return mean_entropies


def uncertainty_consensus_from_entropies(
    answers: Sequence[str],
    mean_entropies: Sequence[float],
    lambda_: float,
    tau: float,
    delta: float,
    eps: float = 1e-12,
    *,
    empty_high_policy: str = "return_p0",
    return_debug: bool = False,
) -> dict[str, float] | tuple[dict[str, float], ConsensusDebugInfo]:
    _validate_answers(answers)
    _validate_common(lambda_, tau, delta, eps)
    if len(mean_entropies) != len(answers):
        raise ValueError("mean_entropies must have the same length as answers.")
    if empty_high_policy not in {"return_p0", "raise"}:
        raise ValueError("empty_high_policy must be 'return_p0' or 'raise'.")

    entropies = np.asarray(mean_entropies, dtype=np.float64)
    if entropies.ndim != 1:
        raise ValueError("mean_entropies must be a 1D sequence.")
    if np.any(~np.isfinite(entropies)) or np.any(entropies < 0):
        raise ValueError("mean_entropies must be finite and nonnegative.")

    weights = np.exp(-float(lambda_) * entropies)
    if np.sum(weights) <= eps:
        raise ValueError("sum of uncertainty weights is numerically zero.")

    consensus_scores: dict[str, float] = {}
    for answer, weight in zip(answers, weights):
        consensus_scores[answer] = consensus_scores.get(answer, 0.0) + float(weight)

    score_sum = sum(consensus_scores.values())
    if score_sum <= eps:
        raise ValueError("sum of consensus scores is numerically zero.")

    p0 = {answer: score / score_sum for answer, score in consensus_scores.items()}
    high_answers = {answer for answer, prob in p0.items() if prob >= tau}
    low_answers = set(p0) - high_answers

    if not low_answers:
        final_p = dict(p0)
    elif not high_answers:
        if empty_high_policy == "raise":
            raise ValueError("A_high is empty; lower tau or use empty_high_policy='return_p0'.")
        final_p = dict(p0)
    else:
        z_high = sum(p0[answer] for answer in high_answers)
        z_low = sum(p0[answer] for answer in low_answers)
        if z_high <= eps or z_low <= eps:
            raise ValueError("cannot redistribute probability mass with zero high or low mass.")
        final_p = {
            answer: (1.0 - delta) * p0[answer] / z_high
            if answer in high_answers
            else delta * p0[answer] / z_low
            for answer in p0
        }

    total = sum(final_p.values())
    if not np.isclose(total, 1.0, atol=max(eps, 1e-10), rtol=0.0):
        raise ValueError(f"final_p must sum to 1, got {total}.")

    if not return_debug:
        return final_p

    debug = ConsensusDebugInfo(
        mean_entropies=[float(x) for x in entropies],
        weights=[float(x) for x in weights],
        consensus_scores=consensus_scores,
        p0=p0,
        high_answers=high_answers,
        low_answers=low_answers,
    )
    return final_p, debug


def uncertainty_aware_consensus_distribution(
    answers: Sequence[str],
    token_probs: Sequence[np.ndarray],
    lambda_: float,
    tau: float,
    delta: float,
    eps: float = 1e-12,
    *,
    empty_high_policy: str = "return_p0",
    return_debug: bool = False,
) -> dict[str, float] | tuple[dict[str, float], ConsensusDebugInfo]:
    _validate_answers(answers)
    if len(token_probs) != len(answers):
        raise ValueError("token_probs must have the same length as answers.")

    mean_entropies = mean_token_entropies_from_probs(token_probs, eps=eps)
    return uncertainty_consensus_from_entropies(
        answers=answers,
        mean_entropies=mean_entropies,
        lambda_=lambda_,
        tau=tau,
        delta=delta,
        eps=eps,
        empty_high_policy=empty_high_policy,
        return_debug=return_debug,
    )


def rewards_from_consensus_distribution(
    answers: Sequence[str],
    final_p: Mapping[str, float],
    invalid_reward: float = 0.0,
) -> list[float]:
    return [float(final_p.get(answer, invalid_reward)) for answer in answers]
