# power check, run before any labeling: how many test prompts do we need for the
# non-inferiority test to have a real shot at passing? it sizes the eval set in step 3.
#
# per test prompt, compare the routed answer to the always-large answer:
#   +1 = large was good enough, routed wasn't (we lost quality)
#   -1 = routed was good enough, large wasn't (small got lucky)
#    0 = same outcome
# quality loss = mean of those. we pass if the upper end of its 95% ci is under the margin.
# this uses a normal-approximation ci because it's fast. the real eval (step 7) uses a
# bootstrap. for a plain mean at n >= 300 they land in the same place, fine for sizing.
import numpy as np

MARGIN = 0.03  # fixed in the pre-flight, 3 points on the good-enough rate
SIMS = 4000


def chance_to_pass(n, true_loss, flip_rate, rng):
    # flip_rate = share of prompts where routed and large disagree at all.
    # you can't lose more than you flip, so true_loss has to fit inside flip_rate.
    assert 0 <= true_loss <= flip_rate <= 1
    # split it into +1s and -1s so that (+1s - -1s) = true_loss.
    p_worse = (flip_rate + true_loss) / 2
    p_better = (flip_rate - true_loss) / 2
    counts = rng.multinomial(n, [p_worse, p_better, 1 - p_worse - p_better], size=SIMS)
    worse, better = counts[:, 0], counts[:, 1]
    loss = (worse - better) / n
    # variance of a {+1, -1, 0} variable is E[d^2] - E[d]^2
    var = (worse + better) / n - loss ** 2
    # 1.96 = upper end of a two-sided 95% ci, i.e. a one-sided test at 2.5%. that's the
    # usual non-inferiority choice and stricter than a one-sided 95% bound (1.645).
    upper = loss + 1.96 * np.sqrt(var / n)
    return float(np.mean(upper < MARGIN))


def main():
    rng = np.random.default_rng(0)
    sizes = [300, 500, 1000, 1500, 2000, 3000]
    print("flip  true_loss  " + "  ".join(f"n={n:>4}" for n in sizes))
    for flip_rate in [0.10, 0.20, 0.30]:
        for true_loss in [0.00, 0.01, 0.02]:
            row = [chance_to_pass(n, true_loss, flip_rate, rng) for n in sizes]
            print(f"{flip_rate:.2f}  {true_loss:.2f}       " + "  ".join(f"{p:6.2f}" for p in row))


if __name__ == "__main__":
    main()
