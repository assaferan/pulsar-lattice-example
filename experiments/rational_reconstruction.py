from sage.all import ceil, log, RealField, lcm, factor, RR, prime_divisors

def reconstruct_denominator(v_tilde, sigma, c=4.0, alpha=0.3, prec=None):
    r"""
    Recover a0 = lcm_i denom(v_i) from a noisy sample v_tilde = v + N(0,sigma),
    each true v_i rational. Per-coordinate CF reconstruction, noise-aware.

    Returns (a_hat, confident, flagged):
      a_hat      -- lcm of denominators over confident coordinates
      confident  -- dict {i: p_i/q_i}
      flagged    -- coords whose denominator exceeds the resolution floor
                    sigma^{-1/2} (information-limited, not recoverable here)
    """
    if prec is None:
        prec = max(53, ceil(2*log(1/sigma, 2)) + 20)   # noise sets the scale
    R = RealField(prec)
    floor = R(sigma)**(-1/2)                            # max recoverable denom
    confident, flagged, qs = {}, [], []

    for i, xi in enumerate(v_tilde):
        x = R(xi)
        r = x.nearby_rational(max_error = R(c*sigma))   # simplest rational within c*sigma
        q = r.denominator()
        stable = all(x.nearby_rational(max_error=R(k*sigma)).denominator() == q
                     for k in (2, 3, 5))                # same denom across a tolerance band
        if q <= alpha*floor and stable:
            confident[i] = r; qs.append(q)
        else:
            flagged.append(i)

    a_hat = lcm(qs) if qs else 1
    return a_hat, confident, flagged


def joint_residual(v_tilde, a_hat, sigma, confident):
    """Worst |a_hat*x - round(a_hat*x)| over confident coords, in units of a_hat*sigma.
       Should sit around the max of n half-normals (~3-4); large => a_hat is wrong."""
    rs = [abs(a_hat*v_tilde[i] - round(a_hat*v_tilde[i])) for i in confident]
    return max(rs) / (a_hat*sigma)


def certify_primes(v_tilde, a_hat, sigma):
    """Drop-one test: each prime power ell^k || a_hat must be *needed* -- some
       coordinate must misfit the coarser grid (a_hat/ell) by >> sigma. Returns
       prime powers that look spurious (no coordinate needs them)."""
    spurious = []
    for ell, k in factor(a_hat):
        coarse = a_hat // ell
        need = max(abs(coarse*x - round(coarse*x)) for x in v_tilde) / (coarse*sigma)
        if need < 4:                                    # nobody needs this factor
            spurious.append((ell, k, RR(need)))
    return spurious

def joint_denominator(v_tilde, sigma, c=4.0, alpha=0.3, thresh=4.5, prec=None):
    r"""
    Recover a0 = lcm_i denom(v_i) from v_tilde = v + N(0,sigma), keeping the
    shared-denominator constraint primary.
      (1) per-coordinate CF PROPOSES candidate factors, admitted strictly (purity);
      (2) answer = minimal common a fitting every trusted coordinate, by stripping
          prime factors no trusted coordinate needs (joint test).
    Returns (a_hat, trusted, flagged).
    """
    a_star, confident, flagged = reconstruct_denominator(v_tilde, sigma, c, alpha, prec)
    trusted = list(confident)

    def fits(a):                      # joint consistency of the common denominator a
        a = int(a)
        worst = max(abs(a*v_tilde[i] - round(a*v_tilde[i])) for i in trusted)
        return worst <= thresh * a * sigma

    a = a_star
    changed = True
    while changed:                    # strip every prime factor that isn't jointly needed
        changed = False
        for ell in prime_divisors(a):
            if a % ell == 0 and fits(a // ell):
                a //= ell; changed = True
    return a, trusted, flagged