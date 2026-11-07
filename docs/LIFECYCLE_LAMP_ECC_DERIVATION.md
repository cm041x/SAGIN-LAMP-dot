# Lifecycle LAMP and event-conditioned critic: algebra and limits

LAMP, discrete SAC, twin critics, and counterfactual baselines are prior work.
This prototype proposes node lifecycle-conditioned damping and an event-conditioned
team-value estimator; these design choices and their novelty are not established
by the following algebra.

For a fixed target-network Bellman target Y, if p(e|h,a)=P(E=e|h,a) and
q(h,a,e)=E[Y|h,a,E=e], the law of total expectation gives
sum_e p(e|h,a)q(h,a,e)=E[Y|h,a]. This does not establish convergence or gains.

For fixed h and other actions, write p_b(e), q_b(e), Q_b=sum_e p_b q_b,
p_bar(e)=sum_b pi_b p_b(e), and B(e)=sum_b pi_b p_b(e)q_b(e)/p_bar(e)
when p_bar(e)>0. For p_bar(e)=0, the event contribution is exactly zero.
Then sum_e p_a(e)[q_a(e)-B(e)] + sum_e[p_a(e)-p_bar(e)]B(e)
= Q_a-sum_b pi_b Q_b. The second term vanishes for action-independent p.
These two terms are algebraic diagnostics, not automatically causal effects.

If max_e |qhat_e-q_e|<=epsilon_q, ||phat-p||_1<=epsilon_p and
max_e |q_e|<=B_q, adding and subtracting sum_e phat_e q_e yields
|Qhat-Q|<=epsilon_q+B_q epsilon_p. With the same policy, the corresponding
advantage error is at most twice that bound. These are conditional on the stated
assumptions; they do not establish the numerical errors in an experiment.

At each snapshot, conditioned damping is in (0,1) and fixed through the L inner
Euler steps. That alone does not guarantee stability for arbitrary weights or
step size. For tanh, fixed n,d,L, epsilon>=0, xi>0 and ||X_t U||_F<=C,
||H_t||_F <= ||H_(t-1)||_F/xi + |mu|C/xi + epsilon L sqrt(n d).
For xi>1 this geometric recursion has limsup bounded by
[|mu|C/xi + epsilon L sqrt(n d)]/[1-1/xi]. The default xi=1 does not satisfy
that sufficient condition; empirical hidden norms are required. Neither this
bound nor a numerical identity proves memory performance or training convergence.
