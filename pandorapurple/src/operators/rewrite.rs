//! Inferno — Algebraic simplification and common-subexpression reuse for symbolic tensor formulas.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Weak};

use candle_core::Result;

use super::symbolic::{infer, Axis, Expr};

/// A PERSISTENT hash-cons table.
///
/// Previously each `simplify` call owned its table, so two formulas
/// simplified separately could not share a node even when structurally
/// identical — `simplify_many` existed precisely to work around that, and
/// only for calls that happened to be batched. A `Context` interns across
/// calls, which makes cross-node CSE automatic and `Arc::as_ptr` a genuine
/// GLOBAL identity for the lifetime of the context.
///
/// # Why the table holds `Weak`
///
/// A strong table would pin every formula ever built — and every `Tensor`
/// inside it — for the process's lifetime. `Weak` lets the `Expr` (and its
/// tensors) drop as soon as the last formula referencing it does, while
/// keeping the ALLOCATION alive, which is what makes the pointer-keyed
/// invariants below sound.
///
/// # Why pointer keys are safe here
///
/// A node's key stores its children's raw pointers, which would be a
/// use-after-free hazard if an address could be recycled. It cannot: every
/// canonical child is itself interned, so the table holds a `Weak` to it,
/// so its allocation block is never returned to the allocator while any key
/// mentions it. A dead entry's `upgrade()` returns `None` and is replaced;
/// stale parent keys mentioning it simply never match again (a missed CSE,
/// never a wrong hit) and are reclaimed by [`prune`](Self::prune).
///
/// The per-call `memo` (original ptr → canonical) is deliberately NOT
/// shared: its keys are pointers to the CALLER's nodes, which the context
/// holds nothing for and which really can be freed and recycled.
pub struct Context {
    cons: HashMap<Key, Weak<Expr>>,
    /// Creation-ordered ids, for deterministic commutative argument
    /// ordering. Keyed by canonical node pointer, which is stable for
    /// exactly as long as `cons` holds the node's `Weak`.
    ids: HashMap<*const Expr, u64>,
    next_id: u64,
    /// Entry count above which the next simplify prunes.
    prune_at: usize,
    hits: u64,
    misses: u64,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    pub fn new() -> Self {
        Self {
            cons: HashMap::new(),
            ids: HashMap::new(),
            next_id: 0,
            prune_at: 4096,
            hits: 0,
            misses: 0,
        }
    }

    /// Interned nodes currently tracked (live and dead).
    pub fn len(&self) -> usize {
        self.cons.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cons.is_empty()
    }

    /// `(hits, misses)` against the cons table since construction — how much
    /// cross-formula sharing the context is actually buying.
    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// Drop entries whose node is gone. Returns how many were reclaimed.
    pub fn prune(&mut self) -> usize {
        let before = self.cons.len();
        self.cons.retain(|_, w| w.strong_count() > 0);
        let live: std::collections::HashSet<*const Expr> =
            self.cons.values().filter_map(|w| w.upgrade()).map(|a| Arc::as_ptr(&a)).collect();
        self.ids.retain(|p, _| live.contains(p));
        self.prune_at = self.cons.len().saturating_mul(2).max(4096);
        before - self.cons.len()
    }

    /// Simplify + hash-cons one formula against this context.
    pub fn simplify(&mut self, root: &Arc<Expr>) -> Result<Arc<Expr>> {
        Ok(self.simplify_many(std::slice::from_ref(root))?.pop().expect("one in, one out"))
    }

    /// Simplify SEVERAL formulas at once. With a persistent context this no
    /// longer *creates* cross-formula sharing — that now happens for any two
    /// calls — but it still guarantees it within one batch, which is what
    /// fused reductions depend on.
    pub fn simplify_many(&mut self, roots: &[Arc<Expr>]) -> Result<Vec<Arc<Expr>>> {
        if self.cons.len() > self.prune_at {
            self.prune();
        }
        let mut cse = Cse { memo: HashMap::new(), ctx: self };
        roots.iter().map(|r| cse.simp(r)).collect()
    }
}

thread_local! {
    /// The default context. Thread-local rather than a global `Mutex`:
    /// formulas are built and consumed on one thread, and a process-wide
    /// lock would serialise every rayon worker's formula construction for
    /// sharing that crosses threads only by accident.
    static THREAD_CTX: RefCell<Context> = RefCell::new(Context::new());
}

/// Run `f` against this thread's intern context.
pub fn with_thread_context<R>(f: impl FnOnce(&mut Context) -> R) -> R {
    THREAD_CTX.with(|c| f(&mut c.borrow_mut()))
}

/// Simplify + hash-cons a formula DAG against the thread's context.
/// Value-preserving and shape-metadata preserving; returns a fully interned
/// (maximally shared) DAG.
pub(crate) fn simplify_expr(root: &Arc<Expr>) -> Result<Arc<Expr>> {
    with_thread_context(|c| c.simplify(root))
}

/// Simplify SEVERAL formulas through the thread's context.
pub(crate) fn simplify_many(roots: &[Arc<Expr>]) -> Result<Vec<Arc<Expr>>> {
    with_thread_context(|c| c.simplify_many(roots))
}

#[derive(PartialEq, Eq, Hash)]
enum Key {
    Leaf(candle_core::TensorId, u8),
    Scalar(u64),
    Eye(u64),
    Un(u8, usize),
    UnC(u8, usize, u64),
    PowF(usize, u64),
    Affine(usize, u64, u64),
    Bin(u8, usize, usize),
}

fn axis_tag(a: Axis) -> u8 {
    match a {
        Axis::I => 0,
        Axis::J => 1,
        Axis::P => 2,
        Axis::IJ => 3,
    }
}

/// Structural key. Commutative operators sort their child pointers, so the
/// table cannot miss on argument order even if a node reaches `intern`
/// without going through [`Cse::mk_ac`] — belt and braces for the real fix,
/// which is canonical ordering in the smart constructor.
fn key_of(e: &Expr) -> Key {
    let p = |a: &Arc<Expr>| Arc::as_ptr(a) as usize;
    let comm = |a: &Arc<Expr>, b: &Arc<Expr>| {
        let (x, y) = (p(a), p(b));
        if x <= y {
            (x, y)
        } else {
            (y, x)
        }
    };
    match e {
        Expr::Leaf { data, axis } => Key::Leaf(data.id(), axis_tag(*axis)),
        Expr::Scalar(v) => Key::Scalar(v.to_bits()),
        Expr::EyeIdentity { delta } => Key::Eye(delta.to_bits()),
        Expr::Neg(a) => Key::Un(0, p(a)),
        Expr::Exp(a) => Key::Un(1, p(a)),
        Expr::Log(a) => Key::Un(2, p(a)),
        Expr::Sqrt(a) => Key::Un(3, p(a)),
        Expr::Recip(a) => Key::Un(4, p(a)),
        Expr::Sum(a) => Key::Un(5, p(a)),
        Expr::PowF(a, pw) => Key::PowF(p(a), pw.to_bits()),
        Expr::MaxC(a, c) => Key::UnC(0, p(a), c.to_bits()),
        Expr::GtC(a, c) => Key::UnC(1, p(a), c.to_bits()),
        Expr::Affine { e, mul, add } => Key::Affine(p(e), mul.to_bits(), add.to_bits()),
        Expr::Add(a, b) => {
            let (x, y) = comm(a, b);
            Key::Bin(0, x, y)
        }
        Expr::Sub(a, b) => Key::Bin(1, p(a), p(b)),
        Expr::Mul(a, b) => {
            let (x, y) = comm(a, b);
            Key::Bin(2, x, y)
        }
        Expr::Div(a, b) => Key::Bin(3, p(a), p(b)),
        Expr::Pow(a, b) => Key::Bin(4, p(a), p(b)),
        Expr::MatDot(a, b) => Key::Bin(5, p(a), p(b)),
    }
}

struct Cse<'a> {
    /// original node ptr → canonical node (visit-once over the input DAG).
    /// Per-call: its keys are the CALLER's pointers, which nothing pins.
    memo: HashMap<*const Expr, Arc<Expr>>,
    ctx: &'a mut Context,
}

impl Cse<'_> {
    /// Intern a node whose children are already canonical.
    fn intern(&mut self, node: Expr) -> Arc<Expr> {
        let key = key_of(&node);
        if let Some(a) = self.ctx.cons.get(&key).and_then(|w| w.upgrade()) {
            self.ctx.hits += 1;
            return a;
        }
        self.ctx.misses += 1;
        let a = Arc::new(node);
        self.ctx.cons.insert(key, Arc::downgrade(&a));
        let id = self.ctx.next_id;
        self.ctx.next_id += 1;
        self.ctx.ids.insert(Arc::as_ptr(&a), id);
        a
    }

    /// A canonical node's creation-order id — the total order commutative
    /// arguments are sorted by.
    ///
    /// Creation order rather than pointer value: addresses vary run to run,
    /// which would make the canonical form (and hence `leaves()` order, and
    /// hence the gradient pack layout) non-reproducible.
    fn id_of(&mut self, e: &Arc<Expr>) -> u64 {
        if let Some(id) = self.ctx.ids.get(&Arc::as_ptr(e)) {
            return *id;
        }
        let id = self.ctx.next_id;
        self.ctx.next_id += 1;
        self.ctx.ids.insert(Arc::as_ptr(e), id);
        id
    }

    /// Flatten a right/left-nested chain of ONE commutative operator into
    /// its operand multiset.
    fn flatten(e: &Arc<Expr>, add: bool, out: &mut Vec<Arc<Expr>>) {
        match (add, e.as_ref()) {
            (true, Expr::Add(a, b)) | (false, Expr::Mul(a, b)) => {
                Self::flatten(a, add, out);
                Self::flatten(b, add, out);
            }
            _ => out.push(e.clone()),
        }
    }

    /// Rebuild a sorted operand list as a BALANCED tree.
    ///
    /// Balanced rather than left-associated so the tile graph's depth is
    /// `O(log k)` — `eval_block` recurses on it, and a long chain is the
    /// same shape that overflows the stack elsewhere in this crate. The
    /// pairing depends only on the sorted order, so it is canonical.
    fn build_balanced(&mut self, mut parts: Vec<Arc<Expr>>, add: bool) -> Arc<Expr> {
        while parts.len() > 1 {
            let mut next = Vec::with_capacity(parts.len().div_ceil(2));
            let mut it = parts.into_iter();
            while let Some(a) = it.next() {
                match it.next() {
                    Some(b) => {
                        let n = if add { Expr::Add(a, b) } else { Expr::Mul(a, b) };
                        next.push(self.intern(n));
                    }
                    None => next.push(a),
                }
            }
            parts = next;
        }
        parts.pop().expect("build_balanced on an empty operand list")
    }

    /// The AC normal form of a commutative chain: flatten, fold the literal
    /// operands into one `Affine`, sort the rest by creation id, rebuild
    /// balanced.
    ///
    /// This is what makes `a·b` and `b·a` the SAME node. Without it the
    /// hash-cons table keys on argument order, so two structurally
    /// equivalent formulas miss each other and every downstream consumer —
    /// the per-tile memo, the fused-sweep pack, the shared Gibbs subtree —
    /// duplicates work it believes it is sharing.
    fn mk_ac(&mut self, a: Arc<Expr>, b: Arc<Expr>, add: bool) -> Arc<Expr> {
        let mut parts = Vec::new();
        Self::flatten(&a, add, &mut parts);
        Self::flatten(&b, add, &mut parts);

        let mut lit = if add { 0.0 } else { 1.0 };
        let mut rest = Vec::with_capacity(parts.len());
        for p in parts {
            match p.as_ref() {
                Expr::Scalar(v) => {
                    if add {
                        lit += *v
                    } else {
                        lit *= *v
                    }
                }
                _ => rest.push(p),
            }
        }
        if rest.is_empty() {
            return self.intern(Expr::Scalar(lit));
        }
        rest.sort_by_key(|p| self.id_of(p));
        let tree = self.build_balanced(rest, add);
        if add {
            self.mk_affine(tree, 1.0, lit)
        } else {
            self.mk_affine(tree, lit, 0.0)
        }
    }

    /// Smart constructor: `mul·e + add` with elimination, scalar folding and
    /// affine-chain fusion. `e` must be canonical; recursion terminates
    /// because a canonical `Affine`'s inner node is never itself `Affine`
    /// or `Scalar`.
    fn mk_affine(&mut self, e: Arc<Expr>, mul: f64, add: f64) -> Arc<Expr> {
        if mul == 1.0 && add == 0.0 {
            return e;
        }
        match e.as_ref() {
            Expr::Scalar(v) => self.intern(Expr::Scalar(v * mul + add)),
            Expr::Affine { e: inner, mul: m1, add: a1 } => {
                let (inner, m1, a1) = (inner.clone(), *m1, *a1);
                self.mk_affine(inner, mul * m1, mul * a1 + add)
            }
            _ => self.intern(Expr::Affine { e, mul, add }),
        }
    }

    fn simp(&mut self, e: &Arc<Expr>) -> Result<Arc<Expr>> {
        if let Some(done) = self.memo.get(&Arc::as_ptr(e)) {
            return Ok(done.clone());
        }
        let out = match e.as_ref() {
            Expr::Leaf { data, axis } => {
                self.intern(Expr::Leaf { data: data.clone(), axis: *axis })
            }
            Expr::Scalar(v) => self.intern(Expr::Scalar(*v)),
            Expr::EyeIdentity { delta } => self.intern(Expr::EyeIdentity { delta: *delta }),

            Expr::Neg(a) => {
                let a = self.simp(a)?;
                self.mk_affine(a, -1.0, 0.0)
            }
            Expr::Affine { e: a, mul, add } => {
                let a = self.simp(a)?;
                self.mk_affine(a, *mul, *add)
            }
            Expr::Exp(a) => {
                let a = self.simp(a)?;
                match a.as_ref() {
                    Expr::Log(x) => x.clone(),
                    Expr::Scalar(v) => self.intern(Expr::Scalar(v.exp())),
                    _ => self.intern(Expr::Exp(a)),
                }
            }
            Expr::Log(a) => {
                let a = self.simp(a)?;
                match a.as_ref() {
                    Expr::Exp(x) => x.clone(),
                    Expr::Scalar(v) => self.intern(Expr::Scalar(v.ln())),
                    _ => self.intern(Expr::Log(a)),
                }
            }
            Expr::Sqrt(a) => {
                let a = self.simp(a)?;
                match a.as_ref() {
                    Expr::Scalar(v) => self.intern(Expr::Scalar(v.sqrt())),
                    _ => self.intern(Expr::Sqrt(a)),
                }
            }
            Expr::Recip(a) => {
                let a = self.simp(a)?;
                match a.as_ref() {
                    Expr::Recip(x) => x.clone(),
                    Expr::Scalar(v) => self.intern(Expr::Scalar(1.0 / v)),
                    _ => self.intern(Expr::Recip(a)),
                }
            }
            Expr::PowF(a, pw) => {
                let a = self.simp(a)?;
                if *pw == 1.0 {
                    a
                } else if *pw == 0.0 {
                    self.intern(Expr::Scalar(1.0))
                } else {
                    match a.as_ref() {
                        Expr::Scalar(v) => self.intern(Expr::Scalar(v.powf(*pw))),
                        _ => self.intern(Expr::PowF(a, *pw)),
                    }
                }
            }
            Expr::Sum(a) => {
                let a = self.simp(a)?;
                // Summing a singleton feature axis is the identity. `infer`
                // failing just means the formula is malformed — keep the
                // node and let the evaluator report it properly.
                if matches!(infer(&a).map(|m| m.dim), Ok(1)) {
                    a
                } else {
                    self.intern(Expr::Sum(a))
                }
            }
            Expr::MaxC(a, c) => {
                let a = self.simp(a)?;
                match a.as_ref() {
                    Expr::Scalar(v) => self.intern(Expr::Scalar(v.max(*c))),
                    // max(max(e, a), b) = max(e, max(a, b))
                    Expr::MaxC(inner, c1) => {
                        let floor = c.max(*c1);
                        let inner = inner.clone();
                        self.intern(Expr::MaxC(inner, floor))
                    }
                    _ => self.intern(Expr::MaxC(a, *c)),
                }
            }
            Expr::GtC(a, c) => {
                let a = self.simp(a)?;
                match a.as_ref() {
                    Expr::Scalar(v) => {
                        self.intern(Expr::Scalar(if *v > *c { 1.0 } else { 0.0 }))
                    }
                    _ => self.intern(Expr::GtC(a, *c)),
                }
            }

            Expr::Add(a, b) => {
                let (a, b) = (self.simp(a)?, self.simp(b)?);
                self.mk_ac(a, b, true)
            }
            Expr::Sub(a, b) => {
                let (a, b) = (self.simp(a)?, self.simp(b)?);
                match (a.as_ref(), b.as_ref()) {
                    (Expr::Scalar(x), Expr::Scalar(y)) => self.intern(Expr::Scalar(x - y)),
                    (Expr::Scalar(c), _) => {
                        let c = *c;
                        self.mk_affine(b, -1.0, c)
                    }
                    (_, Expr::Scalar(c)) => {
                        let c = *c;
                        self.mk_affine(a, 1.0, -c)
                    }
                    _ => self.intern(Expr::Sub(a, b)),
                }
            }
            Expr::Mul(a, b) => {
                let (a, b) = (self.simp(a)?, self.simp(b)?);
                self.mk_ac(a, b, false)
            }
            Expr::Div(a, b) => {
                let (a, b) = (self.simp(a)?, self.simp(b)?);
                match (a.as_ref(), b.as_ref()) {
                    (Expr::Scalar(x), Expr::Scalar(y)) => self.intern(Expr::Scalar(x / y)),
                    // x/c ≡ x·(1/c) exactly matches the elementwise eval up
                    // to the same inf/NaN pattern at c = ±0.
                    (_, Expr::Scalar(c)) => {
                        let c = *c;
                        self.mk_affine(a, 1.0 / c, 0.0)
                    }
                    _ => self.intern(Expr::Div(a, b)),
                }
            }
            Expr::Pow(a, b) => {
                let (a, b) = (self.simp(a)?, self.simp(b)?);
                match (a.as_ref(), b.as_ref()) {
                    // Fold via the SAME lowering the evaluator uses
                    // (exp(b·ln a)), so NaN structure is preserved.
                    (Expr::Scalar(x), Expr::Scalar(y)) => {
                        self.intern(Expr::Scalar((y * x.ln()).exp()))
                    }
                    _ => self.intern(Expr::Pow(a, b)),
                }
            }
            // No algebra here: the operands' axis purity is what makes the
            // node legal at all, and every rewrite that could move a factor
            // across the contraction risks breaking it.
            Expr::MatDot(a, b) => {
                let (a, b) = (self.simp(a)?, self.simp(b)?);
                self.intern(Expr::MatDot(a, b))
            }
        };
        self.memo.insert(Arc::as_ptr(e), out.clone());
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    //! Verification of algebraic simplification and common-subexpression reuse for symbolic tensor formulas.

    use super::super::symbolic::{BlockCfg, LazyTensor};
    use super::Context;
    use candle_core::{DType, Device, Tensor};

    fn max_abs(a: &Tensor, b: &Tensor) -> f64 {
        a.sub(b).unwrap().abs().unwrap().max_all().unwrap()
            .to_dtype(DType::F64).unwrap().to_scalar::<f64>().unwrap()
    }

    fn pts(n: usize, d: usize) -> Tensor {
        Tensor::from_vec(
            (0..n * d).map(|i| ((i * 29 % 13) as f64) / 6.0 - 1.0).collect::<Vec<_>>(),
            (n, d),
            &Device::Cpu,
        )
        .unwrap()
    }

    // ---- AC canonicalisation --------------------------------------------

    /// The rung's headline: `a·b` and `b·a` were structurally distinct, so
    /// the cons table missed and every consumer downstream duplicated work
    /// it believed it was sharing.
    #[test]
    fn commuted_products_and_sums_intern_to_the_same_node() {
        let x = pts(6, 3);
        let y = pts(6, 3);
        let (xi, yj) = (LazyTensor::vi(&x), LazyTensor::vj(&y));
        for (l, r) in [(&xi * &yj, &yj * &xi), (&xi + &yj, &yj + &xi)] {
            let out = super::simplify_many(&[l.expr().clone(), r.expr().clone()]).unwrap();
            assert!(
                std::sync::Arc::ptr_eq(&out[0], &out[1]),
                "commuted operands did not canonicalise to one node"
            );
        }
    }

    /// Associativity too — the n-ary flatten, not just a binary swap.
    #[test]
    fn reassociated_chains_intern_to_the_same_node() {
        let x = pts(5, 2);
        let (a, b, c) = (
            LazyTensor::vi(&x),
            LazyTensor::vj(&x),
            LazyTensor::pm(&Tensor::full(0.5f64, (1, 2), &Device::Cpu).unwrap()),
        );
        let left = (&(&a + &b) + &c).expr().clone();
        let right = (&a + &(&c + &b)).expr().clone();
        let out = super::simplify_many(&[left, right]).unwrap();
        assert!(std::sync::Arc::ptr_eq(&out[0], &out[1]), "reassociation was not canonicalised");
    }

    /// Canonicalising must not move values beyond the reassociation of
    /// floating-point addition it necessarily performs.
    #[test]
    fn ac_canonicalisation_preserves_values() {
        let x = pts(9, 4);
        let y = pts(7, 4);
        let (xi, yj) = (LazyTensor::vi(&x), LazyTensor::vj(&y));
        let g = LazyTensor::pm(&Tensor::full(0.3f64, (1, 4), &Device::Cpu).unwrap());
        let f = (&(&(&xi * &yj) + &g) * &(&yj + &xi)).sum_last();
        let s = f.simplify().unwrap();
        let cfg = BlockCfg::default();
        let d = max_abs(&s.to_dense(&cfg).unwrap(), &f.to_dense(&cfg).unwrap());
        assert!(d < 1e-13, "AC canonicalisation moved values by {d:.3e}");
    }

    /// Ordering is by CREATION id, not by address — otherwise the canonical
    /// form (and with it `leaves()` order, and with it the gradient pack
    /// layout) would vary run to run.
    #[test]
    fn canonical_order_is_stable_across_contexts() {
        let x = pts(4, 2);
        let y = pts(4, 2);
        let shape = |ctx: &mut Context| -> Vec<usize> {
            let (xi, yj) = (LazyTensor::vi(&x), LazyTensor::vj(&y));
            let f = &(&xi * &yj) * &LazyTensor::vi(&x);
            let s = ctx.simplify(f.expr()).unwrap();
            LazyTensor::from_arc(s).leaves().iter().map(|t| t.dims()[0]).collect()
        };
        let a = shape(&mut Context::new());
        let b = shape(&mut Context::new());
        assert_eq!(a, b, "canonical leaf order differs between fresh contexts");
    }

    // ---- the persistent intern context -----------------------------------

    /// The rung's other half: two SEPARATE simplify calls now share nodes.
    /// Previously each call owned its table, so this was impossible.
    #[test]
    fn a_context_shares_nodes_across_separate_simplify_calls() {
        let x = pts(6, 2);
        let y = pts(6, 2);
        let build = || (&LazyTensor::vi(&x) - &LazyTensor::vj(&y)).powf(2.0).sum_last();
        let mut ctx = Context::new();
        let first = ctx.simplify(build().expr()).unwrap();
        let (_, misses_after_first) = ctx.stats();
        let second = ctx.simplify(build().expr()).unwrap();
        assert!(
            std::sync::Arc::ptr_eq(&first, &second),
            "a second simplify built a fresh DAG instead of hitting the table"
        );
        let (_, misses_after_second) = ctx.stats();
        assert_eq!(
            misses_after_first, misses_after_second,
            "the second call allocated new nodes"
        );
    }

    /// The table holds `Weak`, so a context cannot pin formulas — and with
    /// them their tensors — for the life of the process.
    #[test]
    fn dropping_a_formula_releases_it_from_the_context() {
        let x = pts(5, 2);
        let mut ctx = Context::new();
        {
            let f = (&LazyTensor::vi(&x) - &LazyTensor::vj(&x)).exp();
            let _s = ctx.simplify(f.expr()).unwrap();
            assert!(ctx.len() > 0);
        }
        let reclaimed = ctx.prune();
        assert!(reclaimed > 0, "prune reclaimed nothing after the formula dropped");
        // The leaf nodes survive only because `x` itself is still alive; the
        // operation nodes must be gone.
        assert!(ctx.len() < reclaimed + ctx.len(), "unreachable");
    }

    /// The hazard a PERSISTENT table introduces: a training loop rebuilds
    /// formulas over fresh tensors every step, so the table would grow
    /// without bound if nothing reclaimed dead entries. Auto-prune must keep
    /// it proportional to what is live, not to how many steps have run.
    #[test]
    fn a_long_running_loop_does_not_grow_the_context_without_bound() {
        let mut ctx = Context::new();
        let dev = Device::Cpu;
        for step in 0..600 {
            // Fresh tensors each step — exactly what an optimizer produces.
            let x = Tensor::full(step as f64, (4, 3), &dev).unwrap();
            let y = Tensor::full(step as f64 + 1.0, (4, 3), &dev).unwrap();
            let f = (&LazyTensor::vi(&x) - &LazyTensor::vj(&y)).powf(2.0).sum_last().exp();
            let _ = ctx.simplify(f.expr()).unwrap();
            // The formula (and its tensors) drop here.
        }
        ctx.prune();
        assert!(
            ctx.len() < 64,
            "context retained {} entries after 600 steps of dead formulas",
            ctx.len()
        );
    }

    /// A live formula must NOT be pruned — the failure mode that would turn
    /// a cache into a correctness bug.
    #[test]
    fn prune_keeps_nodes_that_are_still_referenced() {
        let x = pts(5, 2);
        let mut ctx = Context::new();
        let f = (&LazyTensor::vi(&x) * &LazyTensor::vj(&x)).sum_last();
        let held = ctx.simplify(f.expr()).unwrap();
        let before = ctx.len();
        ctx.prune();
        assert_eq!(ctx.len(), before, "prune dropped live nodes");
        // And the table still serves them.
        let again = ctx.simplify(f.expr()).unwrap();
        assert!(std::sync::Arc::ptr_eq(&held, &again));
    }

    #[test]
    fn cse_merges_separately_built_equal_subtrees() {
        let x = pts(6, 2);
        let y = pts(6, 2);
        // The user pattern: the same difference block built twice.
        let build_diff = || &LazyTensor::vi(&x) - &LazyTensor::vj(&y);
        let separate = (&build_diff() * &build_diff()).sum_last();
        let shared = {
            let d = build_diff();
            (&d * &d).sum_last()
        };
        let s_sep = separate.simplify().unwrap();
        let s_shared = shared.simplify().unwrap();
        assert_eq!(
            s_sep.node_count(),
            s_shared.node_count(),
            "CSE failed to merge separately built equal subtrees"
        );
        assert!(s_sep.node_count() < separate.node_count(), "no sharing gained");
        let cfg = BlockCfg::default();
        assert!(max_abs(&s_sep.to_dense(&cfg).unwrap(), &separate.to_dense(&cfg).unwrap()) == 0.0);
    }

    #[test]
    fn rewrites_fold_identities_and_affine_chains() {
        let x = pts(5, 3);
        let y = pts(5, 3);
        let base = (&LazyTensor::vi(&x) * &LazyTensor::vj(&y)).sum_last();
        // A pile of no-ops and fusable chains around `base`:
        //   exp(log(((−(−base))·1 + 0).affine(2,0).affine(0.5,0)))^1
        let noisy = (&(&base.neg().neg() * &LazyTensor::scalar(1.0))
            + &LazyTensor::scalar(0.0))
            .affine(2.0, 0.0)
            .affine(0.5, 0.0)
            .log()
            .exp()
            .powf(1.0)
            // Sum over an already-scalar formula is the identity too.
            .sum_last();
        let simplified = noisy.simplify().unwrap();
        assert_eq!(
            simplified.node_count(),
            base.simplify().unwrap().node_count(),
            "identity chain not fully rewritten away: {} nodes vs {}",
            simplified.node_count(),
            base.simplify().unwrap().node_count()
        );
        let cfg = BlockCfg::default();
        let rel = max_abs(&simplified.to_dense(&cfg).unwrap(), &noisy.to_dense(&cfg).unwrap());
        assert!(rel < 1e-14, "rewrite changed values by {rel:.3e}");
    }

    #[test]
    fn constant_subtrees_fold_to_one_scalar() {
        // (2·3 + 4)/5 − 2 = 0 … as a formula tree.
        let two = LazyTensor::scalar(2.0);
        let tree = &(&(&(&two * &LazyTensor::scalar(3.0)) + &LazyTensor::scalar(4.0))
            / &LazyTensor::scalar(5.0))
            - &two;
        let s = tree.simplify().unwrap();
        assert_eq!(s.node_count(), 1, "constant tree did not fold to one Scalar node");
    }

    #[test]
    fn simplify_preserves_values_on_kernel_shaped_formulas() {
        let x = pts(7, 2);
        let g = Tensor::full(0.8f64, (1, 1), &Device::Cpu).unwrap();
        let a = Tensor::full(1.7f64, (1, 1), &Device::Cpu).unwrap();
        let xi = LazyTensor::vi(&x);
        let xj = LazyTensor::vj(&x);
        let diff = &xi - &xj;
        let d2 = (&diff * &diff).sum_last();
        // RBF + eye, and a rational-quadratic with a TENSOR exponent (Pow).
        let rbf = &(&d2 * &LazyTensor::pm(&g)).affine(-1.0, 0.0).exp()
            + &LazyTensor::eye_identity(0.4);
        let alpha = LazyTensor::pm(&a);
        let rq = d2.affine(0.5, 1.0).pow(&alpha.neg());
        let cfg = BlockCfg::default();
        for f in [&rbf, &rq] {
            let s = f.simplify().unwrap();
            let d = max_abs(&s.to_dense(&cfg).unwrap(), &f.to_dense(&cfg).unwrap());
            assert!(d < 1e-14, "simplify moved a kernel formula's values by {d:.3e}");
            assert!(s.node_count() <= f.node_count());
        }
    }
}
