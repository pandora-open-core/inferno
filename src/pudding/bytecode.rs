//! Versioned lane-wise maps. Identical DAG/order for scalar and SIMD lanes.
//! Algebraic maps use runtime-dispatched AVX2 on x86_64, with a scalar tail.
//! Exp/log retain platform libm semantics; other architectures use the scalar map.
//! Transcendentals use Rust's platform f64 functions, not certified intervals.
const MAX_NODES: usize = 4096;
const MAX_INPUTS: usize = 256;
const MAGIC: &[u8; 5] = b"DKMP\x01";

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Input(u16),
    Constant(f64),
    Add(u16, u16),
    Sub(u16, u16),
    Mul(u16, u16),
    Div(u16, u16),
    Exp(u16),
    Log(u16),
}
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    inputs: u16,
    nodes: Vec<Op>,
    root: u16,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Format,
    Budget,
    Reference,
    Domain,
    NonFinite,
    Shape,
}

impl Program {
    pub fn new(inputs: u16, nodes: Vec<Op>, root: u16) -> Result<Self, Error> {
        if inputs as usize > MAX_INPUTS || nodes.is_empty() || nodes.len() > MAX_NODES {
            return Err(Error::Budget);
        }
        if root as usize >= nodes.len() {
            return Err(Error::Reference);
        }
        for (i, op) in nodes.iter().enumerate() {
            let valid = |j: u16| (j as usize) < i;
            match *op {
                Op::Input(j) if j >= inputs => return Err(Error::Reference),
                Op::Constant(c) if !c.is_finite() => return Err(Error::NonFinite),
                Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b)
                    if !valid(a) || !valid(b) =>
                {
                    return Err(Error::Reference)
                }
                Op::Exp(a) | Op::Log(a) if !valid(a) => return Err(Error::Reference),
                _ => (),
            }
        }
        Ok(Self {
            inputs,
            nodes,
            root,
        })
    }
    /// Each input is one map over L independent lanes, with DAG sharing.
    /// Reject a whole packet on any invalid lane, including unused inputs/nodes.
    pub fn eval<const L: usize>(&self, inputs: &[[f64; L]]) -> Result<[f64; L], Error> {
        if L == 0 || L > 64 || inputs.len() != self.inputs as usize {
            return Err(Error::Shape);
        }
        if inputs.iter().flatten().any(|x| !x.is_finite()) {
            return Err(Error::NonFinite);
        }
        let mut values: Vec<[f64; L]> = Vec::with_capacity(self.nodes.len());
        for op in &self.nodes {
            let mut out = [0.; L];
            let binary = match *op {
                Op::Add(a, b) => Some((a, b, 0)),
                Op::Sub(a, b) => Some((a, b, 1)),
                Op::Mul(a, b) => Some((a, b, 2)),
                Op::Div(a, b) => Some((a, b, 3)),
                _ => None,
            };
            if let Some((a, b, kind)) = binary {
                let a = &values[a as usize];
                let b = &values[b as usize];
                if kind == 3 && b.contains(&0.) {
                    return Err(Error::Domain);
                }
                map_binary(a, b, &mut out, kind);
            } else {
                for lane in 0..L {
                    let get = |j: u16| values[j as usize][lane];
                    out[lane] = match *op {
                        Op::Input(j) => inputs[j as usize][lane],
                        Op::Constant(c) => c,
                        Op::Exp(a) => get(a).exp(),
                        Op::Log(a) => {
                            let x = get(a);
                            if x <= 0. {
                                return Err(Error::Domain);
                            }
                            x.ln()
                        }
                        _ => unreachable!("binary map handled above"),
                    };
                }
            }
            for x in &out {
                if !x.is_finite() {
                    return Err(Error::NonFinite);
                }
                if matches!(op, Op::Exp(_)) && *x == 0. {
                    return Err(Error::Domain);
                }
            }
            values.push(out);
        }
        Ok(values[self.root as usize])
    }
    pub fn encode(&self) -> Vec<u8> {
        let mut b = MAGIC.to_vec();
        for n in [self.inputs, self.nodes.len() as u16, self.root] {
            b.extend(n.to_le_bytes());
        }
        for op in &self.nodes {
            let (tag, args): (u8, Vec<u8>) = match *op {
                Op::Input(a) => (0, a.to_le_bytes().to_vec()),
                Op::Constant(c) => (1, c.to_le_bytes().to_vec()),
                Op::Add(a, b) => (2, [a.to_le_bytes(), b.to_le_bytes()].concat()),
                Op::Sub(a, b) => (3, [a.to_le_bytes(), b.to_le_bytes()].concat()),
                Op::Mul(a, b) => (4, [a.to_le_bytes(), b.to_le_bytes()].concat()),
                Op::Div(a, b) => (5, [a.to_le_bytes(), b.to_le_bytes()].concat()),
                Op::Exp(a) => (6, a.to_le_bytes().to_vec()),
                Op::Log(a) => (7, a.to_le_bytes().to_vec()),
            };
            b.push(tag);
            b.extend(args);
        }
        b
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() > 11 + MAX_NODES * 9 || !bytes.starts_with(MAGIC) {
            return Err(Error::Format);
        }
        struct Read<'a> {
            b: &'a [u8],
            at: usize,
        }
        impl Read<'_> {
            fn take<const N: usize>(&mut self) -> Result<[u8; N], Error> {
                let end = self.at.checked_add(N).ok_or(Error::Format)?;
                let a = self
                    .b
                    .get(self.at..end)
                    .ok_or(Error::Format)?
                    .try_into()
                    .map_err(|_| Error::Format)?;
                self.at = end;
                Ok(a)
            }
            fn u16(&mut self) -> Result<u16, Error> {
                Ok(u16::from_le_bytes(self.take()?))
            }
        }
        let mut r = Read { b: bytes, at: 5 };
        let inputs = r.u16()?;
        let count = r.u16()?;
        let root = r.u16()?;
        if count == 0 || count as usize > MAX_NODES {
            return Err(Error::Budget);
        }
        let mut nodes = Vec::with_capacity(count as usize);
        for _ in 0..count {
            nodes.push(match r.take::<1>()?[0] {
                0 => Op::Input(r.u16()?),
                1 => Op::Constant(f64::from_le_bytes(r.take()?)),
                2 => Op::Add(r.u16()?, r.u16()?),
                3 => Op::Sub(r.u16()?, r.u16()?),
                4 => Op::Mul(r.u16()?, r.u16()?),
                5 => Op::Div(r.u16()?, r.u16()?),
                6 => Op::Exp(r.u16()?),
                7 => Op::Log(r.u16()?),
                _ => return Err(Error::Format),
            });
        }
        if r.at != bytes.len() {
            return Err(Error::Format);
        }
        Self::new(inputs, nodes, root)
    }
}

fn map_binary(a: &[f64], b: &[f64], out: &mut [f64], kind: u8) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime feature check; equal slice lengths above are enforced
        // by eval's same-sized arrays. The vector loop checks every full block.
        unsafe {
            map_avx2(a, b, out, kind);
        }
        return;
    }
    for i in 0..out.len() {
        out[i] = binary_scalar(a[i], b[i], kind);
    }
}
fn binary_scalar(a: f64, b: f64, kind: u8) -> f64 {
    match kind {
        0 => a + b,
        1 => a - b,
        2 => a * b,
        3 => a / b,
        _ => unreachable!(),
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn map_avx2(a: &[f64], b: &[f64], out: &mut [f64], kind: u8) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 4 <= out.len() {
        // SAFETY: each full block lies within all three slices; unaligned loads
        // require no extra alignment. No reassociation or fused multiply-add.
        unsafe {
            let x = _mm256_loadu_pd(a.as_ptr().add(i));
            let y = _mm256_loadu_pd(b.as_ptr().add(i));
            let z = match kind {
                0 => _mm256_add_pd(x, y),
                1 => _mm256_sub_pd(x, y),
                2 => _mm256_mul_pd(x, y),
                3 => _mm256_div_pd(x, y),
                _ => unreachable!(),
            };
            _mm256_storeu_pd(out.as_mut_ptr().add(i), z);
        }
        i += 4;
    }
    while i < out.len() {
        out[i] = binary_scalar(a[i], b[i], kind);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packet_scalar_and_roundtrip() {
        let p = Program::new(
            2,
            vec![Op::Input(0), Op::Input(1), Op::Sub(0, 1), Op::Mul(2, 2)],
            3,
        )
        .unwrap();
        assert_eq!(Program::decode(&p.encode()), Ok(p.clone()));
        let golden = include_bytes!("../../tests/pudding_fixtures/squared-map.dkmp");
        assert_eq!(p.encode(), golden);
        assert_eq!(Program::decode(golden), Ok(p.clone()));
        let a = [1., 2., 3., 4., 5., 6., 7., 8.];
        let b = [8., 7., 6., 5., 4., 3., 2., 1.];
        let packet = p.eval(&[a, b]).unwrap();
        for i in 0..8 {
            assert_eq!(packet[i], p.eval(&[[a[i]], [b[i]]]).unwrap()[0]);
        }
    }
    #[test]
    fn algebraic_simd_blocks_and_tail_match_scalar() {
        let a = [1., 2., 3., 4., 5., 6., 7.];
        let b = [8., 7., 6., 5., 4., 3., 2.];
        for kind in 0..4 {
            let mut out = [0.; 7];
            map_binary(&a, &b, &mut out, kind);
            for i in 0..7 {
                assert_eq!(out[i].to_bits(), binary_scalar(a[i], b[i], kind).to_bits());
            }
        }
        let p = Program::new(
            1,
            vec![
                Op::Input(0),
                Op::Log(0),
                Op::Exp(1),
                Op::Constant(2.),
                Op::Div(2, 3),
                Op::Add(4, 3),
            ],
            5,
        )
        .unwrap();
        let packet = p.eval(&[a]).unwrap();
        for i in 0..7 {
            assert_eq!(packet[i].to_bits(), p.eval(&[[a[i]]]).unwrap()[0].to_bits());
        }
    }
    #[test]
    fn rejects_untrusted_graph_and_bytes() {
        assert_eq!(Program::new(0, vec![Op::Exp(0)], 0), Err(Error::Reference));
        assert_eq!(
            Program::new(0, vec![Op::Constant(f64::NAN)], 0),
            Err(Error::NonFinite)
        );
        let p = Program::new(1, vec![Op::Input(0)], 0).unwrap();
        let b = p.encode();
        for len in 0..b.len() {
            assert!(Program::decode(&b[..len]).is_err());
        }
        let mut bad = b.clone();
        bad.push(0);
        assert_eq!(Program::decode(&bad), Err(Error::Format));
        let mut bad = b;
        bad[11] = 255;
        assert_eq!(Program::decode(&bad), Err(Error::Format));
        assert_eq!(p.eval::<0>(&[[]]), Err(Error::Shape));
        assert_eq!(p.eval(&[[f64::INFINITY]]), Err(Error::NonFinite));
    }
    #[test]
    fn rejects_domain_overflow_underflow() {
        for x in [0., -1.] {
            let p = Program::new(1, vec![Op::Input(0), Op::Log(0)], 1).unwrap();
            assert_eq!(p.eval(&[[x]]), Err(Error::Domain));
        }
        let p = Program::new(1, vec![Op::Input(0), Op::Exp(0)], 1).unwrap();
        assert_eq!(p.eval(&[[1000.]]), Err(Error::NonFinite));
        assert_eq!(p.eval(&[[-1000.]]), Err(Error::Domain));
        let p = Program::new(1, vec![Op::Input(0), Op::Div(0, 0)], 1).unwrap();
        assert_eq!(p.eval(&[[0.]]), Err(Error::Domain));
    }
}
