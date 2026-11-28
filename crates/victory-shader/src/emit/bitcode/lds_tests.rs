use super::*;

#[derive(Debug, PartialEq)]
pub(crate) enum GuestEvent {
    Fence,
    Load,
    Store,
}

pub(crate) struct GuestLds {
    pub globals: Vec<(String, u32)>,
    pub events: Vec<Vec<GuestEvent>>,
}

impl Module {
    pub(crate) fn guest_lds_trace(&self) -> GuestLds {
        let plan = self.plan();
        let globals = self
            .globals
            .iter()
            .filter(|g| g.space == 3)
            .map(|g| {
                let TypeDef::Array(n, t) = self.types[g.ty.0 as usize] else {
                    panic!("shared array")
                };
                assert_eq!(self.types[t.0 as usize], TypeDef::Int(32));
                (g.name.clone(), n)
            })
            .collect();
        let mut events: Vec<Vec<_>> = (0..self.blocks.len()).map(|_| Vec::new()).collect();
        let guest = |p| {
            matches!(p, V::Inst(i) if matches!(
            self.insts[i as usize].op, Op::Gep(_, V::Global(g), _) if self.globals[g as usize].name == "lds"))
        };
        for &b in &plan.layout {
            for &i in self.blocks[b]
                .body
                .iter()
                .filter(|&&i| plan.live[i as usize])
            {
                let event = match &self.insts[i as usize].op {
                    // The fixture writes a UAV, so guest phase fences carry the UAV
                    // bit too. Ticket and scalar-publication fences use mode 9.
                    Op::Call(f, a)
                        if self.funcs[*f as usize].name == "dx.op.barrier"
                            && self.int_value(a[1]) == Some(11) =>
                    {
                        GuestEvent::Fence
                    }
                    Op::Load(p) if guest(*p) => GuestEvent::Load,
                    Op::Store(p, _) if guest(*p) => GuestEvent::Store,
                    _ => continue,
                };
                events[b].push(event);
            }
        }
        GuestLds { globals, events }
    }
}
