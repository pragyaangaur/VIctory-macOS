//! D3D12 encode and submission brackets stored in the timeline's fixed buffer.

use std::io::{self, BufWriter, Write};
use std::path::Path;
use windows::Win32::Graphics::Direct3D12::ID3D12CommandList;
use windows::core::Interface;

use super::record::{self, Data, Kind};

#[inline(always)]
pub(crate) fn start() -> u64 {
    if record::metal() { record::ticks() } else { 0 }
}

/// A host command with the shader hash and its first dimension or count.
#[inline(always)]
pub(crate) fn command(
    started: u64,
    name: &'static str,
    list: &impl Interface,
    shader: impl FnOnce() -> Option<&'static str>,
    count: u32,
) {
    if started != 0 && record::metal() {
        let ended = record::ticks();
        let shader = shader()
            .and_then(|s| u64::from_str_radix(s, 16).ok())
            .unwrap_or(0);
        record::put(
            Kind::MetalCall,
            name,
            (started, ended),
            list.as_raw() as u64,
            shader,
            count,
        );
    }
}

/// All lists in one ExecuteCommandLists call share the same bracket.
pub(crate) fn submitted(started: u64, fence: u64, lists: &[Option<ID3D12CommandList>]) {
    if started != 0 && record::metal() {
        let ended = record::ticks();
        for (index, list) in lists.iter().enumerate() {
            if let Some(list) = list {
                record::put(
                    Kind::MetalCall,
                    "submit",
                    (started, ended),
                    list.as_raw() as u64,
                    fence,
                    index as u32,
                );
            }
        }
    }
}

pub(super) fn write(file: &Path, data: Data) -> io::Result<()> {
    let mut out = BufWriter::new(std::fs::File::create(file)?);
    writeln!(out, "# qpc_hz={}", record::frequency())?;
    writeln!(out, "# windows_pid={}", std::process::id())?;
    writeln!(out, "# lost={}", data.lost)?;
    writeln!(
        out,
        "kind\tqpc_start\tqpc_end\twindows_tid\tlist\tshader_or_fence\tcount_or_index"
    )?;
    for e in data
        .events
        .iter()
        .filter(|e| matches!(e.kind, Kind::MetalCall | Kind::Frame))
    {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{:x}\t{:x}\t{}",
            e.name, e.t0, e.t1, e.thread, e.a, e.b, e.stack,
        )?;
    }
    out.flush()
}
