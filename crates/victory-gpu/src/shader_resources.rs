//! Cache the descriptor and workgroup analysis of a title shader.
//! Host translations have their own encoding and cache key.

#[cfg(windows)]
use std::path::PathBuf;
#[cfg(windows)]
use std::sync::{Mutex, OnceLock};

#[cfg(windows)]
use victory_pipeline::{Cache, Converter};
use victory_shader::RootType;

#[cfg(windows)]
static CACHE: OnceLock<Cache> = OnceLock::new();

/// Keeps analysis across runs, independent of host translation settings.
#[cfg(windows)]
pub(crate) fn use_cache(dir: PathBuf) {
    match Cache::open(dir) {
        Ok(cache) => {
            let _ = CACHE.set(cache);
        }
        Err(e) => tracing::warn!(error = %e, "no shader resource analysis cache"),
    }
}

/// What `crate::pipeline` reads from a shader container besides its
/// translation (`pipeline::analyze`): the descriptor ranges its program
/// reads, which of them a buffer store or atomic writes, whether a write
/// could not be traced, and a compute program's group size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Resources {
    pub(crate) used: Vec<victory_shader::UsedResource>,
    pub(crate) used_written: Vec<bool>,
    pub(crate) write_unknown: bool,
    pub(crate) threads: Option<[u32; 3]>,
}

/// Bump when `pipeline::analyze` finds something else in a container than
/// before for a reason `victory_shader::SOURCE_HASH` (part of the key) does
/// not cover: its own code is in this crate.
#[cfg(windows)]
const RESOURCES_VERSION: u32 = 1;

/// The cache entry's converter for [`Resources`]: `victory_shader`'s source
/// hash and the container in, the encoded analysis (or refusal) out; `made`
/// keeps a miss's result, avoiding a second analysis.
#[cfg(windows)]
struct Analyze<F> {
    analyze: F,
    made: Mutex<Option<Result<Resources, String>>>,
}

#[cfg(windows)]
impl<F: Fn(&[u8]) -> Result<Resources, String> + Send + Sync> Converter for Analyze<F> {
    fn id(&self) -> &'static str {
        "shader.resources"
    }
    fn version(&self) -> u32 {
        RESOURCES_VERSION
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        let made = (self.analyze)(&input[victory_shader::SOURCE_HASH.len()..]);
        let bytes = encode_resources(&made);
        *self.made.lock().unwrap_or_else(|e| e.into_inner()) = Some(made);
        Ok(bytes)
    }
}

/// Stores `analyze(container)` alongside the host shader cache. Warm
/// entries save decoding and tracing the program before pipeline creation.
#[cfg(windows)]
pub(crate) fn resources(
    container: &[u8],
    analyze: impl Fn(&[u8]) -> Result<Resources, String> + Send + Sync,
) -> Result<Resources, String> {
    let Some(cache) = CACHE.get() else {
        return analyze(container);
    };
    let input = [victory_shader::SOURCE_HASH.as_bytes(), container].concat();
    let converter = Analyze {
        analyze,
        made: Mutex::new(None),
    };
    let made = || {
        converter
            .made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    };
    match cache.get_or_convert(&converter, &input) {
        Ok(path) => {
            if let Some(made) = made() {
                return made;
            }
            if let Some(found) = crate::read_file(&path)
                .ok()
                .and_then(|b| decode_resources(&b))
            {
                return found;
            }
            tracing::debug!(path = %path.display(), "unreadable shader resources entry; analyzing again");
            let _ = std::fs::remove_file(&path);
        }
        Err(e) => {
            tracing::warn!(
                error = format!("{e:#}"),
                "shader resources entry not written"
            );
        }
    }
    made().unwrap_or_else(|| (converter.analyze)(container))
}

const RESOURCES_MAGIC: &[u8; 4] = b"VRS1";

fn encode_resources(found: &Result<Resources, String>) -> Vec<u8> {
    use victory_shader::ResourceType as K;
    let mut b = RESOURCES_MAGIC.to_vec();
    let r = match found {
        Err(e) => {
            b.push(0);
            b.extend_from_slice(e.as_bytes());
            return b;
        }
        Ok(r) => r,
    };
    b.push(1);
    b.push(u8::from(r.write_unknown));
    match r.threads {
        Some(t) => {
            b.push(1);
            t.iter().for_each(|n| b.extend_from_slice(&n.to_le_bytes()));
        }
        None => b.push(0),
    }
    b.extend_from_slice(&(r.used.len() as u32).to_le_bytes());
    for u in &r.used {
        let (kind, kind_other) = match u.kind {
            K::Srv => (0, 0),
            K::Sampler => (1, 0),
            K::Cbv => (2, 0),
            K::Uav => (3, 0),
            K::Other(n) => (4, n),
        };
        let (root, root_other) = match u.root {
            RootType::Constants => (0, 0),
            RootType::Descriptor => (1, 0),
            RootType::Table => (2, 0),
            RootType::ImmediateSampler => (3, 0),
            RootType::ResourceHeap => (4, 0),
            RootType::SamplerHeap => (5, 0),
            RootType::Other(n) => (6, n),
        };
        b.extend_from_slice(&[kind, kind_other, root]);
        b.extend_from_slice(&root_other.to_le_bytes());
        for n in [u.table_offset_dwords, u.lower_bound, u.stride] {
            b.extend_from_slice(&n.to_le_bytes());
        }
        b.extend_from_slice(&[u.dwords, u.user_data, u.slot]);
    }
    b.extend_from_slice(&(r.used_written.len() as u32).to_le_bytes());
    b.extend(r.used_written.iter().map(|&w| u8::from(w)));
    b
}

/// Bytes read from the front.
struct Cursor<'a>(&'a [u8]);

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (head, rest) = self.0.split_at_checked(n)?;
        self.0 = rest;
        Some(head)
    }
    fn byte(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn flag(&mut self) -> Option<bool> {
        match self.byte()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
}

fn decode_resources(bytes: &[u8]) -> Option<Result<Resources, String>> {
    use victory_shader::ResourceType as K;
    let mut c = Cursor(bytes.strip_prefix(RESOURCES_MAGIC)?);
    if !c.flag()? {
        return Some(Err(String::from_utf8_lossy(c.0).into_owned()));
    }
    let write_unknown = c.flag()?;
    let threads = match c.flag()? {
        true => Some([c.u32()?, c.u32()?, c.u32()?]),
        false => None,
    };
    let count = c.u32()? as usize;
    let mut used = Vec::with_capacity(count.min(4096));
    for _ in 0..count {
        let (kind, kind_other, root) = (c.byte()?, c.byte()?, c.byte()?);
        let root_other = c.u32()?;
        let (table_offset_dwords, lower_bound, stride) = (c.u32()?, c.u32()?, c.u32()?);
        let (dwords, user_data, slot) = (c.byte()?, c.byte()?, c.byte()?);
        used.push(victory_shader::UsedResource {
            kind: match kind {
                0 => K::Srv,
                1 => K::Sampler,
                2 => K::Cbv,
                3 => K::Uav,
                4 => K::Other(kind_other),
                _ => return None,
            },
            root: match root {
                0 => RootType::Constants,
                1 => RootType::Descriptor,
                2 => RootType::Table,
                3 => RootType::ImmediateSampler,
                4 => RootType::ResourceHeap,
                5 => RootType::SamplerHeap,
                6 => RootType::Other(root_other),
                _ => return None,
            },
            table_offset_dwords,
            lower_bound,
            stride,
            dwords,
            user_data,
            slot,
        });
    }
    let written = c.u32()? as usize;
    if written != count {
        return None;
    }
    let used_written = (0..written).map(|_| c.flag()).collect::<Option<Vec<_>>>()?;
    c.0.is_empty().then_some(Ok(Resources {
        used,
        used_written,
        write_unknown,
        threads,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cached analysis reads back as it was written, every kind and root
    /// included; a refusal keeps its message; a cut or lengthened entry is
    /// refused, so it is made again rather than misread.
    #[test]
    fn resources_entries_read_back_whole_or_not_at_all() {
        use victory_shader::{ResourceType as K, UsedResource};
        let kinds = [K::Srv, K::Sampler, K::Cbv, K::Uav, K::Other(9)];
        let roots = [
            RootType::Constants,
            RootType::Descriptor,
            RootType::Table,
            RootType::ImmediateSampler,
            RootType::ResourceHeap,
            RootType::SamplerHeap,
            RootType::Other(0xdead_beef),
        ];
        let used: Vec<UsedResource> = (0..kinds.len() * roots.len())
            .map(|i| UsedResource {
                kind: kinds[i % kinds.len()],
                root: roots[i % roots.len()],
                table_offset_dwords: i as u32 * 8,
                lower_bound: i as u32,
                stride: 32,
                dwords: if i % 2 == 0 { 4 } else { 8 },
                user_data: i as u8,
                slot: 255 - i as u8,
            })
            .collect();
        let found = Resources {
            used_written: (0..used.len()).map(|i| i % 3 == 0).collect(),
            used,
            write_unknown: true,
            threads: Some([64, 1, 1]),
        };
        let bytes = encode_resources(&Ok(found.clone()));
        assert_eq!(decode_resources(&bytes), Some(Ok(found.clone())));
        let none = Resources {
            used: Vec::new(),
            used_written: Vec::new(),
            write_unknown: false,
            threads: None,
        };
        let bytes_none = encode_resources(&Ok(none.clone()));
        assert_eq!(decode_resources(&bytes_none), Some(Ok(none)));
        let refused: Result<Resources, String> = Err("GFX10 code at 0x150c: unsupported".into());
        assert_eq!(decode_resources(&encode_resources(&refused)), Some(refused));
        for cut in [0, 4, 5, 9, bytes.len() / 2, bytes.len() - 1] {
            assert_eq!(decode_resources(&bytes[..cut]), None, "cut at {cut}");
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert_eq!(decode_resources(&longer), None);
        let mut invalid_flag = bytes;
        *invalid_flag.last_mut().unwrap() = 2;
        assert_eq!(decode_resources(&invalid_flag), None);
    }
}
