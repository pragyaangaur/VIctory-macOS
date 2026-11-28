//! The offline batch: every shader found goes through the translator, and
//! what it cannot handle yet is tallied (decision 0003).

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};
use victory_gpu::shader_cache::translation_options;
use victory_shader::emit::host::{self, Host};
use victory_shader::emit::{self, validator::Validator};
use victory_shader::structure::{self, Atom, LaneExit};
use victory_shader::{
    Container, FourCC, HwStage, NativeProgram, RootType, ShaderError, ir, native_programs,
};
use victory_shader::{io, resource};

use crate::shader::{MAX_FILE, Source, sources};

#[derive(Default)]
struct Tally {
    containers: usize,
    duplicates: usize,
    programs: usize,
    extra_programs: usize,
    ir_built: usize,
    runtime_emitted: usize,
    instructions: usize,
    structured: usize,
    vector: usize,
    guarded: usize,
    /// Guards that read a mask the pass cannot see into.
    opaque: usize,
    lane_branches: usize,
    uniform_branches: usize,
    tracked: usize,
    loaded: usize,
    built: usize,
    constant: usize,
    /// Descriptors and scalar accesses with a computed address term.
    dynamic: usize,
    scalar: usize,
    used: usize,
    /// Used-resource entries whose root parameter no descriptor traces
    /// to, by resource and root type.
    untraced_used: BTreeMap<String, usize>,
    /// Root parameters descriptors trace to that no used entry names.
    unlisted: usize,
    /// Vertex and pixel programs, and those whose I/O was raised.
    io_programs: usize,
    io_raised: usize,
    interpolated: usize,
    vertex_values: usize,
    /// Exports by target kind.
    exports: BTreeMap<&'static str, usize>,
    /// Exports still packed as f16 pairs after the export patch.
    compressed: usize,
    /// Vertex and pixel programs every earlier stage handled, and how far
    /// emission got with them.
    emit_programs: usize,
    emitted: usize,
    validated: usize,
    pipelines: usize,
    /// The same for compute programs.
    cs_programs: usize,
    cs_emitted: usize,
    cs_validated: usize,
    cs_pipelines: usize,
    /// The same for mesh programs.
    ms_programs: usize,
    ms_emitted: usize,
    ms_validated: usize,
    ms_pipelines: usize,
    /// Reason → programs stopped by it, and the first one.
    failed: BTreeMap<String, (usize, String)>,
    /// With `--list`: one line per program and outcome.
    list: bool,
    lines: Vec<String>,
}

impl Tally {
    fn fail(&mut self, reason: String, example: impl FnOnce() -> String) {
        if self.list {
            let e = example();
            self.lines.push(format!("{e}  {reason}"));
            self.failed.entry(reason).or_insert((0, e)).0 += 1;
            return;
        }
        self.note_failure(reason, example);
    }

    fn note_failure(&mut self, reason: String, example: impl FnOnce() -> String) {
        self.failed
            .entry(reason)
            .or_insert_with(|| (0, example()))
            .0 += 1;
    }

    fn runtime_outcome(
        &mut self,
        hash: blake3::Hash,
        selected: Option<(HwStage, usize)>,
        result: Result<&emit::HostShader, &ShaderError>,
        example: impl FnOnce() -> String,
    ) {
        self.runtime_emitted += usize::from(result.is_ok());
        if self.list {
            let stage = selected.map_or_else(
                || "-".into(),
                |(stage, index)| format!("{stage:?} #{index}"),
            );
            let outcome = match result {
                Ok(shader) => format!("emitted {}", blake3::hash(&shader.dxil)),
                Err(error) => format!("refused {error}"),
            };
            self.lines
                .push(format!("runtime {hash} {stage} {outcome}  {}", example()));
        }
    }

    fn merge(&mut self, other: Tally) {
        self.containers += other.containers;
        self.duplicates += other.duplicates;
        self.programs += other.programs;
        self.extra_programs += other.extra_programs;
        self.ir_built += other.ir_built;
        self.runtime_emitted += other.runtime_emitted;
        self.instructions += other.instructions;
        self.structured += other.structured;
        self.vector += other.vector;
        self.guarded += other.guarded;
        self.opaque += other.opaque;
        self.lane_branches += other.lane_branches;
        self.uniform_branches += other.uniform_branches;
        self.tracked += other.tracked;
        self.loaded += other.loaded;
        self.built += other.built;
        self.constant += other.constant;
        self.dynamic += other.dynamic;
        self.scalar += other.scalar;
        self.used += other.used;
        for (k, n) in other.untraced_used {
            *self.untraced_used.entry(k).or_default() += n;
        }
        self.unlisted += other.unlisted;
        self.io_programs += other.io_programs;
        self.io_raised += other.io_raised;
        self.interpolated += other.interpolated;
        self.vertex_values += other.vertex_values;
        self.compressed += other.compressed;
        self.emit_programs += other.emit_programs;
        self.emitted += other.emitted;
        self.validated += other.validated;
        self.pipelines += other.pipelines;
        self.cs_programs += other.cs_programs;
        self.cs_emitted += other.cs_emitted;
        self.cs_validated += other.cs_validated;
        self.cs_pipelines += other.cs_pipelines;
        self.ms_programs += other.ms_programs;
        self.ms_emitted += other.ms_emitted;
        self.ms_validated += other.ms_validated;
        self.ms_pipelines += other.ms_pipelines;
        for (k, n) in other.exports {
            *self.exports.entry(k).or_default() += n;
        }
        self.lines.extend(other.lines);
        for (reason, (n, example)) in other.failed {
            self.failed.entry(reason).or_insert((0, example)).0 += n;
        }
    }
}

/// Emitted shaders by label, shared between threads.
type Kept = std::sync::Arc<Mutex<Vec<(String, emit::HostShader)>>>;

/// The optional checks after emission, one set per thread.
struct Checks {
    /// Shaders named in `VICTORY_PAIRS` (`vs+ps,...`, label substrings),
    /// kept to build those pipelines after the walk.
    keep: Option<Kept>,
    validator: Option<Validator>,
    host: Option<(Host, emit::HostShader)>,
    /// Translate for a host of 32-lane waves (`--wave32`, or the GPU
    /// `--pipelines` makes them on).
    wave32: bool,
}

pub fn run(
    paths: Vec<PathBuf>,
    validate: bool,
    pipelines: bool,
    list: bool,
    jobs: Option<usize>,
    wave32: bool,
) -> Result<()> {
    // Dumped shaders and validator reports are game-derived (rule #1).
    for name in ["VICTORY_EMIT_DUMP", "VICTORY_FAIL_DUMP"] {
        if let Some(dir) = std::env::var_os(name) {
            victory_core::ensure_outside_repo(Path::new(&dir))?;
        }
    }
    let paths = crate::default_paths(paths)?;
    let dll = validate.then(Validator::default_path).flatten();
    if validate {
        Validator::load(dll.as_deref()).map_err(anyhow::Error::msg)?;
        println!(
            "validator    {}",
            dll.as_ref()
                .map_or("dxil.dll".into(), |p| p.display().to_string())
        );
    }
    let wave32 = if pipelines {
        let host = Host::new(false).map_err(anyhow::Error::msg)?;
        // As `run` translates for this GPU.
        wave32 || host.wave32().map_err(anyhow::Error::msg)?
    } else {
        wave32
    };
    if wave32 {
        println!("waves        of 32 lanes at most (wave64 programs in halves)");
    }
    let pairs: Vec<(String, String)> = std::env::var("VICTORY_PAIRS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|p| p.split_once('+'))
        .map(|(v, p)| (v.trim().to_owned(), p.trim().to_owned()))
        .collect();
    let kept = std::sync::Arc::new(Mutex::new(Vec::new()));
    let threads = thread_count(jobs);
    let seen = Mutex::new(HashSet::new());
    let total = Mutex::new(Tally::default());
    let started = std::time::Instant::now();
    for root in &paths {
        for entry in walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
        {
            if !entry.file_type().is_file() || entry.metadata().map_or(0, |m| m.len()) > MAX_FILE {
                continue;
            }
            let Ok(data) = std::fs::read(entry.path()) else {
                continue;
            };
            let jobs = sources(&data);
            let next = AtomicUsize::new(0);
            std::thread::scope(|s| {
                let workers: Vec<_> = (0..threads.min(jobs.len()))
                    .map(|_| {
                        s.spawn(|| {
                            let mut t = Tally {
                                list,
                                ..Tally::default()
                            };
                            let mut checks = Checks {
                                keep: (!pairs.is_empty()).then(|| kept.clone()),
                                validator: validate
                                    .then(|| Validator::load(dll.as_deref()).ok())
                                    .flatten(),
                                host: pipelines
                                    .then(|| Host::new(false).ok().map(|h| (h, host::stub_ps())))
                                    .flatten(),
                                wave32,
                            };
                            while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                                translate(job, entry.path(), &seen, &mut t, &mut checks);
                            }
                            lock(&total).merge(t);
                        })
                    })
                    .collect();
                report(
                    entry.path(),
                    jobs.len(),
                    "containers",
                    &next,
                    started,
                    &workers,
                    String::new,
                );
            });
        }
    }
    let t = total.into_inner().unwrap_or_else(|e| e.into_inner());
    println!(
        "containers   {} distinct ({} duplicates skipped)",
        t.containers, t.duplicates
    );
    println!(
        "runtime      {} emitted of {} distinct containers; {} refused",
        t.runtime_emitted,
        t.containers,
        t.containers - t.runtime_emitted
    );
    println!(
        "programs     {} IR builds of {} native programs ({} extra diagnostic programs; {} IR instructions)",
        t.ir_built, t.programs, t.extra_programs, t.instructions
    );
    println!(
        "structured   {} of {} ({} vector instructions, {} guarded, {} on opaque masks; \
         branches {} per lane, {} uniform)",
        t.structured,
        t.ir_built,
        t.vector,
        t.guarded,
        t.opaque,
        t.lane_branches,
        t.uniform_branches
    );
    println!(
        "resources    {} of {} tracked (descriptors {} loaded, {} built, {} constant; \
         {} scalar accesses; {} with a computed term)",
        t.tracked, t.ir_built, t.loaded, t.built, t.constant, t.scalar, t.dynamic
    );
    println!(
        "  used-resource entries {}: {} not traced {:?}; {} traced parameters not listed",
        t.used,
        t.untraced_used.values().sum::<usize>(),
        t.untraced_used,
        t.unlisted
    );
    println!(
        "I/O          {} of {} vertex and pixel programs raised ({} interpolations, \
         {} vertex values; exports {:?}, {} still f16 pairs)",
        t.io_raised, t.io_programs, t.interpolated, t.vertex_values, t.exports, t.compressed
    );
    println!(
        "DXIL         {} emitted of {} vertex and pixel programs; {} validated, {} pipelines",
        t.emitted, t.emit_programs, t.validated, t.pipelines
    );
    println!(
        "             {} emitted of {} compute programs; {} validated, {} pipelines",
        t.cs_emitted, t.cs_programs, t.cs_validated, t.cs_pipelines
    );
    println!(
        "             {} emitted of {} mesh programs; {} validated, {} pipelines",
        t.ms_emitted, t.ms_programs, t.ms_validated, t.ms_pipelines
    );
    if !pairs.is_empty() {
        let host = Host::new(false).map_err(anyhow::Error::msg)?;
        let kept = lock(&kept);
        let find = |name: &str| kept.iter().find(|(l, _)| l.contains(name));
        for (v, p) in &pairs {
            let made = match (find(v), find(p)) {
                (Some((_, vs)), Some((_, ps))) => if vs.stage == HwStage::Ms {
                    host.mesh_pipeline(vs, ps)
                } else {
                    host.pipeline(vs, ps)
                }
                .map(|_| "pipeline created".to_owned())
                .unwrap_or_else(|e| e),
                _ => "not emitted".to_owned(),
            };
            println!("pair {v} + {p}: {made}");
        }
    }
    let mut lines = t.lines;
    lines.sort();
    for l in lines {
        println!("{l}");
    }
    let mut failed: Vec<_> = t.failed.into_iter().collect();
    failed.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
    for (reason, (n, example)) in failed {
        println!("  {n:>7}  {reason}  (e.g. {example})");
    }
    Ok(())
}

/// The warm-up (`--cache`): every distinct container found goes through
/// the shader cache `run` reads (`<cache>/shaders`, decision 0003), by the
/// same code, so a run finds it translated or refused. With `pipelines`,
/// then the shader pack's pipelines go into the pipeline library
/// (`prebuild`).
pub fn warm(
    paths: Vec<PathBuf>,
    jobs: Option<usize>,
    pipelines: bool,
    pipeline_selection: Option<PathBuf>,
) -> Result<()> {
    let paths = crate::default_paths(paths)?;
    let threads = thread_count(jobs);
    let keys_only = std::env::var_os("VICTORY_PREBUILD_KEYS").is_some();
    if let Some(file) = pipeline_selection {
        anyhow::ensure!(!keys_only, "unset VICTORY_PREBUILD_KEYS to pre-build a selection");
        // The selected pipelines translate their own shaders as needed.
        let keys = crate::prebuild::read_selection(&file)?;
        return crate::prebuild::run(&paths, threads, Some(&keys));
    }
    // The pre-build's key listing translates nothing.
    if pipelines && keys_only {
        return crate::prebuild::run(&paths, threads, None);
    }
    let dir = open_shader_cache()?;
    let started = std::time::Instant::now();
    let seen = Mutex::new(HashSet::new());
    // Containers already cached, translated now, refused now.
    let counts = [const { AtomicUsize::new(0) }; 3];
    for root in &paths {
        for entry in walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
        {
            if !entry.file_type().is_file() || entry.metadata().map_or(0, |m| m.len()) > MAX_FILE {
                continue;
            }
            let Ok(data) = std::fs::read(entry.path()) else {
                continue;
            };
            let jobs = sources(&data);
            let next = AtomicUsize::new(0);
            std::thread::scope(|s| {
                let workers: Vec<_> = (0..threads.min(jobs.len()))
                    .map(|_| {
                        s.spawn(|| {
                            while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                                let Some(bytes) = job.bytes() else { continue };
                                let container = declared_container(&bytes);
                                let hash = blake3::hash(container);
                                if !lock(&seen).insert(hash) {
                                    continue;
                                }
                                let one = std::time::Instant::now();
                                let (done, cached) =
                                    victory_gpu::shader_cache::host_shader(container);
                                let k = if cached {
                                    0
                                } else if done.is_ok() {
                                    1
                                } else {
                                    2
                                };
                                counts[k].fetch_add(1, Ordering::Relaxed);
                                tracing::debug!(
                                    hash = %hash,
                                    stage = ?done.as_ref().ok().map(|shader| shader.stage),
                                    dxil_hash = ?done.as_ref().ok().map(|shader| blake3::hash(&shader.dxil)),
                                    reason = done.as_ref().err().map(String::as_str),
                                    ms = one.elapsed().as_millis() as u64,
                                    cached,
                                    ok = done.is_ok(),
                                    "warm-up"
                                );
                            }
                        })
                    })
                    .collect();
                let [c, t, r] = &counts;
                report(
                    entry.path(),
                    jobs.len(),
                    "containers",
                    &next,
                    started,
                    &workers,
                    || {
                        let n = |a: &AtomicUsize| a.load(Ordering::Relaxed);
                        format!(
                            "; all files so far: {} cached, {} translated, {} refused",
                            n(c),
                            n(t),
                            n(r)
                        )
                    },
                );
            });
        }
    }
    let [cached, translated, refused] = counts.map(AtomicUsize::into_inner);
    let bytes: u64 = walkdir::WalkDir::new(&dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum();
    println!("cache        {} ({} MB)", dir.display(), bytes >> 20);
    println!(
        "containers   {} in {:.1} s: {cached} already cached, {translated} translated, {refused} refused",
        cached + translated + refused,
        started.elapsed().as_secs_f64()
    );
    if pipelines {
        crate::prebuild::run(&paths, threads, None)?;
    }
    Ok(())
}

/// Opens the same translated-shader cache as a run.
pub(crate) fn open_shader_cache() -> Result<PathBuf> {
    let shaders = victory_core::Config::load()?.cache_subdir("shaders")?;
    victory_gpu::shader_cache::use_shader_cache(shaders.clone())
        .with_context(|| format!("no shader cache in {}", shaders.display()))
}

/// Shows on stderr how far `workers` are through `file`'s `jobs` (`next`
/// hands them out) until they finish: on a terminal one line rewritten
/// every second, else a line every 10 s; then the file's last line.
pub(crate) fn report<T>(
    file: &Path,
    jobs: usize,
    what: &str,
    next: &AtomicUsize,
    started: std::time::Instant,
    workers: &[std::thread::ScopedJoinHandle<'_, T>],
    extra: impl Fn() -> String,
) {
    use std::io::IsTerminal;
    if workers.is_empty() {
        return;
    }
    let terminal = std::io::stderr().is_terminal();
    let every = std::time::Duration::from_secs(if terminal { 1 } else { 10 });
    let line = || {
        // `next` counts jobs handed out; a worker still running holds one.
        let busy = workers.iter().filter(|w| !w.is_finished()).count();
        let taken = next.load(Ordering::Relaxed).min(jobs);
        let done = taken.saturating_sub(busy);
        format!(
            "{}: {done} of {jobs} {what} ({}%), {:.0} s{}",
            file.display(),
            done * 100 / jobs.max(1),
            started.elapsed().as_secs_f64(),
            extra()
        )
    };
    let mut shown = std::time::Instant::now();
    while !workers.iter().all(|w| w.is_finished()) {
        std::thread::sleep(std::time::Duration::from_millis(100));
        if shown.elapsed() >= every {
            shown = std::time::Instant::now();
            if terminal {
                eprint!("\r{}", line());
            } else {
                eprintln!("{}", line());
            }
        }
    }
    eprintln!("{}{}", if terminal { "\r" } else { "" }, line());
}

fn translate(
    source: &Source,
    path: &Path,
    seen: &Mutex<HashSet<blake3::Hash>>,
    t: &mut Tally,
    checks: &mut Checks,
) {
    let example = || format!("{}{}", path.display(), source.label());
    let Some(bytes) = source.bytes() else {
        return t.fail("corrupt record".into(), example);
    };
    let bytes = declared_container(&bytes);
    let hash = blake3::hash(bytes);
    let short_hash = hash.to_hex()[..16].to_owned();
    // Match the runtime's container identity, including when scanning a larger file.
    static ONLY: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let only = ONLY.get_or_init(|| {
        std::env::var("VICTORY_TRANSLATE_ONLY")
            .unwrap_or_default()
            .split(',')
            .map(|h| h.trim().to_lowercase())
            .filter(|h| !h.is_empty())
            .collect()
    });
    if !only.is_empty() && !only.iter().any(|o| short_hash.starts_with(o.as_str())) {
        return;
    }
    let example = || format!("{}{} #{short_hash}", path.display(), source.label());
    if !lock(seen).insert(hash) {
        t.duplicates += 1;
        return;
    }
    t.containers += 1;

    let container = Container::parse(bytes).and_then(|container| {
        let part = container
            .part(FourCC::DXIL)
            .or(container.part(FourCC::ACCC))
            .ok_or(ShaderError::Malformed("no DXIL part"))?;
        let programs = native_programs(part.data)?;
        Ok((container, programs))
    });
    let (container, programs) = match container {
        Ok(ready) => ready,
        Err(error) => {
            t.runtime_outcome(hash, None, Err(&error), example);
            return t.note_failure(error.to_string(), example);
        }
    };
    let dump = std::env::var_os("VICTORY_FAIL_DUMP");
    let before: Option<BTreeMap<String, usize>> = dump
        .as_ref()
        .map(|_| t.failed.iter().map(|(k, v)| (k.clone(), v.0)).collect());
    let selected = match emit::runtime_program_index(&programs) {
        Ok(index) => Some(index),
        Err(error) => {
            t.runtime_outcome(hash, None, Err(&error), example);
            t.note_failure(error.to_string(), example);
            None
        }
    };
    t.programs += programs.len();
    t.extra_programs += programs.len() - usize::from(selected.is_some());
    let options = translation_options(bytes, checks.wave32);
    for (index, p) in programs.iter().enumerate() {
        let label = || format!("{} {:?} #{index}", example(), p.stage);
        let is_runtime = Some(index) == selected;
        match diagnose_program(&container, p, options, t) {
            Ok(Some(shader)) => {
                if is_runtime {
                    t.runtime_outcome(hash, Some((p.stage, index)), Ok(&shader), example);
                } else if t.list {
                    t.lines.push(format!("{}  extra program emitted", label()));
                }
                check(shader, t, checks, label);
            }
            Ok(None) => {
                if t.list {
                    t.lines.push(format!("{}  extra program analysed", label()));
                }
            }
            Err((step, error)) => {
                if is_runtime {
                    t.runtime_outcome(hash, Some((p.stage, index)), Err(&error), example);
                }
                let (reason, offset) = match error {
                    ShaderError::Unsupported { offset, what } => {
                        (format!("{step}{what}"), Some(offset))
                    }
                    ShaderError::Isa { offset, reason } if step == "structure: " => {
                        (format!("{step}{reason}"), Some(offset))
                    }
                    _ => (format!("{step}{error}"), None),
                };
                let where_ = || match offset {
                    Some(offset) => format!("{} at {offset:#x}", label()),
                    None => label(),
                };
                if is_runtime {
                    t.note_failure(reason, where_);
                } else {
                    t.fail(reason, where_);
                }
            }
        }
    }
    if let (Some(dir), Some(before)) = (dump, before) {
        for (reason, (n, _)) in &t.failed {
            let was = before.get(reason).copied().unwrap_or(0);
            if *n > was && was < 3 {
                let stem: String = reason
                    .chars()
                    .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                    .take(60)
                    .collect();
                let dir = Path::new(&dir);
                let _ = std::fs::create_dir_all(dir);
                let _ = std::fs::write(dir.join(format!("{stem}_{was}.{short_hash}.cdxil")), bytes);
            }
        }
    }
}

/// Counts the intermediate results independently, then returns the first
/// blocking error in the order used by the runtime's emitter.
fn diagnose_program(
    container: &Container<'_>,
    p: &NativeProgram,
    options: emit::Options,
    t: &mut Tally,
) -> Result<Option<emit::HostShader>, (&'static str, ShaderError)> {
    let (program, structured) = victory_shader::range::build(&p.patched_code(), p.wave64, p.vgprs)
        .map_err(|e| {
            (
                if matches!(e, ShaderError::Unsupported { .. }) {
                    "unsupported: "
                } else {
                    ""
                },
                e,
            )
        })?;
    t.ir_built += 1;
    t.instructions += program.blocks.iter().map(|b| b.insts.len()).sum::<usize>();
    let res = resource::track(&program, &p.layout, p.stage, structured.as_ref().ok());
    if let Ok(r) = &res {
        tally_resources(p, r, t);
    }

    let graphics = matches!(p.stage, HwStage::Vs | HwStage::Ps);
    let io = emit::supports_stage(p.stage).then(|| io::raise(&program, p));
    if graphics {
        t.io_programs += 1;
        if let Some(Ok(io)) = &io {
            tally_io(io, t);
        }
    }
    if let Ok(s) = &structured {
        tally_structured(&program, s, t);
    }

    let res = res.map_err(|error| ("resources: ", error))?;
    let io = io.transpose().map_err(|error| ("I/O: ", error))?;
    let st = structured.map_err(|error| ("structure: ", error))?;
    let Some(io) = io else {
        return Ok(None);
    };
    *counter(t, p.stage, 0) += 1;
    let shader = emit::emit_with(container, p, &program, &st, &io, &res, options)
        .map_err(|error| ("emit: ", error))?;
    *counter(t, p.stage, 1) += 1;
    Ok(Some(shader))
}

/// As the title passes it: a raw source runs on to the end of its file.
fn declared_container(bytes: &[u8]) -> &[u8] {
    let size = bytes.get(0x18..0x1c).map_or(0, |size| {
        u32::from_le_bytes(size.try_into().expect("four bytes")) as usize
    });
    &bytes[..size.min(bytes.len())]
}

/// Validates an emitted shader if asked to.
fn check(
    mut shader: emit::HostShader,
    t: &mut Tally,
    checks: &mut Checks,
    label: impl Fn() -> String,
) {
    if let Some(keep) = &checks.keep {
        lock(keep).push((label(), shader.clone()));
    }
    // `VICTORY_EMIT_DUMP` with `--list`: every emitted shader as a file.
    if t.list
        && let Ok(dir) = std::env::var("VICTORY_EMIT_DUMP")
    {
        let name: String = label()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let _ = std::fs::write(
            format!("{dir}/{}.dxil", &name[name.len().saturating_sub(100)..]),
            &shader.dxil,
        );
    }
    if let Some((h, stub_ps)) = &checks.host {
        let made = match shader.stage {
            HwStage::Vs => h.pipeline(&shader, stub_ps).map(drop),
            HwStage::Cs => h.compute_pipeline(&shader).map(drop),
            HwStage::Ms => h.mesh_pipeline(&shader, stub_ps).map(drop),
            _ => host::stub_vs(&shader).and_then(|vs| h.pipeline(&vs, &shader).map(drop)),
        };
        match made {
            Ok(()) => {
                *counter(t, shader.stage, 3) += 1;
                if t.list {
                    t.lines.push(format!("{}  pipeline created", label()));
                }
            }
            Err(e) => t.fail(format!("pipeline: {e}"), &label),
        }
    }
    let Some(v) = checks.validator.as_mut() else {
        return;
    };
    match v.validate(&mut shader.dxil) {
        Ok(()) => {
            *counter(t, shader.stage, 2) += 1;
            if t.list {
                t.lines.push(format!("{}  validated", label()));
            }
        }
        Err(text) => {
            let first = text
                .lines()
                .find(|l| l.contains("error"))
                .unwrap_or(text.lines().next().unwrap_or(""))
                .to_owned();
            if let Ok(dir) = std::env::var("VICTORY_EMIT_DUMP") {
                let n = t
                    .failed
                    .get(&format!("validator: {first}"))
                    .map_or(0, |f| f.0);
                if n < 3 {
                    let name = format!(
                        "{dir}/{}_{n}",
                        first
                            .chars()
                            .filter(char::is_ascii_alphanumeric)
                            .take(40)
                            .collect::<String>()
                    );
                    let _ = std::fs::write(format!("{name}.dxil"), &shader.dxil);
                    let _ = std::fs::write(format!("{name}.txt"), format!("{}\n{text}", label()));
                }
            }
            t.fail(format!("validator: {first}"), label);
        }
    }
}

/// Emission counts of `stage`'s kind: 0 programs, 1 emitted, 2 validated,
/// 3 pipelines.
fn counter(t: &mut Tally, stage: HwStage, k: usize) -> &mut usize {
    let [a, b, c, d] = match stage {
        HwStage::Cs => [
            &mut t.cs_programs,
            &mut t.cs_emitted,
            &mut t.cs_validated,
            &mut t.cs_pipelines,
        ],
        HwStage::Ms => [
            &mut t.ms_programs,
            &mut t.ms_emitted,
            &mut t.ms_validated,
            &mut t.ms_pipelines,
        ],
        _ => [
            &mut t.emit_programs,
            &mut t.emitted,
            &mut t.validated,
            &mut t.pipelines,
        ],
    };
    [a, b, c, d].into_iter().nth(k).expect("0 to 3")
}

fn tally_io(r: &io::Io, t: &mut Tally) {
    t.io_raised += 1;
    for (_, i) in &r.interps {
        match i {
            io::Interp::Interpolated { .. } => t.interpolated += 1,
            io::Interp::Vertex { .. } => t.vertex_values += 1,
        }
    }
    for e in &r.exports {
        let kind = match e.target {
            io::Target::Color(_) => "color",
            io::Target::Depth => "depth",
            io::Target::Null => "null",
            io::Target::Position(_) => "position",
            io::Target::Param { .. } => "param",
            io::Target::Unread => "unread param",
            io::Target::Primitive => "primitive",
            io::Target::PrimitiveParam { .. } => "primitive param",
        };
        *t.exports.entry(kind).or_default() += 1;
        t.compressed += usize::from(e.compressed);
    }
}

fn tally_resources(p: &NativeProgram, r: &resource::Resources, t: &mut Tally) {
    t.tracked += 1;
    t.scalar += r.scalar.len();
    t.dynamic += r.scalar.iter().filter(|s| s.1.dynamic.is_some()).count();
    let address = |load| r.scalar.iter().find(|s| s.0 == load).map(|s| s.1);
    let mut traced = HashSet::new();
    for leaf in r.descriptors.iter().flat_map(|d| d.source.leaves()) {
        let a = match *leaf {
            resource::Source::Loaded { load, .. } | resource::Source::Fetched { load, .. } => {
                t.loaded += 1;
                address(load)
            }
            resource::Source::Built { address, .. } => {
                t.built += 1;
                t.dynamic += usize::from(address.dynamic.is_some());
                Some(address)
            }
            resource::Source::Constant(_) => {
                t.constant += 1;
                None
            }
            resource::Source::Phi { .. }
            | resource::Source::Select { .. }
            | resource::Source::Lanes { .. }
            | resource::Source::LanePhi { .. } => {
                unreachable!("leaves are no phis, selects or per-lane choices")
            }
        };
        // Heaps hold dynamic resources, which no used entry lists; code
        // tables and title memory are no root parameter.
        if let Some(a) = a
            && a.param < p.layout.len()
            && !matches!(
                p.layout[a.param].root,
                RootType::ResourceHeap | RootType::SamplerHeap
            )
        {
            traced.insert(p.layout[a.param].user_data);
        }
    }
    // Root constants are listed as a CBV, read straight from the SGPRs.
    traced.extend(
        r.params
            .iter()
            .map(|&(_, param, _)| &p.layout[param])
            .filter(|l| l.root == RootType::Constants)
            .map(|l| l.user_data),
    );
    let listed: HashSet<u8> = p.used.iter().map(|u| u.user_data).collect();
    t.used += p.used.len();
    // Immediate samplers are literals in the code: the distinct sampler
    // values (decoded) are matched against the listed entries, which do not
    // say which value is whose.
    let mut values: Vec<emit::StaticSampler> = Vec::new();
    for d in r
        .descriptors
        .iter()
        .filter(|d| d.role == resource::Role::Sampler)
    {
        for leaf in d.source.leaves() {
            if let resource::Source::Constant(words) = leaf
                && let Ok(w) = <[u32; 4]>::try_from(&words[..words.len().min(4)])
            {
                let s = emit::StaticSampler::from_words(w);
                if !values.contains(&s) {
                    values.push(s);
                }
            }
        }
    }
    // Engines often declare equal samplers at several registers, so equal
    // values may stand for several entries, but never more values than
    // entries.
    let immediate = p
        .used
        .iter()
        .filter(|u| u.root == RootType::ImmediateSampler)
        .count();
    let samplers_match = !values.is_empty() && values.len() <= immediate;
    let matched = |u: &&victory_shader::UsedResource| {
        if u.root == RootType::ImmediateSampler {
            samplers_match
        } else {
            traced.contains(&u.user_data)
        }
    };
    for u in p.used.iter().filter(|u| !matched(u)) {
        *t.untraced_used
            .entry(format!("{:?} via {:?}", u.kind, u.root))
            .or_default() += 1;
    }
    t.unlisted += traced.difference(&listed).count();
}

fn tally_structured(program: &ir::Program, s: &structure::Structured, t: &mut Tally) {
    t.structured += 1;
    for (block, guards) in program.blocks.iter().zip(&s.guards) {
        t.vector += block
            .insts
            .iter()
            .filter(|i| i.read(ir::Reg::S(ir::EXEC)).is_some())
            .count();
        for &g in guards.iter().flatten() {
            t.guarded += 1;
            let support = s.bools.support(g);
            t.opaque += usize::from(support.iter().any(|a| matches!(a, Atom::Opaque(_))));
        }
    }
    for exit in &s.exits {
        if let LaneExit::Branch { when, .. } = *exit {
            if s.bools
                .support(when)
                .iter()
                .all(|a| matches!(a, Atom::Uniform(_)))
            {
                t.uniform_branches += 1;
            } else {
                t.lane_branches += 1;
            }
        }
    }
}

/// The worker threads to use: `--jobs`, else one per core.
fn thread_count(jobs: Option<usize>) -> usize {
    jobs.unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()))
}

pub(crate) fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    /// Shader dumps are game-derived, so a dump directory in the repo is refused.
    #[test]
    fn emit_dump_inside_repo_is_refused() {
        let repo = env!("CARGO_MANIFEST_DIR");
        if std::env::var("VICTORY_EMIT_DUMP").as_deref() != Ok(repo) {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "translate::tests::emit_dump_inside_repo_is_refused",
                ])
                .env("VICTORY_EMIT_DUMP", repo)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stdout)
            );
            return;
        }
        let missing = std::env::temp_dir().join("victory-translate-no-such-dir");
        let err = super::run(vec![missing], false, false, false, Some(1), false).unwrap_err();
        assert!(
            matches!(
                err.downcast_ref(),
                Some(victory_core::PolicyError::InsideRepo { .. })
            ),
            "{err}"
        );
    }

    /// Each child gets its own process-wide translation switches.
    #[test]
    fn diagnostic_options_match_runtime_translation() {
        use std::process::Command;
        use victory_shader::emit::Options;

        let container = b"an original synthetic container";
        let hash = blake3::hash(container).to_hex();
        if let Ok(case) = std::env::var("VICTORY_TEST_TRANSLATION_OPTIONS") {
            let (wide, configured) = match case.as_str() {
                "configured" => (false, true),
                "wide" => (true, true),
                "disabled" => (true, false),
                other => panic!("unexpected test case {other}"),
            };
            let expected = Options {
                view_marks: true,
                atomic_locks: configured,
                atomic_lock_rounds: configured,
                atomic_umax: configured,
                bias_as_gradients: configured,
                host_wave32: !wide,
                narrow_waves: configured,
                disable_literal_halves: configured,
                per_vertex_center: false,
                uav_size_level0: configured,
                halves_rendezvous: configured,
                thread_limits: false,
            };
            assert_eq!(super::translation_options(container, !wide), expected);
            assert_eq!(
                super::translation_options(b"another container", !wide),
                Options {
                    uav_size_level0: false,
                    ..expected
                },
            );
            return;
        }

        for (case, value, size) in [
            ("configured", "1", &hash[..16]),
            ("wide", "1", &hash[..16]),
            ("disabled", "0", "0"),
        ] {
            let output = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "translate::tests::diagnostic_options_match_runtime_translation",
                ])
                .env("VICTORY_TEST_TRANSLATION_OPTIONS", case)
                .env("VICTORY_ATOMIC_LOCKS", value)
                .env("VICTORY_ATOMIC_LOCK_ROUNDS", value)
                .env("VICTORY_ATOMIC_UMAX", value)
                .env("VICTORY_BIAS_AS_GRADIENTS", value)
                .env("VICTORY_NARROW_WAVES", value)
                .env("VICTORY_HALVES_RENDEZVOUS", value)
                .env(
                    "VICTORY_LITERAL_HALVES",
                    if value == "1" { "0" } else { "1" },
                )
                .env("VICTORY_UAV_SIZE_LEVEL0", size)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{case}: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
