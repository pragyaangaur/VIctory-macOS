use std::path::Path;
use std::process::Command;

use victory_shader::HwStage;
use victory_shader::emit::{self, Options};

#[test]
fn translate_rejects_zero_workers() {
    let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    for extra_args in [&[][..], &["--cache"][..]] {
        let output = Command::new(env!("CARGO_BIN_EXE_victory"))
            .arg("translate")
            .arg(&input)
            .args(["--jobs", "0"])
            .args(extra_args)
            .output()
            .expect("run victory translate");

        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(2), "{extra_args:?}: {stderr}");
        assert!(stderr.contains("--jobs"), "{extra_args:?}: {stderr}");
    }
}

const END: u32 = 0xBF81_0000;

fn container(tag: &[u8; 4], payload: &[u8], digest_byte: u8) -> Vec<u8> {
    let mut bytes = b"DXBC".to_vec();
    bytes.extend_from_slice(&[digest_byte; 16]);
    for word in [1, (44 + payload.len()) as u32, 1, 36] {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes.extend_from_slice(tag);
    bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

fn accc(words: &[u32], threads: u16, lds: bool, descriptor: Option<(u32, u32, u8)>) -> Vec<u8> {
    let mut packet = [0u8; 28];
    for (at, n) in [(12, threads), (14, 1), (16, 1)] {
        packet[at..at + 2].copy_from_slice(&n.to_le_bytes());
    }
    packet[20] = 7; // 32 VGPRs in wave64 mode.
    // One user-data SGPR for the descriptor table, and one 512-byte LDS block.
    let rsrc2 = (u32::from(descriptor.is_some()) << 1) | (u32::from(lds) << 15);
    packet[24..28].copy_from_slice(&rsrc2.to_le_bytes());
    let code: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut metadata = vec![0u8; 24];
    metadata[8..12].copy_from_slice(&(words.len() as u32).to_le_bytes());
    if let Some((offset, dimension, dwords)) = descriptor {
        let mut uav = [0u8; 32];
        uav[..8].copy_from_slice(&[0x41, 0, 0, 0, 3, 3, 0, 0]);
        uav[8..12].copy_from_slice(&offset.to_le_bytes());
        uav[16..20].copy_from_slice(&dimension.to_le_bytes());
        uav[20] = dwords;
        metadata.extend_from_slice(&uav);
    }
    let entries = [(0u32, &packet[..]), (6, &code[..]), (7, &metadata[..])];
    let mut at = 16 + 16 * entries.len();
    let size = at + entries.iter().map(|(_, data)| data.len()).sum::<usize>();
    let mut payload = b"FSPX".to_vec();
    for word in [1, entries.len() as u32, size as u32] {
        payload.extend_from_slice(&word.to_le_bytes());
    }
    for (kind, data) in entries {
        for word in [kind, at as u32, data.len() as u32, 0] {
            payload.extend_from_slice(&word.to_le_bytes());
        }
        at += data.len();
    }
    for (_, data) in entries {
        payload.extend_from_slice(data);
    }
    payload
}

fn divergent_wave() -> Vec<u8> {
    let payload = accc(
        &[
            0xBE84_0300,
            0xBE85_0380,
            0xF408_0A02,
            0xFA00_0040,
            0xBF8C_C07F,
            0x3420_0085,
            0x7E06_0280, // v16 = v0 * 32; v3 = 0
            0x3404_0082,
            0x3A08_00A0,
            0x3408_0882, // v2 = v0 * 4; v4 = (v0 ^ 32) * 4
            0x3602_009F,
            0x7D86_0290, // Active lanes in both halves have (v0 & 31) >= 16.
            0xBE94_246A,
            0xBF88_0006, // Save EXEC and skip the body if empty.
            0xD834_0000,
            0x0000_0002,
            0xBF8C_C07F, // Write lane ID to LDS at v2.
            0xD8D8_0000,
            0x0300_0004,
            0xBF8C_C07F, // Read the other half's ID from v4.
            0xBEFE_0414, // Restore EXEC before writing each lane's result.
            0xE070_1000,
            0x800A_0310,
            END,
        ],
        64,
        true,
        Some((16, 1, 4)),
    );
    container(b"ACCC", &payload, 0)
}

#[test]
fn divergent_wave_translation_depends_on_the_literal_halves_retry() {
    let input = divergent_wave();
    let options = Options {
        host_wave32: true,
        ..Options::default()
    };
    assert_eq!(
        emit::translate_with(&input, options).unwrap().stage,
        HwStage::Cs
    );
    let without_retry = emit::translate_with(
        &input,
        Options {
            disable_literal_halves: true,
            ..options
        },
    )
    .unwrap_err();
    assert!(
        without_retry
            .to_string()
            .contains("LDS one console wave64 shares without barriers"),
        "{without_retry}"
    );
}

fn uav_size_payload() -> Vec<u8> {
    accc(
        &[
            0xBE81_0380,
            0xF40C_0400,
            0xFA00_0000,
            0xBF8C_C07F, // Load the image UAV into s[16:23].
            0x7E00_0280,
            0x7E02_0280, // v0 = v1 = 0 (texel coordinates and query level).
            0xF038_1108,
            0x0004_0200, // image_get_resinfo v2, v0, s[16:23], width only.
            0xF020_1108,
            0x0004_0200, // Write that width to the UAV.
            END,
        ],
        1,
        false,
        Some((0, 4, 8)),
    )
}

fn dxil(include_runtime_stages: bool) -> Vec<u8> {
    let mut fssm = b"FSSM".to_vec();
    fssm.extend_from_slice(&0x101u32.to_le_bytes());
    fssm.extend_from_slice(&[0; 8]);
    let mut payload = vec![0; 0x68];
    payload[..8].copy_from_slice(&6u64.to_le_bytes());
    for at in [0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x48, 0x60] {
        payload[at..at + 8].copy_from_slice(&0xDEAD_BEEF_DEAD_BEEFu64.to_le_bytes());
    }
    // A merged GS precedes the supported VS, then an extra CS.
    for (at, info_size) in [(0x20, 0), (0x28, 0x18c), (0x60, 0x34)]
        .into_iter()
        .take(if include_runtime_stages { 3 } else { 1 })
    {
        let record = payload.len();
        payload[at..at + 8].copy_from_slice(&(record as u64).to_le_bytes());
        let code = record + 0x54 + info_size;
        payload.extend_from_slice(&(code as u64).to_le_bytes());
        payload.extend_from_slice(&4u64.to_le_bytes());
        payload.resize(code, 0);
        if at == 0x60 {
            for axis in 0..3 {
                payload[record + 0x54 + 0x28 + 4 * axis] = 1;
            }
        }
        payload.extend_from_slice(&END.to_le_bytes());
    }
    fssm[8..16].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    fssm.extend_from_slice(&payload);
    // The synthetic XBCH contains one uncompressed deflate block in a zlib stream.
    let size = fssm.len() as u16;
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &fssm {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    let mut stream = vec![0x78, 0x01, 0x01];
    stream.extend_from_slice(&size.to_le_bytes());
    stream.extend_from_slice(&(!size).to_le_bytes());
    stream.extend_from_slice(&fssm);
    stream.extend_from_slice(&(b << 16 | a).to_be_bytes());
    let mut part = vec![0; 0x18];
    part.extend_from_slice(b"XDAT");
    for word in [1u32, 0x24, 0x0d, 0x30, (0x18 + stream.len()) as u32] {
        part.extend_from_slice(&word.to_le_bytes());
    }
    part.extend_from_slice(b"XBCH");
    part.extend_from_slice(&3u32.to_le_bytes());
    part.extend_from_slice(&(stream.len() as u64).to_le_bytes());
    part.extend_from_slice(&u64::from(size).to_le_bytes());
    part.extend_from_slice(&stream);
    part
}

#[test]
fn synthetic_containers_exercise_size_queries_and_runtime_stage_selection() {
    let size = container(b"ACCC", &uav_size_payload(), 0);
    let plain = emit::translate_with(&size, Options::default()).expect("default size query");
    let level0 = emit::translate_with(
        &size,
        Options {
            uav_size_level0: true,
            ..Options::default()
        },
    )
    .expect("level-0 size query");
    assert_ne!(blake3::hash(&plain.dxil), blake3::hash(&level0.dxil));

    let multi = container(b"DXIL", &dxil(true), 0);
    assert_eq!(emit::translate(&multi).unwrap().stage, HwStage::Vs);
    let merged_only = container(b"DXIL", &dxil(false), 0);
    let refusal = emit::translate(&merged_only).unwrap_err();
    assert!(
        refusal
            .to_string()
            .contains("a container without a vertex, pixel, compute or mesh program")
    );
}

#[cfg(windows)]
fn diagnostic_command(input: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_victory"));
    cmd.args(["translate", "--wave32", "--list"])
        .arg(input)
        .env("VICTORY_WARP", "1")
        .env("VICTORY_ATOMIC_LOCKS", "0")
        .env("VICTORY_BIAS_AS_GRADIENTS", "0")
        .env("VICTORY_HALVES_RENDEZVOUS", "0")
        .env("VICTORY_NARROW_WAVES", "0")
        .env_remove("VICTORY_EMIT_DUMP")
        .env_remove("VICTORY_FAIL_DUMP")
        .env_remove("VICTORY_TRANSLATE_ONLY")
        .env_remove("VICTORY_PAIRS")
        .env_remove("VICTORY_UAV_SIZE_LEVEL0");
    cmd
}

#[cfg(windows)]
fn stdout(cmd: &mut Command) -> String {
    let out = cmd.output().expect("run offline diagnostics");
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(
        out.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    stdout
}

#[cfg(windows)]
#[test]
fn diagnostic_filter_uses_the_declared_container_hash() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("synthetic.bin");
    let payload = uav_size_payload();
    let selected = container(b"ACCC", &payload, 1);
    let other = container(b"ACCC", &payload, 2);
    std::fs::write(&file, [&selected[..], &other, b"trailing bytes"].concat()).unwrap();
    let hash = blake3::hash(&selected);
    let output = stdout(diagnostic_command(&file).env(
        "VICTORY_TRANSLATE_ONLY",
        format!(" , {}, ", hash.to_hex()[..16].to_uppercase()),
    ));
    assert!(
        output.contains("1 distinct (0 duplicates skipped)"),
        "{output}"
    );
    assert!(
        output.contains(&format!("runtime {hash} Cs #0 emitted")),
        "{output}"
    );
    assert!(
        !output.contains(&format!("runtime {}", blake3::hash(&other))),
        "{output}"
    );
}

#[cfg(windows)]
#[test]
fn failed_container_dumps_keep_the_declared_bytes_and_bound_the_samples() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("synthetic.bin");
    let dumps = dir.path().join("failures");
    let payload = dxil(false);
    let inputs: Vec<_> = (0..5)
        .map(|digest| container(b"DXIL", &payload, digest))
        .collect();
    std::fs::write(&file, inputs.concat()).unwrap();
    let mut command = diagnostic_command(&file);
    // One worker makes the per-worker diagnostic sample limit deterministic.
    command
        .args(["--jobs", "1"])
        .env("VICTORY_FAIL_DUMP", &dumps);
    let output = stdout(&mut command);
    assert!(output.contains("5 distinct"), "{output}");
    let files: Vec<_> = std::fs::read_dir(&dumps)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert_eq!(files.len(), 3, "{files:?}");
    for path in files {
        let bytes = std::fs::read(&path).unwrap();
        assert!(
            inputs.contains(&bytes),
            "dump includes bytes outside its container"
        );
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(
            name.contains(&blake3::hash(&bytes).to_hex()[..16]),
            "{name}"
        );
    }
}

#[cfg(windows)]
#[test]
fn failed_container_dump_refuses_a_source_tree_destination() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(".victory-root"), "").unwrap();
    let input = dir.path().join("synthetic.bin");
    std::fs::write(&input, container(b"DXIL", &dxil(false), 0)).unwrap();
    let dumps = dir.path().join("failures");
    let output = diagnostic_command(&input)
        .env("VICTORY_FAIL_DUMP", &dumps)
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "source-tree output must be refused"
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("inside the VIctory source tree"));
    assert!(!dumps.exists());
}

#[cfg(windows)]
#[test]
fn wave32_offline_analysis_uses_the_runtime_literal_halves_switch() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("divergent-wave.bin");
    std::fs::write(&file, divergent_wave()).unwrap();
    let on = stdout(diagnostic_command(&file).env("VICTORY_LITERAL_HALVES", "1"));
    assert!(on.contains("Cs") && on.contains("emitted"), "{on}");
    let off = stdout(diagnostic_command(&file).env("VICTORY_LITERAL_HALVES", "0"));
    assert!(
        off.contains("LDS one console wave64 shares without barriers"),
        "{off}"
    );
}

#[cfg(windows)]
#[test]
fn same_payload_containers_keep_distinct_overrides() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("synthetic.bin");
    let payload = uav_size_payload();
    let selected = container(b"ACCC", &payload, 1);
    let default = container(b"ACCC", &payload, 2);
    let multi = container(b"DXIL", &dxil(true), 0);
    let merged_only = container(b"DXIL", &dxil(false), 0);
    let selected_hash = blake3::hash(&selected).to_hex();
    std::fs::write(
        &file,
        [
            &selected[..],
            &default,
            &selected,
            &multi,
            &merged_only,
            b"trailing bytes",
        ]
        .concat(),
    )
    .unwrap();
    let output = stdout(
        diagnostic_command(&file)
            .env("VICTORY_LITERAL_HALVES", "1")
            .env("VICTORY_UAV_SIZE_LEVEL0", &selected_hash[..16]),
    );
    assert!(
        output.contains("4 distinct (1 duplicates skipped)"),
        "{output}"
    );
    assert!(
        output.contains("6 native programs") && output.contains("3 extra diagnostic programs"),
        "{output}"
    );
    for (bytes, level0, stage, index) in [
        (&selected, true, HwStage::Cs, 0),
        (&default, false, HwStage::Cs, 0),
        (&multi, false, HwStage::Vs, 1),
    ] {
        let expected = emit::translate_with(
            bytes,
            Options {
                view_marks: true,
                host_wave32: true,
                uav_size_level0: level0,
                ..Options::default()
            },
        )
        .unwrap();
        assert_eq!(expected.stage, stage);
        let hash = blake3::hash(bytes);
        let dxil = blake3::hash(&expected.dxil);
        assert!(
            output.contains(&format!("runtime {hash} {stage:?} #{index} emitted {dxil}")),
            "{output}"
        );
    }
    assert!(output.contains(&format!(
        "runtime {} - refused GFX10 code at 0x0: a container without a vertex, pixel, compute or mesh program is not supported",
        blake3::hash(&merged_only)
    )), "{output}");
}
