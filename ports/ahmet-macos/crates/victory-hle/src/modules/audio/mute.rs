//! A background run (`VICTORY_BACKGROUND=1`, `victory_gpu::background`) is
//! silent: every audio session of our process, on every output device, is
//! muted before the title starts and whenever a new one appears. A stream
//! takes its session's mute when it is made, so nothing is heard first as
//! long as the title plays in the default session (Grand Theft Auto VI and
//! XAudio2 do); a stream in a session of its own is muted within
//! [`EVERY`]. Only our process's sessions are touched, never a device.
//!
//! shortcut: on Windows the mixer remembers the mute for this executable;
//! it is unmuted there (Volume mixer) by hand. Wine forgets it.

use std::time::Duration;

use windows::Win32::Media::Audio::{
    DEVICE_STATE_ACTIVE, IAudioSessionManager2, IMMDeviceEnumerator, ISimpleAudioVolume,
    MMDeviceEnumerator, eRender,
};
use windows::Win32::System::Com::{CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx};
use windows::core::Interface;

/// How often new sessions are looked for.
const EVERY: Duration = Duration::from_millis(100);

/// Mutes our sessions on every active output; returns how many it muted.
fn mute_all(devices: &IMMDeviceEnumerator) -> windows::core::Result<u32> {
    let mut muted = 0;
    // SAFETY: COM calls on live interfaces, with parameters built here.
    unsafe {
        let outputs = devices.EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE)?;
        for i in 0..outputs.GetCount()? {
            let manager: IAudioSessionManager2 = outputs.Item(i)?.Activate(CLSCTX_ALL, None)?;
            // The default session, made here if no stream has yet.
            let default = manager.GetSimpleAudioVolume(None, 0)?;
            if !default.GetMute()?.as_bool() {
                default.SetMute(true, std::ptr::null())?;
                muted += 1;
            }
            // Sessions streams made under identifiers of their own.
            let Ok(sessions) = manager.GetSessionEnumerator() else {
                continue;
            };
            let count = sessions.GetCount().unwrap_or(0);
            let mut unreachable = 0;
            for k in 0..count {
                // Wine's session controls do not give out their volume.
                let volume = sessions
                    .GetSession(k)
                    .and_then(|s| s.cast::<ISimpleAudioVolume>());
                match volume {
                    Ok(volume) => {
                        if !volume.GetMute()?.as_bool() {
                            volume.SetMute(true, std::ptr::null())?;
                            muted += 1;
                        }
                    }
                    Err(_) => unreachable += 1,
                }
            }
            // More than the default session, and no way to mute the rest:
            // say so, once per count.
            static SEEN: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(1);
            if unreachable > 1
                && SEEN.fetch_max(unreachable, std::sync::atomic::Ordering::Relaxed) < unreachable
            {
                tracing::warn!(
                    sessions = unreachable,
                    "background run: audio sessions besides the default one cannot be muted here"
                );
            }
        }
    }
    Ok(muted)
}

/// In a background run: mutes our audio sessions, returning once the
/// first pass is done (call before the title starts), and keeps muting
/// new ones.
pub fn start() {
    if !victory_gpu::background() {
        return;
    }
    let (first, done) = std::sync::mpsc::channel();
    let spawned = std::thread::Builder::new()
        .name("victory-mute".into())
        .spawn(move || {
            // SAFETY: this thread's own apartment, kept for its lifetime.
            let devices: windows::core::Result<IMMDeviceEnumerator> = unsafe {
                let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)
            };
            let mut first = Some(first);
            let mut failed = false;
            loop {
                let result = devices.as_ref().map_err(Clone::clone).and_then(mute_all);
                match &result {
                    Ok(0) => {}
                    Ok(n) => tracing::info!(sessions = n, "background run: sound muted"),
                    Err(e) if !failed => {
                        failed = true;
                        tracing::warn!(%e, "background run: muting sound failed (no output device?)");
                    }
                    Err(_) => {}
                }
                if let Some(first) = first.take() {
                    let _ = first.send(());
                }
                std::thread::sleep(EVERY);
            }
        });
    if spawned.is_ok() {
        let _ = done.recv_timeout(Duration::from_secs(5));
    }
}
