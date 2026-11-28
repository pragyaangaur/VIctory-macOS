//! Spatial Audio / WASAPI channel downmix shim (`VICTORY_AUDIO_DOWNMIX`).
//!
//! Under Wine on macOS (`winecoreaudio.drv`), render streams ignore
//! `AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM` and hand the client's channel count
//! straight to CoreAudio `AUHAL`, whose default channel map (`[0, 1]` on a
//! stereo device, `[1]` on a 1-channel Bluetooth HFP device) drops channel 2
//! (`FrontCenter`, where surround mixes place character speech) and every
//! higher channel instead of downmixing. In addition, a 1-channel Bluetooth
//! HFP endpoint (`GetMixFormat()->nChannels == 1`) causes titles that check
//! for `2` or `6` channels to fall into a multichannel (e.g. 7.1.4) bed.
//!
//! Controlled by `VICTORY_AUDIO_DOWNMIX` (default: off / `0`):
//! - `0` / unset / `off`: pass `CoCreateInstance` straight through to the host.
//! - `1` / `on` / `auto`: log the host endpoint format; advertise 2-channel
//!   stereo when the host reports 1 channel; and wrap `ISpatialAudioClient`
//!   streams so surround beds (>2 channels) are downmixed into a 2-channel
//!   `FL | FR` bed (and folded to mono on both channels when the host is
//!   1-channel).
//! - `stereo`: like `auto`, and always advertise 2-channel stereo in
//!   `GetMixFormat` even on >2-channel endpoints.
//! - `mono`: like `stereo`, and always fold `L + R` mono into both `FL` and
//!   `FR`.

use std::sync::OnceLock;

/// `AudioObjectType` static bed channel bits (Windows `spatialaudioclient.h`).
pub const OBJ_FRONT_LEFT: u32 = 1 << 1;
pub const OBJ_FRONT_RIGHT: u32 = 1 << 2;
pub const OBJ_FRONT_CENTER: u32 = 1 << 3;
pub const OBJ_LOW_FREQUENCY: u32 = 1 << 4;
pub const OBJ_SIDE_LEFT: u32 = 1 << 5;
pub const OBJ_SIDE_RIGHT: u32 = 1 << 6;
pub const OBJ_BACK_LEFT: u32 = 1 << 7;
pub const OBJ_BACK_RIGHT: u32 = 1 << 8;
pub const OBJ_TOP_FRONT_LEFT: u32 = 1 << 9;
pub const OBJ_TOP_FRONT_RIGHT: u32 = 1 << 10;
pub const OBJ_TOP_BACK_LEFT: u32 = 1 << 11;
pub const OBJ_TOP_BACK_RIGHT: u32 = 1 << 12;
pub const OBJ_BOTTOM_FRONT_LEFT: u32 = 1 << 13;
pub const OBJ_BOTTOM_FRONT_RIGHT: u32 = 1 << 14;
pub const OBJ_BOTTOM_BACK_LEFT: u32 = 1 << 15;
pub const OBJ_BOTTOM_BACK_RIGHT: u32 = 1 << 16;
pub const OBJ_BACK_CENTER: u32 = 1 << 17;

/// Stereo bed mask (`FrontLeft | FrontRight`).
pub const STEREO_MASK: u32 = OBJ_FRONT_LEFT | OBJ_FRONT_RIGHT;

/// -3 dB (`1 / sqrt(2)`), standard ITU-R BS.775 center and surround downmix gain.
const SQRT_HALF: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Operating mode selected by `VICTORY_AUDIO_DOWNMIX`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownmixMode {
    /// Disabled (default): do not wrap COM audio interfaces.
    Off,
    /// Fix 1-channel endpoints and downmix >2-channel beds on <=2-channel hosts.
    Auto,
    /// Always advertise 2-channel stereo and downmix >2-channel beds to stereo.
    Stereo,
    /// Always advertise 2-channel stereo and fold output to mono on both FL/FR.
    Mono,
}

impl DownmixMode {
    pub fn parse(raw: Option<&str>) -> Self {
        let Some(s) = raw else {
            return Self::Off;
        };
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "off" | "false" | "no" => Self::Off,
            "1" | "on" | "true" | "yes" | "auto" => Self::Auto,
            "stereo" | "2" => Self::Stereo,
            "mono" => Self::Mono,
            other => {
                tracing::warn!(
                    value = other,
                    "VICTORY_AUDIO_DOWNMIX: unknown value, using off"
                );
                Self::Off
            }
        }
    }
}

/// Cached process-wide mode from `VICTORY_AUDIO_DOWNMIX` (default: `Off`).
pub fn mode() -> DownmixMode {
    static MODE: OnceLock<DownmixMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let raw = std::env::var("VICTORY_AUDIO_DOWNMIX").ok();
        let m = DownmixMode::parse(raw.as_deref());
        if m != DownmixMode::Off {
            tracing::info!(?m, "audio downmix shim enabled (VICTORY_AUDIO_DOWNMIX)");
        }
        m
    })
}

/// Whether `IAudioClient::GetMixFormat` should report 2-channel stereo to the
/// title so it builds a stereo mix instead of falling back to a surround bed.
pub fn should_spoof_stereo(real_channels: u16, mode: DownmixMode) -> bool {
    match mode {
        DownmixMode::Off => false,
        DownmixMode::Auto => real_channels == 1,
        DownmixMode::Stereo | DownmixMode::Mono => real_channels != 2,
    }
}

/// Whether an `ISpatialAudioObjectRenderStream` with `requested_mask` should be
/// intercepted and downmixed onto a 2-channel `FL | FR` underlying stream.
pub fn should_wrap_stream(requested_mask: u32, real_channels: u16, mode: DownmixMode) -> bool {
    let has_extra_channels = (requested_mask & !STEREO_MASK) != 0;
    match mode {
        DownmixMode::Off => false,
        DownmixMode::Auto => real_channels <= 2 && (real_channels == 1 || has_extra_channels),
        DownmixMode::Stereo => has_extra_channels,
        DownmixMode::Mono => true,
    }
}

/// Whether the downmixed stereo output should be summed to mono and duplicated
/// onto both `FL` and `FR` (so a 1-channel CoreAudio HAL unit with `chmap=[0]`
/// or `chmap=[1]` receives the complete `L + R` mix).
pub fn should_fold_mono(real_channels: u16, mode: DownmixMode) -> bool {
    match mode {
        DownmixMode::Off | DownmixMode::Stereo => false,
        DownmixMode::Auto => real_channels == 1,
        DownmixMode::Mono => true,
    }
}

/// Stereo `(left, right)` downmix gains for one static `AudioObjectType` bit.
pub fn downmix_gains(obj_type: u32) -> (f32, f32) {
    match obj_type {
        OBJ_FRONT_LEFT => (1.0, 0.0),
        OBJ_FRONT_RIGHT => (0.0, 1.0),
        OBJ_FRONT_CENTER => (SQRT_HALF, SQRT_HALF),
        OBJ_LOW_FREQUENCY | OBJ_BACK_CENTER => (0.5, 0.5),
        OBJ_SIDE_LEFT
        | OBJ_BACK_LEFT
        | OBJ_TOP_FRONT_LEFT
        | OBJ_TOP_BACK_LEFT
        | OBJ_BOTTOM_FRONT_LEFT
        | OBJ_BOTTOM_BACK_LEFT => (SQRT_HALF, 0.0),
        OBJ_SIDE_RIGHT
        | OBJ_BACK_RIGHT
        | OBJ_TOP_FRONT_RIGHT
        | OBJ_TOP_BACK_RIGHT
        | OBJ_BOTTOM_FRONT_RIGHT
        | OBJ_BOTTOM_BACK_RIGHT => (0.0, SQRT_HALF),
        _ => (0.0, 0.0),
    }
}

/// Downmixes a slice of `(AudioObjectType, mono_f32_samples)` static bed
/// channels into `out_fl` and `out_fr`. When `fold_mono` is true, `0.5 * (L + R)`
/// is written to both outputs so single-channel outputs never lose a side.
pub fn mix_frame_buffers(
    sources: &[(u32, &[f32])],
    out_fl: &mut [f32],
    out_fr: &mut [f32],
    fold_mono: bool,
) {
    let frames = out_fl.len().min(out_fr.len());
    out_fl[..frames].fill(0.0);
    out_fr[..frames].fill(0.0);

    for &(obj_type, src) in sources {
        let (gl, gr) = downmix_gains(obj_type);
        if gl == 0.0 && gr == 0.0 {
            continue;
        }
        let n = frames.min(src.len());
        if gl != 0.0 && gr != 0.0 {
            for i in 0..n {
                let s = src[i];
                out_fl[i] += s * gl;
                out_fr[i] += s * gr;
            }
        } else if gl != 0.0 {
            for i in 0..n {
                out_fl[i] += src[i] * gl;
            }
        } else {
            for i in 0..n {
                out_fr[i] += src[i] * gr;
            }
        }
    }

    if fold_mono {
        for i in 0..frames {
            let m = (0.5 * (out_fl[i] + out_fr[i])).clamp(-1.0, 1.0);
            out_fl[i] = m;
            out_fr[i] = m;
        }
    } else {
        for i in 0..frames {
            out_fl[i] = out_fl[i].clamp(-1.0, 1.0);
            out_fr[i] = out_fr[i].clamp(-1.0, 1.0);
        }
    }
}

/// Recomputes `WAVEFORMATEX` channel fields for 2-channel output.
/// Returns `(nChannels, nBlockAlign, nAvgBytesPerSec)`.
pub fn stereo_format_fields(bits_per_sample: u16, samples_per_sec: u32) -> (u16, u16, u32) {
    let channels = 2u16;
    let block_align = channels.saturating_mul(bits_per_sample / 8);
    let avg_bytes = samples_per_sec.saturating_mul(u32::from(block_align));
    (channels, block_align, avg_bytes)
}

#[cfg(windows)]
mod com {
    #![allow(non_snake_case)]

    use std::ffi::c_void;
    use std::sync::{Arc, Mutex};

    use windows::Win32::Foundation::{E_NOTIMPL, E_POINTER, HANDLE};
    use windows::Win32::Media::Audio::{
        AUDCLNT_SHAREMODE, AudioObjectType, AudioObjectType_Dynamic, AudioObjectType_FrontLeft,
        AudioObjectType_FrontRight, AudioObjectType_None, DEVICE_STATE, EDataFlow, ERole,
        IAudioClient, IAudioClient_Impl, IAudioFormatEnumerator, IMMDevice, IMMDevice_Impl,
        IMMDeviceCollection, IMMDeviceCollection_Impl, IMMDeviceEnumerator,
        IMMDeviceEnumerator_Impl, IMMNotificationClient, ISpatialAudioClient,
        ISpatialAudioClient_Impl, ISpatialAudioObject, ISpatialAudioObject_Impl,
        ISpatialAudioObjectBase_Impl, ISpatialAudioObjectRenderStream,
        ISpatialAudioObjectRenderStream_Impl, ISpatialAudioObjectRenderStreamBase_Impl,
        MMDeviceEnumerator, SPTLAUDCLNT_E_NO_MORE_OBJECTS, SPTLAUDCLNT_E_OBJECT_ALREADY_ACTIVE,
        SPTLAUDCLNT_E_OUT_OF_ORDER, SPTLAUDCLNT_E_STATIC_OBJECT_NOT_AVAILABLE,
        SpatialAudioObjectRenderStreamActivationParams, WAVEFORMATEX, WAVEFORMATEXTENSIBLE,
    };
    use windows::Win32::System::Com::StructuredStorage::PROPVARIANT;
    use windows::Win32::System::Com::{CLSCTX, CLSCTX_INPROC_SERVER, CoTaskMemFree, STGM};
    use windows::Win32::UI::Shell::PropertiesSystem::IPropertyStore;
    use windows::core::{BOOL, GUID, HRESULT, Interface, PCWSTR, PWSTR, Ref, Result, implement};

    use super::{
        DownmixMode, STEREO_MASK, mix_frame_buffers, mode, should_fold_mono, should_spoof_stereo,
        should_wrap_stream, stereo_format_fields,
    };
    use crate::modules::files::host;
    use crate::{HleModule, hle_export};

    const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
    const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;
    const SPEAKER_STEREO: u32 = 0x3;
    const VT_BLOB: u16 = 0x41;
    const KSDATAFORMAT_SUBTYPE_IEEE_FLOAT: GUID =
        GUID::from_u128(0x00000003_0000_0010_8000_00aa00389b71);

    #[repr(C)]
    struct BlobPropVariant {
        vt: u16,
        reserved: [u16; 3],
        size: u32,
        data: *const SpatialAudioObjectRenderStreamActivationParams,
    }

    fn is_mono_f32_format(fmt: *const WAVEFORMATEX) -> bool {
        if fmt.is_null() {
            return false;
        }
        // SAFETY: caller passed a valid WAVEFORMATEX pointer for activation.
        let base = unsafe { std::ptr::read_unaligned(fmt) };
        if base.nChannels != 1 || base.wBitsPerSample != 32 {
            return false;
        }
        if base.wFormatTag == WAVE_FORMAT_IEEE_FLOAT {
            return true;
        }
        if base.wFormatTag == WAVE_FORMAT_EXTENSIBLE && base.cbSize >= 22 {
            // SAFETY: cbSize >= 22 confirms a full WAVEFORMATEXTENSIBLE.
            let ext = unsafe { std::ptr::read_unaligned(fmt.cast::<WAVEFORMATEXTENSIBLE>()) };
            let subformat = ext.SubFormat;
            return subformat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
        }
        false
    }

    fn query_real_device_channels(device: &IMMDevice) -> Option<u16> {
        let mut raw_client: *mut c_void = std::ptr::null_mut();
        // SAFETY: COM call on live IMMDevice; result is released below.
        let hr = unsafe {
            (Interface::vtable(device).Activate)(
                device.as_raw(),
                &IAudioClient::IID,
                CLSCTX_INPROC_SERVER,
                std::ptr::null(),
                &mut raw_client,
            )
        };
        if hr.is_err() || raw_client.is_null() {
            return None;
        }
        // SAFETY: Activate succeeded and returned an owned IAudioClient.
        let client = unsafe { IAudioClient::from_raw(raw_client) };
        // SAFETY: GetMixFormat returns a CoTaskMemAlloc'd WAVEFORMATEX.
        let Ok(wfx) = (unsafe { client.GetMixFormat() }) else {
            return None;
        };
        if wfx.is_null() {
            return None;
        }
        // SAFETY: non-null WAVEFORMATEX returned by GetMixFormat.
        let ch = unsafe { std::ptr::read_unaligned(wfx).nChannels };
        // SAFETY: allocated by COM via GetMixFormat.
        unsafe { CoTaskMemFree(Some(wfx.cast())) };
        Some(ch)
    }

    #[implement(IMMDeviceEnumerator)]
    struct WrappedDeviceEnumerator {
        inner: IMMDeviceEnumerator,
        mode: DownmixMode,
    }

    impl IMMDeviceEnumerator_Impl for WrappedDeviceEnumerator_Impl {
        fn EnumAudioEndpoints(
            &self,
            dataflow: EDataFlow,
            dwstatemask: DEVICE_STATE,
        ) -> Result<IMMDeviceCollection> {
            // SAFETY: forwarding to live IMMDeviceEnumerator.
            let inner = unsafe { self.inner.EnumAudioEndpoints(dataflow, dwstatemask)? };
            Ok(WrappedDeviceCollection {
                inner,
                mode: self.mode,
            }
            .into())
        }

        fn GetDefaultAudioEndpoint(&self, dataflow: EDataFlow, role: ERole) -> Result<IMMDevice> {
            // SAFETY: forwarding to live IMMDeviceEnumerator.
            let inner = unsafe { self.inner.GetDefaultAudioEndpoint(dataflow, role)? };
            Ok(WrappedDevice {
                inner,
                mode: self.mode,
            }
            .into())
        }

        fn GetDevice(&self, pwstrid: &PCWSTR) -> Result<IMMDevice> {
            // SAFETY: forwarding to live IMMDeviceEnumerator.
            let inner = unsafe { self.inner.GetDevice(*pwstrid)? };
            Ok(WrappedDevice {
                inner,
                mode: self.mode,
            }
            .into())
        }

        fn RegisterEndpointNotificationCallback(
            &self,
            pclient: Ref<IMMNotificationClient>,
        ) -> Result<()> {
            // SAFETY: forwarding to live IMMDeviceEnumerator.
            unsafe { self.inner.RegisterEndpointNotificationCallback(pclient.as_ref()) }
        }

        fn UnregisterEndpointNotificationCallback(
            &self,
            pclient: Ref<IMMNotificationClient>,
        ) -> Result<()> {
            // SAFETY: forwarding to live IMMDeviceEnumerator.
            unsafe {
                self.inner
                    .UnregisterEndpointNotificationCallback(pclient.as_ref())
            }
        }
    }

    #[implement(IMMDeviceCollection)]
    struct WrappedDeviceCollection {
        inner: IMMDeviceCollection,
        mode: DownmixMode,
    }

    impl IMMDeviceCollection_Impl for WrappedDeviceCollection_Impl {
        fn GetCount(&self) -> Result<u32> {
            // SAFETY: forwarding to live IMMDeviceCollection.
            unsafe { self.inner.GetCount() }
        }

        fn Item(&self, ndevice: u32) -> Result<IMMDevice> {
            // SAFETY: forwarding to live IMMDeviceCollection.
            let inner = unsafe { self.inner.Item(ndevice)? };
            Ok(WrappedDevice {
                inner,
                mode: self.mode,
            }
            .into())
        }
    }

    #[implement(IMMDevice)]
    struct WrappedDevice {
        inner: IMMDevice,
        mode: DownmixMode,
    }

    impl IMMDevice_Impl for WrappedDevice_Impl {
        fn Activate(
            &self,
            iid: *const GUID,
            dwclsctx: CLSCTX,
            pactivationparams: *const PROPVARIANT,
            ppinterface: *mut *mut c_void,
        ) -> Result<()> {
            if iid.is_null() || ppinterface.is_null() {
                return Err(E_POINTER.into());
            }
            // SAFETY: forwarding to live IMMDevice vtable.
            unsafe {
                (Interface::vtable(&self.inner).Activate)(
                    self.inner.as_raw(),
                    iid,
                    dwclsctx,
                    pactivationparams,
                    ppinterface,
                )
                .ok()?;
                let raw = *ppinterface;
                if raw.is_null() {
                    return Ok(());
                }
                if *iid == IAudioClient::IID {
                    let inner = IAudioClient::from_raw(raw);
                    let wrapped: IAudioClient = WrappedAudioClient {
                        inner,
                        mode: self.mode,
                    }
                    .into();
                    *ppinterface = wrapped.into_raw();
                } else if *iid == ISpatialAudioClient::IID {
                    let inner = ISpatialAudioClient::from_raw(raw);
                    let real_channels = query_real_device_channels(&self.inner).unwrap_or(2);
                    let wrapped: ISpatialAudioClient = WrappedSpatialAudioClient {
                        inner,
                        real_channels,
                        mode: self.mode,
                    }
                    .into();
                    *ppinterface = wrapped.into_raw();
                }
            }
            Ok(())
        }

        fn OpenPropertyStore(&self, stgmaccess: STGM) -> Result<IPropertyStore> {
            // SAFETY: forwarding to live IMMDevice.
            unsafe { self.inner.OpenPropertyStore(stgmaccess) }
        }

        fn GetId(&self) -> Result<PWSTR> {
            // SAFETY: forwarding to live IMMDevice.
            unsafe { self.inner.GetId() }
        }

        fn GetState(&self) -> Result<DEVICE_STATE> {
            // SAFETY: forwarding to live IMMDevice.
            unsafe { self.inner.GetState() }
        }
    }

    #[implement(IAudioClient)]
    struct WrappedAudioClient {
        inner: IAudioClient,
        mode: DownmixMode,
    }

    impl IAudioClient_Impl for WrappedAudioClient_Impl {
        fn Initialize(
            &self,
            sharemode: AUDCLNT_SHAREMODE,
            streamflags: u32,
            hnsbufferduration: i64,
            hnsperiodicity: i64,
            pformat: *const WAVEFORMATEX,
            audiosessionguid: *const GUID,
        ) -> Result<()> {
            // SAFETY: forwarding to live IAudioClient.
            unsafe {
                self.inner.Initialize(
                    sharemode,
                    streamflags,
                    hnsbufferduration,
                    hnsperiodicity,
                    pformat,
                    (!audiosessionguid.is_null()).then_some(audiosessionguid),
                )
            }
        }

        fn GetBufferSize(&self) -> Result<u32> {
            // SAFETY: forwarding to live IAudioClient.
            unsafe { self.inner.GetBufferSize() }
        }

        fn GetStreamLatency(&self) -> Result<i64> {
            // SAFETY: forwarding to live IAudioClient.
            unsafe { self.inner.GetStreamLatency() }
        }

        fn GetCurrentPadding(&self) -> Result<u32> {
            // SAFETY: forwarding to live IAudioClient.
            unsafe { self.inner.GetCurrentPadding() }
        }

        fn IsFormatSupported(
            &self,
            sharemode: AUDCLNT_SHAREMODE,
            pformat: *const WAVEFORMATEX,
            ppclosestmatch: *mut *mut WAVEFORMATEX,
        ) -> HRESULT {
            // SAFETY: forwarding to live IAudioClient.
            unsafe {
                self.inner.IsFormatSupported(
                    sharemode,
                    pformat,
                    (!ppclosestmatch.is_null()).then_some(ppclosestmatch),
                )
            }
        }

        fn GetMixFormat(&self) -> Result<*mut WAVEFORMATEX> {
            // SAFETY: forwarding to live IAudioClient; mutates CoTaskMemAlloc'd struct in place.
            unsafe {
                let ptr = self.inner.GetMixFormat()?;
                if ptr.is_null() {
                    return Ok(ptr);
                }
                let mut base = std::ptr::read_unaligned(ptr);
                let real_channels = base.nChannels;
                let real_rate = base.nSamplesPerSec;
                let real_bits = base.wBitsPerSample;
                let real_mask = if base.wFormatTag == WAVE_FORMAT_EXTENSIBLE && base.cbSize >= 22 {
                    std::ptr::read_unaligned(ptr.cast::<WAVEFORMATEXTENSIBLE>()).dwChannelMask
                } else {
                    0
                };
                let spoof = should_spoof_stereo(real_channels, self.mode);
                tracing::info!(
                    real_channels,
                    real_rate,
                    real_bits,
                    real_mask = format_args!("{real_mask:#x}"),
                    spoof_stereo = spoof,
                    "audio downmix: host WASAPI mix format"
                );
                if spoof {
                    let (ch, align, avg) = stereo_format_fields(real_bits, real_rate);
                    base.nChannels = ch;
                    base.nBlockAlign = align;
                    base.nAvgBytesPerSec = avg;
                    if base.wFormatTag == WAVE_FORMAT_EXTENSIBLE && base.cbSize >= 22 {
                        let mut ext = std::ptr::read_unaligned(ptr.cast::<WAVEFORMATEXTENSIBLE>());
                        ext.Format = base;
                        ext.dwChannelMask = SPEAKER_STEREO;
                        std::ptr::write_unaligned(ptr.cast::<WAVEFORMATEXTENSIBLE>(), ext);
                    } else {
                        std::ptr::write_unaligned(ptr, base);
                    }
                }
                Ok(ptr)
            }
        }

        fn GetDevicePeriod(
            &self,
            phnsdefaultdeviceperiod: *mut i64,
            phnsminimumdeviceperiod: *mut i64,
        ) -> Result<()> {
            // SAFETY: forwarding to live IAudioClient.
            unsafe {
                self.inner.GetDevicePeriod(
                    (!phnsdefaultdeviceperiod.is_null()).then_some(phnsdefaultdeviceperiod),
                    (!phnsminimumdeviceperiod.is_null()).then_some(phnsminimumdeviceperiod),
                )
            }
        }

        fn Start(&self) -> Result<()> {
            // SAFETY: forwarding to live IAudioClient.
            unsafe { self.inner.Start() }
        }

        fn Stop(&self) -> Result<()> {
            // SAFETY: forwarding to live IAudioClient.
            unsafe { self.inner.Stop() }
        }

        fn Reset(&self) -> Result<()> {
            // SAFETY: forwarding to live IAudioClient.
            unsafe { self.inner.Reset() }
        }

        fn SetEventHandle(&self, eventhandle: HANDLE) -> Result<()> {
            // SAFETY: forwarding to live IAudioClient.
            unsafe { self.inner.SetEventHandle(eventhandle) }
        }

        fn GetService(&self, riid: *const GUID, ppv: *mut *mut c_void) -> Result<()> {
            // SAFETY: forwarding to live IAudioClient vtable.
            unsafe { (Interface::vtable(&self.inner).GetService)(self.inner.as_raw(), riid, ppv).ok() }
        }
    }

    #[implement(ISpatialAudioClient)]
    struct WrappedSpatialAudioClient {
        inner: ISpatialAudioClient,
        real_channels: u16,
        mode: DownmixMode,
    }

    impl ISpatialAudioClient_Impl for WrappedSpatialAudioClient_Impl {
        fn GetStaticObjectPosition(
            &self,
            r#type: AudioObjectType,
            x: *mut f32,
            y: *mut f32,
            z: *mut f32,
        ) -> Result<()> {
            // SAFETY: forwarding to live ISpatialAudioClient.
            unsafe { self.inner.GetStaticObjectPosition(r#type, x, y, z) }
        }

        fn GetNativeStaticObjectTypeMask(&self) -> Result<AudioObjectType> {
            // SAFETY: forwarding to live ISpatialAudioClient.
            unsafe { self.inner.GetNativeStaticObjectTypeMask() }
        }

        fn GetMaxDynamicObjectCount(&self) -> Result<u32> {
            // SAFETY: forwarding to live ISpatialAudioClient.
            unsafe { self.inner.GetMaxDynamicObjectCount() }
        }

        fn GetSupportedAudioObjectFormatEnumerator(&self) -> Result<IAudioFormatEnumerator> {
            // SAFETY: forwarding to live ISpatialAudioClient.
            unsafe { self.inner.GetSupportedAudioObjectFormatEnumerator() }
        }

        fn GetMaxFrameCount(&self, objectformat: *const WAVEFORMATEX) -> Result<u32> {
            // SAFETY: forwarding to live ISpatialAudioClient.
            unsafe { self.inner.GetMaxFrameCount(objectformat) }
        }

        fn IsAudioObjectFormatSupported(&self, objectformat: *const WAVEFORMATEX) -> Result<()> {
            // SAFETY: forwarding to live ISpatialAudioClient.
            unsafe { self.inner.IsAudioObjectFormatSupported(objectformat) }
        }

        fn IsSpatialAudioStreamAvailable(
            &self,
            streamuuid: *const GUID,
            auxiliaryinfo: *const PROPVARIANT,
        ) -> Result<()> {
            // SAFETY: forwarding to live ISpatialAudioClient.
            unsafe {
                self.inner.IsSpatialAudioStreamAvailable(
                    streamuuid,
                    (!auxiliaryinfo.is_null()).then_some(auxiliaryinfo),
                )
            }
        }

        fn ActivateSpatialAudioStream(
            &self,
            activationparams: *const PROPVARIANT,
            riid: *const GUID,
            stream: *mut *mut c_void,
        ) -> Result<()> {
            if riid.is_null() || stream.is_null() {
                return Err(E_POINTER.into());
            }
            // SAFETY: inspects VT_BLOB SpatialAudioObjectRenderStreamActivationParams
            // when activating ISpatialAudioObjectRenderStream, otherwise forwards verbatim.
            unsafe {
                if *riid == ISpatialAudioObjectRenderStream::IID && !activationparams.is_null() {
                    let blob = std::ptr::read_unaligned(activationparams.cast::<BlobPropVariant>());
                    if blob.vt == VT_BLOB
                        && blob.size as usize
                            == size_of::<SpatialAudioObjectRenderStreamActivationParams>()
                        && !blob.data.is_null()
                    {
                        let params = std::ptr::read_unaligned(blob.data);
                        let req_mask = params.StaticObjectTypeMask.0 as u32;
                        let wrap = should_wrap_stream(req_mask, self.real_channels, self.mode)
                            && is_mono_f32_format(params.ObjectFormat);
                        let fold_mono = should_fold_mono(self.real_channels, self.mode);
                        tracing::info!(
                            requested_mask = format_args!("{req_mask:#x}"),
                            requested_channels = req_mask.count_ones(),
                            real_channels = self.real_channels,
                            wrap,
                            fold_mono,
                            "audio downmix: ActivateSpatialAudioStream"
                        );
                        if wrap {
                            let mut stereo_params = params;
                            stereo_params.StaticObjectTypeMask = AudioObjectType(STEREO_MASK as i32);
                            let stereo_blob = BlobPropVariant {
                                vt: VT_BLOB,
                                reserved: [0; 3],
                                size: size_of::<SpatialAudioObjectRenderStreamActivationParams>()
                                    as u32,
                                data: &stereo_params,
                            };
                            let mut raw_inner: *mut c_void = std::ptr::null_mut();
                            (Interface::vtable(&self.inner).ActivateSpatialAudioStream)(
                                self.inner.as_raw(),
                                (&raw const stereo_blob).cast(),
                                riid,
                                &mut raw_inner,
                            )
                            .ok()?;
                            let inner_stream =
                                ISpatialAudioObjectRenderStream::from_raw(raw_inner);
                            let real_fl = inner_stream
                                .ActivateSpatialAudioObject(AudioObjectType_FrontLeft)?;
                            let real_fr = inner_stream
                                .ActivateSpatialAudioObject(AudioObjectType_FrontRight)?;
                            let wrapped: ISpatialAudioObjectRenderStream = DownmixSpatialStream {
                                inner: inner_stream,
                                real_fl,
                                real_fr,
                                requested_mask: req_mask,
                                fold_mono,
                                state: Arc::new(Mutex::new(StreamState::default())),
                            }
                            .into();
                            *stream = wrapped.into_raw();
                            return Ok(());
                        }
                    }
                }
                (Interface::vtable(&self.inner).ActivateSpatialAudioStream)(
                    self.inner.as_raw(),
                    activationparams,
                    riid,
                    stream,
                )
                .ok()
            }
        }
    }

    struct Slot {
        obj_type: u32,
        buf: Mutex<Vec<f32>>,
    }

    #[derive(Default)]
    struct StreamState {
        in_update: bool,
        update_frames: usize,
        slots: Vec<Arc<Slot>>,
    }

    #[implement(ISpatialAudioObjectRenderStream)]
    struct DownmixSpatialStream {
        inner: ISpatialAudioObjectRenderStream,
        real_fl: ISpatialAudioObject,
        real_fr: ISpatialAudioObject,
        requested_mask: u32,
        fold_mono: bool,
        state: Arc<Mutex<StreamState>>,
    }

    impl ISpatialAudioObjectRenderStreamBase_Impl for DownmixSpatialStream_Impl {
        fn GetAvailableDynamicObjectCount(&self) -> Result<u32> {
            Ok(0)
        }

        fn GetService(&self, riid: *const GUID, service: *mut *mut c_void) -> Result<()> {
            if riid.is_null() || service.is_null() {
                return Err(E_POINTER.into());
            }
            // SAFETY: forwarding to live stream vtable.
            unsafe {
                (Interface::vtable(&*self.inner).GetService)(self.inner.as_raw(), riid, service)
                    .ok()
            }
        }

        fn Start(&self) -> Result<()> {
            // SAFETY: forwarding to live ISpatialAudioObjectRenderStream.
            unsafe { self.inner.Start() }
        }

        fn Stop(&self) -> Result<()> {
            // SAFETY: forwarding to live ISpatialAudioObjectRenderStream.
            unsafe { self.inner.Stop() }
        }

        fn Reset(&self) -> Result<()> {
            // SAFETY: forwarding to live ISpatialAudioObjectRenderStream.
            unsafe { self.inner.Reset() }
        }

        fn BeginUpdatingAudioObjects(
            &self,
            availabledynamicobjectcount: *mut u32,
            framecountperbuffer: *mut u32,
        ) -> Result<()> {
            if availabledynamicobjectcount.is_null() || framecountperbuffer.is_null() {
                return Err(E_POINTER.into());
            }
            let mut dyn_count = 0u32;
            let mut frames = 0u32;
            // SAFETY: forwarding to live ISpatialAudioObjectRenderStream.
            unsafe {
                self.inner
                    .BeginUpdatingAudioObjects(&mut dyn_count, &mut frames)?;
            }
            let frame_count = frames as usize;
            {
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                state.in_update = true;
                state.update_frames = frame_count;
                for slot in &state.slots {
                    let mut buf = slot.buf.lock().unwrap_or_else(|e| e.into_inner());
                    buf.resize(frame_count, 0.0);
                    buf.fill(0.0);
                }
            }
            // SAFETY: out pointers verified non-null above.
            unsafe {
                *availabledynamicobjectcount = 0;
                *framecountperbuffer = frames;
            }
            Ok(())
        }

        fn EndUpdatingAudioObjects(&self) -> Result<()> {
            let (frames, slots) = {
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                if !state.in_update {
                    return Err(SPTLAUDCLNT_E_OUT_OF_ORDER.into());
                }
                state.in_update = false;
                (state.update_frames, state.slots.clone())
            };

            if frames > 0 {
                let (mut fl_ptr, mut fl_bytes) = (std::ptr::null_mut::<u8>(), 0u32);
                let (mut fr_ptr, mut fr_bytes) = (std::ptr::null_mut::<u8>(), 0u32);
                // SAFETY: inside Begin/EndUpdatingAudioObjects on self.inner.
                unsafe {
                    self.real_fl.GetBuffer(&mut fl_ptr, &mut fl_bytes)?;
                    self.real_fr.GetBuffer(&mut fr_ptr, &mut fr_bytes)?;
                }
                let out_frames = frames
                    .min((fl_bytes as usize) / size_of::<f32>())
                    .min((fr_bytes as usize) / size_of::<f32>());
                if out_frames > 0 && !fl_ptr.is_null() && !fr_ptr.is_null() {
                    let guards: Vec<_> = slots
                        .iter()
                        .map(|s| (s.obj_type, s.buf.lock().unwrap_or_else(|e| e.into_inner())))
                        .collect();
                    let sources: Vec<(u32, &[f32])> = guards
                        .iter()
                        .map(|(ty, g)| (*ty, &g[..g.len().min(out_frames)]))
                        .collect();
                    // SAFETY: buffers are valid f32 slices of length out_frames
                    // owned by the active spatial audio update pass.
                    unsafe {
                        let out_fl =
                            std::slice::from_raw_parts_mut(fl_ptr.cast::<f32>(), out_frames);
                        let out_fr =
                            std::slice::from_raw_parts_mut(fr_ptr.cast::<f32>(), out_frames);
                        mix_frame_buffers(&sources, out_fl, out_fr, self.fold_mono);
                    }
                }
            }

            // SAFETY: forwarding to live ISpatialAudioObjectRenderStream.
            unsafe { self.inner.EndUpdatingAudioObjects() }
        }
    }

    impl ISpatialAudioObjectRenderStream_Impl for DownmixSpatialStream_Impl {
        fn ActivateSpatialAudioObject(
            &self,
            r#type: AudioObjectType,
        ) -> Result<ISpatialAudioObject> {
            if r#type == AudioObjectType_Dynamic {
                return Err(SPTLAUDCLNT_E_NO_MORE_OBJECTS.into());
            }
            let bits = r#type.0 as u32;
            if r#type != AudioObjectType_None && (bits & !self.requested_mask) != 0 {
                return Err(SPTLAUDCLNT_E_STATIC_OBJECT_NOT_AVAILABLE.into());
            }
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if r#type != AudioObjectType_None
                && state.slots.iter().any(|s| s.obj_type == bits)
            {
                return Err(SPTLAUDCLNT_E_OBJECT_ALREADY_ACTIVE.into());
            }
            let initial_len = state.update_frames.max(480);
            let slot = Arc::new(Slot {
                obj_type: bits,
                buf: Mutex::new(vec![0.0; initial_len]),
            });
            state.slots.push(Arc::clone(&slot));
            Ok(DownmixAudioObject {
                obj_type: r#type,
                stream_state: Arc::clone(&self.state),
                slot,
            }
            .into())
        }
    }

    #[implement(ISpatialAudioObject)]
    struct DownmixAudioObject {
        obj_type: AudioObjectType,
        stream_state: Arc<Mutex<StreamState>>,
        slot: Arc<Slot>,
    }

    impl Drop for DownmixAudioObject {
        fn drop(&mut self) {
            let mut state = self.stream_state.lock().unwrap_or_else(|e| e.into_inner());
            state.slots.retain(|s| !Arc::ptr_eq(s, &self.slot));
        }
    }

    impl ISpatialAudioObjectBase_Impl for DownmixAudioObject_Impl {
        fn GetBuffer(&self, buffer: *mut *mut u8, bufferlength: *mut u32) -> Result<()> {
            if buffer.is_null() || bufferlength.is_null() {
                return Err(E_POINTER.into());
            }
            let frames = {
                let state = self.stream_state.lock().unwrap_or_else(|e| e.into_inner());
                if !state.in_update {
                    return Err(SPTLAUDCLNT_E_OUT_OF_ORDER.into());
                }
                state.update_frames
            };
            let mut buf = self.slot.buf.lock().unwrap_or_else(|e| e.into_inner());
            if buf.len() < frames {
                buf.resize(frames, 0.0);
            }
            let ptr = buf.as_mut_ptr().cast::<u8>();
            // SAFETY: out pointers verified non-null; backing Vec is not
            // reallocated before EndUpdatingAudioObjects.
            unsafe {
                *buffer = ptr;
                *bufferlength = (frames * size_of::<f32>()) as u32;
            }
            Ok(())
        }

        fn SetEndOfStream(&self, _framecount: u32) -> Result<()> {
            Err(E_NOTIMPL.into())
        }

        fn IsActive(&self) -> Result<BOOL> {
            Ok(BOOL(1))
        }

        fn GetAudioObjectType(&self) -> Result<AudioObjectType> {
            Ok(self.obj_type)
        }
    }

    impl ISpatialAudioObject_Impl for DownmixAudioObject_Impl {
        fn SetPosition(&self, _x: f32, _y: f32, _z: f32) -> Result<()> {
            Err(E_NOTIMPL.into())
        }

        fn SetVolume(&self, _volume: f32) -> Result<()> {
            Err(E_NOTIMPL.into())
        }
    }

    /// Title `CoCreateInstance` wrapper: passes every class straight to the host
    /// unless `VICTORY_AUDIO_DOWNMIX` is enabled and the title creates
    /// `MMDeviceEnumerator`.
    pub extern "system" fn CoCreateInstance(
        rclsid: *const GUID,
        p_unk_outer: *mut c_void,
        dw_cls_context: u32,
        riid: *const GUID,
        ppv: *mut *mut c_void,
    ) -> i32 {
        let hr = host!(
            "ole32.dll",
            "CoCreateInstance":
                extern "system" fn(*const GUID, *mut c_void, u32, *const GUID, *mut *mut c_void) -> i32
        )(rclsid, p_unk_outer, dw_cls_context, riid, ppv);

        let downmix_mode = mode();
        if downmix_mode == DownmixMode::Off
            || hr < 0
            || rclsid.is_null()
            || riid.is_null()
            || ppv.is_null()
        {
            return hr;
        }
        // SAFETY: host CoCreateInstance succeeded and populated *ppv.
        unsafe {
            let raw = *ppv;
            if raw.is_null() {
                return hr;
            }
            if *rclsid == MMDeviceEnumerator && *riid == IMMDeviceEnumerator::IID {
                let inner = IMMDeviceEnumerator::from_raw(raw);
                let wrapped: IMMDeviceEnumerator = WrappedDeviceEnumerator {
                    inner,
                    mode: downmix_mode,
                }
                .into();
                *ppv = wrapped.into_raw();
            }
        }
        hr
    }

    pub(crate) fn module() -> HleModule {
        HleModule {
            name: "audio_com",
            dlls: &["ole32.dll", "api-ms-win-core-com-*"],
            exports: vec![hle_export!(CoCreateInstance)],
        }
    }
}

#[cfg(windows)]
pub(crate) use com::module;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_mode_is_off() {
        assert_eq!(DownmixMode::parse(None), DownmixMode::Off);
        assert_eq!(DownmixMode::parse(Some("")), DownmixMode::Off);
        assert_eq!(DownmixMode::parse(Some("0")), DownmixMode::Off);
        assert_eq!(DownmixMode::parse(Some("off")), DownmixMode::Off);
        assert_eq!(DownmixMode::parse(Some("1")), DownmixMode::Auto);
        assert_eq!(DownmixMode::parse(Some("auto")), DownmixMode::Auto);
        assert_eq!(DownmixMode::parse(Some("stereo")), DownmixMode::Stereo);
        assert_eq!(DownmixMode::parse(Some("mono")), DownmixMode::Mono);
    }

    #[test]
    fn auto_mode_triggers_on_mono_hfp_and_surround_beds() {
        // 1-channel Bluetooth HFP: spoof stereo mix format AND wrap + fold mono.
        assert!(should_spoof_stereo(1, DownmixMode::Auto));
        assert!(should_wrap_stream(STEREO_MASK, 1, DownmixMode::Auto));
        assert!(should_fold_mono(1, DownmixMode::Auto));

        // 2-channel speakers: do not spoof mix format; wrap only if surround bed requested.
        assert!(!should_spoof_stereo(2, DownmixMode::Auto));
        assert!(!should_wrap_stream(STEREO_MASK, 2, DownmixMode::Auto));
        let bed_51 = STEREO_MASK | OBJ_FRONT_CENTER | OBJ_LOW_FREQUENCY | OBJ_SIDE_LEFT | OBJ_SIDE_RIGHT;
        assert!(should_wrap_stream(bed_51, 2, DownmixMode::Auto));
        assert!(!should_fold_mono(2, DownmixMode::Auto));

        // 6-channel native surround endpoint: pass through untouched in Auto mode.
        assert!(!should_spoof_stereo(6, DownmixMode::Auto));
        assert!(!should_wrap_stream(bed_51, 6, DownmixMode::Auto));
    }

    #[test]
    fn center_dialogue_downmixes_into_both_stereo_channels() {
        let center = [1.0_f32, -0.5, 0.25, 0.0];
        let mut fl = [0.0_f32; 4];
        let mut fr = [0.0_f32; 4];
        mix_frame_buffers(&[(OBJ_FRONT_CENTER, &center)], &mut fl, &mut fr, false);
        for i in 0..4 {
            let expected = center[i] * SQRT_HALF;
            assert!((fl[i] - expected).abs() < 1e-6);
            assert!((fr[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn mono_fold_duplicates_sum_to_both_channels_so_chmap_one_keeps_left_and_center() {
        let fl_in = [0.8_f32, 0.0];
        let fr_in = [0.0_f32, 0.6];
        let fc_in = [0.0_f32, 0.4];
        let mut out_fl = [0.0_f32; 2];
        let mut out_fr = [0.0_f32; 2];
        mix_frame_buffers(
            &[
                (OBJ_FRONT_LEFT, &fl_in),
                (OBJ_FRONT_RIGHT, &fr_in),
                (OBJ_FRONT_CENTER, &fc_in),
            ],
            &mut out_fl,
            &mut out_fr,
            true,
        );
        // Sample 0: L=0.8, R=0.0 -> mono=0.4 on both FL and FR.
        assert!((out_fl[0] - 0.4).abs() < 1e-6);
        assert!((out_fr[0] - 0.4).abs() < 1e-6);
        // Sample 1: L=0.4*sqrt(0.5), R=0.6+0.4*sqrt(0.5) -> mono=0.3 + 0.4*sqrt(0.5).
        let expected_1 = 0.3 + 0.4 * SQRT_HALF;
        assert!((out_fl[1] - expected_1).abs() < 1e-6);
        assert!((out_fr[1] - expected_1).abs() < 1e-6);
    }

    #[test]
    fn surround_714_mix_preserves_panning_and_clamps() {
        let fl = [0.5_f32];
        let fr = [0.2_f32];
        let fc = [1.0_f32];
        let sl = [0.5_f32];
        let tfr = [0.5_f32];
        let mut out_fl = [0.0_f32; 1];
        let mut out_fr = [0.0_f32; 1];
        mix_frame_buffers(
            &[
                (OBJ_FRONT_LEFT, &fl),
                (OBJ_FRONT_RIGHT, &fr),
                (OBJ_FRONT_CENTER, &fc),
                (OBJ_SIDE_LEFT, &sl),
                (OBJ_TOP_FRONT_RIGHT, &tfr),
            ],
            &mut out_fl,
            &mut out_fr,
            false,
        );
        assert_eq!(out_fl[0], 1.0); // 0.5 + 1.5*0.7071 clamped to 1.0
        assert_eq!(out_fr[0], 1.0); // 0.2 + 1.5*0.7071 clamped to 1.0
    }

    #[test]
    fn stereo_format_fields_match_32bit_float() {
        assert_eq!(stereo_format_fields(32, 16_000), (2, 8, 128_000));
        assert_eq!(stereo_format_fields(32, 48_000), (2, 8, 384_000));
    }
}
