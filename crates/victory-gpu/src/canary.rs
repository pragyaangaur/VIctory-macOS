//! A GPU-side progress canary. A fence can complete after D3DMetal starts
//! ignoring a process's submissions, so completion alone does not prove that
//! the GPU ran the commands. A tiny copy with a new value each submission
//! does; it is checked only after the submission's fence has completed.

#[cfg(windows)]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(windows)]
use windows::Win32::Graphics::Direct3D12::*;
#[cfg(windows)]
use windows::core::Interface;

/// A completed fence whose canary did not reach this submission's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Stall {
    pub(crate) expected: u64,
    pub(crate) observed: u64,
}

/// Checks GPU progress, without a wall-clock threshold: slow work that really
/// ran advances the canary just as fast work does.
pub(crate) fn check_progress(expected: u64, observed: u64) -> Result<(), Stall> {
    if observed == expected {
        Ok(())
    } else {
        Err(Stall { expected, observed })
    }
}

/// One changing upload word copied to readback memory after the title's
/// commands. Both buffers stay mapped; submissions are serialized and waited
/// for by [`super::host::Host::execute`].
#[cfg(windows)]
pub(crate) struct GpuCanary {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    commands: Option<ID3D12CommandList>,
    upload: ID3D12Resource,
    readback: ID3D12Resource,
    upload_data: *mut u64,
    readback_data: *mut u64,
    /// Cleared if a submission's completion could not be established; its
    /// upload word must not be reused by a later call to `Host::execute`.
    active: AtomicBool,
}

#[cfg(windows)]
impl GpuCanary {
    const BYTES: u64 = size_of::<u64>() as u64;

    pub(crate) fn new(host: &super::host::Host) -> windows::core::Result<Self> {
        let (upload, upload_data) = host
            .mapped_buffer(Self::BYTES, false)
            .ok_or_else(windows::core::Error::empty)?;
        let (readback, readback_data) = host
            .mapped_buffer(Self::BYTES, true)
            .ok_or_else(windows::core::Error::empty)?;
        // SAFETY: plain D3D12 object creation with valid descriptions.
        let (allocator, list) = unsafe {
            let allocator = host
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            // CreateCommandList starts in the recording state; close it so
            // the first `record` can reset it like every later submission.
            list.Close()?;
            (allocator, list)
        };
        let commands = list.cast()?;
        super::host::name(&upload, || "GPU progress canary upload".into());
        super::host::name(&readback, || "GPU progress canary readback".into());
        Ok(Self {
            allocator,
            list,
            commands: Some(commands),
            upload,
            readback,
            upload_data: upload_data.cast(),
            readback_data: readback_data.cast(),
            active: AtomicBool::new(true),
        })
    }

    pub(crate) fn active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    pub(crate) fn disable(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    /// Records the copy for `value`. The previous canary has finished before
    /// this is called, so its upload word can be changed and allocator reset.
    pub(crate) fn record(&self, value: u64) -> windows::core::Result<()> {
        // SAFETY: the upload buffer stays mapped for the host's lifetime.
        unsafe { self.upload_data.write_volatile(value) };
        // SAFETY: the previous use finished; the buffers are live and the
        // eight-byte region lies at offset zero in both.
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            self.list
                .CopyBufferRegion(&self.readback, 0, &self.upload, 0, Self::BYTES);
            self.list.Close()?;
        }
        Ok(())
    }

    /// Runs after the title's command lists and before the fence signal.
    pub(crate) fn execute(&self, queue: &ID3D12CommandQueue) {
        // SAFETY: a closed command list and our live queue.
        unsafe { queue.ExecuteCommandLists(std::slice::from_ref(&self.commands)) }
    }

    /// Reads the canary after its fence completed.
    pub(crate) fn check(&self, expected: u64) -> Result<(), Stall> {
        // SAFETY: the readback buffer stays mapped for the host's lifetime;
        // the completed fence orders the GPU's copy before this read.
        let observed = unsafe { self.readback_data.read_volatile() };
        check_progress(expected, observed)
    }
}

#[cfg(test)]
mod tests {
    use super::check_progress;
    use std::time::Duration;

    /// The issue's essential behaviour: a fence completes whether or not the
    /// host ran the submitted work. No real GPU or hang is involved.
    #[derive(Default)]
    struct SyntheticGpu {
        fence: u64,
        readback: u64,
        ignore_submissions: bool,
        elapsed: Duration,
    }

    impl SyntheticGpu {
        fn submit(&mut self, value: u64, wall_time: Duration) {
            self.elapsed += wall_time;
            self.fence = value;
            if !self.ignore_submissions {
                self.readback = value;
            }
        }
    }

    #[test]
    fn synthetic_ignored_submissions_fail_despite_completed_fences() {
        let mut gpu = SyntheticGpu::default();
        for value in 1..=3 {
            gpu.submit(value, Duration::from_millis(1));
            assert_eq!(gpu.fence, value);
            assert_eq!(check_progress(value, gpu.readback), Ok(()));
        }

        gpu.ignore_submissions = true;
        for value in 4..=6 {
            gpu.submit(value, Duration::from_millis(1));
            assert_eq!(gpu.fence, value, "the fence still completes");
            assert_eq!(
                check_progress(value, gpu.readback),
                Err(super::Stall {
                    expected: value,
                    observed: 3,
                })
            );
        }
    }

    #[test]
    fn slow_but_completed_work_is_not_a_stall() {
        let mut gpu = SyntheticGpu::default();
        for value in 1..=4 {
            // Synthetic time only: the detector has no timeout to trip.
            gpu.submit(value, Duration::from_secs(120));
            assert_eq!(gpu.fence, value);
            assert_eq!(check_progress(value, gpu.readback), Ok(()));
        }
        assert_eq!(gpu.elapsed, Duration::from_secs(480));
    }
}
