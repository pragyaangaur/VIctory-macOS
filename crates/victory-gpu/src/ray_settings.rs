//! Startup policy for the BVH builder and indirect lighting.

use std::ffi::OsStr;
use std::sync::OnceLock;

fn builder_opted_in(value: Option<&OsStr>) -> bool {
    value == Some(OsStr::new("1"))
}

// On native hosts only the policy tests use this module.
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn builder_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| builder_opted_in(std::env::var_os("VICTORY_BVH_BUILDER").as_deref()))
}

pub(crate) fn builder_runs(enabled: bool, ready: bool) -> bool {
    enabled && ready
}

pub(crate) struct Ambient {
    pub(crate) enabled: bool,
    pub(crate) estimate_share: Option<f32>,
}

fn ambient_settings(value: Option<&str>, builder_enabled: bool) -> Ambient {
    let value = value.or(if builder_enabled {
        None
    } else {
        Some("estimate")
    });
    Ambient {
        enabled: crate::ambient_set::is_on(value),
        estimate_share: crate::ambient_estimate_set::share(value),
    }
}

#[cfg(windows)]
pub(crate) fn ambient() -> &'static Ambient {
    static AMBIENT: OnceLock<Ambient> = OnceLock::new();
    AMBIENT.get_or_init(|| {
        let value = std::env::var_os("VICTORY_AMBIENT");
        let value = value.as_deref().map(|v| v.to_str().unwrap_or_default());
        ambient_settings(value, builder_enabled())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_builder_requires_an_exact_opt_in() {
        assert!(!builder_opted_in(None));
        for value in ["", "0", "off", "on", "yes", " 1", "1 ", "01"] {
            assert!(!builder_opted_in(Some(OsStr::new(value))), "{value:?}");
        }
        assert!(builder_opted_in(Some(OsStr::new("1"))));
    }

    #[test]
    fn the_builder_needs_both_opt_in_and_ready_pipelines() {
        assert!(!builder_runs(false, true));
        assert!(!builder_runs(false, false));
        assert!(!builder_runs(true, false));
        assert!(builder_runs(true, true));
    }

    #[test]
    fn unset_ambient_uses_the_estimate_only_with_the_builder_disabled() {
        let fallback = ambient_settings(None, false);
        assert!(fallback.enabled);
        assert_eq!(fallback.estimate_share, Some(1.0));

        let title = ambient_settings(None, true);
        assert!(!title.enabled);
        assert_eq!(title.estimate_share, None);
    }

    #[test]
    fn explicit_ambient_keeps_its_meaning() {
        for builder_enabled in [false, true] {
            for (value, enabled, share) in [
                ("", false, None),
                ("0", false, None),
                ("off", false, None),
                ("1", true, None),
                ("estimate", true, Some(1.0)),
                ("estimate:0.25", true, Some(0.25)),
            ] {
                let settings = ambient_settings(Some(value), builder_enabled);
                assert_eq!(
                    (settings.enabled, settings.estimate_share),
                    (enabled, share),
                    "{value:?}"
                );
            }
        }
    }
}
